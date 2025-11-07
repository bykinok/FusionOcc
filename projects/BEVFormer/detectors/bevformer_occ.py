# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Xiaoyu Tian
# ---------------------------------------------

import torch
try:
    from mmcv.runner import force_fp32, auto_fp16
except ImportError:
    # Fallback for newer versions
    def force_fp32(apply_to=None):
        def decorator(func):
            return func
        return decorator
    def auto_fp16(apply_to=None):
        def decorator(func):
            return func
        return decorator

try:
    from mmdet.models import DETECTORS
except ImportError:
    from mmdet3d.registry import MODELS as DETECTORS

try:
    from mmdet3d.core import bbox3d2result
except ImportError:
    try:
        from mmdet3d.structures import bbox3d2result
    except ImportError:
        # Dummy function if not available
        def bbox3d2result(*args, **kwargs):
            return {}

try:
    from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
except ImportError:
    try:
        from mmdet3d.models.detectors import Base3DDetector as MVXTwoStageDetector
    except ImportError:
        # Use torch.nn.Module as base
        from torch.nn import Module as MVXTwoStageDetector
from projects.BEVFormer.utils.grid_mask import GridMask
import time
import copy
import numpy as np
import mmdet3d
from projects.BEVFormer.utils.bricks import run_time


@DETECTORS.register_module()
class BEVFormerOcc(MVXTwoStageDetector):
    """BEVFormer.
    Args:
        video_test_mode (bool): Decide whether to use temporal information during inference.
    """

    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 video_test_mode=False
                 ):

        super(BEVFormerOcc,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False
        
        # CRITICAL: Ensure img_backbone, img_neck, and pts_bbox_head are set
        # (in case parent class doesn't set them properly)
        if not hasattr(self, 'img_backbone') or self.img_backbone is None:
            if img_backbone is not None:
                from mmdet3d.registry import MODELS
                self.img_backbone = MODELS.build(img_backbone)
        if not hasattr(self, 'img_neck') or self.img_neck is None:
            if img_neck is not None:
                from mmdet3d.registry import MODELS
                self.img_neck = MODELS.build(img_neck)
        if not hasattr(self, 'pts_bbox_head') or self.pts_bbox_head is None:
            if pts_bbox_head is not None:
                from mmdet3d.registry import MODELS
                self.pts_bbox_head = MODELS.build(pts_bbox_head)

        # temporal
        self.video_test_mode = video_test_mode
        self.prev_frame_info = {
            'prev_bev': None,
            'scene_token': None,
            'prev_pos': 0,
            'prev_angle': 0,
        }

    def extract_img_feat(self, img, img_metas, len_queue=None):
        """Extract features of images."""
        B = img.size(0)
        if img is not None:
            # Move image to the same device as the model
            if hasattr(self.img_backbone, 'conv1'):
                device = next(self.img_backbone.conv1.parameters()).device
            else:
                device = next(self.img_backbone.parameters()).device
            img = img.to(device)

            # input_shape = img.shape[-2:]
            # # update real input shape of each single img
            # for img_meta in img_metas:
            #     img_meta.update(input_shape=input_shape)

            # Handle 5D input: (B, N, C, H, W) -> (B*N, C, H, W) for backbone
            if img.dim() == 5:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)

            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(B / len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    @auto_fp16(apply_to=('img'))
    def extract_feat(self, img, img_metas=None, len_queue=None):
        """Extract features from images and points."""

        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)

        return img_feats

    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          voxel_semantics,
                          mask_camera,
                          img_metas,
                          gt_bboxes_ignore=None,
                          prev_bev=None):
        """Forward function'
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
            prev_bev (torch.Tensor, optional): BEV features of previous frame.
        Returns:
            dict: Losses of each branch.
        """

        outs = self.pts_bbox_head(
            pts_feats, img_metas, prev_bev)
        loss_inputs = [voxel_semantics, mask_camera, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)
        return losses

    def forward_dummy(self, img):
        dummy_metas = None
        return self.forward_test(img=img, img_metas=[[dummy_metas]])

    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        # Remove 'mode' from kwargs if present (mmengine compatibility)
        kwargs.pop('mode', None)
        
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    def obtain_history_bev(self, imgs_queue, img_metas_list):
        """Obtain history BEV features iteratively. To save GPU memory, gradients are not calculated.
        """
        self.eval()
        
        # Helper function to extract data from DC objects recursively
        def unwrap_dc(data):
            # Check if it's a DataContainer by class name (handles different import paths)
            if type(data).__name__ == 'DataContainer' or type(data).__name__ == 'DC':
                return unwrap_dc(data.data)
            elif isinstance(data, dict):
                return {k: unwrap_dc(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [unwrap_dc(item) for item in data]
            elif isinstance(data, tuple):
                return tuple(unwrap_dc(item) for item in data)
            else:
                return data
        
        img_metas_list = unwrap_dc(img_metas_list)

        with torch.no_grad():
            prev_bev = None
            bs, len_queue, num_cams, C, H, W = imgs_queue.shape
            imgs_queue = imgs_queue.reshape(bs * len_queue, num_cams, C, H, W)
            
            # Ensure imgs_queue is on the same device as the model
            if hasattr(self, 'img_backbone') and self.img_backbone is not None:
                device = next(self.img_backbone.parameters()).device
                imgs_queue = imgs_queue.to(device)
            
            img_feats_list = self.extract_feat(img=imgs_queue, len_queue=len_queue)
            
            # Normalize img_metas_list structure
            # After unwrap_dc, it could be:
            # 1. A dict {0: meta0, 1: meta1, ...}
            # 2. A list of dicts [{0: meta0, 1: meta1, ...}] (batch dimension)
            # 3. A list of lists [[meta0, meta1, ...]] (legacy format)
            if isinstance(img_metas_list, list) and len(img_metas_list) > 0:
                # Get first batch element
                metas_dict = img_metas_list[0]
            else:
                # Already a dict
                metas_dict = img_metas_list
            
            for i in range(len_queue):
                # Access metadata for frame i
                if isinstance(metas_dict, dict) and i in metas_dict:
                    img_metas = [metas_dict[i]]
                elif isinstance(metas_dict, list):
                    # Legacy format: list of metas
                    img_metas = [metas_dict[i]]
                else:
                    raise ValueError(f"Cannot index img_metas at position {i}, type: {type(metas_dict)}")
                    
                if not img_metas[0]['prev_bev_exists']:
                    prev_bev = None
                # img_feats = self.extract_feat(img=img, img_metas=img_metas)
                img_feats = [each_scale[:, i] for each_scale in img_feats_list]
                
                # Ensure img_feats are on the same device as the model
                if len(img_feats) > 0 and hasattr(self.pts_bbox_head, 'transformer'):
                    device = next(self.pts_bbox_head.transformer.parameters()).device
                    img_feats = [feat.to(device) for feat in img_feats]
                
                prev_bev = self.pts_bbox_head(
                    img_feats, img_metas, prev_bev, only_bev=True)
                
                # Ensure prev_bev is on the correct device for the next iteration
                if prev_bev is not None and hasattr(self.pts_bbox_head, 'transformer'):
                    device = next(self.pts_bbox_head.transformer.parameters()).device
                    prev_bev = prev_bev.to(device)
                    
            self.train()
            return prev_bev

    @auto_fp16(apply_to=('img', 'points'))
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      voxel_semantics=None,
                      mask_lidar=None,
                      mask_camera=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      img_depth=None,
                      img_mask=None,
                      ):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        
        # Handle both list and tensor inputs for img
        if isinstance(img, list):
            # img is a list of DC (DataContainer) or tensors
            import torch
            # Extract data from DC if needed
            img_tensors = []
            for item in img:
                if hasattr(item, 'data'):
                    # DC object
                    img_tensors.append(item.data)
                else:
                    # Already a tensor
                    img_tensors.append(item)
            img = torch.stack(img_tensors)
        
        len_queue = img.size(1)
        prev_img = img[:, :-1, ...]
        img = img[:, -1, ...]

        # Helper function to extract data from DC objects recursively
        def unwrap_dc(data):
            # Check if it's a DataContainer by class name (handles different import paths)
            if type(data).__name__ == 'DataContainer' or type(data).__name__ == 'DC':
                return unwrap_dc(data.data)
            elif isinstance(data, dict):
                return {k: unwrap_dc(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [unwrap_dc(item) for item in data]
            elif isinstance(data, tuple):
                return tuple(unwrap_dc(item) for item in data)
            else:
                return data
        
        img_metas = unwrap_dc(img_metas)

        prev_img_metas = copy.deepcopy(img_metas)
        prev_bev = self.obtain_history_bev(prev_img, prev_img_metas)
        
        # Ensure prev_bev is on the same device as the model
        if prev_bev is not None and hasattr(self.pts_bbox_head, 'transformer'):
            device = next(self.pts_bbox_head.transformer.parameters()).device
            prev_bev = prev_bev.to(device)

        # Normalize img_metas structure
        # After unwrap_dc, it could be:
        # 1. A dict {0: meta0, 1: meta1, ...}
        # 2. A list of dicts [{0: meta0, 1: meta1, ...}] (batch dimension)
        # 3. A list of lists [[meta0, meta1, ...]] (legacy format)
        if isinstance(img_metas, list) and len(img_metas) > 0:
            # Get first batch element
            metas_dict = img_metas[0]
        else:
            # Already a dict
            metas_dict = img_metas
        
        # Get metadata for the current (last) frame
        if isinstance(metas_dict, dict) and (len_queue - 1) in metas_dict:
            img_metas = [metas_dict[len_queue - 1]]
        elif isinstance(metas_dict, list):
            # Legacy format: list of metas
            img_metas = [metas_dict[len_queue - 1]]
        else:
            raise ValueError(f"Cannot index img_metas at position {len_queue - 1}, type: {type(metas_dict)}")
        if not img_metas[0]['prev_bev_exists']:
            prev_bev = None
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        
        # Ensure img_feats are on the same device as the model
        if len(img_feats) > 0 and hasattr(self.pts_bbox_head, 'transformer'):
            device = next(self.pts_bbox_head.transformer.parameters()).device
            img_feats = [feat.to(device) for feat in img_feats]
        
        losses = dict()
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, voxel_semantics, mask_camera, img_metas,
                                            gt_bboxes_ignore, prev_bev)

        losses.update(losses_pts)
        return losses

    def forward_test(self, img_metas,
                     img=None,
                     voxel_semantics=None,
                     mask_lidar=None,
                     mask_camera=None,
                     **kwargs):
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        
        # forward_test expects img as list[Tensor], so wrap if it's a Tensor
        if isinstance(img, torch.Tensor):
            img = [img]
        elif img is None:
            img = [img]
        # else: already a list

        if img_metas[0][0]['scene_token'] != self.prev_frame_info['scene_token']:
            # the first sample of each scene is truncated
            self.prev_frame_info['prev_bev'] = None
        # update idx
        self.prev_frame_info['scene_token'] = img_metas[0][0]['scene_token']

        # do not use temporal information
        if not self.video_test_mode:
            self.prev_frame_info['prev_bev'] = None

        # Get the delta of ego position and angle between two timestamps.
        tmp_pos = copy.deepcopy(img_metas[0][0]['can_bus'][:3])
        tmp_angle = copy.deepcopy(img_metas[0][0]['can_bus'][-1])
        if self.prev_frame_info['prev_bev'] is not None:
            img_metas[0][0]['can_bus'][:3] -= self.prev_frame_info['prev_pos']
            img_metas[0][0]['can_bus'][-1] -= self.prev_frame_info['prev_angle']
        else:
            img_metas[0][0]['can_bus'][-1] = 0
            img_metas[0][0]['can_bus'][:3] = 0

        new_prev_bev, occ_results = self.simple_test(
            img_metas[0], img[0], prev_bev=self.prev_frame_info['prev_bev'], **kwargs)
        # During inference, we save the BEV features and ego motion of each timestamp.

        self.prev_frame_info['prev_pos'] = tmp_pos
        self.prev_frame_info['prev_angle'] = tmp_angle
        self.prev_frame_info['prev_bev'] = new_prev_bev
        return occ_results

    def simple_test_pts(self, x, img_metas, prev_bev=None, rescale=False):
        """Test function"""
        outs = self.pts_bbox_head(x, img_metas, prev_bev=prev_bev, test=True)

        occ = self.pts_bbox_head.get_occ(
            outs, img_metas, rescale=rescale)

        return outs['bev_embed'], occ

    def simple_test(self, img_metas, img=None, prev_bev=None, rescale=False):
        """Test function without augmentaiton."""
        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        # bbox_list = [dict() for i in range(len(img_metas))]
        new_prev_bev, occ = self.simple_test_pts(
            img_feats, img_metas, prev_bev, rescale=rescale)
        # for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
        #     result_dict['pts_bbox'] = pts_bbox
        return new_prev_bev, occ

    def test_step(self, data):
        """Test step function for mmengine runner (simplified without MultiScaleFlipAug3D).
        
        Args:
            data (dict or list): The output of dataloader.
            
        Returns:
            list: List of data samples with predictions and ground truth.
        """
        import numpy as np
        
        # Get device from model parameters
        device = next(self.parameters()).device
        
        # Simple unwrap: dataloader may wrap in list
        if isinstance(data, list):
            if len(data) == 0:
                raise ValueError("Empty data list")
            data = data[0]
        
        if not isinstance(data, dict):
            raise TypeError(f"Expected data to be dict, got {type(data)}")
        
        # Extract data directly from dict (like FusionOcc)
        img = data.get('img', None)
        img_metas = data.get('img_metas', None)
        voxel_semantics = data.get('voxel_semantics', None)
        mask_camera = data.get('mask_camera', None)
        mask_lidar = data.get('mask_lidar', None)
        
        # Recursive unwrap for DataContainer (handle nested structures)
        def unwrap(obj, max_depth=10, current_depth=0):
            # Prevent infinite recursion
            if current_depth > max_depth:
                return obj
            
            # Unwrap DataContainer
            if hasattr(obj, 'data'):
                return unwrap(obj.data, max_depth, current_depth + 1)
            # Recursively unwrap lists (but limit depth)
            elif isinstance(obj, list):
                return [unwrap(item, max_depth, current_depth + 1) for item in obj]
            else:
                return obj
        
        # Unwrap and get originals for metric
        voxel_semantics_orig = unwrap(voxel_semantics) if voxel_semantics is not None else None
        mask_camera_orig = unwrap(mask_camera) if mask_camera is not None else None
        mask_lidar_orig = unwrap(mask_lidar) if mask_lidar is not None else None
        
        # img와 img_metas는 이미 tensor 형태로 올 것임 (DefaultFormatBundle3D 처리됨)
        img = unwrap(img)
        img_metas = unwrap(img_metas)
        
        # Convert img from list of tensors to single tensor if needed
        if img is not None and isinstance(img, list):
            # Check if img is nested list (e.g., [[tensor1, ..., tensor6]])
            if len(img) > 0 and isinstance(img[0], list):
                # Flatten one level: [[tensors]] -> [tensors]
                img = img[0]
            
            # Now img should be a list of tensors from different cameras
            # Stack them into a single tensor [N_cams, C, H, W]
            if len(img) > 0 and isinstance(img[0], torch.Tensor):
                img = torch.stack(img)  # [N_cams, C, H, W]
                # Add batch dimension: [1, N_cams, C, H, W]
                img = img.unsqueeze(0)
        
        # Ensure img_metas is nested list: [[meta]]
        # forward_test expects img_metas as [[dict]], so we need proper nesting
        if img_metas is not None:
            if isinstance(img_metas, dict):
                # Single dict -> [[dict]]
                img_metas = [[img_metas]]
            elif isinstance(img_metas, list):
                if len(img_metas) > 0 and isinstance(img_metas[0], dict):
                    # [dict] -> [[dict]]
                    img_metas = [img_metas]
                # else: already [[dict]] or [[...]]
        
        # Move img to device (should already be tensor from DefaultFormatBundle3D)
        if img is not None and isinstance(img, torch.Tensor):
            img = img.to(device)
        
        # Forward pass
        with torch.no_grad():
            result = self(return_loss=False, rescale=True, img=img, img_metas=img_metas)
        
        # Result format from forward_test: could be tensor or list of tensors
        # Original BEVFormer: result.squeeze(dim=0).cpu().numpy().astype(np.uint8)
        if isinstance(result, list):
            # If result is a list, take the first element (batch size = 1)
            if len(result) > 0:
                result = result[0]
        
        if isinstance(result, torch.Tensor):
            # Squeeze batch dimension if present and convert to numpy uint8
            if result.dim() > 3:  # e.g., [1, H, W, D] -> [H, W, D]
                result = result.squeeze(dim=0)
            result = result.cpu().numpy().astype(np.uint8)
        
        # Format as data_sample for OccupancyMetric
        data_sample = {
            'pred_occ': result
        }
        
        # Add ground truth if available
        if voxel_semantics_orig is not None:
            # Flatten nested list structure if present
            if isinstance(voxel_semantics_orig, list):
                if len(voxel_semantics_orig) > 0:
                    voxel_semantics_orig = voxel_semantics_orig[0]
            
            # Convert to numpy array if needed
            if isinstance(voxel_semantics_orig, torch.Tensor):
                data_sample['voxel_semantics'] = voxel_semantics_orig.cpu().numpy()
            elif isinstance(voxel_semantics_orig, memoryview):
                # Convert memoryview to numpy array
                import numpy as np
                data_sample['voxel_semantics'] = np.array(voxel_semantics_orig)
            else:
                data_sample['voxel_semantics'] = voxel_semantics_orig
        
        if mask_lidar_orig is not None:
            if isinstance(mask_lidar_orig, list):
                if len(mask_lidar_orig) > 0:
                    mask_lidar_orig = mask_lidar_orig[0]
            
            if isinstance(mask_lidar_orig, torch.Tensor):
                data_sample['mask_lidar'] = mask_lidar_orig.cpu().numpy()
            elif isinstance(mask_lidar_orig, memoryview):
                import numpy as np
                data_sample['mask_lidar'] = np.array(mask_lidar_orig)
            else:
                data_sample['mask_lidar'] = mask_lidar_orig
        
        if mask_camera_orig is not None:
            if isinstance(mask_camera_orig, list):
                if len(mask_camera_orig) > 0:
                    mask_camera_orig = mask_camera_orig[0]
            
            if isinstance(mask_camera_orig, torch.Tensor):
                data_sample['mask_camera'] = mask_camera_orig.cpu().numpy()
            elif isinstance(mask_camera_orig, memoryview):
                import numpy as np
                data_sample['mask_camera'] = np.array(mask_camera_orig)
            else:
                data_sample['mask_camera'] = mask_camera_orig
        
        return [data_sample]
    
    def val_step(self, data):
        """Validation step function for mmengine runner.
        
        Args:
            data (dict): The output of dataloader.
            
        Returns:
            list: List of data samples (same as test_step).
        """
        # For validation, we use the same logic as test_step
        return self.test_step(data)
