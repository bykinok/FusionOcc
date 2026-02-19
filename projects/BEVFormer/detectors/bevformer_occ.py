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
    
    # Class variable to track sample index during testing
    _test_sample_idx = 0

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
                 video_test_mode=False,
                 depth_supervision=None,
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

        # Auxiliary depth supervision (same as TPVFormer)
        self.depth_head = None
        self._depth_feature_level = 1
        if depth_supervision and depth_supervision.get('enabled'):
            from mmdet3d.registry import MODELS
            depth_cfg = dict(
                type='AuxiliaryDepthHead',
                in_channels=256,
                grid_config=depth_supervision['grid_config'],
                downsample=depth_supervision['downsample'],
                loss_weight=depth_supervision.get('loss_weight', 0.5),
            )
            self.depth_head = MODELS.build(depth_cfg)
            self._depth_feature_level = depth_supervision.get('feature_level', 1)

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

        #breakpoint()

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

            # CRITICAL: Follow original BEVFormer's squeeze logic for batch_size=1
            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
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
        # CRITICAL: For distributed training (DDP), don't call self.eval() directly
        # Instead, use a context manager to temporarily disable BN statistics updates
        # Save the current training mode
        # breakpoint()

        was_training = self.training
        self.eval()
        
        with torch.no_grad():
            prev_bev = None
            bs, len_queue, num_cams, C, H, W = imgs_queue.shape
            imgs_queue = imgs_queue.reshape(bs * len_queue, num_cams, C, H, W)
            
            # Ensure imgs_queue is on the same device as the model
            if hasattr(self, 'img_backbone') and self.img_backbone is not None:
                device = next(self.img_backbone.parameters()).device
                imgs_queue = imgs_queue.to(device)
            
            img_feats_list = self.extract_feat(img=imgs_queue, len_queue=len_queue)

            # CRITICAL: Follow original BEVFormer's img_metas indexing
            # img_metas_list is a list of lists: [[meta0_cam0, meta0_cam1, ...], [meta1_cam0, meta1_cam1, ...], ...]
            for i in range(len_queue):
                # Get metadata for frame i from the list
                img_metas = [each[i] for each in img_metas_list]

                if not img_metas[0]['prev_bev_exists']:
                    prev_bev = None
                # img_feats = self.extract_feat(img=img, img_metas=img_metas)
                img_feats = [each_scale[:, i] for each_scale in img_feats_list]
                
                # Ensure img_feats are on the same device as the model
                if len(img_feats) > 0 and hasattr(self.pts_bbox_head, 'transformer'):
                    device = next(self.pts_bbox_head.transformer.parameters()).device
                    img_feats = [feat.to(device) for feat in img_feats]
                
                # breakpoint()
                prev_bev = self.pts_bbox_head(
                    img_feats, img_metas, prev_bev, only_bev=True)
                
                # Ensure prev_bev is on the correct device for the next iteration
                if prev_bev is not None and hasattr(self.pts_bbox_head, 'transformer'):
                    device = next(self.pts_bbox_head.transformer.parameters()).device
                    prev_bev = prev_bev.to(device)
            
            # Restore the original training mode (no need to explicitly call self.train())
            if was_training:
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
                      gt_depth=None,
                      **kwargs):
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
        
        # CRITICAL: Handle img input format for train mode
        # In train mode, img might come as a list containing DataContainer
        if isinstance(img, list) and len(img) > 0:
            # Unwrap DataContainer if present
            if hasattr(img[0], 'data'):
                img = img[0].data
            else:
                # img is a list of tensors - stack them
                img = torch.stack(img, dim=0) if len(img) > 1 else img[0]
        
        # CRITICAL: Handle different img formats
        # If img is 5D [queue_length, N_cams, C, H, W], add batch dimension
        # Expected format: [B, queue_length, N_cams, C, H, W]
        if img.dim() == 5:
            # Add batch dimension: [queue_length, N_cams, C, H, W] -> [1, queue_length, N_cams, C, H, W]
            img = img.unsqueeze(0)
        
        # CRITICAL: Unwrap img_metas from DataContainer if needed
        if isinstance(img_metas, list) and len(img_metas) > 0 and hasattr(img_metas[0], 'data'):
            img_metas = img_metas[0].data
        
        # CRITICAL: Convert img_metas from dict to list format to match original BEVFormer
        # Original format: img_metas is a list with single dict: [dict]
        # Current format: img_metas might be a dict
        if isinstance(img_metas, dict):
            # Convert dict to list format: [dict]
            img_metas = [img_metas]
        
        # breakpoint()

        # img should now be a tensor [B, queue_length, N_cams, C, H, W]
        len_queue = img.size(1)
        prev_img = img[:, :-1, ...]
        img = img[:, -1, ...]

        prev_img_metas = copy.deepcopy(img_metas)
        prev_bev = self.obtain_history_bev(prev_img, prev_img_metas)
        
        # Ensure prev_bev is on the same device as the model
        if prev_bev is not None and hasattr(self.pts_bbox_head, 'transformer'):
            device = next(self.pts_bbox_head.transformer.parameters()).device
            prev_bev = prev_bev.to(device)

        # CRITICAL: Follow original BEVFormer's img_metas indexing
        # img_metas is a list of lists: [[meta0_cam0, meta0_cam1, ...], [meta1_cam0, meta1_cam1, ...], ...]
        # Get metadata for the current (last) frame
        img_metas = [each[len_queue - 1] for each in img_metas]
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

        # Auxiliary depth supervision (current frame only; gt_depth from BEVFormerPointToMultiViewDepth)
        if self.depth_head is not None and gt_depth is not None:
            if hasattr(gt_depth, 'data'):
                gt_depth = gt_depth.data
            level = self._depth_feature_level
            depth_logits = self.depth_head(img_feats[level])
            if isinstance(gt_depth, (list, tuple)):
                gt_depth = torch.stack([t if torch.is_tensor(t) else torch.as_tensor(t) for t in gt_depth])
            gt_depth = gt_depth.to(depth_logits.device)
            loss_depth = self.depth_head.get_depth_loss(gt_depth, depth_logits)
            losses['loss_depth'] = loss_depth

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
        
        # CRITICAL: img should be wrapped in a list for img[0] to work
        # But if img is already a list (from test_step), keep it as is
        if not isinstance(img, list):
            img = [img] if img is None else [img]

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
        # breakpoint()
        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        # bbox_list = [dict() for i in range(len(img_metas))]
        new_prev_bev, occ = self.simple_test_pts(
            img_feats, img_metas, prev_bev, rescale=rescale)
        # for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
        #     result_dict['pts_bbox'] = pts_bbox
        return new_prev_bev, occ

    def test_step(self, data):
        """Test step function for mmengine runner following original BEVFormer's single_gpu_test logic.
        
        Args:
            data (dict or list): The output of dataloader.
            
        Returns:
            list: List of data samples with predictions and ground truth.
        """
        import numpy as np
        
        # Get current sample index and increment for next call
        sample_idx = self._test_sample_idx
        self._test_sample_idx += 1
        
        # Simple unwrap: dataloader may wrap in list
        if isinstance(data, list):
            if len(data) == 0:
                raise ValueError("Empty data list")
            data = data[0]
        
        if not isinstance(data, dict):
            raise TypeError(f"Expected data to be dict, got {type(data)}")
        
        # Unwrap DataContainer - minimal unwrapping to match original BEVFormer
        def unwrap_dc(obj, depth=0, max_depth=10):
            """Unwrap DataContainer recursively but preserve structure."""
            if depth > max_depth:
                return obj
            if hasattr(obj, 'data'):
                return unwrap_dc(obj.data, depth + 1, max_depth)
            elif isinstance(obj, list):
                return [unwrap_dc(item, depth + 1, max_depth) for item in obj]
            else:
                return obj
        
        # Extract and unwrap all data
        unwrapped_data = {}
        for key, value in data.items():
            unwrapped_data[key] = unwrap_dc(value)
        
        # Separate GT data (for metric) from model input
        voxel_semantics_orig = unwrapped_data.pop('voxel_semantics', None)
        mask_camera_orig = unwrapped_data.pop('mask_camera', None)
        mask_lidar_orig = unwrapped_data.pop('mask_lidar', None)
        
        # Fix img_metas format: forward_test expects [[dict]] but we have [dict]
        if 'img_metas' in unwrapped_data:
            img_metas = unwrapped_data['img_metas']
            if isinstance(img_metas, list) and len(img_metas) > 0:
                if isinstance(img_metas[0], dict):
                    # Convert [dict] -> [[dict]]
                    unwrapped_data['img_metas'] = [img_metas]
        
        # Fix img format: convert [[cam1, cam2, ...]] -> [[1, N_cams, C, H, W]]
        if 'img' in unwrapped_data:
            img = unwrapped_data['img']
            if isinstance(img, list) and len(img) > 0:
                if isinstance(img[0], list) and len(img[0]) > 0:
                    # img = [[cam1, cam2, ...]], stack to [[1, N_cams, C, H, W]]
                    if all(isinstance(t, torch.Tensor) for t in img[0]):
                        stacked = torch.stack(img[0])  # [N_cams, C, H, W]
                        stacked = stacked.unsqueeze(0)  # [1, N_cams, C, H, W]
                        unwrapped_data['img'] = [stacked]  # [[1, N_cams, C, H, W]]
        
        # Forward pass - follow original BEVFormer: model(return_loss=False, rescale=True, **data)
        with torch.no_grad():
            result = self(return_loss=False, rescale=True, **unwrapped_data)
        
        # Follow original BEVFormer's output processing: result.squeeze(dim=0).cpu().numpy().astype(np.uint8)
        if isinstance(result, torch.Tensor):
            result = result.squeeze(dim=0).cpu().numpy().astype(np.uint8)
        elif isinstance(result, list) and len(result) > 0:
            result = result[0]
            if isinstance(result, torch.Tensor):
                result = result.squeeze(dim=0).cpu().numpy().astype(np.uint8)
        
        # Format as data_sample for OccupancyMetric and OccupancyMetricHybrid
        # Use both 'pred_occ' (original BEVFormer) and 'occ_results' (STCOcc metric) for compatibility
        # CRITICAL: occ_results must be a LIST for STCOcc's occupancy_metric.py (line 233-236)
        # because it iterates: for i, id in enumerate(data_id): pred_sem = occ_results[i]
        data_sample = {
            'pred_occ': result,  # For original BEVFormer OccupancyMetric
            'occ_results': [result],  # For STCOcc-based OccupancyMetricHybrid - MUST be list!
            'index': [sample_idx]  # Required by OccupancyMetricHybrid - also as list!
        }
        
        # Add ground truth if available
        def to_numpy(obj):
            """Convert to numpy array."""
            if obj is None:
                return None
            if isinstance(obj, list) and len(obj) > 0:
                obj = obj[0]
            if isinstance(obj, torch.Tensor):
                return obj.cpu().numpy()
            elif isinstance(obj, memoryview):
                return np.array(obj)
            else:
                return obj
        
        if voxel_semantics_orig is not None:
            data_sample['voxel_semantics'] = to_numpy(voxel_semantics_orig)
        if mask_lidar_orig is not None:
            data_sample['mask_lidar'] = to_numpy(mask_lidar_orig)
        if mask_camera_orig is not None:
            data_sample['mask_camera'] = to_numpy(mask_camera_orig)
        
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
