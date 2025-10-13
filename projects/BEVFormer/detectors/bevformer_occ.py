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
        img = [img] if img is None else img

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
