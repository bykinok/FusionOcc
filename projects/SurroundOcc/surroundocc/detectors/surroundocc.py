# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import time
import copy
import os
from typing import Optional, Union

try:
    from mmcv.runner import force_fp32, auto_fp16
except ImportError:
    # MMEngine/mmcv 2.x doesn't have mmcv.runner
    # Use mmengine.runner or create dummy decorators
    try:
        from mmengine.runner import autocast
        # Create compatible decorators
        def force_fp32(apply_to=None):
            def decorator(func):
                return func
            return decorator
        def auto_fp16(apply_to=None):
            def decorator(func):
                return func
            return decorator
    except:
        # Fallback: create dummy decorators
        def force_fp32(apply_to=None):
            def decorator(func):
                return func
            return decorator
        def auto_fp16(apply_to=None):
            def decorator(func):
                return func
            return decorator
from mmdet3d.models.detectors import Base3DDetector
from mmdet3d.registry import MODELS as DET3D_MODELS  
from mmengine.registry import MODELS as ENGINE_MODELS
from mmdet3d.structures.det3d_data_sample import SampleList
from mmdet3d.models.data_preprocessors import Det3DDataPreprocessor
from mmengine.model import BaseModule
from mmdet.registry import MODELS as MMDET_MODELS
from mmdet.models.backbones import ResNet
from mmdet.models.necks import FPN
from mmengine.registry import build_from_cfg

from ..modules import GridMask


@DET3D_MODELS.register_module()
@ENGINE_MODELS.register_module()
class SurroundOcc(Base3DDetector):
    """SurroundOcc: Multi-Camera 3D Occupancy Prediction for Autonomous Driving
    
    Args:
        use_grid_mask (bool): Whether to use grid mask for image augmentation.
        data_preprocessor (dict): Config of data preprocessor.
        img_backbone (dict): Config of image backbone.
        img_neck (dict): Config of image neck.
        pts_bbox_head (dict): Config of occupancy head.
        train_cfg (dict): Training config.
        test_cfg (dict): Testing config.
        use_semantic (bool): Whether to use semantic occupancy prediction.
        is_vis (bool): Whether in visualization mode.
    """
    
    def __init__(self,
                 pts_voxel_encoder: Optional[dict] = None,
                 pts_middle_encoder: Optional[dict] = None, 
                 pts_fusion_layer: Optional[dict] = None,
                 img_backbone: Optional[dict] = None,
                 pts_backbone: Optional[dict] = None,
                 img_neck: Optional[dict] = None,
                 pts_neck: Optional[dict] = None,
                 pts_bbox_head: Optional[dict] = None,
                 img_roi_head: Optional[dict] = None,
                 img_rpn_head: Optional[dict] = None,
                 train_cfg: Optional[dict] = None,
                 test_cfg: Optional[dict] = None,
                 use_grid_mask: bool = False,
                 use_semantic: bool = True,
                 is_vis: bool = False,
                 init_cfg: Optional[dict] = None,
                 data_preprocessor: Optional[dict] = None,
                 **kwargs):

        # Use Base3DDetector's initialization
        super(SurroundOcc, self).__init__(
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
        
        # Build image backbone directly
        if img_backbone is not None:
            if img_backbone['type'] == 'ResNet':
                # Create ResNet directly
                backbone_cfg = img_backbone.copy()
                backbone_cfg.pop('type')
                self.img_backbone = ResNet(**backbone_cfg)
            else:
                self.img_backbone = MMDET_MODELS.build(img_backbone)
        else:
            self.img_backbone = None
            
        # Build image neck directly  
        if img_neck is not None:
            if img_neck['type'] == 'FPN':
                # Create FPN directly
                neck_cfg = img_neck.copy()
                neck_cfg.pop('type')
                self.img_neck = FPN(**neck_cfg)
            else:
                self.img_neck = MMDET_MODELS.build(img_neck)
        else:
            self.img_neck = None
            
        # Build occupancy head using mmengine registry
        if pts_bbox_head is not None:
            self.pts_bbox_head = ENGINE_MODELS.build(pts_bbox_head)
        else:
            self.pts_bbox_head = None
        
        # Initialize other attributes
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.use_semantic = use_semantic
        self.is_vis = is_vis
    
    @property
    def with_img_backbone(self):
        """bool: Whether the detector has image backbone."""
        return hasattr(self, 'img_backbone') and self.img_backbone is not None
    
    @property  
    def with_img_neck(self):
        """bool: Whether the detector has image neck."""
        return hasattr(self, 'img_neck') and self.img_neck is not None
    
    @property
    def with_pts_bbox_head(self):
        """bool: Whether the detector has occupancy head."""
        return hasattr(self, 'pts_bbox_head') and self.pts_bbox_head is not None

    # @auto_fp16(apply_to=('img',))
    def extract_feat(self, img, img_metas=None, len_queue=None):
        """Extract features of images."""
        # Handle case where img is a dict (MMEngine format)
        if isinstance(img, dict):
            if 'imgs' in img:
                img = img['imgs']
            else:
                # Handle camera-based dict format
                cam_keys = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
                cam_imgs = []
                for cam_key in cam_keys:
                    if cam_key in img:
                        cam_data = img[cam_key]
                        # Extract actual tensor from camera dict if it's a dict
                        if isinstance(cam_data, dict):
                            # Look for common image keys
                            for img_key in ['img', 'image', 'data', 'tensor']:
                                if img_key in cam_data:
                                    cam_imgs.append(cam_data[img_key])
                                    break
                            else:
                                raise ValueError(f"No image tensor found in camera dict {cam_key}, keys: {list(cam_data.keys())}")
                        else:
                            cam_imgs.append(cam_data)
                if cam_imgs:
                    import torch
                    img = torch.stack(cam_imgs, dim=0)
                else:
                    raise ValueError(f"Expected 'imgs' key or camera keys in input dict, got keys: {list(img.keys())}")
        
        # Handle case where img is None or a list instead of tensor
        if img is None:
            return None
        elif isinstance(img, list):
            import torch
            img = torch.stack(img, dim=0)

        # breakpoint()
        
        B = img.size(0)
        if img is not None:
            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_(0)
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
            
        if hasattr(self, 'img_neck') and self.img_neck is not None:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(B/len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped


    def loss(self, batch_inputs: dict, batch_data_samples: SampleList, **kwargs) -> dict:
        """Calculate losses from a batch of inputs and data samples."""
        # Handle case where batch_data_samples might be a tuple or other format
        if batch_data_samples is None:
            batch_data_samples = []
        elif isinstance(batch_data_samples, (tuple, list)):
            if len(batch_data_samples) > 0 and not hasattr(batch_data_samples[0], 'metainfo'):
                # If it's nested, flatten it
                flattened_samples = []
                for item in batch_data_samples:
                    if isinstance(item, (list, tuple)):
                        flattened_samples.extend(item)
                    else:
                        flattened_samples.append(item)
                batch_data_samples = flattened_samples
        
        img = batch_inputs.get('imgs', None) if batch_inputs is not None else None
        
        img_metas = []
        for data_sample in batch_data_samples:
            if hasattr(data_sample, 'metainfo'):
                img_metas.append(data_sample.metainfo)
            else:
                img_metas.append({})

        # breakpoint()

        img_feats = self.extract_feat(img, img_metas)
        
        # If no features extracted (img was None), return empty losses
        if img_feats is None:
            import torch
            return dict(loss_occ=torch.tensor(0.0, requires_grad=True))
        
        losses = dict()
        
        # Extract ground truth occupancy
        gt_occ = []
        for data_sample in batch_data_samples:
            if hasattr(data_sample, 'gt_occ'):
                gt_occ.append(data_sample.gt_occ)
            else:
                # Fallback to extract from metainfo
                if hasattr(data_sample, 'metainfo'):
                    gt_occ.append(data_sample.metainfo.get('gt_occ', None))
                else:
                    gt_occ.append(None)
        
        if any(gt is None for gt in gt_occ):
            return dict(loss_occ=torch.tensor(0.0, requires_grad=True))
        
        import torch
        try:
            gt_occ = torch.stack(gt_occ) if not isinstance(gt_occ[0], torch.Tensor) else torch.stack(gt_occ)
        except Exception as e:
            return dict(loss_occ=torch.tensor(0.0, requires_grad=True))
        
        if hasattr(self, 'pts_bbox_head') and self.pts_bbox_head is not None:
            img_metas = [data_sample.metainfo for data_sample in batch_data_samples]
            losses_occ = self.forward_pts_train(img_feats, gt_occ, img_metas)
            losses.update(losses_occ)
        else:
            return dict(loss_occ=torch.tensor(0.0, requires_grad=True))
        
        return losses

    def forward_pts_train(self, pts_feats, gt_occ, img_metas):
        """Forward training function for occupancy head."""
        # breakpoint()
        outs = self.pts_bbox_head(pts_feats, img_metas)
        loss_inputs = [gt_occ, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)
        return losses

    def predict(self, batch_inputs: dict, batch_data_samples: SampleList, **kwargs) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-processing."""
        # Handle case where batch_data_samples might be a tuple or other format
        if batch_data_samples is None:
            batch_data_samples = []
        elif isinstance(batch_data_samples, (tuple, list)):
            if len(batch_data_samples) > 0 and not hasattr(batch_data_samples[0], 'metainfo'):
                # If it's nested, flatten it
                flattened_samples = []
                for item in batch_data_samples:
                    if isinstance(item, (list, tuple)):
                        flattened_samples.extend(item)
                    else:
                        flattened_samples.append(item)
                batch_data_samples = flattened_samples
        
        img_metas = []
        for data_sample in batch_data_samples:
            if hasattr(data_sample, 'metainfo'):
                img_metas.append(data_sample.metainfo)
            else:
                img_metas.append({})
        
        img = batch_inputs.get('imgs', None) if batch_inputs is not None else None
        output = self.simple_test(img_metas, img)
        
        # Handle different output formats from simple_test
        if isinstance(output, list) and len(output) > 0:
            # Check if first element is dict and has occupancy data
            if isinstance(output[0], dict) and 'occupancy' in output[0]:
                if output[0]['occupancy'] is None:
                    # No valid prediction available, return empty data_samples
                    return batch_data_samples
                pred_occ = output[0]['occupancy']
            elif isinstance(output[0], dict) and 'occ_preds' in output[0]:
                pred_occ = output[0]['occ_preds']
            else:
                # Unknown format, return empty data_samples
                return batch_data_samples
        elif isinstance(output, dict) and 'occ_preds' in output:
            pred_occ = output['occ_preds']
        else:
            # No valid prediction available, return empty data_samples
            return batch_data_samples
        if type(pred_occ) == list:
            pred_occ = pred_occ[-1]
        
        if self.is_vis:
            self.generate_output(pred_occ, img_metas)
            return batch_data_samples
        
        # Use the same evaluation approach as the original SurroundOcc
        # Import evaluation function
        from projects.SurroundOcc.surroundocc.evaluation.evaluation_metrics import evaluation_semantic
        
        if self.use_semantic:
            class_num = pred_occ.shape[1]
            pred_softmax = torch.softmax(pred_occ, dim=1)
            _, pred_occ_class = torch.max(pred_softmax, dim=1)
            
            # Collect GT occupancy from data samples
            # Keep as list because sparse GT has different N for each sample
            gt_occ_list = []
            for data_sample in batch_data_samples:
                if hasattr(data_sample, 'gt_occ'):
                    gt_occ = data_sample.gt_occ
                    if isinstance(gt_occ, torch.Tensor):
                        gt_occ_list.append(gt_occ.to(pred_occ.device))
                    elif isinstance(gt_occ, np.ndarray):
                        gt_occ_list.append(torch.from_numpy(gt_occ).to(pred_occ.device))
                    else:
                        # No valid GT, use empty placeholder
                        gt_occ_list.append(torch.zeros((0, 4), dtype=torch.float32, device=pred_occ.device))
                else:
                    # No GT found, use empty placeholder
                    gt_occ_list.append(torch.zeros((0, 4), dtype=torch.float32, device=pred_occ.device))
            
            # Convert list to tensor batch: pad to max length and stack
            # Find max N
            max_n = max([gt.shape[0] for gt in gt_occ_list])
            gt_occ_padded = []
            for gt in gt_occ_list:
                if gt.shape[0] < max_n:
                    # Pad with zeros
                    padding = torch.zeros((max_n - gt.shape[0], 4), dtype=torch.float32, device=pred_occ.device)
                    padding[:, 3] = 255  # Set class to 255 (ignore)
                    gt_padded = torch.cat([gt, padding], dim=0)
                else:
                    gt_padded = gt
                gt_occ_padded.append(gt_padded)
            
            gt_occ = torch.stack(gt_occ_padded, dim=0)
            
            # Use the original evaluation_semantic function to compute metrics
            # This ensures identical evaluation logic to the original SurroundOcc
            eval_results = evaluation_semantic(pred_occ_class, gt_occ, img_metas[0], class_num)
            
            # Store predictions and evaluation results in data samples
            for i, data_sample in enumerate(batch_data_samples):
                if i < len(pred_occ_class):
                    # Store prediction
                    if isinstance(data_sample, dict):
                        data_sample['pred_occ'] = pred_occ_class[i]
                    else:
                        data_sample.pred_occ = pred_occ_class[i]
                    
                    # Store pre-computed evaluation results
                    if i < len(eval_results):
                        if isinstance(data_sample, dict):
                            data_sample['eval_results'] = eval_results[i]
                        else:
                            data_sample.eval_results = eval_results[i]
        else:
            pred_occ_binary = torch.sigmoid(pred_occ[:, 0])
            for i, data_sample in enumerate(batch_data_samples):
                if i < len(pred_occ_binary):
                    data_sample.pred_occ = pred_occ_binary[i]
        
        return batch_data_samples

    def _forward(self, batch_inputs: dict, batch_data_samples: SampleList):
        """Network forward process."""
        img_feats = self.extract_feat(batch_inputs, 
                                    [data_sample.metainfo for data_sample in batch_data_samples])
        img_metas = [data_sample.metainfo for data_sample in batch_data_samples]
        outs = self.pts_bbox_head(img_feats, img_metas)
        return outs

    def simple_test_pts(self, x, img_metas, rescale=False):
        """Test function"""
        outs = self.pts_bbox_head(x, img_metas)
        return outs

    def simple_test(self, img_metas, img=None, rescale=False):
        """Test function without augmentation."""

        # breakpoint()

        img_feats = self.extract_feat(img, img_metas)
        
        # If no features extracted (img was None), return empty output  
        if img_feats is None:
            return [dict(occupancy=None)]

        output = self.simple_test_pts(img_feats, img_metas, rescale=rescale)
        return output

    def generate_output(self, pred_occ, img_metas):
        """Generate visualization output."""
        try:
            import open3d as o3d
        except ImportError:
            print("Open3D not available for visualization")
            return
        
        color_map = np.array([
            [0, 0, 0, 255],
            [255, 120, 50, 255],  # barrier              orangey
            [255, 192, 203, 255],  # bicycle              pink
            [255, 255, 0, 255],  # bus                  yellow
            [0, 150, 245, 255],  # car                  blue
            [0, 255, 255, 255],  # construction_vehicle cyan
            [200, 180, 0, 255],  # motorcycle           dark orange
            [255, 0, 0, 255],  # pedestrian           red
            [255, 240, 150, 255],  # traffic_cone         light yellow
            [135, 60, 0, 255],  # trailer              brown
            [160, 32, 240, 255],  # truck                purple
            [255, 0, 255, 255],  # driveable_surface    dark pink
            [139, 137, 137, 255],
            [75, 0, 75, 255],  # sidewalk             dard purple
            [150, 240, 80, 255],  # terrain              light green
            [230, 230, 250, 255],  # manmade              white
            [0, 175, 0, 255],  # vegetation           green
        ])
        
        if self.use_semantic:
            _, voxel = torch.max(torch.softmax(pred_occ, dim=1), dim=1)
        else:
            voxel = torch.sigmoid(pred_occ[:, 0])
        
        for i in range(voxel.shape[0]):
            x = torch.linspace(0, voxel[i].shape[0] - 1, voxel[i].shape[0])
            y = torch.linspace(0, voxel[i].shape[1] - 1, voxel[i].shape[1])
            z = torch.linspace(0, voxel[i].shape[2] - 1, voxel[i].shape[2])
            X, Y, Z = torch.meshgrid(x, y, z)
            vv = torch.stack([X, Y, Z], dim=-1).to(voxel.device)
        
            vertices = vv[voxel[i] > 0.5]
            vertices[:, 0] = (vertices[:, 0] + 0.5) * (img_metas[i]['pc_range'][3] - img_metas[i]['pc_range'][0]) / img_metas[i]['occ_size'][0] + img_metas[i]['pc_range'][0]
            vertices[:, 1] = (vertices[:, 1] + 0.5) * (img_metas[i]['pc_range'][4] - img_metas[i]['pc_range'][1]) / img_metas[i]['occ_size'][1] + img_metas[i]['pc_range'][1]
            vertices[:, 2] = (vertices[:, 2] + 0.5) * (img_metas[i]['pc_range'][5] - img_metas[i]['pc_range'][2]) / img_metas[i]['occ_size'][2] + img_metas[i]['pc_range'][2]
            
            vertices = vertices.cpu().numpy()
    
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(vertices)
            if self.use_semantic:
                semantics = voxel[i][voxel[i] > 0].cpu().numpy()
                color = color_map[semantics] / 255.0
                pcd.colors = o3d.utility.Vector3dVector(color[..., :3])
                vertices = np.concatenate([vertices, semantics[:, None]], axis=-1)
    
            save_dir = os.path.join('visual_dir', img_metas[i]['occ_path'].replace('.npy', '').split('/')[-1])
            os.makedirs(save_dir, exist_ok=True)

            o3d.io.write_point_cloud(os.path.join(save_dir, 'pred.ply'), pcd)
            np.save(os.path.join(save_dir, 'pred.npy'), vertices)
            for cam_id, cam_path in enumerate(img_metas[i]['filename']):
                os.system('cp {} {}/{}.jpg'.format(cam_path, save_dir, cam_id))

    def forward(self, inputs=None, data_samples=None, mode='tensor', **kwargs):
        """Forward function for SurroundOcc detector.
        
        Args:
            inputs: Input data, typically containing 'imgs'
            data_samples: Data samples containing metadata and ground truth
            mode: Forward mode ('loss', 'predict', 'tensor')
        """
        if mode == 'loss':
            return self.loss(inputs, data_samples, **kwargs)
        elif mode == 'predict':
            return self.predict(inputs, data_samples, **kwargs)
        else:
            # For tensor mode or legacy calls - training mode
            
            if inputs is not None and isinstance(inputs, dict) and 'imgs' in inputs:
                img = inputs['imgs']
            else:
                img = inputs
                
            if data_samples is not None:
                # If we have data_samples with GT, this is training
                has_gt = any(hasattr(sample, 'gt_occ') or 
                           (hasattr(sample, 'metainfo') and 'gt_occ' in sample.metainfo) 
                           for sample in data_samples)
                
                if has_gt and self.training:
                    return self.loss(inputs, data_samples, **kwargs)
                else:
                    return self.predict(inputs, data_samples, **kwargs)
            else:
                # Simple feature extraction
                return self.extract_feat(img)
