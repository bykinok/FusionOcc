import torch
import collections 
import torch.nn.functional as F

from mmdet3d.registry import MODELS as DETECTORS
from mmengine.model import BaseModel
from .bevdepth import BEVDepth_Base, BEVDet
from mmdet3d.registry import MODELS

import numpy as np
import time
import copy
import subprocess
import sys

# Check and install spconv-cu113 if not installed
def check_and_install_spconv_cu113():
    """Check if spconv-cu113 is installed and install it if not found."""
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'show', 'spconv-cu113'],
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode != 0:
            print("spconv-cu113이 설치되어 있지 않습니다. 설치 중...")
            install_result = subprocess.run(
                [sys.executable, '-m', 'pip', 'install', 'spconv-cu113'],
                capture_output=True,
                text=True,
                check=False
            )
            if install_result.returncode == 0:
                print("spconv-cu113이 성공적으로 설치되었습니다.")
            else:
                print(f"spconv-cu113 설치 중 오류 발생: {install_result.stderr}")
        else:
            print("spconv-cu113이 이미 설치되어 있습니다.")
    except Exception as e:
        print(f"spconv-cu113 확인 중 오류 발생: {e}")

# Execute check on module import
check_and_install_spconv_cu113()

@DETECTORS.register_module(force=True)
class OccNet(BaseModel):
    def __init__(self, 
            loss_cfg=None,
            disable_loss_depth=False,
            empty_idx=0,
            num_cls=17,  # Number of classes (17 for nuScenes-Occupancy, 18 for occ3d)
            occ_fuser=None,
            occ_encoder_backbone=None,
            occ_encoder_neck=None,
            loss_norm=False,
            pts_voxel_encoder=None,
            pts_middle_encoder=None,
            pts_voxel_layer=None,
            # Camera-specific components
            img_backbone=None,
            img_neck=None, 
            img_view_transformer=None,
            # BBox head
            pts_bbox_head=None,
            **kwargs):
        
        super().__init__(**kwargs)
                
        self.loss_cfg = loss_cfg
        self.disable_loss_depth = disable_loss_depth
        self.loss_norm = loss_norm
        
        self.record_time = False
        self.time_stats = collections.defaultdict(list)
        self.empty_idx = empty_idx
        self.num_cls = num_cls  # Store number of classes
        
        # Build core components
        self.occ_encoder_backbone = MODELS.build(occ_encoder_backbone)
        self.occ_encoder_neck = MODELS.build(occ_encoder_neck)
        self.occ_fuser = MODELS.build(occ_fuser) if occ_fuser is not None else None
        self.pts_bbox_head = MODELS.build(pts_bbox_head)
        
        # Build camera components if available
        self.img_backbone = MODELS.build(img_backbone) if img_backbone is not None else None
        self.img_neck = MODELS.build(img_neck) if img_neck is not None else None  
        self.img_view_transformer = MODELS.build(img_view_transformer) if img_view_transformer is not None else None
        
        # Build LiDAR-specific components
        self.pts_voxel_encoder = MODELS.build(pts_voxel_encoder) if pts_voxel_encoder is not None else None
        self.pts_middle_encoder = MODELS.build(pts_middle_encoder) if pts_middle_encoder is not None else None
        self.pts_voxel_layer = pts_voxel_layer  # Config only, not a module
        
        # Set modality flags
        self.with_img_backbone = img_backbone is not None
        self.with_img_neck = img_neck is not None
        self.with_pts_bbox = pts_bbox_head is not None
    
    def image_encoder(self, img):
        """Image encoder (copied from original CONet_ori)."""
        imgs = img
        # Handle different input shapes from new mmdet3d version
        if imgs.dim() == 4:
            # Shape: [B*N, C, H, W] - already flattened
            BN, C, imH, imW = imgs.shape
            # Assume 6 cameras
            N = 6
            B = BN // N
            imgs_flat = imgs
        else:
            # Shape: [B, N, C, H, W] - original format
            B, N, C, imH, imW = imgs.shape
            imgs_flat = imgs.view(B * N, C, imH, imW)
        
        # Ensure images are on the same device as the model
        if hasattr(self.img_backbone, 'parameters'):
            device = next(self.img_backbone.parameters()).device
            if imgs_flat.device != device:
                imgs_flat = imgs_flat.to(device)
        
        backbone_feats = self.img_backbone(imgs_flat)
        if self.with_img_neck:
            x = self.img_neck(backbone_feats)
            if type(x) in [list, tuple]:
                x = x[0]
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)
        
        return {'x': x,
                'img_feats': [x.clone()]}
    
    def occ_encoder(self, x):
        """Encode voxel features through backbone and neck."""
        x = self.occ_encoder_backbone(x)
        x = self.occ_encoder_neck(x)
        return x

    def extract_feat(self, points, img, img_metas):
        """Extract features from images and points (copied from original CONet_ori)."""
        img_voxel_feats = None
        pts_voxel_feats, pts_feats = None, None
        depth, img_feats = None, None
        
        # Check if img is not None and not empty, and has valid content
        # For LiDAR-only mode, img might be None or empty list
        has_valid_img = False
        if img is not None and isinstance(img, (list, tuple)) and len(img) > 0:
            # Check if first element is not None (actual image data)
            if img[0] is not None:
                has_valid_img = True
        
        if has_valid_img:
            img_voxel_feats, depth, img_feats = self.extract_img_feat(img, img_metas)
        
        # Only extract point features if points exist AND point encoders are configured
        # For camera-only mode, pts_voxel_encoder will be None
        if points is not None and self.pts_voxel_encoder is not None:
            pts_voxel_feats, pts_feats = self.extract_pts_feat(points)

        if self.occ_fuser is not None:
            voxel_feats = self.occ_fuser(img_voxel_feats, pts_voxel_feats)
        else:
            assert (img_voxel_feats is None) or (pts_voxel_feats is None)
            voxel_feats = img_voxel_feats if pts_voxel_feats is None else pts_voxel_feats

        voxel_feats_enc = self.occ_encoder(voxel_feats)
        if type(voxel_feats_enc) is not list:
            voxel_feats_enc = [voxel_feats_enc]

        return (voxel_feats_enc, img_feats, pts_feats, depth)
    
    def extract_img_feat(self, img, img_metas):
        """Extract features of images (copied from original CONet_ori)."""
        import torch
        # Note: record_time is not used in re-implementation, but kept for compatibility
        
        img_enc_feats = self.image_encoder(img[0])
        x = img_enc_feats['x']
        img_feats = img_enc_feats['img_feats']

        rots, trans, intrins, post_rots, post_trans, bda = img[1:7]
        
        mlp_input = self.img_view_transformer.get_mlp_input(rots, trans, intrins, post_rots, post_trans, bda)
        geo_inputs = [rots, trans, intrins, post_rots, post_trans, bda, mlp_input]
        
        x, depth = self.img_view_transformer([x] + geo_inputs)
        
        return x, depth, img_feats
    
    def _simple_2d_to_3d_conversion(self, x):
        """Simple 2D to 3D conversion for camera-only occupancy model."""
        # Input: x with shape [B, N, C, H, W] where N=6 cameras
        # Output: 3D voxel features with 80 channels for occ_encoder_backbone
        
        B, N, C, H, W = x.shape
        
        # Simple approach: Use multi-view features to create 3D volume
        # 1. Flatten spatial dimensions and aggregate across cameras
        x_flat = x.view(B, N, C, H * W)  # [B, N, C, H*W]
        
        # 2. Aggregate across cameras (simple averaging)
        x_agg = torch.mean(x_flat, dim=1)  # [B, C, H*W]
        
        # 3. Reshape to create 3D volume
        # Target: [B, 80, D, H_new, W_new] for occ_encoder_backbone
        target_channels = 80  # 3D backbone expects 80 channels
        target_depth = 40  # Depth dimension for voxel grid
        target_h = H // 4  # Downsample height
        target_w = W // 4  # Downsample width
        
        # 4. Project to 80 channels for 3D backbone
        if not hasattr(self, '_channel_proj_3d'):
            self._channel_proj_3d = torch.nn.Linear(C, target_channels).to(x.device)
        
        x_proj = self._channel_proj_3d(x_agg.permute(0, 2, 1)).permute(0, 2, 1)  # [B, 80, H*W]
        
        # 5. Reshape to 3D volume  
        # Calculate actual available size
        available_size = x_proj.shape[2]  # H*W size
        total_target_size = target_depth * target_h * target_w
        
        # Adjust dimensions to match available data
        if available_size != total_target_size:
            # Simple fix: make depth dimension match available data
            adjusted_depth = available_size // (target_h * target_w)
            if adjusted_depth == 0:
                adjusted_depth = 1
            x_3d = x_proj.view(B, target_channels, adjusted_depth, target_h, target_w)
        else:
            x_3d = x_proj.view(B, target_channels, target_depth, target_h, target_w)
        
        return x_3d
    
    def extract_pts_feat(self, pts):
        """Extract point cloud features (copied from original CONet_ori)."""
        # For camera-only mode, point encoders may not be configured
        if self.pts_voxel_encoder is None or self.pts_middle_encoder is None:
            return None, None
        
        # Handle case where pts is a list (from DataLoader collation)
        if isinstance(pts, list):
            if len(pts) == 1:
                pts = pts[0]
            else:
                # For batch_size > 1, need to handle differently
                # For now, just use the first sample
                pts = pts[0]
        
        voxels, num_points, coors = self.voxelize(pts)
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        pts_enc_feats = self.pts_middle_encoder(voxel_features, coors, batch_size)
        
        pts_feats = pts_enc_feats['pts_feats']
        return pts_enc_feats['x'], pts_feats
    
    def voxelize(self, points):
        """Convert points to voxels."""
        import torch
        # Fallback between mmdet3d.ops and mmcv.ops
        try:
            from mmdet3d.ops import Voxelization
        except ImportError:
            from mmcv.ops import Voxelization
        if self.pts_voxel_layer is None:
            return None, None, None
        # Unpack raw tensor from LiDARPoints or BasePoints
        pts_input = points
        if hasattr(points, 'tensor'):
            pts_input = points.tensor
        elif hasattr(points, 'points'):
            pts_input = points.points
        
        # Move points to the same device as the model
        if pts_input.device.type == 'cpu' and next(self.parameters()).is_cuda:
            pts_input = pts_input.cuda()
        
        # Initialize voxelization op
        voxelization = Voxelization(**self.pts_voxel_layer)
        # Apply voxelization on raw point tensor
        # Note: Different versions of mmdetection3d/mmcv return different orders
        # mmcv returns (voxels, coors, num_points)
        # We need (voxels, num_points, coors) for compatibility
        result = voxelization(pts_input)
        voxels, coors, num_points = result
        
        # Add batch index if coors is [N, 3] instead of [N, 4]
        if coors.shape[1] == 3:
            # Add batch index column at the beginning
            # Ensure we use the same device and dtype as coors
            batch_idx = torch.zeros((coors.shape[0], 1), dtype=coors.dtype, device=coors.device)
            coors = torch.cat([batch_idx, coors], dim=1)  # [N, 4]: [batch_idx, z, y, x]
        
        return voxels, num_points, coors
    
    # @force_fp32()  # Removed for mmengine compatibility
    def occ_encoder(self, x):
        x = self.occ_encoder_backbone(x)
        x = self.occ_encoder_neck(x)
        return x
    def train_step(self, data, optim_wrapper):
        """Training step for MMEngine compatibility."""
        # Parse inputs from MMDetection3D data structure first
        data = self._parse_inputs(data)
        
        # Extract all necessary data from the parsed dict
        gt_occ = data.get('gt_occ', None)
        points = data.get('points', None)
        img_inputs = data.get('img_inputs', None)
        img_metas = data.get('img_metas', None)
        
        # Ensure gt_occ is on the correct device
        if gt_occ is not None:
            if isinstance(gt_occ, list):
                gt_occ = [item.cuda() if hasattr(item, 'cuda') else item for item in gt_occ]
            elif hasattr(gt_occ, 'cuda'):
                gt_occ = gt_occ.cuda()
        
        # Directly call forward_train to bypass forward() method
        losses = self.forward_train(
            points=points,
            img_metas=img_metas,
            gt_bboxes_3d=None,
            gt_labels_3d=None,
            gt_labels=None,
            gt_bboxes=None,
            img_inputs=img_inputs,
            proposals=None,
            gt_bboxes_ignore=None,
            gt_occ=gt_occ
        )
        
        # Format losses for MMEngine
        parsed_losses, log_vars = self.parse_losses(losses)
        optim_wrapper.update_params(parsed_losses)
        
        return log_vars

    def parse_losses(self, losses):
        """Parse losses for MMEngine compatibility."""
        import torch
        
        log_vars = {}
        parsed_losses = []
        
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                if loss_value.requires_grad:
                    parsed_losses.append(loss_value)
                log_vars[loss_name] = loss_value.item()
            else:
                log_vars[loss_name] = loss_value
                
        # Sum all losses
        if parsed_losses:
            total_loss = sum(parsed_losses)
        else:
            total_loss = sum(losses.values())
            
        log_vars['loss'] = total_loss.item() if hasattr(total_loss, 'item') else total_loss
        
        return total_loss, log_vars

    def loss(self, data_dict):
        """Calculate loss for training.
        
        Args:
            data_dict: Input data containing gt_occ, points, img_metas, etc.
        """
        
        # Extract and parse data
        gt_occ = data_dict.get('gt_occ')
        points = data_dict.get('points')
        img_metas = data_dict.get('img_metas')
        
        # Extract features
        data_parsed = self._parse_inputs(data_dict)
        feats = self.extract_feat(**data_parsed)
        
        # Add loss_depth from view transformer (like BEVDepth.forward_train)
        loss_dict = {}
        
        if hasattr(self, 'img_view_transformer') and not self.disable_loss_depth:
            # Get depth_gt from parsed data
            img_inputs = data_parsed.get('img_inputs')
            if img_inputs is not None and len(img_inputs) > 7:
                depth_gt = img_inputs[7]
                depth = feats.get('depth')
                if depth is not None and depth_gt is not None:
                    loss_depth = self.img_view_transformer.get_depth_loss(depth_gt, depth)
                    loss_dict['loss_depth'] = loss_depth
        
        # Call pts_bbox_head.loss
        loss_dict_occ = self.pts_bbox_head.loss(
            output_voxels=feats.get('output_voxels'),
            output_coords_fine=feats.get('output_coords_fine'),
            output_voxels_fine=feats.get('output_voxels_fine'),
            target_voxels=gt_occ
        )
        
        # Merge loss dictionaries
        loss_dict.update(loss_dict_occ)
        
        return loss_dict
    
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      gt_occ=None):
        """Override forward_train to ensure correct loss calculation for all modalities.
        
        This method handles camera-only, LiDAR-only, and multimodal configurations.
        """
        
        # breakpoint()

        # Extract features based on available modalities
        voxel_feats, img_feats, pts_feats, depth = self.extract_feat(
            points=points, img=img_inputs, img_metas=img_metas)
        
        losses = {}
        
        # Calculate depth loss (following original CONet implementation)
        if not self.disable_loss_depth and depth is not None and img_inputs is not None:
            # Extract depth ground truth from img_inputs
            # img_inputs structure: (imgs, rots, trans, intrins, post_rots, post_trans, bda_rot, img_shape, gt_depths, sensor2sensors)
            # Index -2 is gt_depths
            if isinstance(img_inputs, (list, tuple)) and len(img_inputs) >= 9:
                depth_gt = img_inputs[-2]  # or img_inputs[8] for gt_depths
                losses['loss_depth'] = self.img_view_transformer.get_depth_loss(depth_gt, depth)
        
        # Get predictions from occupancy head using processed voxel_feats
        # Extract transform, handling DataLoader collation (tuple wrapping)
        if img_inputs is not None:
            transform = []
            for i in range(1, min(8, len(img_inputs))):
                item = img_inputs[i]
                # Handle tuple wrapping from DataLoader collation
                if isinstance(item, tuple) and len(item) == 1:
                    transform.append(item[0])
                else:
                    transform.append(item)
        else:
            transform = None
        
        # Handle pts_feats for pts_bbox_head
        if pts_feats is not None and isinstance(pts_feats, dict):
            # Check for different possible keys
            if 'x' in pts_feats:
                pts_feat_tensor = pts_feats['x']
            elif 'voxel_feat' in pts_feats:
                pts_feat_tensor = pts_feats['voxel_feat']
            elif 'pts_feats' in pts_feats:
                # Sometimes pts_feats might be nested
                nested_feats = pts_feats['pts_feats']
                if isinstance(nested_feats, list) and len(nested_feats) > 0:
                    pts_feat_tensor = nested_feats[0]
                else:
                    pts_feat_tensor = nested_feats
            else:
                # If none of the expected keys are found, use the first item
                for key, value in pts_feats.items():
                    if isinstance(value, torch.Tensor):
                        pts_feat_tensor = value
                        break
                    elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], torch.Tensor):
                        pts_feat_tensor = value[0]
                        break
                else:
                    # Fallback - if everything fails, just pass None
                    pts_feat_tensor = None
        else:
            pts_feat_tensor = pts_feats
            
        outs = self.pts_bbox_head(
            voxel_feats=voxel_feats,
            points=points,
            img_metas=img_metas,
            img_feats=img_feats,
            pts_feats=pts_feat_tensor,
            transform=transform,
        )
        
        # Calculate occupancy losses
        loss_dict_occ = self.pts_bbox_head.loss(
            output_voxels=outs['output_voxels'],
            output_coords_fine=outs.get('output_coords_fine'),
            output_voxels_fine=outs.get('output_voxels_fine'),
            target_voxels=gt_occ
        )
        
        losses.update(loss_dict_occ)
        
        return losses

    def forward(self, mode='tensor', **kwargs):
        """Forward method for MMDetection3D v1.4+ compatibility.
        
        Args:
            mode (str): Forward mode - 'loss', 'predict', or 'tensor'
            **kwargs: All input data including 'img_inputs', 'gt_occ', etc.
        """
        
        if mode == 'loss':
            # Check if we have direct data format (from train_step)
            if 'gt_occ' in kwargs and 'points' in kwargs:
                
                # Extract data directly
                gt_occ = kwargs.get('gt_occ')
                points = kwargs.get('points')
                img_metas = kwargs.get('img_metas')
                
                # Ensure gt_occ is on the correct device
                if isinstance(gt_occ, list) and len(gt_occ) > 0:
                    if hasattr(gt_occ[0], 'cuda'):
                        gt_occ = [item.cuda() for item in gt_occ]
                elif hasattr(gt_occ, 'cuda'):
                    gt_occ = gt_occ.cuda()
                
                
                # Extract features directly
                data_parsed = self._parse_inputs(kwargs)
                # For LiDAR-only mode, extract_feat needs points, img=None, img_metas
                feats = self.extract_feat(
                    points=data_parsed['points'], 
                    img=None, 
                    img_metas=data_parsed['img_metas']
                )
                
                
                # Unpack the features tuple (voxel_feats_enc, img_feats, pts_feats, depth)
                voxel_feats_enc, img_feats, pts_feats, depth = feats
                
                # Handle pts_feats for pts_bbox_head
                if pts_feats is not None and isinstance(pts_feats, dict):
                    # Check for different possible keys
                    if 'x' in pts_feats:
                        pts_feat_tensor = pts_feats['x']
                    elif 'voxel_feat' in pts_feats:
                        pts_feat_tensor = pts_feats['voxel_feat']
                    elif 'pts_feats' in pts_feats:
                        # Sometimes pts_feats might be nested
                        nested_feats = pts_feats['pts_feats']
                        if isinstance(nested_feats, list) and len(nested_feats) > 0:
                            pts_feat_tensor = nested_feats[0]
                        else:
                            pts_feat_tensor = nested_feats
                    else:
                        # If none of the expected keys are found, use the first item
                        for key, value in pts_feats.items():
                            if isinstance(value, torch.Tensor):
                                pts_feat_tensor = value
                                break
                            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], torch.Tensor):
                                pts_feat_tensor = value[0]
                                break
                        else:
                            # Fallback - if everything fails, just pass None
                            pts_feat_tensor = None
                else:
                    pts_feat_tensor = pts_feats
                
                # Call pts_bbox_head to get outputs
                outs = self.pts_bbox_head(
                    voxel_feats=voxel_feats_enc,
                    points=None,  # points_occ (not used in this case)
                    img_metas=data_parsed['img_metas'],
                    img_feats=img_feats,
                    pts_feats=pts_feat_tensor,
                    transform=None,
                )
                
                
                # Call pts_bbox_head.loss directly
                loss_dict = self.pts_bbox_head.loss(
                    output_voxels=outs['output_voxels'],
                    output_coords_fine=outs['output_coords_fine'],
                    output_voxels_fine=outs['output_voxels_fine'],
                    target_voxels=gt_occ
                )
                
                return loss_dict
            else:
                # Use MMEngine format parsing
                result = self.loss(kwargs)
                return result
        elif mode == 'predict':
            return self.predict(kwargs) 
        else:
            # Extract input data
            data_dict = self._parse_inputs(kwargs)
            return self.extract_feat(**data_dict)
    
    def _parse_inputs(self, kwargs):
        """Parse inputs from MMDetection3D data structure"""
        
        # Extract data from inputs dictionary (MMDetection3D v1.1+)
        if 'inputs' in kwargs:
            inputs = kwargs['inputs']
                
            if isinstance(inputs, dict):
                # Extract points and img from inputs
                if 'points' in inputs:
                    kwargs['points'] = inputs['points']
                if 'img_inputs' in inputs:
                    kwargs['img_inputs'] = inputs['img_inputs']
                elif 'img' in inputs:
                    kwargs['img_inputs'] = inputs['img']
                # Try alternative keys for images
                elif 'imgs' in inputs:
                    kwargs['img_inputs'] = inputs['imgs']
                elif 'images' in inputs:
                    kwargs['img_inputs'] = inputs['images']
                # Check for gt_occ in inputs
                if 'gt_occ' in inputs:
                    kwargs['gt_occ'] = inputs['gt_occ']
                # Check inside voxels dict for gt_occ
                elif 'voxels' in inputs and isinstance(inputs['voxels'], dict):
                    voxels_dict = inputs['voxels']
                    if 'gt_occ' in voxels_dict:
                        kwargs['gt_occ'] = voxels_dict['gt_occ']
                    
        # Extract data_samples info
        if 'data_samples' in kwargs:
            data_samples = kwargs['data_samples']
            if data_samples is not None:
                # Handle single sample or list of samples
                if not isinstance(data_samples, (list, tuple)):
                    data_samples = [data_samples]
                
                if len(data_samples) > 0:
                    sample = data_samples[0]
                    
                    # Check for img_inputs in data_samples
                    for attr in ['img_inputs', 'img', 'imgs', 'images']:
                        if hasattr(sample, attr):
                            val = getattr(sample, attr)
                            if val is not None and kwargs.get('img_inputs') is None:
                                kwargs['img_inputs'] = val
                                break
                    
                    # Check various possible attribute names
                    for attr in ['gt_occ', 'gt_occ_1_1', 'gt_occupancy', 'occ_gt', 'gt_seg_3d']:
                        if hasattr(sample, attr):
                            val = getattr(sample, attr)
                            if kwargs.get('gt_occ') is None:
                                kwargs['gt_occ'] = val
                    
                    # Check for visible_mask
                    if hasattr(sample, 'visible_mask'):
                        kwargs['visible_mask'] = sample.visible_mask
                                
                    if hasattr(sample, 'metainfo'):
                        kwargs['img_metas'] = [sample.metainfo]
                    
        # Check for gt_occ directly in kwargs (this is where it should be in MMDetection3D)
        if 'gt_occ' in kwargs:
            pass
        else:
            # Try alternative keys
            for key in ['gt_occ_1_1', 'gt_occupancy', 'occ_gt']:
                if key in kwargs and kwargs[key] is not None:
                    kwargs['gt_occ'] = kwargs[key]
                    break
                    
        return kwargs
    
    def loss(self, data_dict, **extra_kwargs):
        """Loss forward function."""
        # Check if data is already in direct format (from train_step)
        if 'gt_occ' in data_dict and 'points' in data_dict:
            # Data is already parsed, use as-is
            pass
        else:
            # Parse inputs from MMDetection3D data structure
            data_dict = self._parse_inputs(data_dict)
        
        # Ensure gt_occ is present
        if data_dict.get('gt_occ') is None:
            raise ValueError("gt_occ must be provided for training. Check your data pipeline and LoadOccupancy configuration.")
        
        # Handle case where gt_occ might be a list (batch of tensors)
        if isinstance(data_dict['gt_occ'], list):
            # Stack list of tensors into a batch tensor
            import torch as torch_mod
            data_dict['gt_occ'] = torch_mod.stack(data_dict['gt_occ'], dim=0)
        
        # Ensure gt_occ has batch dimension if not already present
        if data_dict['gt_occ'].dim() == 3:
            data_dict['gt_occ'] = data_dict['gt_occ'].unsqueeze(0)  # Add batch dimension
        
        # Move gt_occ to same device as model
        device = next(self.parameters()).device
        data_dict['gt_occ'] = data_dict['gt_occ'].to(device)
        
        
        # Extract data from inputs dict
        img_inputs = data_dict.get('img_inputs')
        gt_occ = data_dict.get('gt_occ')
        img_metas = data_dict.get('img_metas', [{}] * len(img_inputs) if img_inputs is not None else [{}])
        
        return self.forward_train(
            points=data_dict.get('points'),
            img_inputs=img_inputs,
            img_metas=img_metas,
            gt_occ=gt_occ,
            **extra_kwargs
        )
    
    def predict(self, data_dict, **extra_kwargs):
        """Predict forward function."""
        # Parse inputs from MMDetection3D data structure
        if 'img_inputs' not in data_dict or 'inputs' in data_dict:
            data_dict = self._parse_inputs(data_dict)
        
        # Extract data from inputs dict
        img_inputs = data_dict.get('img_inputs')
        points = data_dict.get('points')
        gt_occ = data_dict.get('gt_occ')
        visible_mask = data_dict.get('visible_mask')
        img_metas = data_dict.get('img_metas', [{}] * len(img_inputs) if img_inputs is not None else [{}])
        
        return self.forward_test(
            points=points,
            img_inputs=img_inputs,
            img_metas=img_metas,
            gt_occ=gt_occ,
            visible_mask=visible_mask,
            **extra_kwargs
        )
    
    def _forward(self, data_dict, **extra_kwargs):
        """Simple forward function."""
        # Extract data from inputs dict  
        img_inputs = data_dict.get('img_inputs')
        img_metas = data_dict.get('img_metas', [{}] * len(img_inputs) if img_inputs is not None else [{}])
        
        # Extract features only
        voxel_feats, img_feats, pts_feats, depth = self.extract_feat(
            points=None, img=img_inputs, img_metas=img_metas)
        return [voxel_feats], depth
    
    def forward_test(self,
            points=None,
            img_metas=None,
            img_inputs=None,
            gt_occ=None,
            visible_mask=None,
            **kwargs,
        ):
        return self.simple_test(img_metas, img_inputs, points, gt_occ=gt_occ, visible_mask=visible_mask, **kwargs)
    
    def simple_test(self, img_metas, img=None, points=None, rescale=False, points_occ=None, 
            gt_occ=None, visible_mask=None):
        
        # breakpoint()

        voxel_feats, img_feats, pts_feats, depth = self.extract_feat(points, img=img, img_metas=img_metas)

        # Extract transform only if img is not None and has enough elements
        transform = None
        if img is not None and isinstance(img, (list, tuple)) and len(img) > 7:
            transform = img[1:8]
        output = self.pts_bbox_head(
            voxel_feats=voxel_feats,
            points=points_occ,
            img_metas=img_metas,
            img_feats=img_feats,
            pts_feats=pts_feats,
            transform=transform,
        )


        pred_c = output['output_voxels'][0]
        
        SC_metric, _ = self.evaluation_semantic(pred_c, gt_occ, eval_type='SC', visible_mask=visible_mask)
        SSC_metric, SSC_occ_metric = self.evaluation_semantic(pred_c, gt_occ, eval_type='SSC', visible_mask=visible_mask)

        pred_f = None
        SSC_metric_fine = None
        # Check if fine prediction exists (following original code)
        if output.get('output_voxels_fine') is not None and len(output.get('output_voxels_fine', [])) > 0:
            if output.get('output_coords_fine') is not None and len(output.get('output_coords_fine', [])) > 0:
                fine_pred = output['output_voxels_fine'][0]  # [N, ncls]
                fine_coord = output['output_coords_fine'][0]  # [3, N]
                
                # Create pred_f with correct shape: [1, ncls, H, W, D]
                pred_f = self.empty_idx * torch.ones_like(gt_occ).unsqueeze(0).unsqueeze(0).repeat(1, fine_pred.shape[1], 1, 1, 1).float()
                
                # fine_pred: [N, ncls] -> permute: [ncls, N] -> unsqueeze: [1, ncls, N]
                pred_f[:, :, fine_coord[0], fine_coord[1], fine_coord[2]] = fine_pred.permute(1, 0).unsqueeze(0)
            else:
                pred_f = output['output_voxels_fine'][0]
            SC_metric, _ = self.evaluation_semantic(pred_f, gt_occ, eval_type='SC', visible_mask=visible_mask)
            SSC_metric_fine, SSC_occ_metric_fine = self.evaluation_semantic(pred_f, gt_occ, eval_type='SSC', visible_mask=visible_mask)

        # Prepare pred_occ for evaluator (interpolate to gt size and take argmax)
        pred_for_eval = pred_f if pred_f is not None else pred_c
        # Handle gt with or without batch dimension for shape extraction
        if gt_occ.dim() == 3:
            gt_shape = gt_occ.shape
        else:
            _, H, W, D = gt_occ.shape
            gt_shape = (H, W, D)
        
        # Interpolate pred to gt size
        pred_interpolated = F.interpolate(pred_for_eval, size=list(gt_shape), mode='trilinear', align_corners=False).contiguous()
        # Take argmax to get class indices
        pred_occ_np = torch.argmax(pred_interpolated[0], dim=0).cpu().numpy()
        
        # Get gt as numpy (remove batch dim if present)
        if gt_occ.dim() == 4:
            gt_occ_np = gt_occ[0].cpu().numpy()
        else:
            gt_occ_np = gt_occ.cpu().numpy()

        test_output = {
            'SC_metric': SC_metric,
            'SSC_metric': SSC_metric,
            'pred_c': pred_c,
            'pred_f': pred_f,
            'pred_occ': pred_occ_np,  # For compatibility with occ_metric (numpy array)
            'gt_occ': gt_occ_np,  # For compatibility with occ_metric (numpy array)
        }

        if SSC_metric_fine is not None:
            test_output['SSC_metric_fine'] = SSC_metric_fine

        # Return as list for compatibility with MMDetection3D evaluator
        return [test_output]


    def evaluation_semantic(self, pred, gt, eval_type, visible_mask=None):
        # Handle gt with or without batch dimension
        if gt.dim() == 3:
            # No batch dimension, add it
            gt = gt.unsqueeze(0)
        
        _, H, W, D = gt.shape
        pred = F.interpolate(pred, size=[H, W, D], mode='trilinear', align_corners=False).contiguous()
        pred = torch.argmax(pred[0], dim=0).cpu().numpy()
        gt = gt[0].cpu().numpy()
        gt = gt.astype(np.int32)

        # ignore noise
        noise_mask = gt != 255

        if eval_type == 'SC':
            # 0 1 split
            gt[gt != self.empty_idx] = 1
            pred[pred != self.empty_idx] = 1
            return fast_hist(pred[noise_mask], gt[noise_mask], max_label=2), None


        if eval_type == 'SSC':
            hist_occ = None
            if visible_mask is not None:
                visible_mask = visible_mask[0].cpu().numpy()
                mask = noise_mask & (visible_mask!=0)
                hist_occ = fast_hist(pred[mask], gt[mask], max_label=17)

            hist = fast_hist(pred[noise_mask], gt[noise_mask], max_label=17)
            return hist, hist_occ


def fast_hist(pred, label, max_label=18):
    pred = copy.deepcopy(pred.flatten())
    label = copy.deepcopy(label.flatten())
    bin_count = np.bincount(max_label * label.astype(int) + pred, minlength=max_label ** 2)
    return bin_count[:max_label ** 2].reshape(max_label, max_label)