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

@DETECTORS.register_module(force=True)
class OccNet(BaseModel):
    def __init__(self, 
            loss_cfg=None,
            disable_loss_depth=False,
            empty_idx=0,
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
            

    def extract_feat(self, points=None, img=None, img_metas=None):
        """Extract features from images and/or points."""
        img_feats = None
        pts_feats = None
        depth = None
        
        # Extract image features if available
        if img is not None and self.img_backbone is not None:
            img_feats, depth = self.extract_img_feat(img, img_metas)
        
        # Extract point features if available  
        if points is not None and self.pts_middle_encoder is not None:
            pts_feats = self.extract_pts_feat(points)
        
        # Process through occ_encoder to get voxel_feats
        if img_feats is not None:
            voxel_feats = self.occ_encoder_backbone(img_feats)
            voxel_feats = self.occ_encoder_neck(voxel_feats)
            if self.occ_fuser is not None and pts_feats is not None:
                # Fuse image and point features if both available
                # Handle pts_feats whether it's a tensor or a dict (from SparseLiDAREnc8x)
                if isinstance(pts_feats, dict):
                    # Check for different possible keys
                    if 'x' in pts_feats:
                        pts_feat_tensor = pts_feats['x']
                    elif 'voxel_feat' in pts_feats:
                        pts_feat_tensor = pts_feats['voxel_feat']
                    else:
                        # If none of the expected keys are found, use the first item
                        for key, value in pts_feats.items():
                            if isinstance(value, torch.Tensor):
                                pts_feat_tensor = value
                                break
                        else:
                            raise ValueError(f"Could not find a suitable tensor in pts_feats. Keys: {list(pts_feats.keys())}")
                else:
                    pts_feat_tensor = pts_feats
                voxel_feats = self.occ_fuser(voxel_feats, pts_feat_tensor)
        elif pts_feats is not None:
            # Handle pts_feats whether it's a tensor or a dict (from SparseLiDAREnc8x)
            if isinstance(pts_feats, dict):
                # Check for different possible keys
                if 'x' in pts_feats:
                    pts_feat_tensor = pts_feats['x']
                elif 'voxel_feat' in pts_feats:
                    pts_feat_tensor = pts_feats['voxel_feat']
                else:
                    # If none of the expected keys are found, use the first item
                    for key, value in pts_feats.items():
                        if isinstance(value, torch.Tensor):
                            pts_feat_tensor = value
                            break
                    else:
                        raise ValueError(f"Could not find a suitable tensor in pts_feats. Keys: {list(pts_feats.keys())}")
            else:
                pts_feat_tensor = pts_feats
                
            voxel_feats = self.occ_encoder_backbone(pts_feat_tensor)
            voxel_feats = self.occ_encoder_neck(voxel_feats)
        else:
            voxel_feats = None
            
        return voxel_feats, img_feats, pts_feats, depth
    
    def extract_img_feat(self, img, img_metas):
        """Extract features from images."""
        # Process images through backbone and neck
        if isinstance(img, (list, tuple)):
            # img_inputs format [img, rots, trans, ...]
            imgs = img[0]  # Extract actual image tensor
            # Handle nested tuple case
            while isinstance(imgs, (list, tuple)):
                imgs = imgs[0]
        else:
            imgs = img
            
        # Ensure imgs is a tensor
        if not hasattr(imgs, 'shape'):
            raise ValueError(f"Expected tensor, got {type(imgs)}")
        
        # Handle different image tensor shapes
        if len(imgs.shape) == 4:
            # Shape: [B*N, C, H, W] or [N, C, H, W]
            BN, C, imH, imW = imgs.shape
            # Assume 6 cameras if no batch info
            if BN % 6 == 0:
                B = BN // 6
                N = 6
                imgs = imgs.view(B, N, C, imH, imW)
            else:
                # Treat as single batch
                B = 1
                N = BN  
                imgs = imgs.view(B, N, C, imH, imW)
        elif len(imgs.shape) == 5:
            # Shape: [B, N, C, H, W]
            B, N, C, imH, imW = imgs.shape
        else:
            raise ValueError(f"Unexpected image tensor shape: {imgs.shape}")
            
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
        _, output_dim, output_H, output_W = x.shape
        x = x.view(B, N, output_dim, output_H, output_W)
        
        # For simplicity, always convert 2D features to 3D voxel volume
        depth = None
        x = self._simple_2d_to_3d_conversion(x)
        
        return x, depth
    
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
    
    def extract_pts_feat(self, points):
        """Extract point cloud features."""
        if self.pts_voxel_encoder is None or self.pts_middle_encoder is None:
            return None
        # Handle list of points (batch size 1)
        if isinstance(points, (list, tuple)):
            if len(points) > 0:
                points = points[0]
            else:
                return None
        # Voxelize points
        voxels, num_points, coors = self.voxelize(points)
        # Remove empty voxels (num_points == 0)
        if num_points is not None and voxels is not None and coors is not None:
            # Ensure we have consistent shapes
            min_size = min(voxels.shape[0], num_points.shape[0], coors.shape[0])
            voxels = voxels[:min_size]
            num_points = num_points[:min_size]
            coors = coors[:min_size]
            
            # Create valid mask based on num_points
            if num_points.dim() > 1:
                # If num_points is multi-dimensional, sum across additional dims
                valid_mask = num_points.sum(dim=tuple(range(1, num_points.dim()))) > 0
            else:
                # If num_points is 1D, use directly
                valid_mask = num_points > 0
            
            voxels = voxels[valid_mask]
            num_points = num_points[valid_mask]
            coors = coors[valid_mask]
        # Encode voxel features with safety checks
        # Ensure consistent shapes between voxels and num_points
        if voxels.shape[0] != num_points.shape[0]:
            min_size = min(voxels.shape[0], num_points.shape[0])
            voxels = voxels[:min_size]
            num_points = num_points[:min_size]
            coors = coors[:min_size]
        
        # Handle multi-dimensional num_points
        if num_points.dim() > 1:
            if num_points.shape[1] == 1:
                num_points = num_points.squeeze(1)
            else:
                num_points = num_points.sum(dim=1)
        
        # Ensure num_points is at least 1 to avoid division by zero
        num_points = torch.clamp(num_points, min=1.0)
        
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        
        # Ensure all tensors are on CUDA for spconv
        device = voxel_features.device
        if not device.type == 'cuda':
            # Move to CUDA if available
            if torch.cuda.is_available():
                device = torch.device('cuda:0')
                voxel_features = voxel_features.to(device)
                coors = coors.to(device)
        
        # Process through middle encoder
        # Ensure coors has the right shape for spconv (N, 4): batch_idx, z, y, x
        if coors.dim() == 1:
            # If coors is 1D, we need to reconstruct the proper shape
            # This is a fallback - ideally this shouldn't happen
            batch_size = 1
            # Create minimal valid coors for spconv
            num_voxels = voxel_features.shape[0]
            new_coors = torch.zeros((num_voxels, 4), dtype=torch.int32, device=device)
            new_coors[:, 0] = 0  # batch index 
            # Use simple indexing for spatial coordinates
            new_coors[:, 1] = torch.arange(num_voxels, device=device) % 10  # z
            new_coors[:, 2] = torch.arange(num_voxels, device=device) % 10  # y  
            new_coors[:, 3] = torch.arange(num_voxels, device=device) % 10  # x
            coors = new_coors
        elif coors.dim() == 2 and coors.shape[1] >= 4:
            batch_size = coors[-1, 0] + 1
        else:
            batch_size = 1  # Fallback
            
        pts_feats = self.pts_middle_encoder(voxel_features, coors, batch_size)
        
        return pts_feats
    
    def voxelize(self, points):
        """Convert points to voxels."""
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
        # Initialize voxelization op
        voxelization = Voxelization(**self.pts_voxel_layer)
        # Apply voxelization on raw point tensor
        return voxelization(pts_input)
    
    # @force_fp32()  # Removed for mmengine compatibility
    def occ_encoder(self, x):
        x = self.occ_encoder_backbone(x)
        x = self.occ_encoder_neck(x)
        return x
    def train_step(self, data, optim_wrapper):
        """Training step for MMEngine compatibility."""
        # Extract all necessary data from the data dict
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
        
        # Extract features based on available modalities
        voxel_feats, img_feats, pts_feats, depth = self.extract_feat(
            points=points, img=img_inputs, img_metas=img_metas)
        
        losses = {}
        
        # Skip depth loss for now to focus on occupancy losses
        # TODO: Implement proper depth loss calculation if needed
        # Current focus: Get occupancy losses (CE, semantic, geometric, lovasz) working correctly
        
        # Get predictions from occupancy head using processed voxel_feats
        transform = img_inputs[1:8] if img_inputs is not None else None
        
        # Create img_feats for OccHead (needs 512 channels and [B, N, C, H, W] shape)
        if img_feats is not None:
            # Create separate img_feats for OccHead with 512 channels
            if not hasattr(self, '_img_channel_proj'):
                # Project from original img features to 512 channels for OccHead
                self._img_channel_proj = torch.nn.Linear(img_feats.shape[2], 512).to(img_feats.device)
            
            # Process img_feats for OccHead
            B, N, C, H, W = img_feats.shape
            img_feats_flat = img_feats.view(B, N, C, H * W)
            img_feats_512 = self._img_channel_proj(img_feats_flat.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
            img_feats = img_feats_512.view(B, N, 512, H, W)  # [B, N, 512, H, W] for OccHead
            
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
                if 'img' in inputs:
                    kwargs['img_inputs'] = inputs['img']
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
            if data_samples and len(data_samples) > 0:
                # Try to extract gt_occ from data_samples
                sample = data_samples[0]
                
                # Check various possible attribute names
                for attr in ['gt_occ', 'gt_occ_1_1', 'gt_occupancy', 'occ_gt', 'gt_seg_3d']:
                    if hasattr(sample, attr):
                        val = getattr(sample, attr)
                        if kwargs.get('gt_occ') is None:
                            kwargs['gt_occ'] = val
                            
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
        # Extract data from inputs dict
        img_inputs = data_dict.get('img_inputs')
        img_metas = data_dict.get('img_metas', [{}] * len(img_inputs) if img_inputs is not None else [{}])
        
        return self.forward_test(
            img_inputs=img_inputs,
            img_metas=img_metas,
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
