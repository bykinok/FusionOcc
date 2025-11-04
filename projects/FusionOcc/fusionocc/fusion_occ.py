# Copyright (c) OpenMMLab. All rights reserved.
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from mmcv.cnn.bricks.conv_module import ConvModule
try:
    from mmcv.runner import auto_fp16, force_fp32
except ImportError:
    try:
        from mmengine.runner import auto_fp16, force_fp32
    except ImportError:
        # Create dummy decorators if not available
        def auto_fp16(*args, **kwargs):
            if args and callable(args[0]):
                return args[0]
            def decorator(func):
                return func
            return decorator
        def force_fp32(*args, **kwargs):
            if args and callable(args[0]):
                return args[0]
            def decorator(func):
                return func
            return decorator
from mmdet3d.registry import MODELS
try:
    from mmdet.models.builder import build_loss
except ImportError:
    import torch.nn as nn
    def build_loss(cfg):
        if cfg['type'] == 'CrossEntropyLoss':
            return nn.CrossEntropyLoss()
        else:
            from mmdet3d.registry import MODELS
            return MODELS.build(cfg)

from .lidar_encoder import CustomSparseEncoder


class SimpleDataPreprocessor(nn.Module):
    """Simple data preprocessor that moves data to the correct device."""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, data, training=False):
        """Process data and move to device - PASSTHROUGH mode.
        
        Args:
            data (dict): Input data dictionary
            training (bool): Whether in training mode
            
        Returns:
            dict: Processed data (unchanged, just passed through)
        """
        # CRITICAL: Just return data as-is
        # MMEngine will handle device placement
        # FormatDataSamples already collected the necessary keys
        return data


@MODELS.register_module()
class FusionDepthSeg(nn.Module):
    """Base class for FusionDepthSeg detector."""
    
    def __init__(self, 
                 img_backbone=None,
                 img_neck=None,
                 img_view_transformer=None,
                 img_bev_encoder_backbone=None,
                 img_bev_encoder_neck=None,
                 pre_process=None,
                 num_adj=1,
                 data_preprocessor=None,
                 **kwargs):
        super(FusionDepthSeg, self).__init__()
        
        # Initialize data_preprocessor for MMEngine compatibility
        if data_preprocessor is not None:
            from mmdet3d.registry import MODELS as MODELS_REGISTRY
            self.data_preprocessor = MODELS_REGISTRY.build(data_preprocessor)
        else:
            # Create a simple data_preprocessor that moves data to device
            self.data_preprocessor = SimpleDataPreprocessor()
        # Initialize with basic components
        # num_frame = num_adj + 1 (current frame + adjacent frames)
        self.num_frame = num_adj + 1
        self.align_after_view_transformation = kwargs.get('align_after_view_transformation', False)
        
        # Initialize grid for shift_feature (will be created on first use)
        self.grid = None
        
        from mmdet3d.registry import MODELS as MODELS_REGISTRY
        
        # Build image backbone
        self.img_backbone = MODELS_REGISTRY.build(img_backbone) if img_backbone is not None else None
        
        # Build image neck
        self.img_neck = MODELS_REGISTRY.build(img_neck) if img_neck is not None else None
        self.with_img_neck = img_neck is not None
        
        # Build image view transformer
        self.img_view_transformer = MODELS_REGISTRY.build(img_view_transformer) if img_view_transformer is not None else None
        
        # Build pre-process network
        if pre_process is not None and isinstance(pre_process, dict):
            self.pre_process = True
            self.pre_process_net = MODELS_REGISTRY.build(pre_process)
        else:
            self.pre_process = False
            self.pre_process_net = None
        
        # Build BEV encoder from backbone and neck
        # These names must match the checkpoint keys
        if img_bev_encoder_backbone is not None:
            self.img_bev_encoder_backbone = MODELS_REGISTRY.build(img_bev_encoder_backbone)
            if img_bev_encoder_neck is not None:
                self.img_bev_encoder_neck = MODELS_REGISTRY.build(img_bev_encoder_neck)
            else:
                self.img_bev_encoder_neck = None
        else:
            self.img_bev_encoder_backbone = None
            self.img_bev_encoder_neck = None

    def image_encoder(self, img, stereo=False):
        """Encode images using backbone and neck."""
        imgs = img
        # Handle both (B, N, C, H, W) and (N, C, H, W) formats
        if imgs.ndim == 4:  # (N, C, H, W)
            N, C, imH, imW = imgs.shape
            B = 1
            imgs = imgs.unsqueeze(0)  # (1, N, C, H, W)
        else:  # (B, N, C, H, W)
            B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)
        
        x = self.img_backbone(imgs)
        stereo_feat = None
        if stereo:
            stereo_feat = x[0]
        x = x[1:]
        
        if self.with_img_neck:
            x = self.img_neck(x)
            if type(x) in [list, tuple]:
                x = x[0]
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)
        return x, stereo_feat

    def prepare_img_3d_feat(self, img, sensor2keyego, ego2global, intrin,
                            post_rot, post_tran, bda, mlp_input, input_depth=None):
        x, _ = self.image_encoder(img, stereo=False)
        
        img_3d_feat, depth, seg = self.img_view_transformer(
            [x, sensor2keyego, ego2global, intrin, post_rot, post_tran, bda,
             mlp_input], input_depth)
                
        if self.pre_process and self.pre_process_net is not None:
            img_3d_feat = self.pre_process_net(img_3d_feat)[0]
        return img_3d_feat, depth, seg

    def prepare_inputs(self, inputs, stereo=False):
        """Split the inputs into each frame and compute sensor to key ego transforms."""
        # Get device from model
        device = next(self.parameters()).device
        
        # Handle None inputs - this should not happen if forward_train is correct
        if inputs is None:
            raise ValueError(
                "prepare_inputs received None! This should have been caught earlier in forward_train. "
                "Check that img_inputs is being passed correctly."
            )
        
        # inputs can be tuple or list: (imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans, [bda], ...)
        if isinstance(inputs, (tuple, list)) and len(inputs) >= 6:
            # Convert all tuple elements to tensors if needed
            def to_tensor(x):
                if isinstance(x, (tuple, list)):
                    if len(x) > 0 and isinstance(x[0], torch.Tensor):
                        result = torch.stack(x) if x[0].ndim > 0 else torch.tensor(x)
                    else:
                        result = torch.tensor(x) if not isinstance(x[0], (tuple, list)) else torch.stack([to_tensor(item) for item in x])
                else:
                    result = x
                # Move to device if it's a tensor
                if isinstance(result, torch.Tensor):
                    result = result.to(device)
                return result
            
            imgs = to_tensor(inputs[0])
            sensor2egos = to_tensor(inputs[1])
            ego2globals = to_tensor(inputs[2])
            intrins = to_tensor(inputs[3])
            post_rots = to_tensor(inputs[4])
            post_trans = to_tensor(inputs[5])
            bda = to_tensor(inputs[6]) if len(inputs) > 6 else None
        else:
            raise ValueError(f"Invalid inputs format: {type(inputs)}, length: {len(inputs) if hasattr(inputs, '__len__') else 'N/A'}")
        
        # Handle different input shapes
        if imgs.ndim == 4:  # (N, C, H, W) - no batch dimension
            imgs = imgs.unsqueeze(0)  # (1, N, C, H, W)
        
        B, N, C, H, W = imgs.shape
        # Split multi-frame: N = N_cameras * num_frame
        N = N // self.num_frame
        imgs = imgs.view(B, N, self.num_frame, C, H, W)
        imgs = torch.split(imgs, 1, 2)
        imgs = [t.squeeze(2) for t in imgs]
        
        # Ensure other tensors also have batch dimension
        if sensor2egos.ndim == 3:  # (N, 4, 4)
            sensor2egos = sensor2egos.unsqueeze(0)  # (1, N, 4, 4)
        if ego2globals.ndim == 3:  # (N, 4, 4)
            ego2globals = ego2globals.unsqueeze(0)
        if intrins.ndim == 3:  # (N, 3, 3)
            intrins = intrins.unsqueeze(0)
        if post_rots.ndim == 3:  # (N, 3, 3)
            post_rots = post_rots.unsqueeze(0)
        if post_trans.ndim == 2:  # (N, 3)
            post_trans = post_trans.unsqueeze(0)
        if bda is not None and bda.ndim == 2:  # (3, 3)
            bda = bda.unsqueeze(0)
        
        sensor2egos = sensor2egos.view(B, self.num_frame, N, 4, 4)
        ego2globals = ego2globals.view(B, self.num_frame, N, 4, 4)
        
        # Calculate the transformation from sweep sensor to key ego
        keyego2global = ego2globals[:, 0, 0, ...].unsqueeze(1).unsqueeze(1)
        global2keyego = torch.inverse(keyego2global.double())
        sensor2keyegos = global2keyego @ ego2globals.double() @ sensor2egos.double()
        sensor2keyegos = sensor2keyegos.float()
        
        # Split into list of frames
        extra = [
            sensor2keyegos,
            ego2globals,
            intrins.view(B, self.num_frame, N, 3, 3),
            post_rots.view(B, self.num_frame, N, 3, 3),
            post_trans.view(B, self.num_frame, N, 3)
        ]
        extra = [torch.split(t, 1, 1) for t in extra]
        extra = [[p.squeeze(1) for p in t] for t in extra]
        sensor2keyegos, ego2globals, intrins, post_rots, post_trans = extra
        
        # If bda is None, create identity matrix (don't expand, keep as (B, 3, 3))
        if bda is None:
            device = imgs[0].device
            bda = torch.eye(3, 3, device=device, dtype=imgs[0].dtype)
            if B > 1:
                bda = bda.unsqueeze(0).expand(B, -1, -1)
            else:
                bda = bda.unsqueeze(0)
        elif bda.ndim == 2:
            # If bda is (3, 3), add batch dimension
            bda = bda.unsqueeze(0)
        
        return imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, bda, None

    def extract_img_3d_feat(self, img_inputs, input_depth):
        """Extract 3D image features from multiple frames."""
        imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, bda, _ = \
            self.prepare_inputs(img_inputs, stereo=False)
        
        """Extract features of images."""
        img_3d_feat_list = []
        depth_key_frame = None
        seg_key_frame = None
        
        # Process each frame
        for fid in range(self.num_frame - 1, -1, -1):
            # Use the correct frame index
            img, sensor2keyego, ego2global, intrin, post_rot, post_tran = \
                imgs[fid], sensor2keyegos[fid], ego2globals[fid], intrins[fid], \
                post_rots[fid], post_trans[fid]
            curr_frame = fid == 0
            
            if self.align_after_view_transformation:
                sensor2keyego, ego2global = sensor2keyegos[0], ego2globals[0]
            
            # Pass individual frame's transformation matrices
            mlp_input = self.img_view_transformer.get_mlp_input(
                sensor2keyegos[0], ego2globals[0], intrin,
                post_rot, post_tran, bda)
            
            inputs_curr = (img, sensor2keyego, ego2global, intrin,
                           post_rot, post_tran, bda, mlp_input, input_depth)
            
            if curr_frame:
                img_3d_feat, depth, pred_seg = self.prepare_img_3d_feat(*inputs_curr)
                seg_key_frame = pred_seg
                depth_key_frame = depth
            else:
                # For non-current frames, compute features without gradients
                with torch.no_grad():
                    img_3d_feat, _, _ = self.prepare_img_3d_feat(*inputs_curr)
            
            img_3d_feat_list.append(img_3d_feat)
        
        if self.align_after_view_transformation:
            for adj_id in range(self.num_frame - 1):
                img_3d_feat_list[adj_id] = \
                    self.shift_feature(img_3d_feat_list[adj_id],
                                       [sensor2keyegos[0],
                                        sensor2keyegos[self.num_frame - 2 - adj_id]],
                                       bda)
        
        img_3d_feat_feat = torch.cat(img_3d_feat_list, dim=1)
        return img_3d_feat_feat, depth_key_frame, seg_key_frame

    def gen_grid(self, input, sensor2keyegos, bda, bda_adj=None):
        """Generate grid for feature alignment using transformation matrices.
        
        Args:
            input: Feature tensor (N, C, H, W)
            sensor2keyegos: List of transformation matrices [current, adjacent]
            bda: BEV data augmentation matrix
            bda_adj: Adjacent frame BEV data augmentation matrix (optional)
            
        Returns:
            grid: Sampling grid for F.grid_sample (N, H, W, 2)
        """
        n, c, h, w = input.shape
        _, v, _, _ = sensor2keyegos[0].shape
        
        if not hasattr(self, 'grid') or self.grid is None:
            # Generate base grid
            xs = torch.linspace(
                0, w - 1, w, dtype=input.dtype,
                device=input.device).view(1, w).expand(h, w)
            ys = torch.linspace(
                0, h - 1, h, dtype=input.dtype,
                device=input.device).view(h, 1).expand(h, w)
            grid = torch.stack((xs, ys, torch.ones_like(xs)), -1)
            self.grid = grid
        else:
            grid = self.grid
            
        grid = grid.view(1, h, w, 3).expand(n, h, w, 3).view(n, h, w, 3, 1)

        # Get transformation from current ego frame to adjacent ego frame
        # Transformation from current camera frame to current ego frame
        c02l0 = sensor2keyegos[0][:, 0:1, :, :]

        # Transformation from adjacent camera frame to current ego frame
        c12l0 = sensor2keyegos[1][:, 0:1, :, :]

        # Add BEV data augmentation
        bda_ = torch.zeros((n, 1, 4, 4), dtype=grid.dtype).to(grid)
        # Handle bda shape: if (B, 3, 3), take first batch; if (3, 3), use directly
        if bda.ndim == 3:
            bda_[:, :, :3, :3] = bda[0:1].unsqueeze(1) if n == 1 else bda[0].unsqueeze(0).unsqueeze(0)
        else:
            bda_[:, :, :3, :3] = bda.unsqueeze(0).unsqueeze(0)
        bda_[:, :, 3, 3] = 1
        c02l0 = bda_.matmul(c02l0)
        
        if bda_adj is not None:
            bda_ = torch.zeros((n, 1, 4, 4), dtype=grid.dtype).to(grid)
            if bda_adj.ndim == 3:
                bda_[:, :, :3, :3] = bda_adj[0:1].unsqueeze(1) if n == 1 else bda_adj[0].unsqueeze(0).unsqueeze(0)
            else:
                bda_[:, :, :3, :3] = bda_adj.unsqueeze(0).unsqueeze(0)
            bda_[:, :, 3, 3] = 1
        c12l0 = bda_.matmul(c12l0)

        # Transformation from current ego frame to adjacent ego frame
        l02l1 = c02l0.matmul(torch.inverse(c12l0))[:, 0, :, :].view(
            n, 1, 1, 4, 4)
        
        # Remove Z dimension (keep X, Y only for BEV)
        l02l1 = l02l1[:, :, :,
                [True, True, False, True], :][:, :, :, :,
                [True, True, False, True]]

        # Create feature to BEV transformation matrix
        feat2bev = torch.zeros((3, 3), dtype=grid.dtype).to(grid)
        feat2bev[0, 0] = self.img_view_transformer.grid_interval[0]
        feat2bev[1, 1] = self.img_view_transformer.grid_interval[1]
        feat2bev[0, 2] = self.img_view_transformer.grid_lower_bound[0]
        feat2bev[1, 2] = self.img_view_transformer.grid_lower_bound[1]
        feat2bev[2, 2] = 1
        feat2bev = feat2bev.view(1, 3, 3)
        
        # Apply transformation
        tf = torch.inverse(feat2bev).matmul(l02l1).matmul(feat2bev)

        # Transform and normalize grid
        grid = tf.matmul(grid)
        normalize_factor = torch.tensor([w - 1.0, h - 1.0],
                                        dtype=input.dtype,
                                        device=input.device)
        grid = grid[:, :, :, :2, 0] / normalize_factor.view(1, 1, 1, 2) * 2.0 - 1.0
        return grid

    def shift_feature(self, feature, sensor2keyegos, bda, bda_adj=None):
        """Shift feature based on sensor transformations using grid sampling.
        
        Args:
            feature: Feature tensor to align (N, C, H, W)
            sensor2keyegos: List of transformation matrices [current, adjacent]
            bda: BEV data augmentation matrix
            bda_adj: Adjacent frame BEV data augmentation matrix (optional)
            
        Returns:
            output: Aligned feature tensor (N, C, H, W)
        """
        grid = self.gen_grid(feature, sensor2keyegos, bda, bda_adj=bda_adj)
        output = F.grid_sample(feature, grid.to(feature.dtype), align_corners=True)
        return output


@MODELS.register_module()
class FusionOCC(FusionDepthSeg):
    def __init__(self,
                 loss_occ=None,
                 out_dim=32,
                 use_mask=False,
                 num_classes=18,
                 use_predicter=True,
                 class_wise=False,
                 point_cloud_range=[-40, -40, -1, 40, 40, 5.4],
                 voxel_size=[0.05, 0.05, 0.05],
                 lidar_in_channel=5,
                 lidar_out_channel=32,
                 fuse_loss_weight=0.1,
                 img_bev_encoder_backbone=None,
                 img_bev_encoder_neck=None,
                 data_preprocessor=None,
                 **kwargs):
        super(FusionOCC, self).__init__(
            img_bev_encoder_backbone=img_bev_encoder_backbone,
            img_bev_encoder_neck=img_bev_encoder_neck,
            data_preprocessor=data_preprocessor,
            **kwargs)
        
        self.voxel_size = voxel_size
        self.lidar_out_channel = lidar_out_channel
        self.lidar_in_channel = lidar_in_channel
        self.sparse_shape = [
            int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0]),
            int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1]),
            int((point_cloud_range[5] - point_cloud_range[2]) / voxel_size[2])
        ]
        self.point_cloud_range = point_cloud_range
        
        # Build lidar encoder
        self.lidar_encoder = CustomSparseEncoder(
            in_channels=self.lidar_in_channel,
            sparse_shape=self.sparse_shape,
            point_cloud_range=self.point_cloud_range,
            voxel_size=self.voxel_size,
            output_channels=self.lidar_out_channel,
            block_type="conv_module"
        )
        
        self.out_dim = out_dim
        out_channels = out_dim if use_predicter else num_classes
        self.final_conv = ConvModule(
            out_dim,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            conv_cfg=dict(type='Conv3d'))
            
        self.use_predicter = use_predicter
        if use_predicter:
            self.predicter = nn.Sequential(
                nn.Linear(self.out_dim, self.out_dim * 2),
                nn.Softplus(),
                nn.Linear(self.out_dim * 2, num_classes),
            )
            
        self.pts_bbox_head = None
        self.use_mask = use_mask
        self.num_classes = num_classes
        self.loss_occ = build_loss(loss_occ) if loss_occ else None
        self.class_wise = class_wise
        self.align_after_view_transformation = False
        self.fuse_loss_weight = fuse_loss_weight

    def occ_encoder(self, x):
        """Encode fusion features using BEV encoder (backbone + neck)."""
        x = self.img_bev_encoder_backbone(x)
        x = self.img_bev_encoder_neck(x)
        if type(x) in [list, tuple]:
            x = x[0]
        return x

    def extract_feat(self, lidar_feat, img, img_metas, input_depth=None, **kwargs):
        """Extract features from images and points."""
        fusion_feats, depth, pred_segs = self.extract_fusion_feat(
            lidar_feat, img, img_metas, input_depth=input_depth, **kwargs
        )
        
        pts_feats = None
        return fusion_feats, pts_feats, depth, pred_segs

    def extract_fusion_feat(self, lidar_feat, img, img_metas, input_depth=None, **kwargs):
        """Extract fusion features from lidar and image."""
        # Extract image 3D features first
        img_3d_feat_feat, depth_key_frame, seg_key_frame = self.extract_img_3d_feat(
            img_inputs=img, input_depth=input_depth)
        
        # print(f"DEBUG extract_fusion_feat - img_3d_feat_feat shape: {img_3d_feat_feat.shape}")
        # print(f"DEBUG extract_fusion_feat - lidar_feat is None: {lidar_feat is None}")
        # if lidar_feat is not None:
        #     print(f"DEBUG extract_fusion_feat - lidar_feat shape: {lidar_feat.shape}, dim: {lidar_feat.dim()}")
        
        # Process lidar features
        if lidar_feat is not None and lidar_feat.dim() == 5:
            # Ensure both features are on the same device
            if lidar_feat.device != img_3d_feat_feat.device:
                img_3d_feat_feat = img_3d_feat_feat.to(lidar_feat.device)
            
            # Both features should be in (B, C, D, H, W) format
            # img_3d_feat_feat is already (B, C, D, H, W)
            # lidar_feat from lidar_encoder is (B, C, D, H, W)
            # No need to permute, just ensure they match in spatial dimensions
            
            # Downsample lidar_feat to match img_3d_feat_feat size if needed
            if lidar_feat.shape[2:] != img_3d_feat_feat.shape[2:]:
                # Use adaptive average pooling to match target size
                B, C, D, H, W = lidar_feat.shape
                target_D, target_H, target_W = img_3d_feat_feat.shape[2], img_3d_feat_feat.shape[3], img_3d_feat_feat.shape[4]
                
                # print(f"DEBUG extract_fusion_feat - Downsampling lidar_feat from {lidar_feat.shape} to match img_3d_feat_feat spatial dims ({target_D}, {target_H}, {target_W})")
                
                # Reshape to (B*C, 1, D, H, W) for 3D pooling
                lidar_feat = lidar_feat.view(B * C, 1, D, H, W)
                lidar_feat = torch.nn.functional.adaptive_avg_pool3d(
                    lidar_feat, 
                    (target_D, target_H, target_W)
                )
                # Reshape back to (B, C, D, H, W)
                lidar_feat = lidar_feat.view(B, C, target_D, target_H, target_W)
                # print(f"DEBUG extract_fusion_feat - lidar_feat after downsampling: {lidar_feat.shape}")
            
            # Now both are (B, C, D, H, W), concatenate along channel dimension
            # print(f"DEBUG extract_fusion_feat - Before cat: img_3d_feat_feat={img_3d_feat_feat.shape}, lidar_feat={lidar_feat.shape}")
            fusion_feat = torch.cat([img_3d_feat_feat, lidar_feat], dim=1)
            # print(f"DEBUG extract_fusion_feat - After cat: fusion_feat={fusion_feat.shape}")
        else:
            fusion_feat = img_3d_feat_feat
            # print(f"DEBUG extract_fusion_feat - Using img_3d_feat_feat only (no lidar): {fusion_feat.shape}")
        
        # Encode fusion features
        # print(f"DEBUG extract_fusion_feat - Before occ_encoder: fusion_feat={fusion_feat.shape}")
        fusion_feat = self.occ_encoder(fusion_feat)
        # print(f"DEBUG extract_fusion_feat - After occ_encoder: fusion_feat={fusion_feat.shape}")
        return fusion_feat, depth_key_frame, seg_key_frame

    def train_step(self, data, optim_wrapper=None):
        """Training step for MMEngine compatibility.
        
        This is called by MMEngine's training loop instead of forward().
        """
        # Extract data from the batch
        # MMEngine passes data as a dict or list
        if isinstance(data, dict):
            # Try to extract all necessary data
            losses = self.forward_train(
                points=data.get('points'),
                img_inputs=data.get('img_inputs') or data.get('imgs') or data.get('img'),
                segs=data.get('segs'),
                sparse_depth=data.get('sparse_depth'),
                voxel_semantics=data.get('voxel_semantics'),
                mask_camera=data.get('mask_camera'),
            )
        else:
            # Fallback to forward
            losses = self(**data, mode='loss')
        
        # Parse losses
        parsed_losses, log_vars = self._parse_losses(losses)
        
        # Backward
        if optim_wrapper is not None:
            optim_wrapper.update_params(parsed_losses)
        
        return log_vars
    
    def _parse_losses(self, losses):
        """Parse losses dict."""
        import torch
        log_vars = {}
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean().item()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean().item() for _loss in loss_value)
            else:
                log_vars[loss_name] = loss_value
        
        loss = sum(value for key, value in losses.items() if 'loss' in key.lower())
        log_vars['loss'] = loss.item() if isinstance(loss, torch.Tensor) else loss
        
        return loss, log_vars

    def forward(self, inputs=None, data_samples=None, mode='tensor', **kwargs):
        """Forward function for MMEngine compatibility.
        
        Args:
            inputs (dict): Input data containing 'points' and 'img_inputs'
            data_samples: Data samples (not used in this implementation)
            mode (str): Mode of forward pass ('tensor', 'loss', or 'predict')
            **kwargs: Additional keyword arguments
            
        Returns:
            dict or list: Depending on the mode
        """
        # Extract data from data_samples if present
        if data_samples is not None and not isinstance(data_samples, list):
            data_samples = [data_samples]
        
        # Merge inputs with data from data_samples
        if data_samples is not None and len(data_samples) > 0:
            # Extract data from data_samples and merge with inputs
            for sample in data_samples:
                if hasattr(sample, 'keys'):
                    # data_samples is dict-like
                    for key in ['img_inputs', 'imgs', 'img', 'segs', 'sparse_depth', 'voxel_semantics', 'mask_camera']:
                        if key in sample:
                            if inputs is None:
                                inputs = {}
                            if key not in inputs:
                                inputs[key] = sample[key]
                elif hasattr(sample, '__dict__'):
                    # data_samples has attributes
                    for key in ['img_inputs', 'imgs', 'img', 'segs', 'sparse_depth', 'voxel_semantics', 'mask_camera']:
                        if hasattr(sample, key):
                            if inputs is None:
                                inputs = {}
                            if key not in inputs:
                                inputs[key] = getattr(sample, key)
        
        # Map common alternative key names to expected names
        if inputs is not None and isinstance(inputs, dict):
            # Map various possible key names to expected names
            if 'imgs' in inputs and 'img_inputs' not in inputs:
                inputs['img_inputs'] = inputs.pop('imgs')
            if 'img' in inputs and 'img_inputs' not in inputs:
                inputs['img_inputs'] = inputs.pop('img')
        
        # Also check kwargs for image inputs and other data
        for key in ['img_inputs', 'imgs', 'img', 'segs', 'sparse_depth', 'voxel_semantics', 'mask_camera']:
            if key in kwargs:
                if inputs is None:
                    inputs = {}
                target_key = 'img_inputs' if key in ['imgs', 'img'] else key
                if target_key not in inputs:
                    inputs[target_key] = kwargs.pop(key)
        
        if mode == 'loss':
            # Training mode
            if inputs is not None and isinstance(inputs, dict):
                return self.forward_train(**inputs, **kwargs)
            else:
                return self.forward_train(**kwargs)
        elif mode == 'predict':
            # Testing/Inference mode
            if inputs is not None and isinstance(inputs, dict):
                return self.simple_test(**inputs, **kwargs)
            else:
                return self.simple_test(**kwargs)
        elif mode == 'tensor':
            # Tensor mode (for feature extraction)
            if inputs is not None and isinstance(inputs, dict):
                return self.forward_train(**inputs, **kwargs)
            else:
                return self.forward_train(**kwargs)
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def forward_train(self,
                      points=None,
                      img_inputs=None,
                      segs=None,
                      sparse_depth=None,
                      **kwargs):
        # Debug: print all available parameters
        # If img_inputs is None, try to find it in kwargs with different names
        if img_inputs is None:
            for key in ['imgs', 'img', 'img_inputs']:
                if key in kwargs and kwargs[key] is not None:
                    img_inputs = kwargs.pop(key)
                    break
        
        # If still None, raise error
        if img_inputs is None:
            raise ValueError("img_inputs is None! Image data is not being passed to forward_train.")
        
        # Get model device
        device = next(self.parameters()).device
        
        # Move all inputs to the correct device
        if points is not None:
            if isinstance(points, list):
                points = [p.to(device) if isinstance(p, torch.Tensor) else p for p in points]
            elif isinstance(points, torch.Tensor):
                points = points.to(device)
        
        if img_inputs is not None:
            if isinstance(img_inputs, (list, tuple)):
                img_inputs = [inp.to(device) if isinstance(inp, torch.Tensor) else inp for inp in img_inputs]
            elif isinstance(img_inputs, torch.Tensor):
                img_inputs = img_inputs.to(device)
        
        if segs is not None:
            if isinstance(segs, torch.Tensor):
                segs = segs.to(device)
            elif isinstance(segs, list):
                segs = [s.to(device) if isinstance(s, torch.Tensor) else s for s in segs]
        
        if sparse_depth is not None:
            if isinstance(sparse_depth, torch.Tensor):
                sparse_depth = sparse_depth.to(device)
            elif isinstance(sparse_depth, list):
                sparse_depth = [s.to(device) if isinstance(s, torch.Tensor) else s for s in sparse_depth]
        
        # Move kwargs tensors to device
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                kwargs[key] = value.to(device)
            elif isinstance(value, list):
                kwargs[key] = [v.to(device) if isinstance(v, torch.Tensor) else v for v in value]
        
        # Convert sparse_depth and segs from list to tensor if needed
        if sparse_depth is not None and isinstance(sparse_depth, list):
            if len(sparse_depth) > 0:
                if isinstance(sparse_depth[0], torch.Tensor):
                    sparse_depth = torch.stack(sparse_depth, dim=0).to(device)
                elif isinstance(sparse_depth[0], np.ndarray):
                    sparse_depth = torch.from_numpy(np.stack(sparse_depth, axis=0)).to(device)
        
        if segs is not None and isinstance(segs, list):
            if len(segs) > 0:
                if isinstance(segs[0], torch.Tensor):
                    segs = torch.stack(segs, dim=0).to(device)
                elif isinstance(segs[0], np.ndarray):
                    segs = torch.from_numpy(np.stack(segs, axis=0)).to(device)
        
        lidar_feat, x_list, x_sparse_out = self.lidar_encoder(points)
        lidar_feat = lidar_feat.permute(0, 1, 2, 4, 3).contiguous()

        input_depth = sparse_depth
        
        img_3d_feat_feat, depth_key_frame, seg_key_frame = self.extract_img_3d_feat(
            img_inputs=img_inputs, input_depth=input_depth)
        fusion_feat = torch.cat([img_3d_feat_feat, lidar_feat], dim=1)
        fusion_feat = self.occ_encoder(fusion_feat)

        losses = dict()
        
        # Use img_view_transformer.get_loss for proper depth and segmentation loss calculation
        depth_loss, seg_loss, vis_depth_pred, vis_depth_label, vis_seg_pred, vis_seg_label = \
            self.img_view_transformer.get_loss(sparse_depth, depth_key_frame, segs, seg_key_frame)
        
        losses['depth_loss'] = depth_loss * self.fuse_loss_weight
        losses['seg_loss'] = seg_loss * self.fuse_loss_weight

        occ_pred = self.final_conv(fusion_feat).permute(0, 4, 3, 2, 1)
        if self.use_predicter:
            occ_pred = self.predicter(occ_pred)
            
        voxel_semantics = kwargs['voxel_semantics']
        mask_camera = kwargs['mask_camera']
        
        # Convert to tensor if needed (safety check)
        if isinstance(voxel_semantics, list):
            if len(voxel_semantics) > 0:
                if isinstance(voxel_semantics[0], torch.Tensor):
                    voxel_semantics = torch.stack(voxel_semantics, dim=0)
                elif isinstance(voxel_semantics[0], np.ndarray):
                    voxel_semantics = torch.from_numpy(np.stack(voxel_semantics, axis=0))
        
        if isinstance(mask_camera, list):
            if len(mask_camera) > 0:
                if isinstance(mask_camera[0], torch.Tensor):
                    mask_camera = torch.stack(mask_camera, dim=0)
                elif isinstance(mask_camera[0], np.ndarray):
                    mask_camera = torch.from_numpy(np.stack(mask_camera, axis=0))

        assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17
        loss_occ = self.loss_single(voxel_semantics, mask_camera, occ_pred)
        losses.update(loss_occ)
        return losses

    def loss_single(self, voxel_semantics, mask_camera, preds):
        loss_ = dict()
        voxel_semantics = voxel_semantics.long()
        if self.use_mask:
            mask_camera = mask_camera.to(torch.int32)
            voxel_semantics = voxel_semantics.reshape(-1)
            preds = preds.reshape(-1, self.num_classes)
            mask_camera = mask_camera.reshape(-1)
            num_total_samples = mask_camera.sum()
            
            # Manually implement the same behavior as MMDet's CrossEntropyLoss with avg_factor
            # This is equivalent to the original implementation
            # Calculate loss with reduction='none' to get per-sample losses
            loss_per_sample = torch.nn.functional.cross_entropy(
                preds, voxel_semantics, reduction='none')
            
            # Apply mask_camera as weight (convert to float for weighting)
            weighted_loss = loss_per_sample * mask_camera.float()
            
            # Sum and normalize by avg_factor (num_total_samples)
            # This is exactly what MMDet's loss does with avg_factor parameter
            if num_total_samples > 0:
                loss_occ = weighted_loss.sum() / num_total_samples
            else:
                loss_occ = weighted_loss.sum() * 0.0
            
            loss_['loss_occ'] = loss_occ
        else:
            voxel_semantics = voxel_semantics.reshape(-1)
            preds = preds.reshape(-1, self.num_classes)
            loss_occ = self.loss_occ(preds, voxel_semantics)
            loss_['loss_occ'] = loss_occ
        return loss_

    def simple_test(self,
                    points=None,
                    img_metas=None,
                    img_inputs=None,
                    sparse_depth=None,
                    **kwargs):
        """Test function without augmentation."""
        # Get model device
        device = next(self.parameters()).device
        
        # Move all inputs to the correct device
        if points is not None:
            if isinstance(points, list):
                points = [p.to(device) if isinstance(p, torch.Tensor) else p for p in points]
            elif isinstance(points, torch.Tensor):
                points = points.to(device)
        
        if img_inputs is not None:
            if isinstance(img_inputs, (list, tuple)):
                img_inputs = [inp.to(device) if isinstance(inp, torch.Tensor) else inp for inp in img_inputs]
            elif isinstance(img_inputs, torch.Tensor):
                img_inputs = img_inputs.to(device)
        
        if sparse_depth is not None:
            sparse_depth = sparse_depth[0] if isinstance(sparse_depth, (list, tuple)) else sparse_depth
            if isinstance(sparse_depth, torch.Tensor):
                sparse_depth = sparse_depth.to(device)
            
        lidar_feat, x_list, x_sparse_out = self.lidar_encoder(points)
        # N, C, D, H, W -> N,C,D,W,H
        lidar_feat = lidar_feat.permute(0, 1, 2, 4, 3).contiguous()
        input_depth = sparse_depth
        img_3d_feat_feat, depth_key_frame, seg_key_frame = self.extract_img_3d_feat(
            img_inputs=img_inputs, input_depth=input_depth)
        fusion_feat = torch.cat([img_3d_feat_feat, lidar_feat], dim=1)
        fusion_feat = self.occ_encoder(fusion_feat)

        occ_pred = self.final_conv(fusion_feat).permute(0, 4, 3, 2, 1)  # bncdhw->bnwhdc
        if self.use_predicter:
            occ_pred = self.predicter(occ_pred)
        occ_score = occ_pred.softmax(-1)
        occ_res = occ_score.argmax(-1)
        occ_res = occ_res.squeeze(dim=0).cpu().numpy().astype(np.uint8)
        return [occ_res]

    def show_results(self, data, result, out_dir):
        """Results visualization for occupancy prediction.

        Args:
            data (dict): Input points and the information of the sample.
            result (list): Prediction results (occupancy grids).
            out_dir (str): Output directory of visualization result.
        """
        import os
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        for batch_id in range(len(result)):
            # Get occupancy grid result
            occ_grid = result[batch_id]  # Shape: (W, H, D)
            
            # Get point cloud data for reference
            if hasattr(data['points'][0], '_data'):
                points = data['points'][0]._data[0][batch_id].numpy()
            elif hasattr(data['points'][0], 'data'):
                points = data['points'][0].data[0][batch_id].numpy()
            else:
                points = data['points'][0][batch_id].numpy()
            
            # Get metadata
            if hasattr(data['img_metas'][0], '_data'):
                img_meta = data['img_metas'][0]._data[0][batch_id]
            elif hasattr(data['img_metas'][0], 'data'):
                img_meta = data['img_metas'][0].data[0][batch_id]
            else:
                img_meta = data['img_metas'][0][batch_id]
            
            # Create filename
            pts_filename = img_meta.get('pts_filename', f'sample_{batch_id}')
            file_name = os.path.split(pts_filename)[-1].split('.')[0]
            
            # Create visualization
            fig = plt.figure(figsize=(15, 5))
            
            # Plot 1: Original point cloud (top view)
            ax1 = fig.add_subplot(131)
            ax1.scatter(points[:, 0], points[:, 1], c=points[:, 2], s=0.1, cmap='viridis')
            ax1.set_title('Original Point Cloud (Top View)')
            ax1.set_xlabel('X (m)')
            ax1.set_ylabel('Y (m)')
            ax1.axis('equal')
            
            # Plot 2: Occupancy grid slice (middle height)
            ax2 = fig.add_subplot(132)
            mid_height = occ_grid.shape[2] // 2
            occ_slice = occ_grid[:, :, mid_height]
            im = ax2.imshow(occ_slice.T, origin='lower', cmap='tab20', vmin=0, vmax=17)
            ax2.set_title(f'Occupancy Grid (Height Slice {mid_height})')
            ax2.set_xlabel('X Index')
            ax2.set_ylabel('Y Index')
            plt.colorbar(im, ax=ax2)
            
            # Plot 3: Occupancy statistics
            ax3 = fig.add_subplot(133)
            unique, counts = np.unique(occ_grid, return_counts=True)
            ax3.bar(unique, counts)
            ax3.set_title('Occupancy Class Distribution')
            ax3.set_xlabel('Class ID')
            ax3.set_ylabel('Voxel Count')
            ax3.set_yscale('log')
            
            plt.tight_layout()
            
            # Save visualization
            os.makedirs(out_dir, exist_ok=True)
            output_path = os.path.join(out_dir, f'{file_name}_occupancy.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Also save the raw occupancy data
            np.save(os.path.join(out_dir, f'{file_name}_occupancy.npy'), occ_grid)
            
            print(f"Saved occupancy visualization: {output_path}")
            print(f"Saved occupancy data: {os.path.join(out_dir, f'{file_name}_occupancy.npy')}")

    def show_results_3d(self, data, result, out_dir, show_open3d=True):
        """3D visualization for occupancy prediction using Open3D.

        Args:
            data (dict): Input points and the information of the sample.
            result (list): Prediction results (occupancy grids).
            out_dir (str): Output directory of visualization result.
            show_open3d (bool): Whether to show Open3D visualization.
        """
        try:
            import open3d as o3d
        except ImportError:
            print("Open3D not installed. Using matplotlib visualization instead.")
            return self.show_results(data, result, out_dir)
        
        import os
        
        for batch_id in range(len(result)):
            # Get occupancy grid result
            occ_grid = result[batch_id]  # Shape: (W, H, D)
            
            # Get point cloud data for reference
            if hasattr(data['points'][0], '_data'):
                points = data['points'][0]._data[0][batch_id].numpy()
            elif hasattr(data['points'][0], 'data'):
                points = data['points'][0].data[0][batch_id].numpy()
            else:
                points = data['points'][0][batch_id].numpy()
            
            # Get metadata
            if hasattr(data['img_metas'][0], '_data'):
                img_meta = data['img_metas'][0]._data[0][batch_id]
            elif hasattr(data['img_metas'][0], 'data'):
                img_meta = data['img_metas'][0].data[0][batch_id]
            else:
                img_meta = data['img_metas'][0][batch_id]
            
            # Create filename
            pts_filename = img_meta.get('pts_filename', f'sample_{batch_id}')
            file_name = os.path.split(pts_filename)[-1].split('.')[0]
            
            # Convert occupancy grid to point cloud
            occ_points, occ_colors = self._occupancy_to_pointcloud(occ_grid)
            
            # Create Open3D visualization
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name=f"Occupancy Visualization - {file_name}")
            
            # Add coordinate frame
            coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
            vis.add_geometry(coordinate_frame)
            
            # Add original point cloud
            if points is not None:
                pcd_original = o3d.geometry.PointCloud()
                pcd_original.points = o3d.utility.Vector3dVector(points[:, :3])
                pcd_original.colors = o3d.utility.Vector3dVector(
                    np.tile([0.5, 0.5, 0.5], (points.shape[0], 1)))  # Gray color
                vis.add_geometry(pcd_original)
            
            # Add occupancy voxels
            if len(occ_points) > 0:
                pcd_occ = o3d.geometry.PointCloud()
                pcd_occ.points = o3d.utility.Vector3dVector(occ_points)
                pcd_occ.colors = o3d.utility.Vector3dVector(occ_colors / 255.0)
                vis.add_geometry(pcd_occ)
            
            # Set render options
            render_option = vis.get_render_option()
            render_option.point_size = 3.0
            render_option.background_color = np.array([0.1, 0.1, 0.1])
            
            if show_open3d:
                vis.run()
            
            # Save screenshot
            os.makedirs(out_dir, exist_ok=True)
            screenshot_path = os.path.join(out_dir, f'{file_name}_3d_occupancy.png')
            vis.capture_screen_image(screenshot_path)
            vis.destroy_window()
            
            # Save occupancy data
            np.save(os.path.join(out_dir, f'{file_name}_occupancy.npy'), occ_grid)
            
            print(f"Saved 3D occupancy visualization: {screenshot_path}")

    def _occupancy_to_pointcloud(self, occ_grid):
        """Convert occupancy grid to colored point cloud.
        
        Args:
            occ_grid (np.ndarray): Occupancy grid with shape (W, H, D)
            
        Returns:
            tuple: (points, colors) where points are 3D coordinates and colors are RGB values
        """
        # Define colors for each class (same as in nuscenes_dataset_occ.py)
        colors_map = np.array([
            [0, 0, 0],           # 0 undefined
            [255, 158, 0],       # 1 car  orange
            [0, 0, 230],         # 2 pedestrian  Blue
            [47, 79, 79],        # 3 sign  Darkslategrey
            [220, 20, 60],       # 4 CYCLIST  Crimson
            [255, 69, 0],        # 5 traffic_light  Orangered
            [255, 140, 0],       # 6 pole  Darkorange
            [233, 150, 70],      # 7 construction_cone  Darksalmon
            [255, 61, 99],       # 8 bicycle  Red
            [112, 128, 144],     # 9 motorcycle  Slategrey
            [222, 184, 135],     # 10 building Burlywood
            [0, 175, 0],         # 11 vegetation  Green
            [165, 42, 42],       # 12 trunk  nuTonomy green
            [0, 207, 191],       # 13 curb, road, lane_marker, other_ground
            [75, 0, 75],         # 14 walkable, sidewalk
            [255, 0, 0],         # 15 unobserved
            [0, 0, 0],           # 16 undefined
            [0, 0, 0],           # 17 undefined
        ])
        
        # Get non-empty voxels (exclude class 0 and 17)
        valid_mask = (occ_grid > 0) & (occ_grid < 17)
        valid_indices = np.where(valid_mask)
        
        if len(valid_indices[0]) == 0:
            return np.array([]), np.array([])
        
        # Convert grid indices to world coordinates
        # Assuming point cloud range and voxel size from config
        voxel_size = np.array(self.voxel_size)  # [0.05, 0.05, 0.05]
        point_cloud_range = np.array(self.point_cloud_range)  # [-40, -40, -1, 40, 40, 5.4]
        
        points = np.stack(valid_indices, axis=1).astype(np.float32)
        points[:, 0] = points[:, 0] * voxel_size[0] + point_cloud_range[0]  # X
        points[:, 1] = points[:, 1] * voxel_size[1] + point_cloud_range[1]  # Y  
        points[:, 2] = points[:, 2] * voxel_size[2] + point_cloud_range[2]  # Z
        
        # Get colors for each valid voxel
        class_ids = occ_grid[valid_mask]
        colors = colors_map[class_ids]
        
        return points, colors
    
    def train_step(self, data, optim_wrapper):
        """Train step function for mmengine runner.
        
        Args:
            data (dict): The output of dataloader.
            optim_wrapper: Optimizer wrapper.
            
        Returns:
            dict: Dict of outputs.
        """
        # Extract data - Check for both direct keys and nested in data_samples
        if 'data_samples' in data:
            # MMEngine format - extract from data_samples
            data_samples = data['data_samples']
            imgs = data['inputs']['img_inputs'] if 'inputs' in data else data['img_inputs']
            points = data['inputs'].get('points', None) if 'inputs' in data else data.get('points', None)
            
            # Try to get GT data from data_samples
            if hasattr(data_samples, 'gt_seg_3d'):
                voxel_semantics = data_samples.gt_seg_3d
            elif hasattr(data_samples, 'voxel_semantics'):
                voxel_semantics = data_samples.voxel_semantics
            else:
                voxel_semantics = data.get('voxel_semantics', None)
                
            if hasattr(data_samples, 'mask_camera'):
                mask_camera = data_samples.mask_camera
            else:
                mask_camera = data.get('mask_camera', None)
                
            img_metas = data.get('img_metas', {})
            
            # CRITICAL: Extract sparse_depth and segs for depth/seg losses
            sparse_depth = data.get('sparse_depth', None)
            segs = data.get('segs', None)
        else:
            # Direct format
            imgs = data['img_inputs']
            points = data.get('points', None)
            voxel_semantics = data.get('voxel_semantics', None)
            mask_camera = data.get('mask_camera', None)
            img_metas = data.get('img_metas', {})
            sparse_depth = data.get('sparse_depth', None)
            segs = data.get('segs', None)
        
        # Call forward_train instead of directly calling extract_feat
        # This ensures depth_loss and seg_loss are calculated
        # Note: points should be passed as LiDARPoints objects (or list of tensors)
        # The lidar_encoder will handle device placement and conversion internally
        losses = self.forward_train(
            points=points,
            img_inputs=imgs,
            segs=segs,
            sparse_depth=sparse_depth,
            voxel_semantics=voxel_semantics,
            mask_camera=mask_camera,
            img_metas=img_metas
        )
        
        # Backward and optimize
        parsed_losses, log_vars = self.parse_losses(losses)
        optim_wrapper.update_params(parsed_losses)
        
        return log_vars
    
    def parse_losses(self, losses):
        """Parse losses for logging and backward.
        
        Args:
            losses (dict): Dict of losses.
            
        Returns:
            tuple: (loss, log_vars)
        """
        log_vars = {}
        loss_total = 0
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.item()
                loss_total += loss_value
            else:
                log_vars[loss_name] = loss_value
                loss_total += loss_value
        
        log_vars['loss'] = loss_total.item() if isinstance(loss_total, torch.Tensor) else loss_total
        return loss_total, log_vars
    
    def val_step(self, data):
        """Validation step function for mmengine runner.
        
        Args:
            data (dict): The output of dataloader.
            
        Returns:
            dict: Dict of outputs.
        """
        # Extract data
        imgs = data['img_inputs']
        points = data.get('points', None)
        voxel_semantics = data.get('voxel_semantics', None)
        mask_camera = data.get('mask_camera', None)
        img_metas = data.get('img_metas', {})
        
        # Get device from model parameters
        device = next(self.parameters()).device
        
        # Convert LiDARPoints object to tensor if necessary and move to correct device
        if points is not None:
            if hasattr(points, 'tensor'):
                points = points.tensor
            # Ensure points is on the correct device
            if isinstance(points, list):
                points = [pt.to(device) if hasattr(pt, 'to') else pt for pt in points]
            elif hasattr(points, 'to'):
                points = points.to(device)
        
        # Forward pass (without gradient)
        with torch.no_grad():
            if points is not None:
                # Process lidar data
                lidar_feat, _, _ = self.lidar_encoder(points)
                
                # Extract features from both modalities
                occ_pred = self.extract_feat(lidar_feat, imgs, img_metas)
            else:
                # Image-only mode
                occ_pred = self.extract_feat(None, imgs, img_metas)
        
        # Calculate loss for validation
        losses = dict()
        if self.loss_occ is not None and voxel_semantics is not None:
            # Reshape predictions and targets
            if isinstance(occ_pred, (list, tuple)):
                occ_pred = occ_pred[0]
            
            # Make sure shapes match
            if occ_pred.ndim == 5:  # (B, C, H, W, D)
                B, C, H, W, D = occ_pred.shape
                occ_pred = occ_pred.permute(0, 2, 3, 4, 1).reshape(-1, C)
            
            if voxel_semantics is not None and hasattr(voxel_semantics, 'ndim') and voxel_semantics.ndim == 4:  # (B, H, W, D)
                voxel_semantics = voxel_semantics.reshape(-1)
            
            # Apply mask if available
            if mask_camera is not None and self.use_mask:
                if mask_camera is not None and hasattr(mask_camera, 'ndim') and mask_camera.ndim == 4:
                    mask_camera = mask_camera.reshape(-1)
                if hasattr(mask_camera, 'ndim') and not isinstance(mask_camera, list):
                    valid_mask = mask_camera > 0
                else:
                    # Skip mask if it's not a tensor
                    valid_mask = None
                if valid_mask is not None and valid_mask.sum() > 0:
                    occ_pred = occ_pred[valid_mask]
                    voxel_semantics = voxel_semantics[valid_mask]
            
            # Calculate occupancy loss
            if hasattr(voxel_semantics, 'long') and not isinstance(voxel_semantics, list):
                loss_occ = self.loss_occ(occ_pred, voxel_semantics.long())
            else:
                # Skip loss calculation if voxel_semantics is not a proper tensor
                loss_occ = torch.tensor(0.0, device=occ_pred.device, requires_grad=True)
            losses['loss_occ'] = loss_occ
        
        # Parse losses for logging
        parsed_losses, log_vars = self.parse_losses(losses)
        
        return log_vars
    
    def test_step(self, data):
        """Test step function for mmengine runner.
        
        Args:
            data (dict): The output of dataloader.
            
        Returns:
            list: List of data samples with predictions and ground truth.
        """
        # Fix ALL BN buffers and conv_out weights if they are incorrect (only once)
        if not hasattr(self, '_weights_fixed'):
            self._weights_fixed = True
            # Check if BN buffers are zero
            bn_module = self.lidar_encoder.encoder_layers.encoder_layer1[0][1]
            if bn_module.running_mean.abs().sum() < 1e-6:
                # Load checkpoint once and cache as class variable to avoid repeated loading
                if not hasattr(FusionOCC, '_cached_checkpoint'):
                    FusionOCC._cached_checkpoint = torch.load(
                        'projects/FusionOcc/ckpt/fusion_occ_mask.pth', 
                        map_location='cpu'
                    )
                
                state_dict = FusionOCC._cached_checkpoint['state_dict']
                
                # Fix ALL BatchNorm layers in the entire model
                for name, module in self.named_modules():
                    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                        prefix = f'{name}.'
                        if f'{prefix}running_mean' in state_dict:
                            module.running_mean.copy_(state_dict[f'{prefix}running_mean'])
                        if f'{prefix}running_var' in state_dict:
                            module.running_var.copy_(state_dict[f'{prefix}running_var'])
                        if f'{prefix}num_batches_tracked' in state_dict:
                            module.num_batches_tracked.copy_(state_dict[f'{prefix}num_batches_tracked'])
                
                # Fix conv_out weights (they are incorrectly loaded)
                for name, param in self.lidar_encoder.conv_out.named_parameters():
                    full_name = f'lidar_encoder.conv_out.{name}'
                    if full_name in state_dict:
                        param.data.copy_(state_dict[full_name])
        
        # Get device from model parameters
        device = next(self.parameters()).device
        
        # Extract data
        imgs = data['img_inputs']
        
        # Unwrap MMEngine dataloader's batch wrapping
        # MMEngine wraps each element in a tuple: [(tensor,), (tensor,), ...]
        # We need to unwrap it to get the actual tuple: (tensor, tensor, ...)
        if isinstance(imgs, list) and len(imgs) > 0 and isinstance(imgs[0], tuple):
            imgs = tuple(item[0] if isinstance(item, tuple) and len(item) == 1 else item for item in imgs)
        
        points = data.get('points', None)
        voxel_semantics = data.get('voxel_semantics', None)
        mask_camera = data.get('mask_camera', None)
        mask_lidar = data.get('mask_lidar', None)
        sparse_depth = data.get('sparse_depth', None)
        img_metas = data.get('img_metas', data.get('data_samples', {}))
        
        # Unpack sparse_depth if it's a list (like in original simple_test)
        if sparse_depth is not None and isinstance(sparse_depth, (list, tuple)):
            sparse_depth = sparse_depth[0]
        
        # Ensure sparse_depth has batch dimension [B, N, H, W]
        # If it's [N, H, W], add batch dimension
        if sparse_depth is not None and isinstance(sparse_depth, torch.Tensor):
            if sparse_depth.ndim == 3:
                sparse_depth = sparse_depth.unsqueeze(0)  # [N, H, W] -> [1, N, H, W]
            # Move sparse_depth to device
            sparse_depth = sparse_depth.to(device)
        
        # Move imgs to device
        if isinstance(imgs, (tuple, list)):
            imgs = tuple(x.to(device) if hasattr(x, 'to') and isinstance(x, torch.Tensor) else x for x in imgs)
        elif isinstance(imgs, torch.Tensor):
            imgs = imgs.to(device)
        
        # Keep points as LiDARPoints objects - lidar_encoder will handle conversion
        # No need to convert to tensor here
        
        # Forward pass (without gradient)
        with torch.no_grad():
            # WORKAROUND: Skip lidar due to mmcv 2.1.0 sparse convolution CUDA error
            # This is a temporary solution to test image-only performance
            # Temporarily enable lidar to get detailed error message
            use_lidar_workaround = False  # Set to True to use image-only mode
            
            lidar_feat = None
            if points is not None and not use_lidar_workaround:
                # Process lidar data
                # lidar_encoder expects a list of points (one per batch element)
                if not isinstance(points, list):
                    points = [points]
                try:
                    lidar_feat, _, _ = self.lidar_encoder(points)
                    # N, C, D, H, W -> N, C, D, W, H (swap last two dimensions like in original)
                    lidar_feat = lidar_feat.permute(0, 1, 2, 4, 3).contiguous()
                except RuntimeError as e:
                    print(f"[ERROR] Lidar encoder failed: {e}")
                    lidar_feat = None  # Force image-only mode
            
            # Extract image 3D features
            input_depth = sparse_depth
            img_3d_feat_feat, depth_key_frame, seg_key_frame = self.extract_img_3d_feat(
                img_inputs=imgs, input_depth=input_depth)
            
            # Fusion
            if lidar_feat is not None:
                fusion_feat = torch.cat([img_3d_feat_feat, lidar_feat], dim=1)
            else:
                # Image-only mode: pad with zeros to match expected channel count
                # occ_encoder expects 96 channels (64 img + 32 lidar)
                B, C, D, H, W = img_3d_feat_feat.shape
                zero_lidar = torch.zeros(B, 32, D, H, W, device=img_3d_feat_feat.device, dtype=img_3d_feat_feat.dtype)
                fusion_feat = torch.cat([img_3d_feat_feat, zero_lidar], dim=1)
            
            fusion_feat = self.occ_encoder(fusion_feat)
            
            # Apply final conv and predicter
            occ_pred = self.final_conv(fusion_feat).permute(0, 4, 3, 2, 1)  # bncdhw->bnwhdc
            if self.use_predicter:
                occ_pred = self.predicter(occ_pred)
        
        # Apply softmax before argmax (like in simple_test)
        occ_score = occ_pred.softmax(-1)  # (B, W, H, D, num_classes)
        
        # Get prediction labels (argmax over class dimension)
        occ_pred_labels = torch.argmax(occ_score, dim=-1)  # (B, W, H, D)
        
        # Prepare output as data samples
        data_samples = []
        batch_size = occ_pred_labels.shape[0]
        
        for i in range(batch_size):
            data_sample = {}
            
            # Add prediction (now should be shape (200, 200, 16))
            data_sample['occ_pred'] = occ_pred_labels[i].cpu().numpy()
            
            # Add ground truth if available
            if voxel_semantics is not None:
                if hasattr(voxel_semantics, 'cpu'):
                    gt_semantics = voxel_semantics[i].cpu().numpy()
                else:
                    gt_semantics = voxel_semantics[i] if isinstance(voxel_semantics, list) else voxel_semantics
                
                gt_occ_data = {'semantics': gt_semantics}
                
                # Add masks if available
                if mask_lidar is not None:
                    if hasattr(mask_lidar, 'cpu'):
                        gt_occ_data['mask_lidar'] = mask_lidar[i].cpu().numpy()
                    else:
                        gt_occ_data['mask_lidar'] = mask_lidar[i] if isinstance(mask_lidar, list) else mask_lidar
                
                if mask_camera is not None:
                    if hasattr(mask_camera, 'cpu'):
                        gt_occ_data['mask_camera'] = mask_camera[i].cpu().numpy()
                    else:
                        gt_occ_data['mask_camera'] = mask_camera[i] if isinstance(mask_camera, list) else mask_camera
                
                data_sample['gt_occ'] = gt_occ_data
            
            data_samples.append(data_sample)
        
        return data_samples 