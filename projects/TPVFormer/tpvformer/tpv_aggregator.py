import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule

from mmdet3d.registry import MODELS


@MODELS.register_module()
class TPVAggregator(BaseModule):
    """TPVFormer Occupancy Aggregator for 3D occupancy prediction.
    
    This module aggregates TPV features from three orthogonal views (H-W, Z-H, W-Z)
    to predict 3D occupancy grid.
    """

    def __init__(self,
                 tpv_h,
                 tpv_w,
                 tpv_z,
                 nbr_classes=18,
                 in_dims=256,
                 hidden_dims=512,
                 out_dims=256,
                 scale_h=1,
                 scale_w=1,
                 scale_z=1,
                 use_checkpoint=True):
        super().__init__()
        self.tpv_h = tpv_h
        self.tpv_w = tpv_w
        self.tpv_z = tpv_z
        self.nbr_classes = nbr_classes
        self.scale_h = scale_h
        self.scale_w = scale_w
        self.scale_z = scale_z
        self.use_checkpoint = use_checkpoint

        # Feature decoder
        self.decoder = nn.Sequential(
            nn.Linear(in_dims, hidden_dims),
            nn.Softplus(),
            nn.Linear(hidden_dims, out_dims)
        )

        # Final classifier for occupancy prediction
        self.classifier = nn.Linear(out_dims, nbr_classes)
        self.classes = nbr_classes

    def forward(self, tpv_list, points=None):
        """Forward function for occupancy prediction.
        
        Args:
            tpv_list: List of TPV features [tpv_hw, tpv_zh, tpv_wz]
                - tpv_hw: (B, H*W, C) - Height-Width view
                - tpv_zh: (B, Z*H, C) - Depth-Height view  
                - tpv_wz: (B, W*Z, C) - Width-Depth view
            points: Optional point cloud coordinates for point-wise prediction
                    (B, N, 3) in grid coordinates
        
        Returns:
            If points is not None:
                tuple: (logits_vox, logits_pts)
                    - logits_vox: (B, C, W, H, Z) voxel occupancy logits
                    - logits_pts: (B, C, N, 1, 1) point occupancy logits
            Else:
                torch.Tensor: logits of shape (B, C, W, H, Z)
        """

        # breakpoint()

        tpv_hw, tpv_zh, tpv_wz = tpv_list[0], tpv_list[1], tpv_list[2]
        bs, _, c = tpv_hw.shape
        
        # Reshape TPV features to 3D spatial dimensions
        tpv_hw = tpv_hw.permute(0, 2, 1).reshape(bs, c, self.tpv_h, self.tpv_w)
        tpv_zh = tpv_zh.permute(0, 2, 1).reshape(bs, c, self.tpv_z, self.tpv_h)
        tpv_wz = tpv_wz.permute(0, 2, 1).reshape(bs, c, self.tpv_w, self.tpv_z)
        
        # Scale features if needed
        if self.scale_h != 1 or self.scale_w != 1:
            tpv_hw = F.interpolate(
                tpv_hw,
                size=(self.tpv_h * self.scale_h, self.tpv_w * self.scale_w),
                mode='bilinear')
        if self.scale_z != 1 or self.scale_h != 1:
            tpv_zh = F.interpolate(
                tpv_zh,
                size=(self.tpv_z * self.scale_z, self.tpv_h * self.scale_h),
                mode='bilinear')
        if self.scale_w != 1 or self.scale_z != 1:
            tpv_wz = F.interpolate(
                tpv_wz,
                size=(self.tpv_w * self.scale_w, self.tpv_z * self.scale_z),
                mode='bilinear')
        
        if points is not None:
            # Process both voxel and point predictions (following tpv04 implementation)
            _, n, _ = points.shape
            
            points = points.reshape(bs, 1, n, 3)
            
            # Normalize points to [-1, 1] for grid_sample
            points[..., 0] = points[..., 0] / (self.tpv_w * self.scale_w) * 2 - 1
            points[..., 1] = points[..., 1] / (self.tpv_h * self.scale_h) * 2 - 1
            points[..., 2] = points[..., 2] / (self.tpv_z * self.scale_z) * 2 - 1
            
            # Sample features at point locations
            # Original checkpoint was trained with PyTorch < 1.3.0 where align_corners=True was default
            sample_loc = points[:, :, :, [0, 1]]
            tpv_hw_pts = F.grid_sample(tpv_hw, sample_loc, align_corners=True).squeeze(2)  # bs, c, n
            sample_loc = points[:, :, :, [1, 2]]
            tpv_zh_pts = F.grid_sample(tpv_zh, sample_loc, align_corners=True).squeeze(2)
            sample_loc = points[:, :, :, [2, 0]]
            tpv_wz_pts = F.grid_sample(tpv_wz, sample_loc, align_corners=True).squeeze(2)
            
            # Expand TPV features to 3D volume
            tpv_hw_vox = tpv_hw.unsqueeze(-1).permute(0, 1, 3, 2, 4).expand(-1, -1, -1, -1, self.scale_z * self.tpv_z)
            tpv_zh_vox = tpv_zh.unsqueeze(-1).permute(0, 1, 4, 3, 2).expand(-1, -1, self.scale_w * self.tpv_w, -1, -1)
            tpv_wz_vox = tpv_wz.unsqueeze(-1).permute(0, 1, 2, 4, 3).expand(-1, -1, -1, self.scale_h * self.tpv_h, -1)
            
            # Fuse features
            fused_vox = (tpv_hw_vox + tpv_zh_vox + tpv_wz_vox).flatten(2)
            fused_pts = tpv_hw_pts + tpv_zh_pts + tpv_wz_pts
            fused = torch.cat([fused_vox, fused_pts], dim=-1)  # bs, c, whz+n
            
            fused = fused.permute(0, 2, 1)
            if self.use_checkpoint:
                fused = torch.utils.checkpoint.checkpoint(self.decoder, fused)
                logits = torch.utils.checkpoint.checkpoint(self.classifier, fused)
            else:
                fused = self.decoder(fused)
                logits = self.classifier(fused)
            logits = logits.permute(0, 2, 1)
            logits_vox = logits[:, :, :(-n)].reshape(bs, self.classes, self.scale_w * self.tpv_w, self.scale_h * self.tpv_h, self.scale_z * self.tpv_z)
            logits_pts = logits[:, :, (-n):].reshape(bs, self.classes, n, 1, 1)
            return logits_vox, logits_pts
            
        else:
            # Voxel-only prediction
            tpv_hw = tpv_hw.unsqueeze(-1).permute(0, 1, 3, 2, 4).expand(-1, -1, -1, -1, self.scale_z * self.tpv_z)
            tpv_zh = tpv_zh.unsqueeze(-1).permute(0, 1, 4, 3, 2).expand(-1, -1, self.scale_w * self.tpv_w, -1, -1)
            tpv_wz = tpv_wz.unsqueeze(-1).permute(0, 1, 2, 4, 3).expand(-1, -1, -1, self.scale_h * self.tpv_h, -1)
            
            fused = tpv_hw + tpv_zh + tpv_wz
            fused = fused.permute(0, 2, 3, 4, 1)
            if self.use_checkpoint:
                fused = torch.utils.checkpoint.checkpoint(self.decoder, fused)
                logits = torch.utils.checkpoint.checkpoint(self.classifier, fused)
            else:
                fused = self.decoder(fused)
                logits = self.classifier(fused)
            logits = logits.permute(0, 4, 1, 2, 3)
            
            return logits
