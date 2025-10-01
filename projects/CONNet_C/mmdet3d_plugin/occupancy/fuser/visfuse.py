import random
from typing import List

import torch
from torch import nn
import torch.nn.functional as F

from mmdet3d.registry import MODELS as FUSION_LAYERS
from mmcv.cnn import build_norm_layer


@FUSION_LAYERS.register_module()
class VisFuser(nn.Module):
    def __init__(self, in_channels, out_channels, norm_cfg=None) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if norm_cfg is None:
            norm_cfg = dict(type='BN3d', eps=1e-3, momentum=0.01)
            
        # We'll create the convolution layers on first forward pass
        # This lets us dynamically adapt to the actual input channel dimensions
        self.img_enc = None
        self.pts_enc = None
        self.norm_cfg = norm_cfg
        self.vis_enc = nn.Sequential(
            nn.Conv3d(2*out_channels, 16, 3, padding=1, bias=False),
            build_norm_layer(norm_cfg, 16)[1],
            # nn.BatchNorm3d(16),
            nn.ReLU(True),
            nn.Conv3d(16, 1, 1, padding=0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, img_voxel_feats, pts_voxel_feats):
        device = img_voxel_feats.device if hasattr(img_voxel_feats, 'device') else \
                 pts_voxel_feats.device if hasattr(pts_voxel_feats, 'device') else \
                 torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                 
        # Handle list input from ResNet3D
        if isinstance(img_voxel_feats, list):
            # Use the last feature map (highest level)
            img_voxel_feats = img_voxel_feats[-1]
        
        # Handle list input from pts encoder
        if isinstance(pts_voxel_feats, list):
            # Use the last feature map (highest level)
            pts_voxel_feats = pts_voxel_feats[-1]
            
        # Create encoders dynamically on first forward pass
        if self.img_enc is None:
            img_in_channels = img_voxel_feats.shape[1]  # Get actual channel count
            self.img_enc = nn.Sequential(
                nn.Conv3d(img_in_channels, self.out_channels, 7, padding=3, bias=False),
                build_norm_layer(self.norm_cfg, self.out_channels)[1],
                nn.ReLU(True),
            ).to(device)
            
        if self.pts_enc is None:
            pts_in_channels = pts_voxel_feats.shape[1]  # Get actual channel count
            self.pts_enc = nn.Sequential(
                nn.Conv3d(pts_in_channels, self.out_channels, 7, padding=3, bias=False),
                build_norm_layer(self.norm_cfg, self.out_channels)[1],
                nn.ReLU(True),
            ).to(device)
            
        # Apply encoding
        img_voxel_feats = self.img_enc(img_voxel_feats)
        pts_voxel_feats = self.pts_enc(pts_voxel_feats)
        
        # Ensure shapes match for concatenation
        if img_voxel_feats.shape[2:] != pts_voxel_feats.shape[2:]:
            # Resize pts_voxel_feats to match img_voxel_feats
            pts_voxel_feats = F.interpolate(
                pts_voxel_feats, 
                size=img_voxel_feats.shape[2:],
                mode='trilinear',
                align_corners=False
            )
            
        # Calculate visibility weights and fuse features
        vis_weight = self.vis_enc(torch.cat([img_voxel_feats, pts_voxel_feats], dim=1))
        voxel_feats = vis_weight * img_voxel_feats + (1 - vis_weight) * pts_voxel_feats

        return voxel_feats
