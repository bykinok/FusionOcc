# Copyright (c) OpenMMLab. All rights reserved.
"""Auxiliary depth head for TPVFormer (Depth Supervision).

Uses LiDAR-derived gt_depth in ego frame (same as TPVFormer lidar2img=ego2img).
BCE loss on depth bins; only used at training. Inference does not use this head.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmdet3d.registry import MODELS


def _get_num_depth_bins(grid_config_depth):
    """Number of depth bins from grid_config['depth'] = [min, max, step]."""
    d_min, d_max, step = grid_config_depth[0], grid_config_depth[1], grid_config_depth[2]
    return int(round((d_max - d_min) / step))


@MODELS.register_module()
class AuxiliaryDepthHead(BaseModule):
    """Auxiliary depth prediction head for auxiliary depth supervision.

    Predicts per-pixel depth distribution (bins) from one FPN level;
    supervised by LiDAR-derived gt_depth (ego-frame) with BCE.
    """

    def __init__(self,
                 in_channels=256,
                 grid_config=None,
                 downsample=16,
                 loss_weight=0.5,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        if grid_config is None:
            grid_config = dict(depth=[1.0, 45.0, 0.5])
        self.grid_config = grid_config
        self.downsample = downsample
        self.loss_weight = loss_weight
        self.D = _get_num_depth_bins(grid_config['depth'])
        self.sid = False
        self.depth_conv = nn.Conv2d(in_channels, self.D, kernel_size=1, padding=0)

    def get_downsampled_gt_depth(self, gt_depths):
        """Convert gt depth maps to one-hot depth bins at feature resolution.

        Args:
            gt_depths: [B, N, H, W]
        Returns:
            [B*N*h*w, D] one-hot (or multi-hot) depth labels
        """
        if isinstance(gt_depths, (list, tuple)):
            gt_depths = torch.stack([t if torch.is_tensor(t) else torch.tensor(t) for t in gt_depths])
        B, N, H, W = gt_depths.shape
        # Crop to multiples of downsample so view() is valid (pipeline may produce non-divisible sizes)
        H_crop = (H // self.downsample) * self.downsample
        W_crop = (W // self.downsample) * self.downsample
        if H_crop == 0 or W_crop == 0:
            raise ValueError(
                f'get_downsampled_gt_depth: H or W too small for downsample={self.downsample}. '
                f'Got H={H}, W={W} -> H_crop={H_crop}, W_crop={W_crop}.')
        gt_depths = gt_depths[..., :H_crop, :W_crop]
        H, W = H_crop, W_crop
        gt_depths = gt_depths.view(
            B * N, H // self.downsample, self.downsample,
            W // self.downsample, self.downsample, 1)
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(-1, self.downsample * self.downsample)
        gt_depths_tmp = torch.where(
            gt_depths == 0.0,
            1e5 * torch.ones_like(gt_depths),
            gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(B * N, H // self.downsample, W // self.downsample)

        d0, d1, d2 = self.grid_config['depth'][0], self.grid_config['depth'][1], self.grid_config['depth'][2]
        if not self.sid:
            gt_depths = (gt_depths - (d0 - d2)) / d2
        else:
            gt_depths = torch.log(gt_depths) - torch.log(torch.tensor(d0, dtype=gt_depths.dtype, device=gt_depths.device))
            gt_depths = gt_depths * (self.D - 1) / torch.log(
                torch.tensor(d1 - 1.0, dtype=gt_depths.dtype, device=gt_depths.device) / d0)
            gt_depths = gt_depths + 1.0
        gt_depths = torch.where(
            (gt_depths < self.D + 1) & (gt_depths >= 0.0),
            gt_depths,
            torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(gt_depths.long(), num_classes=self.D + 1).view(-1, self.D + 1)[:, 1:]
        return gt_depths.float()

    def get_depth_loss(self, depth_labels, depth_preds):
        """BCE loss between one-hot gt depth bins and predicted depth distribution."""
        # GT grid size (same as in get_downsampled_gt_depth)
        if isinstance(depth_labels, (list, tuple)):
            depth_labels = torch.stack([t if torch.is_tensor(t) else torch.as_tensor(t) for t in depth_labels])
        B, N, H, W = depth_labels.shape
        H_crop = (H // self.downsample) * self.downsample
        W_crop = (W // self.downsample) * self.downsample
        h_gt, w_gt = H_crop // self.downsample, W_crop // self.downsample

        depth_labels = self.get_downsampled_gt_depth(depth_labels)
        # depth_preds: [B, N, D, H_feat, W_feat] — interpolate to (h_gt, w_gt) to match GT grid
        depth_preds = F.interpolate(
            depth_preds.view(B * N, self.D, depth_preds.shape[3], depth_preds.shape[4]),
            size=(h_gt, w_gt),
            mode='bilinear',
            align_corners=False,
        )
        depth_preds = depth_preds.view(B * N, self.D, h_gt, w_gt).permute(0, 2, 3, 1).contiguous().view(-1, self.D)
        fg_mask = torch.max(depth_labels, dim=1).values > 0.0
        depth_labels = depth_labels[fg_mask]
        depth_preds = depth_preds[fg_mask]
        if fg_mask.sum() == 0:
            return depth_preds.sum() * 0.0
        # depth_preds are logits; use BCEWithLogits (expects logits, not [0,1])
        with torch.amp.autocast('cuda', enabled=False):
            depth_loss = F.binary_cross_entropy_with_logits(
                depth_preds.float(),
                depth_labels,
                reduction='none',
            ).sum() / max(1.0, fg_mask.sum().float())
        return self.loss_weight * depth_loss

    def forward(self, x):
        """x: [B, N, C, H, W] -> [B, N, D, H, W] logits."""
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        logits = self.depth_conv(x)
        return logits.view(B, N, self.D, H, W)
