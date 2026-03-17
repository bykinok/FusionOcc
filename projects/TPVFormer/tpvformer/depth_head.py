# Copyright (c) OpenMMLab. All rights reserved.
"""Auxiliary depth head for TPVFormer (Depth Supervision).

Uses LiDAR-derived gt_depth in ego frame (same as TPVFormer lidar2img=ego2img).
BCE loss on depth bins; only used at training. Inference does not use this head.

Loss normalization follows the per-element mean convention:
  sum(BCE) / (num_fg_pixels × D)
which gives loss values comparable in scale to the occupancy CE/Lovász losses.
This differs from the raw BEVDet-style formula (sum / num_fg only), which
produces D×-inflated loss values requiring a very small loss_weight to compensate.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmdet3d.registry import MODELS
from mmdet.models.backbones.resnet import BasicBlock
from mmcv.cnn import build_conv_layer, build_norm_layer


def _get_num_depth_bins(grid_config_depth):
    """Number of depth bins from grid_config['depth'] = [min, max, step]."""
    d_min, d_max, step = grid_config_depth[0], grid_config_depth[1], grid_config_depth[2]
    return int(round((d_max - d_min) / step))


@MODELS.register_module()
class AuxiliaryDepthHead(BaseModule):
    """Auxiliary depth prediction head for auxiliary depth supervision.

    Predicts per-pixel depth distribution (bins) from one FPN level;
    supervised by LiDAR-derived gt_depth (ego-frame) with BCE.

    The BCE loss is normalized per element (num_fg × D), so loss_weight
    controls what fraction of the total occupancy loss the depth term contributes.
    With loss_weight=0.5 the depth term is approximately half the occupancy loss.
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
        """BCE loss between one-hot gt depth bins and predicted depth distribution.

        Normalization: sum(BCE) / (num_fg_pixels × D)
        This is equivalent to reduction='mean' over foreground elements only,
        giving a per-element average BCE that is directly comparable in scale
        to the occupancy CE/Lovász losses.
        """
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
        # Normalize by num_fg × D (per-element mean) so that loss_weight has an
        # intuitive scale relative to the occupancy losses.
        with torch.amp.autocast('cuda', enabled=False):
            depth_loss = F.binary_cross_entropy_with_logits(
                depth_preds.float(),
                depth_labels,
                reduction='none',
            ).sum() / max(1.0, fg_mask.sum().float() * self.D)
        return self.loss_weight * depth_loss

    def forward(self, x):
        """x: [B, N, C, H, W] -> [B, N, D, H, W] logits."""
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        logits = self.depth_conv(x)
        return logits.view(B, N, self.D, H, W)


# ---------------------------------------------------------------------------
# BEVDet-style Depth Auxiliary Head
# Ref: BEVDepth (Li et al., AAAI 2023) — depth branch only, no camera conditioning
# ---------------------------------------------------------------------------

class _ASPPModule(nn.Module):
    """Atrous convolution block used inside ASPP."""

    def __init__(self, inplanes, planes, kernel_size, padding, dilation, norm_cfg):
        super().__init__()
        self.atrous_conv = nn.Conv2d(
            inplanes, planes, kernel_size=kernel_size,
            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = build_norm_layer(norm_cfg, planes)[1]
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.atrous_conv(x)))


class _ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling for multi-scale depth context."""

    def __init__(self, inplanes, mid_channels, norm_cfg=None):
        super().__init__()
        if norm_cfg is None:
            norm_cfg = dict(type='BN2d', eps=1e-3, momentum=0.01)
        self.aspp1 = _ASPPModule(inplanes, mid_channels, 1,  padding=0,  dilation=1,  norm_cfg=norm_cfg)
        self.aspp2 = _ASPPModule(inplanes, mid_channels, 3,  padding=6,  dilation=6,  norm_cfg=norm_cfg)
        self.aspp3 = _ASPPModule(inplanes, mid_channels, 3,  padding=12, dilation=12, norm_cfg=norm_cfg)
        self.aspp4 = _ASPPModule(inplanes, mid_channels, 3,  padding=18, dilation=18, norm_cfg=norm_cfg)
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, mid_channels, 1, bias=False),
            build_norm_layer(norm_cfg, mid_channels)[1],
            nn.ReLU(inplace=True),
        )
        self.conv1 = nn.Conv2d(mid_channels * 5, mid_channels, 1, bias=False)
        self.bn1 = build_norm_layer(norm_cfg, mid_channels)[1]
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = F.interpolate(
            self.global_avg_pool(x), size=x4.shape[2:],
            mode='bilinear', align_corners=True)
        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        return self.dropout(self.relu(self.bn1(self.conv1(x))))


@MODELS.register_module()
class BEVDetStyleAuxDepthHead(AuxiliaryDepthHead):
    """BEVDet-style auxiliary depth head (no camera conditioning).

    Architecture: reduce_conv(3×3,BN,ReLU) → 3×BasicBlock → ASPP → DCN → depth_conv(1×1)

    Compared with AuxiliaryDepthHead (single 1×1 conv), this provides:
      - Larger receptive field via stacked BasicBlocks
      - Multi-scale atrous context via ASPP (dilation 1/6/12/18)
      - Deformable spatial attention via DCN

    Loss computation (get_downsampled_gt_depth, get_depth_loss) is fully inherited
    from AuxiliaryDepthHead; only the feature extraction network is upgraded.

    Args:
        in_channels (int): FPN output channels (BEVFormer/TPVFormer: 256).
        mid_channels (int): Internal channel width (default 256).
        grid_config / downsample / loss_weight: forwarded to AuxiliaryDepthHead.
        norm_cfg (dict): Normalisation for BN layers (default BN2d).
    """

    def __init__(self,
                 in_channels=256,
                 mid_channels=256,
                 grid_config=None,
                 downsample=16,
                 loss_weight=0.5,
                 norm_cfg=None,
                 init_cfg=None):
        super().__init__(
            in_channels=in_channels,
            grid_config=grid_config,
            downsample=downsample,
            loss_weight=loss_weight,
            init_cfg=init_cfg,
        )
        if norm_cfg is None:
            norm_cfg = dict(type='BN2d', eps=1e-3, momentum=0.01)

        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            build_norm_layer(norm_cfg, mid_channels)[1],
            nn.ReLU(inplace=True),
        )
        # Override the simple 1×1 depth_conv from AuxiliaryDepthHead
        self.depth_conv = nn.Sequential(
            BasicBlock(mid_channels, mid_channels, norm_cfg=norm_cfg),
            BasicBlock(mid_channels, mid_channels, norm_cfg=norm_cfg),
            BasicBlock(mid_channels, mid_channels, norm_cfg=norm_cfg),
            _ASPP(mid_channels, mid_channels, norm_cfg=norm_cfg),
            build_conv_layer(
                cfg=dict(type='DCN', in_channels=mid_channels, out_channels=mid_channels,
                         kernel_size=3, padding=1, groups=4, im2col_step=128)),
            nn.Conv2d(mid_channels, self.D, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        """x: [B, N, C, H, W] -> [B, N, D, H, W] logits."""
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        x = self.reduce_conv(x)
        logits = self.depth_conv(x)
        return logits.view(B, N, self.D, H, W)
