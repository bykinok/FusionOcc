"""Auxiliary depth supervision heads for SparseOcc.

BEV augmentation 이후 ego-frame LiDAR point cloud를 카메라별 depth map으로 변환한
gt_depth를 ground truth로 사용하는 보조 depth supervision head.

Loss normalization:
  sum(BCE) / (num_fg_pixels × D)
→ per-element mean BCE로, occupancy CE/Lovász loss와 스케일을 맞춘다.
  loss_weight=0.5이면 depth 손실이 occupancy 손실의 약 절반.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmdet3d.registry import MODELS
from mmdet.models.backbones.resnet import BasicBlock
from mmcv.cnn import build_conv_layer, build_norm_layer


def _num_depth_bins(grid_config_depth):
    """grid_config['depth'] = [d_min, d_max, step] → bin 수."""
    d_min, d_max, step = grid_config_depth
    return int(round((d_max - d_min) / step))


@MODELS.register_module()
class AuxDepthHead(BaseModule):
    """보조 depth prediction head (단순 1×1 conv 버전).

    FPN 한 레벨의 feature [B, N, C, H, W]를 받아
    per-pixel depth distribution [B, N, D, H, W] logits를 예측하고,
    gt_depth [B, N, H_img, W_img]에 대해 BCE loss를 계산한다.

    Args:
        in_channels (int): FPN 출력 채널 수 (기본 256).
        grid_config (dict): depth 범위 dict, 예 ``dict(depth=[1.0, 45.0, 0.5])``.
        downsample (int): 이미지 대비 depth map 다운샘플 배율 (기본 16).
        loss_weight (float): depth loss 가중치 (기본 0.5).
    """

    def __init__(self,
                 in_channels: int = 256,
                 grid_config: dict = None,
                 downsample: int = 16,
                 loss_weight: float = 0.5,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        if grid_config is None:
            grid_config = dict(depth=[1.0, 45.0, 0.5])
        self.grid_config = grid_config
        self.downsample = downsample
        self.loss_weight = loss_weight
        self.D = _num_depth_bins(grid_config['depth'])
        self.sid = False  # linear binning (not scale-invariant)
        self.depth_conv = nn.Conv2d(in_channels, self.D, kernel_size=1, padding=0)

    # ------------------------------------------------------------------
    # GT 처리
    # ------------------------------------------------------------------

    def _get_downsampled_gt_depth(self, gt_depths):
        """gt depth map [B, N, H, W] → one-hot depth bins [B*N*h*w, D].

        각 spatial cell에서 가장 가까운(최솟값) depth를 대표 depth로 사용.
        """
        if isinstance(gt_depths, (list, tuple)):
            gt_depths = torch.stack(
                [t if torch.is_tensor(t) else torch.tensor(t) for t in gt_depths])
        B, N, H, W = gt_depths.shape
        H_crop = (H // self.downsample) * self.downsample
        W_crop = (W // self.downsample) * self.downsample
        if H_crop == 0 or W_crop == 0:
            raise ValueError(
                f'H or W too small for downsample={self.downsample}. '
                f'Got H={H}, W={W}.')
        gt_depths = gt_depths[..., :H_crop, :W_crop].contiguous()
        H, W = H_crop, W_crop

        gt_depths = gt_depths.view(
            B * N,
            H // self.downsample, self.downsample,
            W // self.downsample, self.downsample, 1)
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(-1, self.downsample * self.downsample)
        # 0(무효)를 제외하고 최솟값 선택
        gt_depths_tmp = torch.where(
            gt_depths == 0.0,
            1e5 * torch.ones_like(gt_depths),
            gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(
            B * N, H // self.downsample, W // self.downsample)

        d0, d1, d2 = (self.grid_config['depth'][0],
                      self.grid_config['depth'][1],
                      self.grid_config['depth'][2])
        gt_depths = (gt_depths - (d0 - d2)) / d2
        gt_depths = torch.where(
            (gt_depths < self.D + 1) & (gt_depths >= 0.0),
            gt_depths,
            torch.zeros_like(gt_depths))
        gt_depths = (F.one_hot(gt_depths.long(), num_classes=self.D + 1)
                     .view(-1, self.D + 1)[:, 1:])
        return gt_depths.float()

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def get_depth_loss(self, gt_depths, depth_preds):
        """BCE loss: gt depth bins vs predicted depth logits.

        Args:
            gt_depths  (Tensor): [B, N, H_img, W_img]
            depth_preds (Tensor): [B, N, D, H_feat, W_feat]

        Returns:
            Tensor: scalar depth loss.
        """
        if isinstance(gt_depths, (list, tuple)):
            gt_depths = torch.stack(
                [t if torch.is_tensor(t) else torch.as_tensor(t) for t in gt_depths])
        B, N, H, W = gt_depths.shape
        H_crop = (H // self.downsample) * self.downsample
        W_crop = (W // self.downsample) * self.downsample
        h_gt = H_crop // self.downsample
        w_gt = W_crop // self.downsample

        depth_labels = self._get_downsampled_gt_depth(gt_depths)

        # pred를 gt 해상도에 맞게 보간
        depth_preds = F.interpolate(
            depth_preds.view(B * N, self.D,
                             depth_preds.shape[3], depth_preds.shape[4]),
            size=(h_gt, w_gt),
            mode='bilinear',
            align_corners=False,
        )
        depth_preds = (depth_preds
                       .view(B * N, self.D, h_gt, w_gt)
                       .permute(0, 2, 3, 1)
                       .contiguous()
                       .view(-1, self.D))

        fg_mask = torch.max(depth_labels, dim=1).values > 0.0
        depth_labels = depth_labels[fg_mask]
        depth_preds = depth_preds[fg_mask]

        if fg_mask.sum() == 0:
            return depth_preds.sum() * 0.0

        with torch.amp.autocast('cuda', enabled=False):
            loss = F.binary_cross_entropy_with_logits(
                depth_preds.float(),
                depth_labels,
                reduction='none',
            ).sum() / max(1.0, fg_mask.sum().float() * self.D)

        return self.loss_weight * loss

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x):
        """x: [B, N, C, H, W] → [B, N, D, H, W] logits."""
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        logits = self.depth_conv(x)
        return logits.view(B, N, self.D, H, W)


# ---------------------------------------------------------------------------
# ASPP helper modules
# ---------------------------------------------------------------------------

class _ASPPModule(nn.Module):
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
    """Atrous Spatial Pyramid Pooling (다중 스케일 context)."""

    def __init__(self, inplanes, mid_channels, norm_cfg=None):
        super().__init__()
        if norm_cfg is None:
            norm_cfg = dict(type='BN2d', eps=1e-3, momentum=0.01)
        self.aspp1 = _ASPPModule(inplanes, mid_channels, 1,  0,  1,  norm_cfg)
        self.aspp2 = _ASPPModule(inplanes, mid_channels, 3,  6,  6,  norm_cfg)
        self.aspp3 = _ASPPModule(inplanes, mid_channels, 3,  12, 12, norm_cfg)
        self.aspp4 = _ASPPModule(inplanes, mid_channels, 3,  18, 18, norm_cfg)
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


# ---------------------------------------------------------------------------
# BEVDet-style Depth Auxiliary Head
# ---------------------------------------------------------------------------

@MODELS.register_module()
class BEVDetStyleAuxDepthHead(AuxDepthHead):
    """BEVDet-style 보조 depth head (ASPP + BasicBlock + DCN).

    ``AuxDepthHead`` 의 단순 1×1 conv 대신
    3×BasicBlock → ASPP → DCN → 1×1 conv 구조로 receptive field를 키운다.
    loss 계산 로직은 ``AuxDepthHead`` 에서 그대로 상속.

    Args:
        in_channels (int): FPN 출력 채널 수 (기본 256).
        mid_channels (int): 내부 채널 수 (기본 256).
        grid_config / downsample / loss_weight: ``AuxDepthHead`` 에 전달.
        norm_cfg (dict): BN 설정 (기본 BN2d).
    """

    def __init__(self,
                 in_channels: int = 256,
                 mid_channels: int = 256,
                 grid_config: dict = None,
                 downsample: int = 16,
                 loss_weight: float = 0.5,
                 norm_cfg: dict = None,
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
            nn.Conv2d(in_channels, mid_channels,
                      kernel_size=3, stride=1, padding=1),
            build_norm_layer(norm_cfg, mid_channels)[1],
            nn.ReLU(inplace=True),
        )
        # AuxDepthHead의 depth_conv(1×1)를 ASPP+DCN+conv로 교체
        self.depth_conv = nn.Sequential(
            BasicBlock(mid_channels, mid_channels, norm_cfg=norm_cfg),
            BasicBlock(mid_channels, mid_channels, norm_cfg=norm_cfg),
            BasicBlock(mid_channels, mid_channels, norm_cfg=norm_cfg),
            _ASPP(mid_channels, mid_channels, norm_cfg=norm_cfg),
            build_conv_layer(
                cfg=dict(type='DCN',
                         in_channels=mid_channels,
                         out_channels=mid_channels,
                         kernel_size=3, padding=1,
                         groups=4, im2col_step=128)),
            nn.Conv2d(mid_channels, self.D, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        """x: [B, N, C, H, W] → [B, N, D, H, W] logits."""
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        x = self.reduce_conv(x)
        logits = self.depth_conv(x)
        return logits.view(B, N, self.D, H, W)
