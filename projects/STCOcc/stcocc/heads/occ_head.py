import copy
import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmengine.model import BaseModule
from mmdet3d.registry import MODELS
from mmdet.models.losses.utils import reduce_loss

from ..modules.basic_block import BasicBlock3D

@MODELS.register_module()
class OccHead(BaseModule):
    def __init__(
            self,
            in_channels,
            out_channels,
            num_classes=17,
            conv_before_predictor=True,
            class_weights=None,
            up_sample=True,
            last_occ_unsample=True,
            empty_idx=16,
            use_group_refine=False,
            group_list=None,
            foreground_idx=None,
            background_idx=None,
            class_frequency=None,
            bev_w=200,
            bev_h=200,
            bev_z=16,
    ):
        super(OccHead, self).__init__()
        if conv_before_predictor:
            self.voxel_conv = BasicBlock3D(
                channels_in=in_channels,
                channels_out=out_channels,
                stride=1,
            )

        self.predicter = nn.Sequential(
            nn.Linear(out_channels, out_channels * 2),
            nn.Softplus(),
            nn.Linear(out_channels * 2, num_classes),
        )

        self.use_group_refine = use_group_refine
        self.empty_idx = empty_idx
        self.conv_before_predictor = conv_before_predictor
        self.last_occ_unsample = last_occ_unsample
        self.up_sample = up_sample
        self.foreground_idx = foreground_idx
        self.background_idx = background_idx
        self.class_frequency = class_frequency
        self.group_list = group_list
        self.bev_w = bev_w
        self.bev_h = bev_h
        self.bev_z = bev_z
        self.num_classes = num_classes

    # @force_fp32()  # deprecated decorator
    def forward(self, voxel_feats, last_occ_pred=None):
        
        # required [bs, c, z, h, w] voxel feats
        if self.conv_before_predictor:
            pred_voxel_feats = self.voxel_conv(voxel_feats)
        else:
            pred_voxel_feats = voxel_feats

        if self.up_sample:
            pred_voxel_feats = F.interpolate(pred_voxel_feats, scale_factor=2, mode='trilinear', align_corners=False)

        pred_voxel_feats = pred_voxel_feats.permute(0, 4, 3, 2, 1)  # [bs, c, z, h, w] - > [bs, w, h, z, c]

        pred_voxel_semantic = self.predicter(pred_voxel_feats)         # [bs, w, h, z, c]

        if last_occ_pred is not None:
            # last_occ_pred: [bs, w, h, z, c]
            if self.last_occ_unsample:
                last_occ_pred = last_occ_pred.permute(0, 4, 3, 1, 2)  # [bs, c, z, w, h]
                last_occ_pred = F.interpolate(last_occ_pred, scale_factor=2, mode='trilinear', align_corners=False)
                last_occ_pred = last_occ_pred.permute(0, 3, 4, 2, 1)  # [bs, w, h, z, c]
            pred_voxel_semantic = pred_voxel_semantic + 0.5*last_occ_pred

            # After softmax
            pred_softmax = pred_voxel_semantic.softmax(-1)
            pred_argmax = pred_softmax.argmax(-1)
        
        return pred_voxel_semantic, pred_voxel_feats


# Helper function for CustomFocalLoss
def sigmoid_focal_loss(inputs, targets, alpha=0.25, gamma=2.0, reduction='mean'):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = 0.25.
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


class CustomFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', bev_h=200, bev_w=200):
        super(CustomFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bev_h = bev_h
        self.bev_w = bev_w

    def forward(self, inputs, targets, weight=None, avg_factor=None):
        loss = sigmoid_focal_loss(inputs, targets, self.alpha, self.gamma, 'none')
        
        if weight is not None:
            if weight.shape != loss.shape:
                if weight.size(0) == loss.size(0):
                    # For class-wise weights
                    weight = weight.view(-1, 1)
            loss = weight * loss
        
        loss = reduce_loss(loss, self.reduction, avg_factor)
        return loss