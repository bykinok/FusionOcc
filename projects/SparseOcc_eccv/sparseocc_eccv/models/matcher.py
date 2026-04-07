"""
Modified from https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/modeling/matcher.py
"""
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from scipy.optimize import linear_sum_assignment
# 구버전 mmcv.runner.BaseModule → compat 경유
from ..compat import BaseModule, build_match_cost


def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor, mask_camera: torch.Tensor):
    if mask_camera is not None:
        inputs = inputs[:, mask_camera]
        targets = targets[:, mask_camera]
    
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


batch_dice_loss_jit = torch.jit.script(batch_dice_loss)


def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor, mask_camera: torch.Tensor):
    hw = inputs.shape[1]
    
    if mask_camera is not None:
        mask_camera = mask_camera.to(torch.int32)
        mask_camera = mask_camera[None].expand(inputs.shape[0], mask_camera.shape[-1])
        
        pos = F.binary_cross_entropy_with_logits(
            inputs, torch.ones_like(inputs), mask_camera, reduction="none"
        )
        neg = F.binary_cross_entropy_with_logits(
            inputs, torch.zeros_like(inputs), mask_camera, reduction="none"
        )
    else:
        pos = F.binary_cross_entropy_with_logits(
            inputs, torch.ones_like(inputs), reduction="none"
        )
        neg = F.binary_cross_entropy_with_logits(
            inputs, torch.zeros_like(inputs), reduction="none"
        )

    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum(
        "nc,mc->nm", neg, (1 - targets)
    )

    return loss / hw


batch_sigmoid_ce_loss_jit = torch.jit.script(batch_sigmoid_ce_loss)


class HungarianMatcher(BaseModule):
    def __init__(self, cost_class: float = 1, cost_mask: float = 1, cost_dice: float = 1):
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice

        self.loss_focal = build_match_cost(dict(type='FocalLossCost', weight=2.0))

        assert cost_class != 0 or cost_mask != 0 or cost_dice != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, mask_pred, class_pred, mask_gt, class_gt, mask_camera):
        bs, num_queries = class_pred.shape[:2]
        indices = []

        for b in range(bs):
            mask_camera_b = mask_camera[b] if mask_camera is not None else None
            tgt_ids = class_gt[b]
            num_instances = tgt_ids.shape[0]

            out_prob = class_pred[b]
            cost_class = self.loss_focal(out_prob, tgt_ids.long())

            out_mask = mask_pred[b]
            tgt_mask = mask_gt[b]
            
            tgt_mask = (tgt_mask.unsqueeze(-1) == torch.arange(num_instances).to(mask_gt.device))
            tgt_mask = tgt_mask.permute(1, 0)

            tgt_mask = tgt_mask.view(tgt_mask.shape[0], -1)
            out_mask = out_mask.view(out_mask.shape[0], -1)

            with autocast(enabled=False):
                out_mask = out_mask.float()
                tgt_mask = tgt_mask.float()
                cost_mask = batch_sigmoid_ce_loss(out_mask, tgt_mask, mask_camera_b)
                cost_dice = batch_dice_loss(out_mask, tgt_mask, mask_camera_b)
            
            C = (
                self.cost_mask * cost_mask
                + self.cost_class * cost_class
                + self.cost_dice * cost_dice
            )
            C = C.reshape(num_queries, -1).cpu()
            
            indices.append(linear_sum_assignment(C))
            
        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]
