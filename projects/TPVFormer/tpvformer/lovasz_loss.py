import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.registry import MODELS


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    if p == 0:
        return torch.zeros_like(gt_sorted)
    if gts == 0 or gts == p:
        return torch.zeros_like(gt_sorted)
    
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return probas * 0.
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float()  # foreground for class c
        if (classes == 'present' and fg.sum() == 0):
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (fg - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, lovasz_grad(fg_sorted)))
    return torch.mean(torch.stack(losses))


def flatten_probas(probas, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    if probas.dim() == 3:
        # assumes output of a sigmoid layer
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels


@MODELS.register_module()
class LovaszLoss(nn.Module):
    """Lovasz-Softmax loss for multi-class segmentation.
    
    Compatible with tpv04_occupancy.py implementation.
    """
    
    def __init__(self, 
                 loss_type='multi_class',
                 classes='present', 
                 ignore_index=0,
                 loss_weight=1.0):
        super().__init__()
        self.classes = classes
        self.ignore_index = ignore_index
        self.loss_weight = loss_weight
    
    def forward(self, inputs, targets, **kwargs):
        """
        Forward function.
        
        Args:
            inputs (torch.Tensor): Predictions with shape [B, C, H, W, Z] or [B, C, N]
            targets (torch.Tensor): Ground truth with shape [B, H, W, Z] or [B, N]
            
        Returns:
            torch.Tensor: Lovasz loss value
        """
        # Convert to probabilities (same as tpv04: torch.nn.functional.softmax)
        if inputs.dim() == 5:  # 3D case [B, C, H, W, Z]
            probas = F.softmax(inputs, dim=1)
            # Flatten to 2D
            B, C, H, W, Z = probas.shape
            probas = probas.permute(0, 2, 3, 4, 1).contiguous().view(-1, C)
            targets = targets.view(-1)
        else:  # Already flattened
            probas = F.softmax(inputs, dim=1)
            if targets.dim() > 1:
                targets = targets.view(-1)
        
        # Remove ignore_index pixels
        if self.ignore_index is not None:
            valid_mask = (targets != self.ignore_index)
            probas = probas[valid_mask]
            targets = targets[valid_mask]
        
        if self.per_image:
            # Not implemented for per_image mode in this version
            loss = lovasz_softmax_flat(probas, targets, classes=self.classes)
        else:
            loss = lovasz_softmax_flat(probas, targets, classes=self.classes)
        
        return loss * self.loss_weight
