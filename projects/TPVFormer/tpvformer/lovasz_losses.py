"""
Lovasz-Softmax and Jaccard hinge loss in PyTorch
Maxim Berman 2018 ESAT-PSI KU Leuven (MIT License)
"""

import torch
from torch.autograd import Variable
import torch.nn.functional as F

try:
    from itertools import ifilterfalse
except ImportError:  # py3k
    from itertools import filterfalse as ifilterfalse


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_softmax(probas, labels, classes='present', per_image=False, ignore=None):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """
    if per_image:
        loss = mean(lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
                          for prob, lab in zip(probas, labels))
    else:
        flattened = flatten_probas(probas, labels, ignore)
        # Debug: check what we got after flattening
        if not hasattr(lovasz_softmax, '_debug_flatten'):
            vprobas, vlabels = flattened
            print(f"[DEBUG LOVASZ FLATTEN] After flatten - vprobas shape: {vprobas.shape}, vlabels shape: {vlabels.shape}")
            print(f"[DEBUG LOVASZ FLATTEN] Unique labels after filter: {torch.unique(vlabels)}")
            print(f"[DEBUG LOVASZ FLATTEN] Number of valid samples: {len(vlabels)}")
            lovasz_softmax._debug_flatten = True
        loss = lovasz_softmax_flat(*flattened, classes=classes)
    return loss


def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        print("[DEBUG LOVASZ FLAT] probas.numel() == 0, returning zero")
        return probas * 0.
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    
    if not hasattr(lovasz_softmax_flat, '_debug_classes'):
        print(f"[DEBUG LOVASZ FLAT] classes mode: {classes}, C: {C}")
        print(f"[DEBUG LOVASZ FLAT] Unique labels in probas: {torch.unique(labels)}")
        lovasz_softmax_flat._debug_classes = True
    
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
        errors = (Variable(fg) - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    
    if not hasattr(lovasz_softmax_flat, '_debug_losses'):
        print(f"[DEBUG LOVASZ FLAT] Number of class losses: {len(losses)}")
        if len(losses) > 0:
            print(f"[DEBUG LOVASZ FLAT] First few losses: {[l.item() for l in losses[:3]]}")
        lovasz_softmax_flat._debug_losses = True
    
    # Ensure we return a tensor
    if len(losses) == 0:
        print("[DEBUG LOVASZ FLAT] len(losses) == 0, returning zero")
        return probas.sum() * 0.  # Return zero tensor with gradient
    loss_mean = mean(losses)
    # Convert to tensor if it's a scalar
    if not isinstance(loss_mean, torch.Tensor):
        return torch.tensor(loss_mean, device=probas.device, dtype=probas.dtype)
    return loss_mean


def flatten_probas(probas, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    if probas.dim() == 3:
        # assumes output of a sigmoid layer
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    elif probas.dim() == 5:
        # 3D segmentation
        B, C, L, H, W = probas.size()
        probas = probas.contiguous().view(B, C, L, H*W)
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    vprobas = probas[valid.nonzero(as_tuple=False).squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels


def isnan(x):
    return x != x


def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    Returns a tensor (not a Python scalar).
    """
    l = list(l)  # Convert to list to handle properly
    if len(l) == 0:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    
    if ignore_nan:
        l = [x for x in l if not isnan(x)]
    
    if len(l) == 0:
        return empty
    
    # Use torch.stack to ensure we return a tensor
    if isinstance(l[0], torch.Tensor):
        return torch.stack(l).mean()
    else:
        return sum(l) / len(l)

