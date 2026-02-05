import torch
import torch.nn as nn
import torch.nn.functional as F

def multiscale_supervision(gt_occ, ratio, gt_shape):
    '''
    change ground truth shape as (B, W, H, Z) for each level supervision
    '''
    gt = torch.zeros([gt_shape[0], gt_shape[2], gt_shape[3], gt_shape[4]]).to(gt_occ.device).type(torch.float) 
    for i in range(gt.shape[0]):
        coords = gt_occ[i][:, :3].type(torch.long) // ratio
        gt[i, coords[:, 0], coords[:, 1], coords[:, 2]] = gt_occ[i][:, 3]
    
    return gt

def geo_scal_loss(pred, ssc_target, semantic=True, empty_idx=0):
    """Geometric scale loss for occupancy prediction.
    
    Args:
        pred: Predicted occupancy logits
        ssc_target: Ground truth semantic labels
        semantic: Whether to use semantic mode (softmax) or binary mode (sigmoid)
        empty_idx: Index of the empty/free class (0 for original, 17 for occ3d)
        
    Returns:
        Loss value combining precision, recall, and specificity
    """
    # Get softmax probabilities
    if semantic:
        pred = F.softmax(pred, dim=1)
        # Compute empty and nonempty probabilities
        # For original SurroundOcc: empty_idx=0 (class 0 is empty)
        # For Occ3D: empty_idx=17 (class 17 is free)
        empty_probs = pred[:, empty_idx, :, :, :]
    else:
        empty_probs = 1 - torch.sigmoid(pred)
    nonempty_probs = 1 - empty_probs

    # Remove unknown voxels
    mask = ssc_target != 255
    # nonempty = not empty_idx
    # For original: nonempty = (target != 0)
    # For occ3d: nonempty = (target != 17)
    nonempty_target = ssc_target != empty_idx
    nonempty_target = nonempty_target[mask].float()
    nonempty_probs = nonempty_probs[mask]
    empty_probs = empty_probs[mask]

    intersection = (nonempty_target * nonempty_probs).sum()
    precision = intersection / nonempty_probs.sum()
    recall = intersection / nonempty_target.sum()
    spec = ((1 - nonempty_target) * (empty_probs)).sum() / (1 - nonempty_target).sum()
    return (
        F.binary_cross_entropy(precision, torch.ones_like(precision))
        + F.binary_cross_entropy(recall, torch.ones_like(recall))
        + F.binary_cross_entropy(spec, torch.ones_like(spec))
    )


def sem_scal_loss(pred, ssc_target, empty_idx=0):
    """Semantic scale loss for class-wise precision/recall/specificity.
    
    Args:
        pred: Predicted occupancy logits
        ssc_target: Ground truth semantic labels
        empty_idx: Index of the empty/free class (0 for original, 17 for occ3d)
        
    Returns:
        Average loss across all classes (excluding empty/free class)
    """
    # Get softmax probabilities
    pred = F.softmax(pred, dim=1)
    loss = 0
    count = 0
    mask = ssc_target != 255
    n_classes = pred.shape[1]
    # CRITICAL FIX: Exclude empty/free class from loss calculation
    # This aligns training objective with evaluation metric (which excludes free class)
    # For original SurroundOcc: empty_idx=0, evaluate classes 1-16
    # For Occ3D: empty_idx=17, evaluate classes 0-16
    for i in range(0, n_classes):
        # Skip empty/free class to align with evaluation metric
        if i == empty_idx:
            continue
        # Get probability of class i
        p = pred[:, i, :, :, :]

        # Remove unknown voxels
        target_ori = ssc_target
        p = p[mask]
        target = ssc_target[mask]

        completion_target = torch.ones_like(target)
        completion_target[target != i] = 0
        completion_target_ori = torch.ones_like(target_ori).float()
        completion_target_ori[target_ori != i] = 0
        if torch.sum(completion_target) > 0:
            count += 1.0
            nominator = torch.sum(p * completion_target)
            loss_class = 0
            if torch.sum(p) > 0:
                precision = nominator / (torch.sum(p))
                loss_precision = F.binary_cross_entropy(
                    precision, torch.ones_like(precision)
                )
                loss_class += loss_precision
            if torch.sum(completion_target) > 0:
                recall = nominator / (torch.sum(completion_target))
                loss_recall = F.binary_cross_entropy(recall, torch.ones_like(recall))
                loss_class += loss_recall
            if torch.sum(1 - completion_target) > 0:
                specificity = torch.sum((1 - p) * (1 - completion_target)) / (
                    torch.sum(1 - completion_target)
                )
                loss_specificity = F.binary_cross_entropy(
                    specificity, torch.ones_like(specificity)
                )
                loss_class += loss_specificity
            loss += loss_class
    return loss / count

