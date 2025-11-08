import numpy as np
import torch

def gt_to_voxel(gt, img_metas):
    """Convert sparse GT representation to dense voxel grid.
    
    Args:
        gt (np.ndarray): Ground truth in sparse format [N, 4] where columns are [x, y, z, class_label]
        img_metas (dict): Image metadata containing occ_size
        
    Returns:
        np.ndarray: Dense voxel grid with shape occ_size, filled with class labels
    """
    # IMPORTANT: Initialize with zeros (not 255) to match original SurroundOcc
    # Occupied voxels will be filled with their class labels
    # Empty voxels remain as 0
    voxel = np.zeros(img_metas['occ_size'])
    
    # Fill voxel grid with class labels from sparse GT
    if len(gt) > 0:
        voxel[
            gt[:, 0].astype(np.int32), 
            gt[:, 1].astype(np.int32), 
            gt[:, 2].astype(np.int32)
        ] = gt[:, 3]

    return voxel

def evaluation_semantic(pred_occ, gt_occ, img_metas, class_num):
    """Evaluate semantic occupancy prediction.
    
    Args:
        pred_occ (torch.Tensor): Predicted occupancy with shape [B, H, W, D]
        gt_occ (torch.Tensor): Ground truth occupancy in sparse format [B, N, 4]
        img_metas (dict or list): Image metadata
        class_num (int): Number of semantic classes (including class 0)
        
    Returns:
        np.ndarray: Evaluation scores with shape [B, class_num, 3] where the last dimension is [tp, gt_count, pred_count]
    """
    results = []

    # Handle case where img_metas is a list (one dict per batch item)
    if isinstance(img_metas, list):
        img_metas_list = img_metas
    else:
        img_metas_list = [img_metas] * pred_occ.shape[0]

    for i in range(pred_occ.shape[0]):
        gt_i, pred_i = gt_occ[i].cpu().numpy(), pred_occ[i].cpu().numpy()
        img_meta = img_metas_list[i]
        
        # Check if GT is sparse (N, 4) or dense voxel grid
        if gt_i.ndim == 2 and gt_i.shape[1] == 4:
            # Sparse format: convert to dense voxel grid
            gt_i = gt_to_voxel(gt_i, img_meta)
        elif gt_i.ndim == 3:
            # Already dense voxel grid, use as-is
            pass
        else:
            print(f"Warning: Unexpected GT shape {gt_i.shape}, skipping sample {i}")
            continue
        
        # Create mask for valid voxels (exclude 255 which is invalid/ignore)
        mask = (gt_i != 255)
        
        # Calculate scores for each class
        score = np.zeros((class_num, 3))
        for j in range(class_num):
            if j == 0:  # class 0 for geometry IoU (any occupied voxel)
                score[j][0] += ((gt_i[mask] != 0) * (pred_i[mask] != 0)).sum()  # true positives
                score[j][1] += (gt_i[mask] != 0).sum()  # ground truth positive count
                score[j][2] += (pred_i[mask] != 0).sum()  # prediction positive count
            else:  # class-specific IoU
                score[j][0] += ((gt_i[mask] == j) * (pred_i[mask] == j)).sum()  # true positives
                score[j][1] += (gt_i[mask] == j).sum()  # ground truth count for class j
                score[j][2] += (pred_i[mask] == j).sum()  # prediction count for class j

        results.append(score)
    
    return np.stack(results, axis=0)

