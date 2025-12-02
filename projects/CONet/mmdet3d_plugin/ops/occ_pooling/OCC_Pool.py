import torch

from projects.CONet.mmdet3d_plugin.ops.occ_pooling import occ_pool_ext

__all__ = ["occ_pool"]


def occ_pool_pure_pytorch(feats, coords, B, D, H, W):
    """Pure PyTorch implementation of occ_pool.
    
    This implementation is compatible with PyTorch 2.x + CUDA 12.1.
    Uses index_add for efficient scatter-add operations.
    
    Args:
        feats: Input features, FloatTensor[N, C]
        coords: Input coordinates, IntTensor[N, 4] where each row is [x, y, z, batch_idx]
        B: Batch size
        D: Depth of output voxel grid
        H: Height of output voxel grid
        W: Width of output voxel grid
    
    Returns:
        Output voxel features, FloatTensor[B, C, D, H, W]
    """
    # Convert parameters to integers if they are tensors
    if hasattr(B, 'item'):
        B = int(B.item())
    if hasattr(D, 'item'):
        D = int(D.item())
    if hasattr(H, 'item'):
        H = int(H.item())
    if hasattr(W, 'item'):
        W = int(W.item())
    
    # Ensure coords is on the same device as feats
    if coords.device != feats.device:
        coords = coords.to(feats.device)
    
    # Initialize output tensor: [B, D, H, W, C]
    C = feats.shape[1]
    out = torch.zeros((B, D, H, W, C), dtype=feats.dtype, device=feats.device)
    
    # Compute flat indices for scatter-add
    # coords shape: [N, 4] where each row is [x, y, z, batch_idx]
    batch_idx = coords[:, 3].long()
    z_idx = coords[:, 2].long()
    x_idx = coords[:, 0].long()
    y_idx = coords[:, 1].long()
    
    # Flatten the 4D indices to 1D
    flat_idx = (
        batch_idx * (D * H * W) +
        z_idx * (H * W) +
        x_idx * W +
        y_idx
    )
    
    # Sort for better cache locality
    sorted_idx = flat_idx.argsort()
    flat_idx_sorted = flat_idx[sorted_idx]
    feats_sorted = feats[sorted_idx]
    
    # Reshape out to [B*D*H*W, C] for index_add
    out_flat = out.view(-1, C)
    
    # Scatter-add features to their voxel locations
    out_flat.index_add_(0, flat_idx_sorted, feats_sorted)
    
    # Reshape back and permute to [B, C, D, H, W]
    out = out_flat.view(B, D, H, W, C)
    out = out.permute(0, 4, 1, 2, 3).contiguous()
    
    return out


def occ_pool(feats, coords, B, D, H, W):
    """Occupancy pooling function.
    
    Pools point features into a 3D voxel grid.
    Uses Pure PyTorch implementation for PyTorch 2.x compatibility.
    
    Args:
        feats: Input features, FloatTensor[N, C]
        coords: Input coordinates, IntTensor[N, 4] where each row is [x, y, z, batch_idx]
        B: Batch size
        D: Depth of output voxel grid
        H: Height of output voxel grid
        W: Width of output voxel grid
    
    Returns:
        Output voxel features, FloatTensor[B, C, D, H, W]
    """
    assert feats.shape[0] == coords.shape[0], "feats and coords must have same number of points"

    # Convert parameters to integers if they are tensors
    if hasattr(B, 'item'):
        B = int(B.item())
    if hasattr(D, 'item'):
        D = int(D.item())
    if hasattr(H, 'item'):
        H = int(H.item())
    if hasattr(W, 'item'):
        W = int(W.item())

    return occ_pool_pure_pytorch(feats, coords, B, D, H, W)
