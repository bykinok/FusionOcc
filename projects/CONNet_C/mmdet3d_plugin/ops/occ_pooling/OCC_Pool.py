import torch

# CUDA extension - may need to be compiled separately
# from projects.occ_plugin.ops.occ_pooling import occ_pool_ext
try:
    from . import occ_pool_ext
except ImportError:
    occ_pool_ext = None

__all__ = ["occ_pool"]


class QuickCumsum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, geom_feats, ranks):
        x = x.cumsum(0)
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        kept[:-1] = ranks[1:] != ranks[:-1]

        x, geom_feats = x[kept], geom_feats[kept]
        x = torch.cat((x[:1], x[1:] - x[:-1]))

        # save kept for backward
        ctx.save_for_backward(kept)

        # no gradient for geom_feats
        ctx.mark_non_differentiable(geom_feats)

        return x, geom_feats

    @staticmethod
    def backward(ctx, gradx, gradgeom):
        (kept,) = ctx.saved_tensors
        back = torch.cumsum(kept, 0)
        back[kept] -= 1

        val = gradx[back]

        return val, None, None


class QuickCumsumCuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, geom_feats, ranks, B, D, H, W):
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        kept[1:] = ranks[1:] != ranks[:-1]
        interval_starts = torch.where(kept)[0].int()
        interval_lengths = torch.zeros_like(interval_starts)
        interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
        interval_lengths[-1] = x.shape[0] - interval_starts[-1]
        geom_feats = geom_feats.int()

        if occ_pool_ext is not None:
            out = occ_pool_ext.occ_pool_forward(
                x,
                geom_feats,
                interval_lengths,
                interval_starts,
                B,
                D,
                H,
                W,
            )
        else:
            # Fallback implementation: proper voxel pooling with averaging
            c = x.shape[1]  # number of channels
            out = torch.zeros((B, D, H, W, c), device=x.device, dtype=x.dtype)
            
            # Filter valid coordinates and features
            # Coordinate order: [x, y, z, batch]
            valid_mask = ((geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < H) & 
                         (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < W) & 
                         (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < D) & 
                         (geom_feats[:, 3] >= 0) & (geom_feats[:, 3] < B))
            
            # Initialize counts for all voxels
            counts = torch.zeros(B * D * H * W, device=x.device, dtype=torch.float)
            
            if valid_mask.any():
                valid_coords = geom_feats[valid_mask]  # [valid_points, 4]
                valid_features = x[valid_mask]  # [valid_points, c]
                
                # Convert 4D coordinates to flat indices  
                # Coordinate order: [x, y, z, batch] -> [batch, z, x, y]
                flat_indices = (valid_coords[:, 3] * (D * H * W) + 
                               valid_coords[:, 2] * (H * W) + 
                               valid_coords[:, 0] * W + 
                               valid_coords[:, 1])  # [valid_points]
                flat_indices = flat_indices.to(device=x.device, dtype=torch.long)
                
                # Reshape output for scatter operations
                out_flat = out.view(-1, c)  # [B*D*H*W, c]
                
                # Count how many points go to each voxel
                counts.scatter_add_(0, flat_indices, torch.ones_like(flat_indices, dtype=torch.float))
                
                # Sum features for each voxel
                out_flat.scatter_add_(0, flat_indices.unsqueeze(1).expand(-1, c), valid_features)
                
                # Average: only divide voxels that have points (counts > 0)
                non_empty_mask = counts > 0
                out_flat[non_empty_mask] = out_flat[non_empty_mask] / counts[non_empty_mask].unsqueeze(1)
                
                # Reshape back
                out = out_flat.view(B, D, H, W, c)

        ctx.save_for_backward(interval_starts, interval_lengths, geom_feats)
        ctx.saved_shapes = B, D, H, W
        ctx.input_shape = x.shape  # Save input shape for backward
        ctx.input_device = x.device  # Save input device for backward
        
        # Save counts for backward pass averaging (always save, even if empty)
        if occ_pool_ext is None:
            ctx.voxel_counts = counts
        else:
            ctx.voxel_counts = None
        return out

    @staticmethod
    def backward(ctx, out_grad):
        interval_starts, interval_lengths, geom_feats = ctx.saved_tensors
        B, D, H, W = ctx.saved_shapes
        input_shape = ctx.input_shape
        input_device = ctx.input_device

        out_grad = out_grad.contiguous()
        if occ_pool_ext is not None:
            x_grad = occ_pool_ext.occ_pool_backward(
                out_grad,
                geom_feats,
                interval_lengths,
                interval_starts,
                B,
                D,
                H,
                W,
            )
        else:
            # Fallback implementation: gradient computation with averaging
            x_grad = torch.zeros(input_shape, device=input_device, dtype=torch.float)
            
            # Filter valid coordinates
            # Coordinate order: [x, y, z, batch]
            valid_mask = ((geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < H) & 
                         (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < W) & 
                         (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < D) & 
                         (geom_feats[:, 3] >= 0) & (geom_feats[:, 3] < B))
            
            if valid_mask.any():
                valid_coords = geom_feats[valid_mask]  # [valid_points, 4]
                valid_indices = torch.where(valid_mask)[0]  # [valid_points]
                
                # Convert 4D coordinates to flat indices  
                # Coordinate order: [x, y, z, batch] -> [batch, z, x, y]
                flat_indices = (valid_coords[:, 3] * (D * H * W) + 
                               valid_coords[:, 2] * (H * W) + 
                               valid_coords[:, 0] * W + 
                               valid_coords[:, 1])  # [valid_points]
                flat_indices = flat_indices.to(device=input_device, dtype=torch.long)
                
                # Reshape gradient for gathering
                out_grad_flat = out_grad.view(-1, out_grad.shape[-1])  # [B*D*H*W, c]
                
                # Gather gradients from corresponding voxel locations
                gathered_grads = out_grad_flat[flat_indices]  # [valid_points, c]
                
                # Apply averaging gradient: divide by voxel counts
                if ctx.voxel_counts is not None:
                    # Get counts for each point's voxel
                    point_counts = ctx.voxel_counts[flat_indices]  # [valid_points]
                    # Only divide where counts > 0 to avoid division by zero
                    valid_count_mask = point_counts > 0
                    gathered_grads[valid_count_mask] = gathered_grads[valid_count_mask] / point_counts[valid_count_mask].unsqueeze(1)
                
                # Assign to valid positions
                x_grad[valid_indices] = gathered_grads

        return x_grad, None, None, None, None, None, None


def occ_pool(feats, coords, B, D, H, W):
    assert feats.shape[0] == coords.shape[0]

    ranks = (
        coords[:, 0] * (W * D * B)
        + coords[:, 1] * (D * B)
        + coords[:, 2] * B
        + coords[:, 3]
    )
    indices = ranks.argsort()
    feats, coords, ranks = feats[indices], coords[indices], ranks[indices]

    x = QuickCumsumCuda.apply(feats, coords, ranks, B, D, H, W)
    x = x.permute(0, 4, 1, 2, 3).contiguous()
    return x