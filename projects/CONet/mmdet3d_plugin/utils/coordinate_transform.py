
import torch

def coarse_to_fine_coordinates(coarse_cor, ratio, topk=30000):
    """
    Args:
        coarse_cor (torch.Tensor): [3, N]"""

    fine_cor = coarse_cor * ratio
    fine_cor = fine_cor[None].repeat(ratio**3, 1, 1)  # [8, 3, N]

    device = fine_cor.device
    value = torch.meshgrid([torch.arange(ratio).to(device), torch.arange(ratio).to(device), torch.arange(ratio).to(device)])
    value = torch.stack(value, dim=3).reshape(-1, 3)

    fine_cor = fine_cor + value[:,:,None]

    if fine_cor.shape[-1] < topk:
        return fine_cor.permute(1,0,2).reshape(3,-1)
    else:
        fine_cor = fine_cor[:,:,torch.randperm(fine_cor.shape[-1])[:topk]]
        return fine_cor.permute(1,0,2).reshape(3,-1)



def project_points_on_img(points, rots, trans, intrins, post_rots, post_trans, bda_mat, pts_range,
                        W_img, H_img, W_occ, H_occ, D_occ):
    """Simplified version from original CONet for multimodal setup."""
    # Handle W_img and H_img being tuple/list (from DataLoader collation)
    if isinstance(W_img, (tuple, list)):
        W_img = W_img[0]
    if isinstance(H_img, (tuple, list)):
        H_img = H_img[0]
    
    with torch.no_grad():
        device = points.device
        if not isinstance(W_img, torch.Tensor):
            W_img = torch.tensor(W_img, device=device, dtype=points.dtype)
        else:
            W_img = W_img.to(device)
        
        if not isinstance(H_img, torch.Tensor):
            H_img = torch.tensor(H_img, device=device, dtype=points.dtype)
        else:
            H_img = H_img.to(device)

        voxel_size = ((pts_range[3:] - pts_range[:3]) / torch.tensor([W_occ-1, H_occ-1, D_occ-1])).to(points.device)
        points = points * voxel_size[None, None] + pts_range[:3][None, None].to(points.device)

        # project 3D point cloud (after bev-aug) onto multi-view images for corresponding 2D coordinates
        # Handle bda_mat shape and ensure it's on the same device as points
        if bda_mat.dim() == 2:  # [3, 3]
            # For 3x3 matrix, add batch dimension and extend to 4x4
            bda_4x4 = torch.eye(4, device=points.device, dtype=bda_mat.dtype)
            bda_4x4[:3, :3] = bda_mat.to(points.device)
            inv_bda = bda_4x4.inverse()
        else:  # [1, 3, 3] or [1, 4, 4]
            if bda_mat.shape[-1] == 3:
                # Convert 3x3 to 4x4
                bda_4x4 = torch.eye(4, device=points.device, dtype=bda_mat.dtype).unsqueeze(0).repeat(bda_mat.shape[0], 1, 1)
                bda_4x4[:, :3, :3] = bda_mat.to(points.device)
                inv_bda = bda_4x4.inverse()[0]  # [4, 4]
            else:
                inv_bda = bda_mat.to(points.device).inverse()[0]  # [4, 4]
        
        # Apply BDA transformation (points: [1, N, 3])
        points_homo = torch.cat([points, torch.ones(*points.shape[:-1], 1, device=points.device)], dim=-1)  # [1, N, 4]
        points = (inv_bda @ points_homo.unsqueeze(-1)).squeeze(-1)[..., :3]  # [1, N, 3]
        
        # from lidar to camera
        # Move all transformation matrices to the same device as points
        trans = trans.to(points.device)
        rots = rots.to(points.device)
        intrins = intrins.to(points.device)
        post_rots = post_rots.to(points.device)
        post_trans = post_trans.to(points.device)
        
        points = points.view(-1, 1, 3)  # [N, 1, 3]
        points = points - trans.view(1, -1, 3)  # Broadcasting: [N, n_cam, 3]
        inv_rots = rots.inverse().unsqueeze(0)  # [1, n_cam, 3, 3]
        points = (inv_rots @ points.unsqueeze(-1))  # [N, n_cam, 3, 1]
        
        # from camera to raw pixel
        points = (intrins.unsqueeze(0) @ points).squeeze(-1)  # [N, n_cam, 3]
        points_d = points[..., 2:3]  # [N, n_cam, 1]
        points_uv = points[..., :2] / (points_d + 1e-5)  # [N, n_cam, 2]
        
        # from raw pixel to transformed pixel
        points_uv = post_rots[..., :2, :2].unsqueeze(0) @ points_uv.unsqueeze(-1)  # [N, n_cam, 2, 1]
        points_uv = points_uv.squeeze(-1) + post_trans[..., :2].unsqueeze(0)  # [N, n_cam, 2]

        points_uv[..., 0] = (points_uv[..., 0] / (W_img-1) - 0.5) * 2
        points_uv[..., 1] = (points_uv[..., 1] / (H_img-1) - 0.5) * 2

        mask = (points_d[..., 0] > 1e-5) \
            & (points_uv[..., 0] > -1) & (points_uv[..., 0] < 1) \
            & (points_uv[..., 1] > -1) & (points_uv[..., 1] < 1)
    
    # points_uv shape: [B, N, n_cam, 2] -> [n_cam, B, N, 2] -> [n_cam, 1, N, 2] (when B=1)
    # permute(2, 0, 1, 3): dim 2->0, dim 0->1, dim 1->2, dim 3->3
    return points_uv.permute(2, 0, 1, 3), mask  # [n_cam, 1, N, 2], [N, n_cam]

def project_points_on_img_complex(points, rots, trans, intrins, post_rots, post_trans, bda_mat, pts_range,
                        W_img, H_img, W_occ, H_occ, D_occ):
    
    def ensure_device(tensor, target_device):
        """Helper function to move tensor to target device"""
        if isinstance(tensor, tuple):
            return tensor
        if hasattr(tensor, 'device') and tensor.device != target_device:
            return tensor.to(target_device)
        return tensor
        
    with torch.no_grad():
        # Ensure all tensors are on the same device as points
        target_device = points.device
        rots = ensure_device(rots, target_device)
        trans = ensure_device(trans, target_device)
        intrins = ensure_device(intrins, target_device)
        post_rots = ensure_device(post_rots, target_device)
        post_trans = ensure_device(post_trans, target_device)
        bda_mat = ensure_device(bda_mat, target_device)
        
        voxel_size = ((pts_range[3:] - pts_range[:3]) / torch.tensor([W_occ-1, H_occ-1, D_occ-1])).to(points.device)
        points = points * voxel_size[None, None] + pts_range[:3][None, None].to(points.device)

        # project 3D point cloud (after bev-aug) onto multi-view images for corresponding 2D coordinates
        if isinstance(bda_mat, tuple):
            # Handle case where bda_mat is a tuple - use identity transformation
            inv_bda = torch.eye(4, device=points.device, dtype=points.dtype)
            use_bda = True
        else:
            # Check if bda_mat is 3x3 or 4x4
            if bda_mat.shape[-1] == 3:
                # 3x3 matrix, skip bda transformation for now
                use_bda = False
            else:
                # 4x4 matrix, use normal bda transformation
                inv_bda = bda_mat.inverse()
                use_bda = True
        
        if use_bda:
            # Convert to homogeneous coordinates for 4x4 transformation
            points_homo = torch.cat([points, torch.ones(points.shape[:-1] + (1,), device=points.device, dtype=points.dtype)], dim=-1)
            # points_homo: [N, n_cam, 4], inv_bda: [4, 4] -> reshape and use matrix multiplication
            original_shape = points_homo.shape
            points_homo_flat = points_homo.view(-1, 4)  # [N*n_cam, 4]
            points_homo_flat = (inv_bda @ points_homo_flat.T).T  # [N*n_cam, 4]
            points_homo = points_homo_flat.view(original_shape)  # [N, n_cam, 4]
            points = points_homo[..., :3]  # Extract 3D coordinates
        
        # from lidar to camera
        # Reshape to [N*n_cam, 1, 3] for easier processing
        original_shape = points.shape  # [N, n_cam, 3]
        points = points.view(-1, 1, 3)  # [N*n_cam, 1, 3]
        
        # Handle case where trans is a tuple
        if isinstance(trans, tuple):
            # Use zero translation if trans is not a proper tensor
            trans_tensor = torch.zeros((points.shape[0], 1, 3), device=points.device, dtype=points.dtype)
        else:
            if trans.dim() == 2:  # [N_cam, 3]
                # Calculate actual number of points
                n_cam = trans.shape[0]
                total_points = points.shape[0]  # This is N*n_cam
                n_points = total_points // n_cam  # This gives us N
                
                # Create the correct pattern for each point-camera combination
                # We want: [cam0, cam1, ..., cam5, cam0, cam1, ..., cam5, ...]
                trans_indices = torch.arange(n_cam, device=trans.device).repeat(n_points)  # [0,1,2,3,4,5,0,1,2,3,4,5,...]
                trans_pattern = trans[trans_indices]  # [N*N_cam, 3]
                trans_tensor = trans_pattern.unsqueeze(1)  # [N*N_cam, 1, 3]
            else:  # [B, N_cam, 3] or other  
                trans_tensor = trans.view(-1, 1, 3)
        
        points = points - trans_tensor
        
        # Handle case where rots is a tuple
        if isinstance(rots, tuple):
            # Use identity rotation if rots is not a proper tensor
            inv_rots = torch.eye(3, device=points.device, dtype=points.dtype).unsqueeze(0)
        else:
            if rots.dim() == 2:  # [3, 3] - single camera rotation, expand to all cameras
                # Get number of cameras from trans
                n_cams = trans_tensor.shape[1] if hasattr(trans_tensor, 'shape') and trans_tensor.dim() > 1 else 1
                rots_expanded = rots.unsqueeze(0).expand(n_cams, -1, -1)  # [N_cam, 3, 3]
                inv_rots = rots_expanded.inverse().unsqueeze(0)  # [1, N_cam, 3, 3]
            elif rots.dim() == 3:  # [B, 3, 3] or [N_cam, 3, 3]
                if rots.shape[0] == 1:  # [1, 3, 3] - single camera, expand
                    n_cams = trans_tensor.shape[1] if hasattr(trans_tensor, 'shape') and trans_tensor.dim() > 1 else 1
                    rots_expanded = rots.expand(n_cams, -1, -1)  # [N_cam, 3, 3]
                    inv_rots = rots_expanded.inverse().unsqueeze(0)  # [1, N_cam, 3, 3]
                else:
                    inv_rots = rots.inverse().unsqueeze(0)  # [1, N_cam, 3, 3]
            else:  # [B, N_cam, 3, 3]
                inv_rots = rots.inverse()
        # points: [N*n_cam, 1, 3], inv_rots: [1, n_cam, 3, 3] or [n_cam, 3, 3]
        # Expand inv_rots to match points
        if inv_rots.dim() == 4:  # [1, n_cam, 3, 3]
            n_points = original_shape[0]
            inv_rots_expanded = inv_rots.repeat(1, n_points, 1, 1).view(-1, 3, 3)  # [N*n_cam, 3, 3]
        else:  # [n_cam, 3, 3]
            n_points = original_shape[0] 
            inv_rots_expanded = inv_rots.unsqueeze(1).repeat(1, n_points, 1, 1).view(-1, 3, 3)  # [N*n_cam, 3, 3]
        
        points = (inv_rots_expanded @ points.unsqueeze(-1)).squeeze(-1)
        
        # from camera to raw pixel
        # Handle case where intrins is a tuple
        if isinstance(intrins, tuple):
            # Use identity intrinsics if intrins is not a proper tensor
            intrins_tensor = torch.eye(3, device=points.device, dtype=points.dtype).unsqueeze(0)
        else:
            if intrins.dim() == 2:  # [3, 3] - single camera intrinsics, expand to all cameras
                # Get number of cameras from trans or use 6 as default
                n_cams = trans_tensor.shape[1] if hasattr(trans_tensor, 'shape') and trans_tensor.dim() > 1 else 6
                intrins_expanded = intrins.unsqueeze(0).expand(n_cams, -1, -1)  # [N_cam, 3, 3]
                intrins_tensor = intrins_expanded.unsqueeze(0)  # [1, N_cam, 3, 3]
            elif intrins.dim() == 3:  # [B, 3, 3] or [N_cam, 3, 3]
                if intrins.shape[0] == 1:  # [1, 3, 3] - single camera, expand
                    n_cams = trans_tensor.shape[1] if hasattr(trans_tensor, 'shape') and trans_tensor.dim() > 1 else 6
                    intrins_expanded = intrins.expand(n_cams, -1, -1)  # [N_cam, 3, 3]
                    intrins_tensor = intrins_expanded.unsqueeze(0)  # [1, N_cam, 3, 3]
                else:
                    intrins_tensor = intrins.unsqueeze(0)  # [1, N_cam, 3, 3]
            else:  # [B, N_cam, 3, 3]
                intrins_tensor = intrins
        
        # points: [N*n_cam, 1, 3], intrins_tensor: [1, n_cam, 3, 3] or [n_cam, 3, 3]
        # Expand intrins_tensor to match points
        if intrins_tensor.dim() == 4:  # [1, n_cam, 3, 3]
            n_points = original_shape[0]
            intrins_expanded = intrins_tensor.repeat(1, n_points, 1, 1).view(-1, 3, 3)  # [N*n_cam, 3, 3]
        else:  # [n_cam, 3, 3]
            n_points = original_shape[0]
            intrins_expanded = intrins_tensor.unsqueeze(1).repeat(1, n_points, 1, 1).view(-1, 3, 3)  # [N*n_cam, 3, 3]
        
        points = (intrins_expanded @ points.unsqueeze(-1)).squeeze(-1)
        points_d = points[..., 2:3]
        points_uv = points[..., :2] / (points_d + 1e-5)
        
        # from raw pixel to transformed pixel
        # Handle case where post_rots is a tuple
        if isinstance(post_rots, tuple):
            # Use identity post_rots if post_rots is not a proper tensor
            n_cams = trans_tensor.shape[1] if hasattr(trans_tensor, 'shape') and trans_tensor.dim() > 1 else 6
            post_rots_tensor = torch.eye(2, device=points.device, dtype=points.dtype).unsqueeze(0).unsqueeze(0).expand(1, n_cams, -1, -1)
        else:
            if post_rots.dim() == 2:  # [2, 2] - single camera, expand
                n_cams = trans_tensor.shape[1] if hasattr(trans_tensor, 'shape') and trans_tensor.dim() > 1 else 6
                post_rots_expanded = post_rots[:2, :2].unsqueeze(0).expand(n_cams, -1, -1)  # [N_cam, 2, 2]
                post_rots_tensor = post_rots_expanded.unsqueeze(0)  # [1, N_cam, 2, 2]
            elif post_rots.dim() == 3:  # [B, 2, 2] or [N_cam, 2, 2] or [N_cam, 3, 3]
                if post_rots.shape[0] == 1:  # [1, 3, 3] or similar, expand
                    n_cams = trans_tensor.shape[1] if hasattr(trans_tensor, 'shape') and trans_tensor.dim() > 1 else 6
                    post_rots_expanded = post_rots[0, :2, :2].unsqueeze(0).expand(n_cams, -1, -1)  # [N_cam, 2, 2]
                    post_rots_tensor = post_rots_expanded.unsqueeze(0)  # [1, N_cam, 2, 2]
                else:
                    post_rots_tensor = post_rots[..., :2, :2].unsqueeze(0)  # [1, N_cam, 2, 2]
            else:  # [B, N_cam, 3, 3] or similar
                post_rots_tensor = post_rots[..., :2, :2]
        
        # Handle case where post_trans is a tuple
        if isinstance(post_trans, tuple):
            # Use zero post_trans if post_trans is not a proper tensor
            post_trans_tensor = torch.zeros((points.shape[0], 1, 2), device=points.device, dtype=points.dtype)
        else:
            if post_trans.dim() == 2:  # [N_cam, 2] or [N_cam, 3]
                # Apply same pattern as trans_tensor
                n_cam = post_trans.shape[0]
                total_points = points.shape[0]  # This is N*n_cam
                n_points = total_points // n_cam  # This gives us N
                
                # Create the correct pattern for each point-camera combination
                post_trans_indices = torch.arange(n_cam, device=post_trans.device).repeat(n_points)
                post_trans_pattern = post_trans[post_trans_indices, :2]  # [N*N_cam, 2]
                post_trans_tensor = post_trans_pattern.unsqueeze(1)  # [N*N_cam, 1, 2]
            else:  # Other cases
                post_trans_tensor = post_trans[..., :2].view(-1, 1, 2)
        
        # points_uv: [N*n_cam, 1, 2], post_rots_tensor: [1, n_cam, 2, 2] or [n_cam, 2, 2]
        # Expand post_rots_tensor to match points_uv
        if post_rots_tensor.dim() == 4:  # [1, n_cam, 2, 2]
            n_points = original_shape[0]
            post_rots_expanded = post_rots_tensor.repeat(1, n_points, 1, 1).view(-1, 2, 2)  # [N*n_cam, 2, 2]
        else:  # [n_cam, 2, 2]
            n_points = original_shape[0]
            post_rots_expanded = post_rots_tensor.unsqueeze(1).repeat(1, n_points, 1, 1).view(-1, 2, 2)  # [N*n_cam, 2, 2]
        
        points_uv = (post_rots_expanded @ points_uv.unsqueeze(-1)).squeeze(-1)
        points_uv = points_uv + post_trans_tensor

        # Handle case where W_img and H_img are tuples
        if isinstance(W_img, tuple):
            W_img_val = 800  # Default width
        else:
            W_img_val = W_img
            
        if isinstance(H_img, tuple):
            H_img_val = 450  # Default height
        else:
            H_img_val = H_img
        
        points_uv[..., 0] = (points_uv[..., 0] / (W_img_val-1) - 0.5) * 2
        points_uv[..., 1] = (points_uv[..., 1] / (H_img_val-1) - 0.5) * 2

        # Reshape back to [N, n_cam, 2]
        points_uv = points_uv.view(original_shape[0], original_shape[1], 2)
        points_d = points_d.view(original_shape[0], original_shape[1], 1)

        mask = (points_d[..., 0] > 1e-5) \
            & (points_uv[..., 0] > -1) & (points_uv[..., 0] < 1) \
            & (points_uv[..., 1] > -1) & (points_uv[..., 1] < 1)
    
    # points_uv should have shape [N, n_cam, 2] -> after permute: [n_cam, 1, N, 2]
    # Add dimension for grid_sample format: [n_cam, 1, N, 2]
    points_uv = points_uv.permute(1, 0, 2).unsqueeze(1)  # [n_cam, 1, N, 2]
    
    return points_uv, mask