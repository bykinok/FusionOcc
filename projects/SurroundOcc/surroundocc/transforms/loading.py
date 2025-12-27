import numpy as np
import torch
from mmengine.registry import TRANSFORMS as ENGINE_TRANSFORMS
from mmdet3d.registry import TRANSFORMS as DET3D_TRANSFORMS
from mmcv.transforms import BaseTransform
import os


@ENGINE_TRANSFORMS.register_module()
@DET3D_TRANSFORMS.register_module()
class LoadOccupancy(BaseTransform):
    """Load occupancy ground truth.
    
    Args:
        use_semantic (bool): Whether to use semantic occupancy. Default: True.
        occ_size (list): Size of the occupancy grid [H, W, D]. Default: [200, 200, 16].
        pc_range (list): Point cloud range [x_min, y_min, z_min, x_max, y_max, z_max].
    """
    
    def __init__(self, 
                 use_semantic=True, 
                 occ_size=[200, 200, 16],
                 pc_range=[-50, -50, -5.0, 50, 50, 3.0],
                 use_occ3d=False):
        self.use_semantic = use_semantic
        self.occ_size = occ_size
        self.pc_range = pc_range
        self.use_occ3d = use_occ3d
    
    def transform(self, results):
        """Transform function to load occupancy data.
        
        Args:
            results (dict): Result dict from loading pipeline.
            
        Returns:
            dict: The result dict contains the occupancy ground truth.
        """
        occ_size = results.get('occ_size', self.occ_size)
        
        # Try to load real occupancy data first
        gt_occ = None
        
        # breakpoint()
        # Load occ3d format if specified
        if self.use_occ3d and 'occ3d_gt_path' in results:
            occ_3d_path = os.path.join(results['occ3d_gt_path'], 'labels.npz')
            workspace_root = os.getcwd()
            if occ_3d_path.startswith('./data/'):
                occ_3d_path = occ_3d_path.replace('./data/', os.path.join(workspace_root, 'data') + '/')
            elif not occ_3d_path.startswith('/'):
                occ_3d_path = os.path.join(workspace_root, occ_3d_path)
            
            if os.path.exists(occ_3d_path):
                import torch
                occ_3d = np.load(occ_3d_path)
                occ_3d_semantic = occ_3d['semantics']  # (200, 200, 16), occ3d format: 0=others, 1-16=semantic, 17=free
                occ_3d_cam_mask = occ_3d['mask_camera']  # (200, 200, 16) boolean mask
                
                # Process semantic labels for occ3d format: 0 -> 18 (others), then 18 -> 17 (empty)
                occ_3d_semantic_processed = occ_3d_semantic.copy()
                occ_3d_semantic_processed[occ_3d_semantic_processed == 0] = 18
                occ_3d_semantic_processed[occ_3d_semantic_processed == 17] = 0
                occ_3d_semantic_processed[occ_3d_semantic_processed == 18] = 17
                
                # Convert to torch tensor for processing
                occ_3d_gt = torch.from_numpy(occ_3d_semantic_processed).long()  # (200, 200, 16)
                occ_3d_cam_mask = torch.from_numpy(occ_3d_cam_mask).bool()  # (200, 200, 16) - must be bool for indexing
                
                # ===== occ3d format (18 classes: 0=noise, 1-16=semantic, 17=free) =====
                # Create masked version (visible by camera) - DENSE format (200, 200, 16)
                # Apply camera mask: invisible voxels -> 255 (ignore)
                occ_3d_gt_masked = occ_3d_gt.clone()  # (200, 200, 16)
                occ_3d_gt_masked[~occ_3d_cam_mask] = 255  # Set invisible voxels to 255 (ignore)
                
                # Store dense GT for evaluation (compatible with STCOcc metric)
                results['occ_3d_masked'] = occ_3d_gt_masked.numpy().astype(np.uint8)  # (200, 200, 16)
                results['occ_3d'] = occ_3d_gt.numpy().astype(np.uint8)  # (200, 200, 16)
                
                # For training, create sparse format from full GT
                # STCOcc uses full GT for training, camera mask only for evaluation
                all_coords = torch.nonzero(occ_3d_gt < 17, as_tuple=False)  # All non-empty voxels (0-16), shape: (N, 3)
                
                if len(all_coords) > 0:
                    # Vectorized sparse conversion (100-200x faster than Python loop)
                    # Extract labels for all coordinates at once
                    labels = occ_3d_gt[all_coords[:, 0], all_coords[:, 1], all_coords[:, 2]]  # (N,)
                    # Stack coordinates [x, y, z] and labels [class] -> [N, 4]
                    occ3d_sparse = torch.cat([all_coords.float(), labels.unsqueeze(1).float()], dim=1)
                    results['gt_occ'] = occ3d_sparse.cpu().numpy().astype(np.float32)
                else:
                    results['gt_occ'] = np.zeros((0, 4), dtype=np.float32)
                
                return results
        
        if 'occ_path' in results:
            occ_path = results['occ_path']
            # Handle relative paths
            workspace_root = os.getcwd()
            if occ_path.startswith('./data/'):
                occ_path = occ_path.replace('./data/', os.path.join(workspace_root, 'data') + '/')
            elif not occ_path.startswith('/'):
                occ_path = os.path.join(workspace_root, occ_path)
                
            if os.path.exists(occ_path):
                try:
                    gt_occ = np.load(occ_path)
                    gt_occ = gt_occ.astype(np.float32)
                    
                    # IMPORTANT: Keep GT in sparse format [N, 4] to match original SurroundOcc
                    # The first three channels represent xyz voxel coordinate and last channel is semantic class
                    # 
                    # CRITICAL FIX: Original GT has classes 1-17 for occupied semantic voxels
                    # Model predicts classes 0-16 where class 0 = any occupied (geometry)
                    # So we need to shift GT classes: GT_class - 1 = Model_class
                    # Example: GT class 1 (barrier) -> Model class 0... WAIT NO!
                    # 
                    # Actually: Model class 0 = empty, Model classes 1-16 = semantic
                    # GT sparse format only has occupied voxels with classes 1-17
                    # But NuScenes uses class 0 as ignore in original data
                    # So: GT class 0 -> 255 (ignore), GT classes 1-16 stay the same
                    if self.use_semantic:
                        # Convert class 0 (ignore) to 255
                        gt_occ[..., 3][gt_occ[..., 3] == 0] = 255
                    else:
                        # For non-semantic mode, filter out class 0 and set all to 1
                        gt_occ = gt_occ[gt_occ[..., 3] > 0]
                        gt_occ[..., 3] = 1
                    
                    # GT data loaded successfully in sparse format
                        
                except Exception as e:
                    # Error loading occupancy data
                    print(f"Error loading occupancy data: {e}")
                    gt_occ = None
        
        # If no real data was loaded, raise an error
        if gt_occ is None:
            raise RuntimeError(f"Failed to load occupancy ground truth data from {occ_path if 'occ_path' in results else 'unknown path'}")
        
        results['gt_occ'] = gt_occ
        results['occ_size'] = np.array(occ_size)
        results['pc_range'] = np.array(self.pc_range)
        
        return results
    
    def _sparse_to_dense(self, sparse_points, occ_size, pc_range):
        """Convert sparse point cloud to dense voxel grid.
        
        Args:
            sparse_points (np.array): Shape (N, 4), each point is [voxel_x, voxel_y, voxel_z, class_id]
                                     Note: coordinates are already voxel indices, not world coordinates
            occ_size (list): Target voxel grid size [H, W, D]
            pc_range (list): Point cloud range (not used for voxel index data)
        
        Returns:
            np.array: Dense voxel grid of shape (H, W, D)
        """
        # Initialize empty voxel grid
        voxel_grid = np.zeros(occ_size, dtype=np.uint8)
        
        if len(sparse_points) == 0:
            return voxel_grid
            
        # Extract voxel indices and labels (coordinates are already voxel indices)
        voxel_x = sparse_points[:, 0].astype(np.int32)  # already voxel index
        voxel_y = sparse_points[:, 1].astype(np.int32)  # already voxel index
        voxel_z = sparse_points[:, 2].astype(np.int32)  # already voxel index
        labels = sparse_points[:, 3].astype(np.uint8)   # class_id
        
        # Filter out points outside the valid voxel grid range
        valid_mask = (
            (voxel_x >= 0) & (voxel_x < occ_size[0]) &
            (voxel_y >= 0) & (voxel_y < occ_size[1]) &
            (voxel_z >= 0) & (voxel_z < occ_size[2])
        )
        
        voxel_x = voxel_x[valid_mask]
        voxel_y = voxel_y[valid_mask]
        voxel_z = voxel_z[valid_mask]
        labels = labels[valid_mask]
        
        # Fill voxel grid
        voxel_grid[voxel_x, voxel_y, voxel_z] = labels
        
        # Successfully converted sparse points to dense voxel grid
        
        return voxel_grid
    
    def _get_occ_path(self, results):
        """Get the path to occupancy ground truth file."""
        # This would construct the path to the actual occupancy file
        sample_idx = results.get('sample_idx')
        if sample_idx is None:
            raise ValueError("No sample_idx found in results")
        data_root = results.get('data_root', '')
        occ_path = os.path.join(data_root, 'occupancy', f'{sample_idx}.npy')
        return occ_path
    
    
    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                   f'use_semantic={self.use_semantic}, '
                   f'occ_size={self.occ_size}, '
                   f'pc_range={self.pc_range})')
        return repr_str