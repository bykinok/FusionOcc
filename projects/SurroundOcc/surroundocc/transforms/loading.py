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
                 pc_range=[-50, -50, -5.0, 50, 50, 3.0]):
        self.use_semantic = use_semantic
        self.occ_size = occ_size
        self.pc_range = pc_range
    
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
        
        if 'occ_path' in results:
            occ_path = results['occ_path']
            # Handle relative paths
            if occ_path.startswith('./data/'):
                occ_path = occ_path.replace('./data/', '/home/h00323/Projects/occfrmwrk/data/')
            elif not occ_path.startswith('/'):
                occ_path = os.path.join('/home/h00323/Projects/occfrmwrk', occ_path)
                
            if os.path.exists(occ_path):
                try:
                    gt_occ_raw = np.load(occ_path)
                           # GT data loaded successfully
                    
                    # Check if data is sparse (N x 4) or dense (H x W x D)
                    if len(gt_occ_raw.shape) == 2 and gt_occ_raw.shape[1] == 4:
                        # Sparse format: convert to dense voxel grid
                        # Converting sparse GT to dense voxel grid
                        gt_occ = self._sparse_to_dense(gt_occ_raw, occ_size, results.get('pc_range', self.pc_range))
                    else:
                        # Already dense format
                        gt_occ = gt_occ_raw
                        
                        # Resize GT to match occ_size if needed
                        if list(gt_occ.shape) != occ_size:
                            try:
                                from skimage.transform import resize
                                # Use skimage resize with preserve_range and nearest neighbor
                                gt_occ_resized = resize(
                                    gt_occ, 
                                    occ_size, 
                                    order=0,  # nearest neighbor
                                    preserve_range=True, 
                                    anti_aliasing=False
                                ).astype(gt_occ.dtype)
                                gt_occ = gt_occ_resized
                            except ImportError:
                                try:
                                    from scipy.ndimage import zoom
                                    zoom_factors = [
                                        occ_size[0] / gt_occ.shape[0],
                                        occ_size[1] / gt_occ.shape[1], 
                                        occ_size[2] / gt_occ.shape[2]
                                    ]
                                    # Use nearest neighbor interpolation for occupancy labels
                                    gt_occ = zoom(gt_occ, zoom_factors, order=0).astype(gt_occ.dtype)
                                except Exception as e:
                                    # Resize failed, keeping original size
                                    occ_size = list(gt_occ.shape)
                        
                except Exception as e:
                    # Error loading occupancy data
                    gt_occ = None
        
        # If no real data was loaded, raise an error
        if gt_occ is None:
            raise RuntimeError(f"Failed to load occupancy ground truth data from {occ_path if occ_path else 'unknown path'}")
        
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