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
        use_occ3d (bool): Whether to use occ3d format. Default: False.
        use_mask_camera (bool): Whether to use camera mask for loss calculation. Default: False.
            If True, voxels not visible from camera are marked as ignore (255) for loss.
            If False, all voxels keep their original labels.
    """
    
    def __init__(self, 
                 use_semantic=True, 
                 occ_size=[200, 200, 16],
                 pc_range=[-50, -50, -5.0, 50, 50, 3.0],
                 use_occ3d=False,
                 use_mask_camera=False,
                 use_mask_camera_1_2=False,
                 use_mask_camera_1_4=False,
                 use_mask_camera_1_8=False):
        self.use_semantic = use_semantic
        self.occ_size = occ_size
        self.pc_range = pc_range
        self.use_occ3d = use_occ3d
        self.use_mask_camera = use_mask_camera
        self.use_mask_camera_1_2 = use_mask_camera_1_2
        self.use_mask_camera_1_4 = use_mask_camera_1_4
        self.use_mask_camera_1_8 = use_mask_camera_1_8
    
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
        
        # Load occ3d format if specified
        if self.use_occ3d and 'occ3d_gt_path' in results:
                        
            occ3d_gt_label = os.path.join(results['occ3d_gt_path'], 'labels.npz')
            occ3d_gt_label_1_2 = os.path.join(results['occ3d_gt_path'], 'labels_1_2.npz')
            occ3d_gt_label_1_4 = os.path.join(results['occ3d_gt_path'], 'labels_1_4.npz')
            occ3d_gt_label_1_8 = os.path.join(results['occ3d_gt_path'], 'labels_1_8.npz')

            # Save GT path to results for debugging (will be passed to img_metas)
            results['occ3d_gt_path'] = occ3d_gt_label
            
            if os.path.exists(occ3d_gt_label):
                import torch
                occ_3d = np.load(occ3d_gt_label)
                occ_3d_semantic = occ_3d['semantics']  # (200, 200, 16), occ3d format: 0=others, 1-16=semantic, 17=free
                occ_3d_cam_mask = occ_3d['mask_camera']  # (200, 200, 16) boolean mask

                # gt_occ = occ_3d_semantic.astype(np.int32)               
                
                if self.use_mask_camera:
                    occ_3d_gt_masked = np.where(occ_3d_cam_mask, occ_3d_semantic, 255).astype(np.uint8)
                else:
                    occ_3d_gt_masked = occ_3d_semantic.astype(np.uint8)

                # Convert to torch tensor for processing
                gt_occ_tensor = torch.from_numpy(occ_3d_gt_masked).long()
                
                # For training, create sparse format including ALL classes (0-17) and ignore label (255)
                # When use_mask_camera=True, invisible voxels are marked as 255
                # This allows the model to distinguish between others(0) and free(17)
                # and properly ignore masked voxels in loss calculation
                all_coords = torch.nonzero(gt_occ_tensor >= 0, as_tuple=False)  # All voxels including 255, shape: (N, 3)
                
                if len(all_coords) > 0:
                    # Vectorized sparse conversion (100-200x faster than Python loop)
                    # Extract labels for all coordinates (0-17 for valid, 255 for masked)
                    labels = gt_occ_tensor[all_coords[:, 0], all_coords[:, 1], all_coords[:, 2]]  # (N,)
                    # Stack coordinates [x, y, z] and labels [class] -> [N, 4]
                    occ3d_sparse = torch.cat([all_coords.float(), labels.unsqueeze(1).float()], dim=1)
                    results['gt_occ'] = occ3d_sparse.cpu().numpy().astype(np.float32)
                else:
                    results['gt_occ'] = np.zeros((0, 4), dtype=np.float32)
            else:
                raise FileNotFoundError(f"Occ3D ground truth file not found: {occ3d_gt_label}")

            results['voxel_semantics'] = occ_3d_gt_masked
            
            # breakpoint()
            # Load multiscale GT based on individual flags
            if os.path.exists(occ3d_gt_label_1_2):
                occ_3d_1_2 = np.load(occ3d_gt_label_1_2)
                occ_3d_1_2_semantic = occ_3d_1_2['semantics']
                occ_3d_1_2_cam_mask = occ_3d_1_2['mask_camera']
                if self.use_mask_camera_1_2:
                    occ_3d_1_2_gt_masked = np.where(occ_3d_1_2_cam_mask, occ_3d_1_2_semantic, 255).astype(np.uint8)
                else:
                    occ_3d_1_2_gt_masked = occ_3d_1_2_semantic.astype(np.uint8)
                results['voxel_semantics_1_2'] = occ_3d_1_2_gt_masked
            else:
                raise FileNotFoundError(f"Occ3D ground truth file not found: {occ3d_gt_label_1_2}")

            if os.path.exists(occ3d_gt_label_1_4):
                occ_3d_1_4 = np.load(occ3d_gt_label_1_4)
                occ_3d_1_4_semantic = occ_3d_1_4['semantics']
                occ_3d_1_4_cam_mask = occ_3d_1_4['mask_camera']
                if self.use_mask_camera_1_4:
                    occ_3d_1_4_gt_masked = np.where(occ_3d_1_4_cam_mask, occ_3d_1_4_semantic, 255).astype(np.uint8)
                else:
                    occ_3d_1_4_gt_masked = occ_3d_1_4_semantic.astype(np.uint8)
                results['voxel_semantics_1_4'] = occ_3d_1_4_gt_masked
            else:
                raise FileNotFoundError(f"Occ3D ground truth file not found: {occ3d_gt_label_1_4}")

            if os.path.exists(occ3d_gt_label_1_8):
                occ_3d_1_8 = np.load(occ3d_gt_label_1_8)
                occ_3d_1_8_semantic = occ_3d_1_8['semantics']
                occ_3d_1_8_cam_mask = occ_3d_1_8['mask_camera']
                if self.use_mask_camera_1_8:
                    occ_3d_1_8_gt_masked = np.where(occ_3d_1_8_cam_mask, occ_3d_1_8_semantic, 255).astype(np.uint8)
                else:
                    occ_3d_1_8_gt_masked = occ_3d_1_8_semantic.astype(np.uint8)
                results['voxel_semantics_1_8'] = occ_3d_1_8_gt_masked
            else:
                raise FileNotFoundError(f"Occ3D ground truth file not found: {occ3d_gt_label_1_8}")

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