# Transform points from lidar to ego coordinate (FusionOcc approach)
import numpy as np
from mmdet3d.registry import TRANSFORMS
from pyquaternion import Quaternion


@TRANSFORMS.register_module()
class PointsLidar2Ego(object):
    """Transform points from lidar coordinate to ego coordinate.
    
    This transform converts point coordinates from the LiDAR sensor frame
    to the ego vehicle frame using the lidar2ego transformation matrix.
    Follows FusionOcc approach for compatibility with Occ3D GT.
    """
    
    def __call__(self, input_dict):
        """Transform points from lidar to ego coordinate.
        
        Args:
            input_dict (dict): Result dict with 'points' and lidar2ego transformation.
                             Expects 'lidar2ego_rotation' and 'lidar2ego_translation' in input_dict.
        
        Returns:
            dict: Results with transformed points.
        """
        points = input_dict['points']
        
        # Get lidar2ego transformation
        if 'lidar2ego_rotation' in input_dict and 'lidar2ego_translation' in input_dict:
            # Quaternion format (fusionocc/CONet style)
            lidar2ego_rot = Quaternion(input_dict['lidar2ego_rotation']).rotation_matrix
            lidar2ego_trans = np.array(input_dict['lidar2ego_translation'])
        else:
            # No transformation available, skip
            return input_dict
        
        # Transform points: p_ego = R @ p_lidar + t
        # Handle both tensor (torch) and numpy array
        if hasattr(points, 'tensor'):
            # mmdet3d LiDARPoints or BasePoints with tensor attribute
            import torch
            if not isinstance(lidar2ego_rot, torch.Tensor):
                lidar2ego_rot = torch.from_numpy(lidar2ego_rot).float()
            if not isinstance(lidar2ego_trans, torch.Tensor):
                lidar2ego_trans = torch.from_numpy(lidar2ego_trans).float()
            
            points.tensor[:, :3] = (points.tensor[:, :3] @ lidar2ego_rot.T) + lidar2ego_trans
        else:
            # Plain numpy array
            points[:, :3] = (points[:, :3] @ lidar2ego_rot.T) + lidar2ego_trans
        
        input_dict['points'] = points
        return input_dict
