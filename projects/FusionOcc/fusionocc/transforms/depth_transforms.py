# Copyright (c) OpenMMLab. All rights reserved.
"""Depth transformation for FusionOcc."""

import torch
import numpy as np
from mmdet3d.registry import TRANSFORMS
from pyquaternion import Quaternion


@TRANSFORMS.register_module()
class PointToMultiViewDepth(object):
    """Project LiDAR points to multiple camera views to generate sparse depth maps.
    
    This is a critical transform for FusionOcc that creates sparse depth supervision
    by projecting LiDAR points onto each camera image.
    
    Args:
        grid_config (dict): Grid configuration for depth range.
        downsample (int): Downsample factor for the depth map. Default: 1.
    """
    
    def __init__(self, grid_config, downsample=1):
        self.downsample = downsample
        self.grid_config = grid_config

    def points2depthmap(self, points, height, width):
        """Convert points to depth map.
        
        Args:
            points: Points in image coordinates (u, v, depth)
            height: Image height
            width: Image width
            
        Returns:
            torch.Tensor: Sparse depth map
        """
        height, width = height // self.downsample, width // self.downsample
        depth_map = torch.zeros((height, width), dtype=torch.float32)
        coor = torch.round(points[:, :2] / self.downsample)
        depth = points[:, 2]
        
        # Filter out points outside image and depth range
        kept1 = (coor[:, 0] >= 0) & (coor[:, 0] < width) & \
                (coor[:, 1] >= 0) & (coor[:, 1] < height) & \
                (depth < self.grid_config['depth'][1]) & \
                (depth >= self.grid_config['depth'][0])
        coor, depth = coor[kept1], depth[kept1]
        
        # Sort by position and depth (keep closest depth for each pixel)
        ranks = coor[:, 0] + coor[:, 1] * width
        sort = (ranks + depth / 100.).argsort()
        coor, depth, ranks = coor[sort], depth[sort], ranks[sort]

        # Keep only one depth per pixel (the closest one)
        kept2 = torch.ones(coor.shape[0], device=coor.device, dtype=torch.bool)
        kept2[1:] = (ranks[1:] != ranks[:-1])
        coor, depth = coor[kept2], depth[kept2]
        coor = coor.to(torch.long)
        depth_map[coor[:, 1], coor[:, 0]] = depth
        return depth_map

    def __call__(self, results):
        """Generate sparse depth maps from LiDAR points.
        
        Args:
            results (dict): Results containing points and camera information.
            
        Returns:
            dict: Results with added 'sparse_depth' key.
        """

        

        # Get points (should be in ego coordinate after PointsLidar2Ego)
        points_lidar = results['curr_points']
        
        # Get camera information from img_inputs (unpack in two lines to handle bda)
        imgs, sensor2egos, ego2globals, intrins = results['img_inputs'][:4]
        post_rots, post_trans, bda = results['img_inputs'][4:]
        
        # Remove batch dimension (all tensors have batch dim from PrepareImageSeg)
        imgs = imgs.squeeze(0)              # [1, N, C, H, W] -> [N, C, H, W]
        sensor2egos = sensor2egos.squeeze(0)  # [1, N, 4, 4] -> [N, 4, 4]
        ego2globals = ego2globals.squeeze(0)  # [1, N, 4, 4] -> [N, 4, 4]
        intrins = intrins.squeeze(0)          # [1, N, 3, 3] -> [N, 3, 3]
        post_rots = post_rots.squeeze(0)      # [1, N, 3, 3] -> [N, 3, 3]
        post_trans = post_trans.squeeze(0)    # [1, N, 3] -> [N, 3]
        bda = bda.squeeze(0)                  # [1, 3, 3] -> [3, 3]
        
        # Support both fusionocc and occfrmwrk formats
        if 'curr' in results:
            # fusionocc format
            curr_info = results['curr']
            cam_data = curr_info.get('cams', {})
            cam_names = results.get('cam_names', list(cam_data.keys()))
            data_format = 'fusionocc'
        elif 'images' in results:
            # occfrmwrk format
            cam_data = results['images']
            # Use data_config camera order (should match PrepareImageSeg order)
            cam_names = results.get('cam_names', list(cam_data.keys()))
            data_format = 'occfrmwrk'
        else:
            # Fallback: create dummy sparse depth
            num_cams = intrins.shape[0]
            H, W = imgs.shape[2], imgs.shape[3]
            results['sparse_depth'] = torch.zeros(num_cams, H // self.downsample, W // self.downsample)
            return results

        # breakpoint()
        
        depth_map_list = []
        for cid in range(len(results['cam_names'])):
            cam_name = results['cam_names'][cid]
            
            if data_format == 'fusionocc':
                # Get lidar2ego transformation
                lidar2lidarego = np.eye(4, dtype=np.float32)
                lidar2lidarego[:3, :3] = Quaternion(
                    curr_info['lidar2ego_rotation']).rotation_matrix
                lidar2lidarego[:3, 3] = curr_info['lidar2ego_translation']
                lidar2lidarego = torch.from_numpy(lidar2lidarego)

                lidarego2global = np.eye(4, dtype=np.float32)
                lidarego2global[:3, :3] = Quaternion(
                    curr_info['ego2global_rotation']).rotation_matrix
                lidarego2global[:3, 3] = curr_info['ego2global_translation']
                lidarego2global = torch.from_numpy(lidarego2global)

                cam2camego = np.eye(4, dtype=np.float32)
                cam2camego[:3, :3] = Quaternion(
                    curr_info['cams'][cam_name]['sensor2ego_rotation']).rotation_matrix
                cam2camego[:3, 3] = curr_info['cams'][cam_name]['sensor2ego_translation']
                cam2camego = torch.from_numpy(cam2camego)

                camego2global = np.eye(4, dtype=np.float32)
                camego2global[:3, :3] = Quaternion(
                    curr_info['cams'][cam_name]['ego2global_rotation']).rotation_matrix
                camego2global[:3, 3] = curr_info['cams'][cam_name]['ego2global_translation']
                camego2global = torch.from_numpy(camego2global)
            else:
                # occfrmwrk format - use matrices from img_inputs (already computed)
                # sensor2egos and ego2globals are already computed by PrepareImageSeg
                # We can use them directly
                lidar2lidarego = results.get('lidar2ego', torch.eye(4))
                if isinstance(lidar2lidarego, np.ndarray):
                    lidar2lidarego = torch.from_numpy(lidar2lidarego).float()
                else:
                    lidar2lidarego = torch.tensor(lidar2lidarego).float()
                    
                lidarego2global = results.get('ego2global', torch.eye(4))
                if isinstance(lidarego2global, np.ndarray):
                    lidarego2global = torch.from_numpy(lidarego2global).float()
                else:
                    lidarego2global = torch.tensor(lidarego2global).float()
                
                # For camera transformation, use sensor2egos and ego2globals from img_inputs
                cam2camego = sensor2egos[cid]  # Already torch tensor
                camego2global = ego2globals[cid]  # Already torch tensor

            # Create cam2img matrix
            cam2img = torch.eye(4, dtype=torch.float32)
            cam2img[:3, :3] = intrins[cid]

            # Compute lidar2cam transformation
            
            # Compute inverse and clean up numerical errors in last row
            cam2global = camego2global.matmul(cam2camego)
            cam2global_inv = torch.inverse(cam2global)
            threshold = 1e-10
            cam2global_inv = torch.where(torch.abs(cam2global_inv) < threshold, 
                                         torch.zeros_like(cam2global_inv), 
                                         cam2global_inv)
            lidar2cam = cam2global_inv.matmul(lidarego2global.matmul(lidar2lidarego))
            lidar2cam = lidar2cam.float()

            # lidar2cam = torch.inverse(camego2global.matmul(cam2camego)).matmul(
            #     lidarego2global.matmul(lidar2lidarego))
            lidar2img = cam2img.matmul(lidar2cam)
            
            # breakpoint()

            # Project points to image
            points_img = points_lidar.tensor[:, :3].matmul(
                lidar2img[:3, :3].T) + lidar2img[:3, 3].unsqueeze(0)
            
            points_img = torch.cat(
                [points_img[:, :2] / points_img[:, 2:3], points_img[:, 2:3]], 1)
            
            # Apply post-transformation (data augmentation)
            points_img = points_img.matmul(
                post_rots[cid].T) + post_trans[cid:cid + 1, :]
            
            # Generate depth map
            depth_map = self.points2depthmap(points_img, imgs.shape[2], imgs.shape[3])
            depth_map_list.append(depth_map)
        
        depth_map = torch.stack(depth_map_list)
        results['sparse_depth'] = depth_map
        
        # breakpoint()   

        return results
    
    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(downsample={self.downsample}, '
        repr_str += f'grid_config={self.grid_config})'
        return repr_str

