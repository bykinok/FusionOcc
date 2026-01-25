import os.path as osp
from typing import Callable, List, Union

from mmengine.dataset import BaseDataset

from mmdet3d.registry import DATASETS


@DATASETS.register_module()
class NuScenesOccupancyDataset(BaseDataset):
    r"""NuScenes Occupancy Dataset for 3D occupancy prediction.

    This class serves as the API for experiments on the NuScenes Dataset
    for occupancy prediction task.

    Please refer to `NuScenes Dataset <https://www.nuscenes.org/download>`_
    for data downloading.

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        pipeline (list[dict]): Pipeline used for data processing.
            Defaults to [].
        test_mode (bool): Store `True` when building test or val dataset.
    """
    METAINFO = {
        'classes': (
            'noise', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
            'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
            'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
            'vegetation', 'occupied'  # Added 'occupied' class for occupancy
        ),
        'ignore_index': 0,
        'label_mapping': dict([
            (1, 0), (5, 0), (7, 0), (8, 0), (10, 0), (11, 0), (13, 0),
            (19, 0), (20, 0), (0, 0), (29, 0), (31, 0), (9, 1), (14, 2),
            (15, 3), (16, 3), (17, 4), (18, 5), (21, 6), (2, 7), (3, 7),
            (4, 7), (6, 7), (12, 8), (22, 9), (23, 10), (24, 11), (25, 12),
            (26, 13), (27, 14), (28, 15), (30, 16), (32, 17)  # Added mapping for occupied
        ]),
        'palette': [
            [0, 0, 0],      # noise
            [255, 120, 50], # barrier              orange
            [255, 192, 203], # bicycle              pink
            [255, 255, 0],   # bus                  yellow
            [0, 150, 245],   # car                  blue
            [0, 255, 255],   # construction_vehicle cyan
            [255, 127, 0],   # motorcycle           dark orange
            [255, 0, 0],     # pedestrian           red
            [255, 240, 150], # traffic_cone         light yellow
            [135, 60, 0],    # trailer              brown
            [160, 32, 240],  # truck                purple
            [255, 0, 255],   # driveable_surface    dark pink
            [139, 137, 137], # other_flat           dark red
            [75, 0, 75],     # sidewalk             dark purple
            [150, 240, 80],  # terrain              light green
            [230, 230, 250], # manmade              white
            [0, 175, 0],     # vegetation           green
            [128, 128, 128], # occupied             gray
        ]
    }

    def __init__(self,
                 data_root: str,
                 ann_file: str,
                 pipeline: List[Union[dict, Callable]] = [],
                 test_mode: bool = False,
                 **kwargs) -> None:
        metainfo = dict(label2cat={
            i: cat_name
            for i, cat_name in enumerate(self.METAINFO['classes'])
        })
        super().__init__(
            ann_file=ann_file,
            data_root=data_root,
            metainfo=metainfo,
            pipeline=pipeline,
            test_mode=test_mode,
            **kwargs)

    def parse_data_info(self, info: dict) -> Union[List[dict], dict]:
        """Process the raw data info.

        Args:
            info (dict): Raw info dict.

        Returns:
            List[dict] or dict: Has `ann_info` in training stage. And
            all path has been converted to absolute path.
        """
        data_list = []
        
        # Add scene_name and scene_token (following STCOcc implementation)
        # Extract scene_name from occ_path if available
        if 'occ_path' in info and info['occ_path']:
            # occ_path format: './data/nuscenes/gts/scene-xxxx/token.npz'
            # Extract 'scene-xxxx' from path
            path_parts = info['occ_path'].split('/')
            for part in path_parts:
                if part.startswith('scene-'):
                    info['scene_name'] = part
                    break
            
            # Fallback if scene name not found in path
            if 'scene_name' not in info:
                info['scene_name'] = 'unknown'
        elif 'scene_name' not in info:
            # Use 'unknown' as fallback
            info['scene_name'] = 'unknown'
        
        # Process lidar points
        if 'lidar_points' in info:
            info['lidar_points']['lidar_path'] = \
                osp.join(
                    self.data_prefix.get('pts', ''),
                    info['lidar_points']['lidar_path'])
            
            # 최상위 레벨에 lidar_path 설정 (Pack3DDetInputs가 metainfo로 저장하기 위해)
            info['lidar_path'] = info['lidar_points']['lidar_path']
            
            # pts_filename도 설정 (일부 transform에서 사용)
            if 'pts_filename' not in info:
                info['pts_filename'] = info['lidar_path']

        # Process camera images
        for cam_id, img_info in info['images'].items():
            if 'img_path' in img_info:
                img_info['img_path'] = osp.join(
                    self.data_prefix.get(cam_id, ''),
                    img_info['img_path'])

        # Process annotations for occupancy
        if 'instances_3d' in info:
            for instance in info['instances_3d']:
                if 'bbox_3d' in instance:
                    # Convert bbox_3d to occupancy grid if needed
                    pass

        data_list.append(info)
        return data_list

    def get_data_info(self, index: int) -> dict:
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - index (int): Dataset index (for STCOcc metric compatibility).
                - pts_filename (str): Filename of point clouds.
                - img_filename (str): Filename of images.
                - ann_info (dict): Annotation info.
        """
        data_info = super().get_data_info(index)
        
        # CRITICAL: Add 'index' for STCOcc metric compatibility
        # STCOcc detector expects img_meta['index'] to match predictions with GT
        data_info['index'] = index
        
        # Add occupancy-specific information
        if 'instances_3d' in data_info:
            # Process 3D instances for occupancy prediction
            for instance in data_info['instances_3d']:
                if 'bbox_3d' in instance:
                    # Convert 3D bounding box to occupancy grid coordinates
                    bbox_3d = instance['bbox_3d']
                    # Add occupancy grid information
                    instance['occupancy_grid'] = self._bbox_to_occupancy_grid(bbox_3d)

        return data_info

    def _bbox_to_occupancy_grid(self, bbox_3d):
        """Convert 3D bounding box to occupancy grid coordinates.
        
        Args:
            bbox_3d (dict): 3D bounding box information
            
        Returns:
            dict: Occupancy grid coordinates
        """
        # This is a placeholder implementation
        # In practice, you would implement the conversion logic
        # from 3D bounding box to occupancy grid coordinates
        return {
            'grid_coords': [0, 0, 0],  # Placeholder
            'grid_size': [1, 1, 1]     # Placeholder
        }
