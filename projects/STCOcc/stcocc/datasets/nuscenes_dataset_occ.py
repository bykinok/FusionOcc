# Copyright (c) OpenMMLab. All rights reserved.
import os
import mmcv
import torch
import cv2
import math
import numpy as np
from tqdm import tqdm
from typing import Any

from nuscenes.nuscenes import NuScenes

from mmdet3d.registry import DATASETS
from mmdet3d.datasets.nuscenes_dataset import NuScenesDataset
from .nuscenes_ego_pose_loader import nuScenesDataset
from .nuscenes_utils import nuscenes_get_rt_matrix
from .ray_metrics_occ3d import main as ray_based_miou_occ3d
from .ray_metrics_openocc import main as ray_based_miou_openocc

occ3d_colors_map = np.array(
    [
        [0,   0,   0, 255],  # 0 undefined
        [255, 158, 0, 255],  # 1 car  orange
        [0, 0, 230, 255],    # 2 pedestrian  Blue
        [47, 79, 79, 255],   # 3 sign  Darkslategrey
        [220, 20, 60, 255],  # 4 CYCLIST  Crimson
        [255, 69, 0, 255],   # 5 traiffic_light  Orangered
        [255, 140, 0, 255],  # 6 pole  Darkorange
        [233, 150, 70, 255], # 7 construction_cone  Darksalmon
        [255, 61, 99, 255],  # 8 bycycle  Red
        [112, 128, 144, 255],# 9 motorcycle  Slategrey
        [222, 184, 135, 255],# 10 building Burlywood
        [0, 175, 0, 255],    # 11 vegetation  Green
        [165, 42, 42, 255],  # 12 trunk  nuTonomy green
        [0, 207, 191, 255],  # 13 curb, road, lane_marker, other_ground
        [75, 0, 75, 255], # 14 walkable, sidewalk
        [255, 0, 0, 255], # 15 unobsrvd
        [0, 0, 0, 0],   # 16 undefined
        [0, 0, 0, 0],  # 16 undefined
    ]
)

openocc_colors_map = np.array(
    [
        [0, 150, 245],      # car                  blue         √
        [160, 32, 240],     # truck                purple       √
        [135, 60, 0],       # trailer              brown        √
        [255, 255, 0],      # bus                  yellow       √
        [0, 255, 255],      # construction_vehicle cyan         √
        [255, 192, 203],    # bicycle              pink         √
        [255, 127, 0],      # motorcycle           dark orange  √
        [255, 0, 0],        # pedestrian           red          √
        [255, 240, 150],    # traffic_cone         light yellow
        [255, 120, 50],     # barrier              orange
        [255, 0, 255],      # driveable_surface    dark pink
        [139, 137, 137],    # other_flat           dark red
        [75, 0, 75],        # sidewalk             dard purple
        [150, 240, 80],     # terrain              light green
        [230, 230, 250],    # manmade              white
        [0, 175, 0],        # vegetation           green
        [255, 255, 255],    # Free                 White
    ]
)



@DATASETS.register_module()
class NuScenesDatasetOccpancy(NuScenesDataset):
    # Semantic label mapping for occupancy prediction
    SegLabelMapping = dict([(1, 0), (5, 0), (7, 0), (8, 0), (10, 0), (11, 0), (13, 0),
                           (19, 0), (20, 0), (0, 0), (29, 0), (31, 0), (9, 1), (14, 2),
                           (15, 3), (16, 3), (17, 4), (18, 5), (21, 6), (2, 7), (3, 7),
                           (4, 7), (6, 7), (12, 8), (22, 9), (23, 10), (24, 11), (25, 12),
                           (26, 13), (27, 14), (28, 15), (30, 16)])
    
    def __init__(self, 
                 classes=None,  # Handle classes parameter
                 stereo=False,  # Handle stereo parameter
                 sequences_split_num=1,  # Handle sequences_split_num
                 work_dir=None,  # Handle work_dir parameter
                 dataset_name=None,  # Handle dataset_name parameter
                 eval_metric=None,  # Handle eval_metric parameter
                 eval_show=None,  # Handle eval_show parameter
                 img_info_prototype=None,  # Handle img_info_prototype parameter
                 multi_adj_frame_id_cfg=None,  # Handle multi_adj_frame_id_cfg parameter
                 use_sequence_group_flag=None,  # Handle use_sequence_group_flag parameter
                 load_interval=1,  # Add load_interval parameter (default=1, same as original)
                 box_type_3d='LiDAR',  # CRITICAL: Add box_type_3d parameter
                 **kwargs):
        # Store custom parameters separately
        self._classes = classes
        self.stereo = stereo
        self.sequences_split_num = sequences_split_num
        self.work_dir = work_dir
        self.dataset_name = dataset_name
        self.eval_metric = eval_metric
        self.eval_show = eval_show
        self.img_info_prototype = img_info_prototype
        self.multi_adj_frame_id_cfg = multi_adj_frame_id_cfg
        self.use_sequence_group_flag = use_sequence_group_flag
        self.load_interval = load_interval  # Store load_interval
        
        # Initialize show_ins_var attribute (required by Det3DDataset)
        self.show_ins_var = kwargs.get('show_ins_var', False)
        
        # CRITICAL: Store box_type_3d and compute box_mode_3d
        self.box_type_3d = box_type_3d
        from mmdet3d.structures.bbox_3d import get_box_type
        self.box_type_3d, self.box_mode_3d = get_box_type(box_type_3d)
        
        # Set up METAINFO for proper class statistics
        if classes is not None:
            if not hasattr(self, 'METAINFO'):
                self.METAINFO = {}
            self.METAINFO['classes'] = classes
        
        # Remove unsupported kwargs before passing to parent
        filtered_kwargs = {k: v for k, v in kwargs.items() 
                          if k not in ['sequences_split_num', 'stereo', 'work_dir', 
                                     'dataset_name', 'eval_metric', 'eval_show',
                                     'img_info_prototype', 'multi_adj_frame_id_cfg',
                                     'use_sequence_group_flag', 'classes']}
        
        # Store original lazy_init value and show_ins_var setting
        original_lazy_init = filtered_kwargs.get('lazy_init', False)
        original_show_ins_var = filtered_kwargs.get('show_ins_var', False)
        
        # Temporarily disable statistics output during parent initialization
        filtered_kwargs['show_ins_var'] = False
        super().__init__(**filtered_kwargs)
        
        # Restore show_ins_var setting
        self.show_ins_var = original_show_ins_var
        
        # Calculate and show corrected statistics if not lazy_init
        if not original_lazy_init:
            self._calculate_num_ins_per_cat()
            self._show_correct_statistics()
        
        # CRITICAL: After parent __init__, self.data_infos may be empty, reload it!
        # Note: This happens before load_data_list is called by parent class
        # So we need to ensure load_data_list sets data_infos correctly
        
        # CRITICAL: Set sequence group flag for temporal fusion
        # Must be called after load_data_list (which does sorting) - same as original
        if self.use_sequence_group_flag:
            # CRITICAL: mmengine's BaseDataset uses lazy_init, but we need data NOW
            # Original model loads data in __init__ immediately, so we must do the same
            
            # Force data loading by explicitly calling load_data_list
            if not hasattr(self, 'data_list') or len(self.data_list) == 0:
                # Directly call load_data_list to populate data
                loaded_data = self.load_data_list()
                if loaded_data and len(loaded_data) > 0:
                    self.data_list = loaded_data
                    self.data_infos = loaded_data  # Sync for backward compatibility
                    print(f"DEBUG: Force-loaded {len(self.data_list)} samples for sequence flag")
            
            # Double-check data is loaded
            if hasattr(self, 'data_list') and len(self.data_list) > 0:
                # Ensure data_infos is synced
                if not hasattr(self, 'data_infos') or len(self.data_infos) == 0:
                    self.data_infos = self.data_list
                
                # Now set sequence group flag
                self._set_sequence_group_flag()
                print(f"DEBUG: Sequence group flag set, flag shape: {self.flag.shape}")
            else:
                print(f"WARNING: Cannot set sequence group flag - data not loaded")
                self.flag = np.array([], dtype=np.int64)

    def load_data_list(self):
        """Load annotations from file.
        
        Adapts the old format STCOcc annotations to the new mmengine format.
        CRITICAL: This method MUST return the data_list AND set self.data_infos
        CRITICAL: Match original behavior - sort by timestamp and apply load_interval
        """
        import mmengine
        
        # Load the original annotation file
        raw_data = mmengine.load(self.ann_file)
        
        # Extract data_infos based on format
        data_infos = None
        
        # Check if it's already in the new format
        if isinstance(raw_data, dict) and 'data_list' in raw_data and 'metainfo' in raw_data:
            data_infos = raw_data['data_list']
        elif isinstance(raw_data, list):
            data_infos = raw_data
        elif isinstance(raw_data, dict) and 'infos' in raw_data:
            data_infos = raw_data['infos']
        else:
            # Fallback: treat the whole data as data_list
            data_infos = raw_data if isinstance(raw_data, list) else [raw_data]
        
        # CRITICAL: Sort by timestamp to match original behavior
        # Original: data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))
        if data_infos and len(data_infos) > 0:
            # Check if timestamp key exists
            if 'timestamp' in data_infos[0]:
                data_infos = list(sorted(data_infos, key=lambda e: e['timestamp']))
            else:
                print(f"WARNING: 'timestamp' key not found in data_infos, skipping sort")
        
        # CRITICAL: Apply load_interval sampling to match original behavior
        # Original: data_infos = data_infos[::self.load_interval]
        load_interval = getattr(self, 'load_interval', 1)
        if load_interval > 1:
            original_len = len(data_infos)
            data_infos = data_infos[::load_interval]
        
        # Store metadata if available (for backward compatibility)
        if isinstance(raw_data, dict):
            if 'metadata' in raw_data:
                self.metadata = raw_data['metadata']
                if 'version' in self.metadata:
                    self.version = self.metadata['version']
            elif 'metainfo' in raw_data:
                self.metadata = raw_data['metainfo']
        
        # CRITICAL: Set data_infos for backward compatibility
        self.data_infos = data_infos
        
        # CRITICAL: Also ensure data_list is set (mmengine BaseDataset will overwrite this, but we set it for consistency)
        if not hasattr(self, 'data_list') or len(self.data_list) == 0:
            self.data_list = data_infos
        
        return data_infos

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        input_dict = super(NuScenesDatasetOccpancy, self).get_data_info(index)
        
        # CRITICAL: Ensure data_list is loaded before accessing it
        # In mmengine, full_init() calls load_data_list(), but get_data_info might be called before full_init()
        # So we need to ensure data_list is loaded here
        if not hasattr(self, 'data_list') or len(self.data_list) == 0:
            # Load data_list if not already loaded
            self.load_data_list()
        
        # CRITICAL: Ensure data_infos is synced with data_list (mmengine format)
        if not hasattr(self, 'data_infos') or len(self.data_infos) == 0:
            if hasattr(self, 'data_list') and len(self.data_list) > 0:
                self.data_infos = self.data_list
        
        # CRITICAL: Add index to match original format
        input_dict['index'] = index
        
        # CRITICAL: Get data_info from data_list or data_infos (use data_list first as it's mmengine format)
        data_info = None
        if hasattr(self, 'data_list') and len(self.data_list) > 0 and index < len(self.data_list):
            data_info = self.data_list[index]
        elif hasattr(self, 'data_infos') and len(self.data_infos) > 0 and index < len(self.data_infos):
            data_info = self.data_infos[index]
        
        # CRITICAL: Set sample_idx from token to match original format
        # Original: sample_idx=info['token'] (string token like '30e55a3ec6184d8cb1944b39ba19d622')
        if 'sample_idx' not in input_dict or input_dict.get('sample_idx') == index:
            if data_info is not None and 'token' in data_info:
                input_dict['sample_idx'] = data_info['token']
            elif 'token' in input_dict:
                input_dict['sample_idx'] = input_dict['token']
        
        # CRITICAL: Ensure pts_filename is properly set (match original format)
        # Priority: 1) data_info['lidar_path'], 2) input_dict['lidar_path'], 3) data_info['lidar_points']['lidar_path']
        if 'pts_filename' not in input_dict or not input_dict.get('pts_filename'):
            if data_info is not None:
                # STCOcc data uses 'lidar_path' key
                if 'lidar_path' in data_info and data_info['lidar_path']:
                    input_dict['pts_filename'] = data_info['lidar_path']
                # Also check if lidar_points has lidar_path (mmdet3d format)
                elif 'lidar_points' in data_info and isinstance(data_info['lidar_points'], dict):
                    if 'lidar_path' in data_info['lidar_points']:
                        input_dict['pts_filename'] = data_info['lidar_points']['lidar_path']
            
            # Fallback: use lidar_path from input_dict if already set by parent
            if ('pts_filename' not in input_dict or not input_dict.get('pts_filename')) and 'lidar_path' in input_dict:
                input_dict['pts_filename'] = input_dict['lidar_path']
        
        # CRITICAL: Add scene_name to match original format
        # Priority: 1) data_info['occ_path'], 2) input_dict['occ_path']
        if 'scene_name' not in input_dict or not input_dict.get('scene_name'):
            if data_info is not None:
                if 'occ_path' in data_info and data_info['occ_path']:
                    input_dict['scene_name'] = data_info['occ_path'].split('/')[-2]
            
            # Fallback: use occ_path from input_dict if already set by parent
            if ('scene_name' not in input_dict or not input_dict.get('scene_name')) and 'occ_path' in input_dict and input_dict['occ_path']:
                input_dict['scene_name'] = input_dict['occ_path'].split('/')[-2]
        
        # Super() already provides perfect data structure with cams info
        # Just add curr key pointing to the same data for STCOcc compatibility
        input_dict['curr'] = input_dict.copy()
        
        # Add specific STCOcc fields if they exist in the data
        if 'occ_path' in input_dict:
            input_dict['occ_gt_path'] = input_dict['occ_path']
        
        # Handle multi-frame adjacency if needed (STCOcc specific)
        # This follows the original STCOcc_ori implementation
        # CRITICAL: Use data_list (mmengine format) - it should be loaded by now
        # Match original condition: '4d' in img_info_prototype
        if (self.multi_adj_frame_id_cfg and 
            hasattr(self, 'img_info_prototype') and 
            '4d' in self.img_info_prototype):
            
            # CRITICAL: data_list should be loaded by now, but check again
            data_source = None
            if hasattr(self, 'data_list') and len(self.data_list) > 0:
                data_source = self.data_list
            elif hasattr(self, 'data_infos') and len(self.data_infos) > 0:
                data_source = self.data_infos

            # breakpoint()
            
            if data_source is not None and len(data_source) > 0 and index < len(data_source):
                info = data_source[index]
                adjacent_list = []
                
                # Build adjacent frame id list (same as get_adj_info)
                adj_id_list = list(range(*self.multi_adj_frame_id_cfg))
                
                # For stereo mode, add an additional frame
                if self.stereo:
                    assert self.multi_adj_frame_id_cfg[0] == 1
                    assert self.multi_adj_frame_id_cfg[2] == 1
                    adj_id_list.append(self.multi_adj_frame_id_cfg[1])
                
                # Get adjacent frames (PAST frames, not future!)
                for select_id in adj_id_list:
                    # Use PAST frame: index - select_id (same as get_adj_info)
                    adj_data_idx = max(index - select_id, 0)
                    
                    # Check if it's the same scene, otherwise use current frame
                    if adj_data_idx < len(data_source) and 'scene_token' in info and 'scene_token' in data_source[adj_data_idx]:
                        if not data_source[adj_data_idx]['scene_token'] == info['scene_token']:
                            # Different scene, use current frame instead
                            adjacent_list.append(info)
                        else:
                            # Same scene, use the adjacent frame
                            adjacent_list.append(data_source[adj_data_idx])
                    elif adj_data_idx < len(data_source):
                        # No scene_token info, use the adjacent frame
                        adjacent_list.append(data_source[adj_data_idx])
                    else:
                        # Index out of bounds, use current frame
                        adjacent_list.append(info)
                
                if adjacent_list:
                    input_dict['adjacent'] = adjacent_list
        
        input_dict['seg_label_mapping'] = self.SegLabelMapping
        
        # CRITICAL: Update can_bus information to match original format
        # Original code overwrites can_bus pos&rot information with ego2global values
        if 'can_bus' in input_dict and 'curr' in input_dict:
            from pyquaternion import Quaternion
            from nuscenes.eval.common.utils import quaternion_yaw
            
            rotation = Quaternion(input_dict['curr']['ego2global_rotation'])
            translation = input_dict['curr']['ego2global_translation']
            can_bus = input_dict['can_bus'].copy()  # Make a copy to avoid modifying original
            # Overwrite the canbus pos&rot information.
            can_bus[:3] = translation
            can_bus[3:7] = rotation
            patch_angle = quaternion_yaw(rotation) / np.pi * 180
            if patch_angle < 0:
                patch_angle += 360
            can_bus[-2] = patch_angle / 180 * np.pi
            can_bus[-1] = patch_angle
            input_dict['can_bus'] = can_bus
        elif 'can_bus' in input_dict and 'ego2global_rotation' in input_dict and 'ego2global_translation' in input_dict:
            # Fallback: use direct keys if 'curr' key doesn't exist
            from pyquaternion import Quaternion
            from nuscenes.eval.common.utils import quaternion_yaw
            
            rotation = Quaternion(input_dict['ego2global_rotation'])
            translation = input_dict['ego2global_translation']
            can_bus = input_dict['can_bus'].copy()  # Make a copy to avoid modifying original
            # Overwrite the canbus pos&rot information.
            can_bus[:3] = translation
            can_bus[3:7] = rotation
            patch_angle = quaternion_yaw(rotation) / np.pi * 180
            if patch_angle < 0:
                patch_angle += 360
            can_bus[-2] = patch_angle / 180 * np.pi
            can_bus[-1] = patch_angle
            input_dict['can_bus'] = can_bus
        
        return input_dict
    
    def _set_sequence_group_flag(self):
        """
        Set each sequence to be a different group
        """
        # CRITICAL: Ensure data_infos is populated (sync from data_list if needed)
        if not hasattr(self, 'data_infos') or len(self.data_infos) == 0:
            if hasattr(self, 'data_list') and len(self.data_list) > 0:
                self.data_infos = self.data_list
            else:
                print(f"WARNING: Both data_list and data_infos are empty, cannot set sequence group flag")
                self.flag = np.array([], dtype=np.int64)
                return
        
        # CRITICAL: Match original - use self.data_infos directly (not data_source)
        res = []
        curr_sequence = 0
        for idx in range(len(self.data_infos)):
            # CRITICAL: Match original - direct access to 'prev' key (not .get())
            if idx != 0 and len(self.data_infos[idx]['prev']) == 0:
                # Not first frame and # of sweeps is 0 -> new sequence
                curr_sequence += 1
            res.append(curr_sequence)

        self.flag = np.array(res, dtype=np.int64)

        # breakpoint()

        if self.sequences_split_num != 1:
            if self.sequences_split_num == 'all':
                self.flag = np.array(range(len(self.data_infos)), dtype=np.int64)
            else:
                bin_counts = np.bincount(self.flag)
                new_flags = []
                curr_new_flag = 0
                for curr_flag in range(len(bin_counts)):
                    curr_sequence_length = np.array(
                        list(range(0,
                                bin_counts[curr_flag],
                                math.ceil(bin_counts[curr_flag] / self.sequences_split_num)))
                        + [bin_counts[curr_flag]])
                    for sub_seq_idx in (curr_sequence_length[1:] - curr_sequence_length[:-1]):
                        for _ in range(sub_seq_idx):
                            new_flags.append(curr_new_flag)
                        curr_new_flag += 1

                # breakpoint()

                assert len(new_flags) == len(self.flag)
                assert len(np.bincount(new_flags)) == len(np.bincount(self.flag)) * self.sequences_split_num
                self.flag = np.array(new_flags, dtype=np.int64)

    def prepare_data(self, idx: int) -> Any:
        """Prepare data for the given index.
        
        CRITICAL: Completely override BaseDataset.prepare_data() to ensure 
        'adjacent' and temporal information are passed to the pipeline.
        """
        # Get data info
        data_info = self.get_data_info(idx)
        
        # CRITICAL: Add box_mode_3d and box_type_3d to match original format
        # These are normally added by pre_pipeline in mmdet3d, but we need to add them explicitly
        if hasattr(self, 'box_type_3d') and hasattr(self, 'box_mode_3d'):
            data_info['box_type_3d'] = self.box_type_3d
            data_info['box_mode_3d'] = self.box_mode_3d
        else:
            # Fallback: try to get from parent class or set defaults
            from mmdet3d.structures.bbox_3d import LiDARInstance3DBoxes
            data_info['box_type_3d'] = 'LiDAR'
            data_info['box_mode_3d'] = LiDARInstance3DBoxes
        
        # Add temporal fusion information
        if self.use_sequence_group_flag:
            
            # breakpoint()

            # CRITICAL: Ensure flag is set and has correct length
            if not hasattr(self, 'flag') or len(self.flag) == 0:
                # Flag not set yet, set it now
                if hasattr(self, 'data_list') and len(self.data_list) > 0:
                    if not hasattr(self, 'data_infos') or len(self.data_infos) == 0:
                        self.data_infos = self.data_list
                    self._set_sequence_group_flag()
                else:
                    # Fallback: use index as sequence_group_idx
                    data_info['sequence_group_idx'] = 0
                    data_info['start_of_sequence'] = True
                    data_info['curr_to_prev_ego_rt'] = torch.eye(4).float()
                    data_info['curr_to_prev_lidar_rt'] = torch.eye(4).float()  # CRITICAL: Add this
                    return self.pipeline(data_info)
            
            # Ensure idx is within bounds
            if idx >= len(self.flag):
                print(f"WARNING: idx {idx} >= len(flag) {len(self.flag)}, using 0 as default")
                data_info['sequence_group_idx'] = 0
            else:
                data_info['sample_index'] = idx
                data_info['sequence_group_idx'] = int(self.flag[idx])
            
            data_info['start_of_sequence'] = idx == 0 or self.flag[idx - 1] != self.flag[idx]
            # data_info['start_of_sequence'] = True
            
            # CRITICAL: Get transformation matrices from current to previous frame
            # Match original format: both curr_to_prev_lidar_rt and curr_to_prev_ego_rt
            if not data_info['start_of_sequence'] and idx > 0:
                # Ensure data_infos is available
                if not hasattr(self, 'data_infos') or len(self.data_infos) == 0:
                    if hasattr(self, 'data_list') and len(self.data_list) > 0:
                        self.data_infos = self.data_list
                
                if hasattr(self, 'data_infos') and len(self.data_infos) > idx and len(self.data_infos) > idx - 1:
                    # CRITICAL: Add curr_to_prev_lidar_rt to match original format
                    data_info['curr_to_prev_lidar_rt'] = torch.FloatTensor(nuscenes_get_rt_matrix(
                        self.data_infos[idx], self.data_infos[idx - 1],
                        "lidar", "lidar"))
                    data_info['curr_to_prev_ego_rt'] = torch.FloatTensor(nuscenes_get_rt_matrix(
                        self.data_infos[idx], self.data_infos[idx - 1],
                        "ego", "ego"))
                else:
                    data_info['curr_to_prev_lidar_rt'] = torch.eye(4).float()
                    data_info['curr_to_prev_ego_rt'] = torch.eye(4).float()
            else:
                # Match original: calculate transformation matrix even for start_of_sequence
                # Original uses nuscenes_get_rt_matrix with same frame for both src and dest
                if hasattr(self, 'data_infos') and len(self.data_infos) > idx:
                    data_info['curr_to_prev_lidar_rt'] = torch.eye(4).float()
                    data_info['curr_to_prev_ego_rt'] = torch.FloatTensor(nuscenes_get_rt_matrix(
                        self.data_infos[idx], self.data_infos[idx],
                        "ego", "ego"))
                else:
                    data_info['curr_to_prev_lidar_rt'] = torch.eye(4).float()
                    data_info['curr_to_prev_ego_rt'] = torch.eye(4).float()
        
        # Apply pipeline transforms
        # Deepcopy to avoid modifying cached data_info
        import copy
        data = copy.deepcopy(data_info)
        
        # Execute the pipeline
        data = self.pipeline(data)
        
        return data
        
    def get_cat_ids(self, idx):
        """Get category ids by index. Override to handle class statistics properly."""
        # Try multiple ways to access data
        info = None
        if hasattr(self, 'data_list') and len(self.data_list) > idx:
            info = self.data_list[idx]
        elif hasattr(self, 'data_infos') and len(self.data_infos) > idx:
            info = self.data_infos[idx]
        else:
            # Fallback: Load data directly from file if needed
            try:
                import mmengine
                raw_data = mmengine.load(self.ann_file)
                if isinstance(raw_data, dict) and 'infos' in raw_data:
                    infos = raw_data['infos']
                elif isinstance(raw_data, list):
                    infos = raw_data
                else:
                    return []
                    
                if idx < len(infos):
                    info = infos[idx]
                else:
                    return []
            except Exception:
                return []
        
        if info is not None:
            if 'gt_names' in info and len(info['gt_names']) > 0:
                    
                # Create NuScenes class name mapping to handle different formats
                nusc_class_mapping = {
                    'movable_object.pushable_pullable': 'others',
                    'movable_object.barrier': 'barrier', 
                    'vehicle.bicycle': 'bicycle',
                    'vehicle.bus.bendy': 'bus',
                    'vehicle.bus.rigid': 'bus',
                    'vehicle.car': 'car',
                    'vehicle.construction': 'construction_vehicle',
                    'vehicle.motorcycle': 'motorcycle',
                    'human.pedestrian.adult': 'pedestrian',
                    'human.pedestrian.child': 'pedestrian',
                    'human.pedestrian.construction_worker': 'pedestrian',
                    'human.pedestrian.personal_mobility': 'pedestrian',
                    'human.pedestrian.police_officer': 'pedestrian',
                    'human.pedestrian.stroller': 'pedestrian',
                    'human.pedestrian.wheelchair': 'pedestrian',
                    'movable_object.trafficcone': 'traffic_cone',
                    'vehicle.trailer': 'trailer',
                    'vehicle.truck': 'truck'
                }
                
                # Get the class list from config or METAINFO
                if hasattr(self, '_classes') and self._classes:
                    class_names = self._classes
                elif hasattr(self, 'METAINFO') and 'classes' in self.METAINFO:
                    class_names = self.METAINFO['classes']
                else:
                    # Default to the classes used in config
                    class_names = ['others','barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
                                 'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
                                 'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade', 'vegetation','free']
                
                
                cat_ids = []
                for name in info['gt_names']:
                    # Convert bytes to string if needed
                    if isinstance(name, bytes):
                        name = name.decode('utf-8')
                    
                    # Apply mapping if exists
                    mapped_name = nusc_class_mapping.get(name, name)
                    
                    # Find in class list
                    if mapped_name in class_names:
                        cat_ids.append(class_names.index(mapped_name))
                    else:
                        # Map unknown classes to 'others' if it exists
                        if 'others' in class_names:
                            cat_ids.append(class_names.index('others'))
                
                return cat_ids
            else:
                pass
        return []
        
    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: annotation information that contains the following keys:
                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): 3D ground truth bboxes.
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """
        if hasattr(self, 'data_infos') and index < len(self.data_infos):
            info = self.data_infos[index]
            
            # Get annotation data
            ann_info = {}
            
            if 'gt_boxes' in info:
                ann_info['gt_bboxes_3d'] = info['gt_boxes']
                
            if 'gt_names' in info:
                gt_names = info['gt_names']
                if isinstance(gt_names[0], bytes):
                    gt_names = [name.decode('utf-8') for name in gt_names]
                ann_info['gt_names'] = gt_names
                
                # Convert class names to indices
                gt_labels = []
                class_names = self._classes if self._classes else []
                
                # NuScenes class mapping
                nusc_class_mapping = {
                    'movable_object.pushable_pullable': 'others',
                    'movable_object.barrier': 'barrier', 
                    'vehicle.bicycle': 'bicycle',
                    'vehicle.bus.bendy': 'bus',
                    'vehicle.bus.rigid': 'bus',
                    'vehicle.car': 'car',
                    'vehicle.construction': 'construction_vehicle',
                    'vehicle.motorcycle': 'motorcycle',
                    'human.pedestrian.adult': 'pedestrian',
                    'human.pedestrian.child': 'pedestrian',
                    'human.pedestrian.construction_worker': 'pedestrian',
                    'human.pedestrian.personal_mobility': 'pedestrian',
                    'human.pedestrian.police_officer': 'pedestrian',
                    'human.pedestrian.stroller': 'pedestrian',
                    'human.pedestrian.wheelchair': 'pedestrian',
                    'movable_object.trafficcone': 'traffic_cone',
                    'vehicle.trailer': 'trailer',
                    'vehicle.truck': 'truck'
                }
                
                for name in gt_names:
                    mapped_name = nusc_class_mapping.get(name, name)
                    if mapped_name in class_names:
                        gt_labels.append(class_names.index(mapped_name))
                    else:
                        # Default to 'others' class
                        if 'others' in class_names:
                            gt_labels.append(class_names.index('others'))
                        else:
                            gt_labels.append(0)  # fallback to first class
                            
                ann_info['gt_labels_3d'] = np.array(gt_labels, dtype=np.int64)
                
            return ann_info
        
    def _calculate_num_ins_per_cat(self):
        """Calculate the number of instances per category in the dataset."""
        from collections import defaultdict
        import mmengine
        
        # Initialize counters
        num_ins_per_cat = defaultdict(int)
        
        # Get classes - only include object detection classes for statistics
        all_classes = self._classes if self._classes is not None else []
        if not all_classes and hasattr(self, 'METAINFO') and 'classes' in self.METAINFO:
            all_classes = self.METAINFO['classes']
            
        # Filter out voxel semantic classes that are not in object annotations
        # These classes are for occupancy prediction, not object detection
        voxel_semantic_classes = {
            'driveable_surface', 'other_flat', 'sidewalk', 
            'terrain', 'manmade', 'vegetation', 'free'
        }
        
        # Only include object detection classes in statistics
        classes = [cls for cls in all_classes if cls not in voxel_semantic_classes]
        
        # Initialize all classes to 0
        for class_name in classes:
            num_ins_per_cat[class_name] = 0
            
        # Load data once and reuse it - much faster than loading for each sample
        try:
            raw_data = mmengine.load(self.ann_file)
            if isinstance(raw_data, dict) and 'infos' in raw_data:
                infos = raw_data['infos']
            elif isinstance(raw_data, list):
                infos = raw_data
            else:
                self.num_ins_per_cat = dict(num_ins_per_cat)
                return
        except Exception:
            self.num_ins_per_cat = dict(num_ins_per_cat)
            return
            
        # Create NuScenes class name mapping to handle different formats
        nusc_class_mapping = {
            'movable_object.pushable_pullable': 'others',
            'movable_object.barrier': 'barrier', 
            'vehicle.bicycle': 'bicycle',
            'vehicle.bus.bendy': 'bus',
            'vehicle.bus.rigid': 'bus',
            'vehicle.car': 'car',
            'vehicle.construction': 'construction_vehicle',
            'vehicle.motorcycle': 'motorcycle',
            'human.pedestrian.adult': 'pedestrian',
            'human.pedestrian.child': 'pedestrian',
            'human.pedestrian.wheelchair': 'pedestrian',
            'human.pedestrian.stroller': 'pedestrian',
            'human.pedestrian.personal_mobility': 'pedestrian',
            'human.pedestrian.police_officer': 'pedestrian',
            'human.pedestrian.construction_worker': 'pedestrian',
            'movable_object.trafficcone': 'traffic_cone',
            'vehicle.trailer': 'trailer',
            'vehicle.truck': 'truck',
        }
        
        # Count instances in a subset of samples for speed
        sample_count = min(len(infos), 1000)  # First 1000 samples
        for idx in range(sample_count):
            try:
                info = infos[idx]
                if 'gt_names' in info and len(info['gt_names']) > 0:
                    for name in info['gt_names']:
                        # Convert bytes to string if needed
                        if isinstance(name, bytes):
                            name = name.decode('utf-8')
                        
                        # Map to standard class name
                        mapped_name = nusc_class_mapping.get(name, name)
                        
                        # Find in class list
                        if mapped_name in classes:
                            num_ins_per_cat[mapped_name] += 1
                        else:
                            # Map unknown classes to 'others' if it exists
                            if 'others' in classes:
                                num_ins_per_cat['others'] += 1
            except Exception:
                continue
                
        # Store the results
        self.num_ins_per_cat = dict(num_ins_per_cat)
        
            
    def _show_correct_statistics(self):
        """Show corrected dataset statistics."""
        from mmengine.logging import print_log
        try:
            from terminaltables import AsciiTable
        except ImportError:
            from mmdet.utils import AsciiTable
        
        # Show statistics similar to Det3DDataset
        print_log('-' * 30, 'current')
        print_log(f'The length of the dataset: {len(self)}', 'current')
        
        if hasattr(self, 'num_ins_per_cat') and self.num_ins_per_cat:
            content_show = [['category', 'number']]
            for cat_name, num in self.num_ins_per_cat.items():
                content_show.append([cat_name, num])
            table = AsciiTable(content_show)
            print_log(
                f'The number of instances per category in the dataset:\n{table.table}',
                'current')
        else:
            print_log('No class statistics available', 'current')

    def evaluate_rayioU(self, results, logger=None, dataset_name='openocc'):
        if self.eval_show:
            mmcv.mkdir_or_exist(self.work_dir)

        pred_sems, gt_sems = [], []
        pred_flows, gt_flows = [], []
        lidar_origins = []
        data_index = []

        print('\nStarting Evaluation...')
        processed_set = set()
        for index, result in enumerate(results):
            data_id = result['index']
            for i, id in enumerate(data_id):
                if id in processed_set: continue
                processed_set.add(id)

                pred_sem = result['occ_results'][i]

                if 'flow_results' not in result:
                    pred_flow = np.zeros(pred_sem.shape + (2, ))
                else:
                    pred_flow = result['flow_results'][i]

                data_index.append(id)
                pred_sems.append(pred_sem)
                pred_flows.append(pred_flow)

        nusc = NuScenes('v1.0-trainval', 'data/nuscenes/')
        nusdata = nuScenesDataset(nusc, 'val')

        for index in data_index:
            if index >= len(self.data_infos):
                break
            info = self.data_infos[index]

            occ_path = info['occ_path']
            if dataset_name == 'openocc':
                occ_path = occ_path.replace('gts', 'openocc_v2')
            occ_path = os.path.join(occ_path, 'labels.npz')
            occ_gt = np.load(occ_path, allow_pickle=True)

            gt_semantics = occ_gt['semantics'].astype(np.uint8)
            if dataset_name == 'occ3d':
                gt_flow = np.zeros((200, 200, 16, 2), dtype=np.float16)
            elif dataset_name == 'openocc':
                gt_flow = occ_gt['flow'].astype(np.float16)

            gt_sems.append(gt_semantics)
            gt_flows.append(gt_flow)

            # get lidar
            ref_sample_token, output_origin_tensor = nusdata.__getitem__(index)
            lidar_origins.append(output_origin_tensor.unsqueeze(0))

        # visualization
        # if self.eval_show:
        #     for index in range(len(data_index)):
        #         if index >= len(self.data_infos):
        #             break
        #         info = self.data_infos[data_index[index]]
        #         if dataset_name == 'openocc':
        #             occ_bev_vis = self.vis_occ(pred_sems[index], color_map=openocc_colors_map, empty_idx=16)
        #             occ_bev_vis_gt = self.vis_occ(gt_sems[index], color_map=openocc_colors_map, empty_idx=16)
        #         elif dataset_name == 'occ3d':
        #             occ_bev_vis = self.vis_occ(pred_sems[index], color_map=occ3d_colors_map, empty_idx=17)
        #             occ_bev_vis_gt = self.vis_occ(gt_sems[index], color_map=occ3d_colors_map, empty_idx=17)
        #         scene_token = info['token']
        #         occ_bev_vis = np.concatenate([occ_bev_vis, occ_bev_vis_gt], axis=1)
        #         cv2.imwrite(os.path.join(self.work_dir, f'{scene_token}.png'), occ_bev_vis)

        if dataset_name == 'openocc':
            miou, mave, occ_score = ray_based_miou_openocc(pred_sems, gt_sems, pred_flows, gt_flows, lidar_origins, logger=logger)
        elif dataset_name == 'occ3d':
            miou, mave, occ_score = ray_based_miou_occ3d(pred_sems, gt_sems, pred_flows, gt_flows, lidar_origins, logger=logger)

        eval_dict = {
            'miou':miou,
            'mave':mave
        }
        return eval_dict

    def evaluate_miou(self, results, logger=None, dataset_name='openocc'):
        pred_sems, gt_sems = [], []
        data_index = []

        num_classes = 17 if dataset_name == 'openocc' else 18
        use_image_mask = True if dataset_name == 'occ3d' else False
        self.miou_metric = Metric_mIoU(
            num_classes=num_classes,
            use_lidar_mask=False,
            use_image_mask=use_image_mask,
            logger=logger
        )

        print('\nStarting Evaluation...')
        processed_set = set()
        for result in results:
            data_id = result['index']
            for i, id in enumerate(data_id):
                if id in processed_set: continue
                processed_set.add(id)

                pred_sem = result['occ_results'][i]
                data_index.append(id)
                pred_sems.append(pred_sem)

        for index in tqdm(data_index):
            if index >= len(self.data_infos):
                break
            info = self.data_infos[index]

            occ_path = info['occ_path']
            if dataset_name == 'openocc':
                occ_path = occ_path.replace('gts', 'openocc_v2')
            occ_path = os.path.join(occ_path, 'labels.npz')
            occ_gt = np.load(occ_path, allow_pickle=True)

            gt_semantics = occ_gt['semantics']
            pr_semantics = pred_sems[data_index.index(index)]

            if dataset_name == 'occ3d':
                mask_camera = occ_gt['mask_camera'].astype(bool)
            else:
                mask_camera = None

            self.miou_metric.add_batch(pr_semantics, gt_semantics, None, mask_camera)

        _, miou, _, _, _ = self.miou_metric.count_miou()
        eval_dict = {
            'miou':miou,
        }
        return eval_dict

    def evaluate(self, occ_results, logger=None, runner=None, show_dir=None, **eval_kwargs):
        if self.eval_metric == 'rayiou':
            return self.evaluate_rayioU(occ_results, logger, dataset_name=self.dataset_name)
        elif self.eval_metric == 'miou':
            return self.evaluate_miou(occ_results, logger, dataset_name=self.dataset_name)


    def vis_occ(self, semantics, empty_idx, color_map=None):
        # simple visualization of result in BEV
        semantics_valid = np.logical_not(semantics == empty_idx)
        d = np.arange(16).reshape(1, 1, 16)
        d = np.repeat(d, 200, axis=0)
        d = np.repeat(d, 200, axis=1).astype(np.float32)
        d = d * semantics_valid
        selected = np.argmax(d, axis=2)

        selected_torch = torch.from_numpy(selected)
        semantics_torch = torch.from_numpy(semantics)

        occ_bev_torch = torch.gather(semantics_torch, dim=2, index=selected_torch.unsqueeze(-1))
        occ_bev = occ_bev_torch.numpy()

        occ_bev = occ_bev.flatten().astype(np.int32)
        occ_bev_vis = color_map[occ_bev].astype(np.uint8)
        occ_bev_vis = occ_bev_vis.reshape(200, 200, 3)[::-1, ::-1, :3]
        occ_bev_vis = cv2.resize(occ_bev_vis, (200, 200))

        occ_bev_vis = cv2.resize(occ_bev_vis, (600, 600))
        occ_bev_vis = cv2.cvtColor(occ_bev_vis, cv2.COLOR_BGR2RGB)
        return occ_bev_vis