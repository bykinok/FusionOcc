# Copyright (c) OpenMMLab. All rights reserved.
import copy
import numpy as np
import os
from mmdet3d.datasets import NuScenesDataset
from mmdet3d.registry import DATASETS
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x


@DATASETS.register_module()
class FusionOccDataset(NuScenesDataset):
    """FusionOcc dataset for 3D occupancy prediction.
    
    This dataset is based on nuScenes dataset but adds occupancy-specific
    annotations and processing.
    """
    
    def __init__(self, 
                 data_root,
                 ann_file,
                 pipeline=None,
                 classes=None,
                 modality=None,
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 test_mode=False,
                 use_valid_flag=False,
                 **kwargs):
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            use_valid_flag=use_valid_flag,
            **kwargs)
    
    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data
                preprocessing pipelines.
        """

        data_info = super().get_data_info(index)
        
        # Add FusionOcc specific information
        if 'ann_infos' in data_info:
            # Add occupancy ground truth path
            if 'occ_path' in data_info:
                data_info['occ_gt_path'] = data_info['occ_path']
            
            # Add annotation information for occupancy
            if 'ann_infos' in data_info:
                data_info['gt_boxes_3d'] = data_info['ann_infos'][0]  # boxes
                data_info['gt_labels_3d'] = data_info['ann_infos'][1]  # labels
        
        return data_info
    
    def prepare_train_data(self, index):
        """Training data preparation."""
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None

        # NuScenesDatasetOccpancy는 Det3DDataset을 직접 상속받지 않으므로 pre_pipeline 대신 직접 처리
        # deepcopy here to avoid inplace modification in pipeline.
        input_dict = copy.deepcopy(input_dict)

        # box_type_3d (str): 3D box type.
        input_dict['box_type_3d'] = self.box_type_3d
        # box_mode_3d (str): 3D box mode.
        input_dict['box_mode_3d'] = self.box_mode_3d

        # Add FusionOcc specific data preparation
        if self.modality is None:
            input_dict['modality'] = dict(use_lidar=True, use_camera=True)

        # Add occupancy-specific keys
        input_dict['voxel_semantics'] = None  # Will be loaded by pipeline
        input_dict['mask_camera'] = None  # Will be loaded by pipeline

        try:
            data = self.pipeline(input_dict)
            return data
        except Exception as e:
            # 디버깅을 위한 에러 로깅
            import traceback
            print(f"ERROR in pipeline for index {index}: {e}")
            print(f"input_dict keys: {input_dict.keys()}")
            if 'curr' in input_dict:
                print(f"curr keys: {input_dict['curr'].keys() if isinstance(input_dict['curr'], dict) else 'not dict'}")
            elif 'images' in input_dict:
                print(f"images keys: {list(input_dict['images'].keys()) if isinstance(input_dict['images'], dict) else 'not dict'}")
            traceback.print_exc()
            return None  # None을 반환하면 BaseDataset이 재시도

    def prepare_test_data(self, index):
        """Testing data preparation."""
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None

        # NuScenesDatasetOccpancy는 Det3DDataset을 직접 상속받지 않으므로 pre_pipeline 대신 직접 처리
        # deepcopy here to avoid inplace modification in pipeline.
        input_dict = copy.deepcopy(input_dict)

        # box_type_3d (str): 3D box type.
        input_dict['box_type_3d'] = self.box_type_3d
        # box_mode_3d (str): 3D box mode.
        input_dict['box_mode_3d'] = self.box_mode_3d

        # Add FusionOcc specific data preparation
        if self.modality is None:
            input_dict['modality'] = dict(use_lidar=True, use_camera=True)

        try:
            data = self.pipeline(input_dict)
            return data
        except Exception as e:
            import traceback
            print(f"ERROR in test pipeline for index {index}: {e}")
            traceback.print_exc()
            raise  # 테스트 모드에서는 예외를 다시 발생시킴


@DATASETS.register_module()
class NuScenesDatasetOccpancy(NuScenesDataset):
    """NuScenes Occupancy dataset for 3D occupancy prediction.
    
    This dataset is compatible with the original FusionOcc implementation
    and supports occupancy ground truth loading.
    """
    
    def __init__(self, 
                 use_mask=True,
                 classes=None,
                 **kwargs):
        # Remove problematic kwargs that are not supported by MMEngine BaseDataset
        unsupported_args = ['classes', 'stereo', 'filter_empty_gt', 'img_info_prototype', 
                           'multi_adj_frame_id_cfg', 'multi_adj_frame_id_cfg_lidar']
        self.use_mask = use_mask
        
        # Store removed arguments as instance variables
        for arg in unsupported_args:
            if arg in kwargs:
                setattr(self, arg, kwargs.pop(arg))
        
        # Store classes
        if classes is not None:
            self.CLASSES = classes
        elif hasattr(self, 'classes'):
            self.CLASSES = self.classes
        
        # Store original lazy_init value and show_ins_var parameter
        original_lazy_init = kwargs.get('lazy_init', False)
        show_ins_var = kwargs.pop('show_ins_var', False)
        
        # Suppress parent class statistics output temporarily
        kwargs['lazy_init'] = True
        super().__init__(**kwargs)
        
        # Initialize attributes that were skipped due to lazy_init=True
        self.show_ins_var = show_ins_var
        
        # breakpoint()

        # Ensure data_list is properly loaded (workaround for BaseDataset issue)
        if not hasattr(self, 'data_list') or not self.data_list:
            self.data_list = self.load_data_list()
        
        # Fix num_ins_per_cat calculation before any statistics output
        self._calculate_num_ins_per_cat()
        
        # Properly initialize _fully_initialized attribute if not set by parent
        if not hasattr(self, '_fully_initialized'):
            self._fully_initialized = True
        
        # Now trigger parent class statistics output with correct values
        if not original_lazy_init:
            self._show_correct_statistics()
    
    def load_data_list(self):
        """Load data list from annotation file."""
        import pickle
        
        # Load data from pkl file
        ann_file = self.ann_file
        if not ann_file.startswith('/'):
            # Make relative paths absolute
            import os
            ann_file = os.path.abspath(ann_file)
            
        with open(ann_file, 'rb') as f:
            data = pickle.load(f)
        
        

        # Extract data list - support both fusionocc format ('infos') and mmdet3d format ('data_list')
        if isinstance(data, dict):
            if 'data_list' in data:
                data_list = data['data_list']
            elif 'infos' in data:
                data_list = data['infos']
            else:
                raise ValueError(f"Dict in {ann_file} has neither 'data_list' nor 'infos' key. Keys: {data.keys()}")
        elif isinstance(data, list):
            data_list = data
        else:
            raise ValueError(f"Unsupported data format in {ann_file}: {type(data)}")
        
        # CRITICAL: Sort by timestamp to match original behavior
        # Original code: data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))
        if len(data_list) > 0 and 'timestamp' in data_list[0]:
            data_list = list(sorted(data_list, key=lambda e: e['timestamp']))
        
        # Apply load_interval if specified (matching original behavior)
        # Original code: data_infos = data_infos[::self.load_interval]
        if hasattr(self, 'load_interval') and self.load_interval > 1:
            data_list = data_list[::self.load_interval]

        # breakpoint()
        
        return data_list
    
    
    def get_adj_info(self, info: dict, index: int) -> list:
        """Get adjacent frame information for temporal fusion.
        
        Args:
            info (dict): Current frame info
            index (int): Current frame index
            
        Returns:
            list: List of adjacent frame infos
        """
        info_adj_list = []
        if not hasattr(self, 'multi_adj_frame_id_cfg'):
            return info_adj_list
            
        adj_id_list = list(range(*self.multi_adj_frame_id_cfg))
        
        for select_id in adj_id_list:
            orig_select_id = select_id
            select_id = max(index - select_id, 0)
            
            # Check if adjacent frame is in the same scene
            # If select_id == index, it means we can't go back (first frame of scene), use current frame
            if (select_id == index or 
                select_id >= len(self.data_list) or 
                self.data_list[select_id].get('scene_token') != info.get('scene_token')):
                # If not same scene, out of bounds, or same index, use current frame
                info_adj_list.append(info)
            else:
                # Use previous frame from same scene
                info_adj_list.append(self.data_list[select_id])
        
        return info_adj_list
    
    def get_adj_info_lidar(self, info: dict, index: int) -> list:
        """Get adjacent lidar frame information for temporal fusion.
        
        Args:
            info (dict): Current frame info
            index (int): Current frame index
            
        Returns:
            list: List of adjacent lidar frame infos
        """
        info_adj_list = []
        if not hasattr(self, 'multi_adj_frame_id_cfg_lidar'):
            return info_adj_list
            
        adj_id_list = list(range(*self.multi_adj_frame_id_cfg_lidar))
        
        for select_id in adj_id_list:
            select_id = max(index - select_id, 0)
            # Check if adjacent frame is in the same scene
            # If select_id == index, it means we can't go back (first frame of scene), use current frame
            if (select_id == index or 
                select_id >= len(self.data_list) or 
                self.data_list[select_id].get('scene_token') != info.get('scene_token')):
                # If not same scene, out of bounds, or same index, use current frame
                info_adj_list.append(info)
            else:
                # Use previous frame from same scene
                info_adj_list.append(self.data_list[select_id])
        
        return info_adj_list

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
                - occ_gt_path (str): Path to occupancy ground truth.
        """
        # breakpoint()
        # Ensure data_list is properly loaded and check bounds
        if not hasattr(self, 'data_list') or not self.data_list:
            self.data_list = self.load_data_list()
        
        if index >= len(self.data_list):
            raise IndexError(f"Index {index} out of range for data_list with length {len(self.data_list)}")
        
        info = self.data_list[index]
        
        # Create input_dict in the format expected by the pipeline
        input_dict = dict()
        
        # Basic info
        if 'token' in info:
            input_dict['sample_idx'] = info['token']
        else:
            input_dict['sample_idx'] = info.get('sample_idx', index)
        input_dict['token'] = info.get('token', '')
        input_dict['timestamp'] = info.get('timestamp', 0.0)
        
        # Lidar information - support both formats
        if 'lidar_points' in info:
            # occfrmwrk format: nested dict with 4x4 transformation matrix
            lidar_info = info['lidar_points']
            lidar_path = lidar_info.get('lidar_path', '')
            
            # Handle path: occfrmwrk uses relative filename
            if lidar_path and not lidar_path.startswith('data/') and not lidar_path.startswith('./'):
                lidar_path = f'data/nuscenes/samples/LIDAR_TOP/{lidar_path}'
            elif lidar_path.startswith('./'):
                lidar_path = lidar_path[2:]
                
            input_dict['pts_filename'] = lidar_path
            
            # lidar2ego: occfrmwrk provides 4x4 transformation matrix as nested list
            lidar2ego_matrix = lidar_info.get('lidar2ego', np.eye(4))
            input_dict['lidar2ego'] = np.array(lidar2ego_matrix).reshape(4, 4)
            
            input_dict['lidar_points'] = {
                'lidar_path': lidar_path,
                'num_pts_feats': lidar_info.get('num_pts_feats', 5)
            }
        elif 'lidar_path' in info:
            # fusionocc format: direct path with rotation and translation separate
            lidar_path = info['lidar_path']
            # Remove leading './' if present
            if lidar_path.startswith('./'):
                lidar_path = lidar_path[2:]
            input_dict['pts_filename'] = lidar_path
            
            # Build 4x4 transformation matrix from rotation quaternion and translation
            if 'lidar2ego_rotation' in info and 'lidar2ego_translation' in info:
                from pyquaternion import Quaternion
                lidar2ego_rotation = Quaternion(info['lidar2ego_rotation'])
                lidar2ego_translation = np.array(info['lidar2ego_translation'])
                
                lidar2ego = np.eye(4)
                lidar2ego[:3, :3] = lidar2ego_rotation.rotation_matrix
                lidar2ego[:3, 3] = lidar2ego_translation
                input_dict['lidar2ego'] = lidar2ego
            else:
                input_dict['lidar2ego'] = np.eye(4)
                
            input_dict['lidar_points'] = {
                'lidar_path': lidar_path,
                'num_pts_feats': 5
            }
        
        # Image information - 원본과 동일하게 modality 체크 추가
        # 원본: if self.modality['use_camera']: ... else: input_dict.update(dict(curr=info))
        if self.modality is None or self.modality.get('use_camera', True):
            if 'images' in info:
                # occfrmwrk format
                input_dict['images'] = info['images']
            elif 'cams' in info:
                # fusionocc format: pass entire info as 'curr' for compatibility (원본과 동일)
                # 원본 코드: input_dict.update(dict(curr=info))
                input_dict['curr'] = info
                
                # Add adjacent frames for temporal fusion (원본과 동일)
                # 원본: if 'fusion' in self.img_info_prototype: ...
                if hasattr(self, 'img_info_prototype') and 'fusion' in self.img_info_prototype:
                    info_adj_list = self.get_adj_info(info, index)
                    if info_adj_list:
                        input_dict['adjacent'] = info_adj_list
                    
                    info_adj_list_lidar = self.get_adj_info_lidar(info, index)
                    if info_adj_list_lidar:
                        input_dict['lidar_adjacent'] = info_adj_list_lidar
        
        # Ego to global transformation - support both formats
        if 'ego2global' in info:
            # occfrmwrk format: 4x4 transformation matrix
            ego2global_matrix = info['ego2global']
            input_dict['ego2global'] = np.array(ego2global_matrix).reshape(4, 4)
        elif 'ego2global_rotation' in info and 'ego2global_translation' in info:
            # fusionocc format: rotation quaternion and translation separate
            from pyquaternion import Quaternion
            ego2global_rotation = Quaternion(info['ego2global_rotation'])
            ego2global_translation = np.array(info['ego2global_translation'])
            
            ego2global = np.eye(4)
            ego2global[:3, :3] = ego2global_rotation.rotation_matrix
            ego2global[:3, 3] = ego2global_translation
            input_dict['ego2global'] = ego2global
            
        # Instances (annotations)
        if 'instances' in info:
            input_dict['ann_info'] = dict()
            instances = info['instances']
            if len(instances) > 0:
                # Convert instances to the expected format
                gt_bboxes_3d = []
                gt_labels_3d = []
                
                for instance in instances:
                    if 'bbox_3d' in instance and instance.get('bbox_3d_isvalid', True):
                        gt_bboxes_3d.append(instance['bbox_3d'])
                        gt_labels_3d.append(instance.get('bbox_label_3d', instance.get('bbox_label', 0)))
                
                if gt_bboxes_3d:
                    input_dict['ann_info']['gt_bboxes_3d'] = np.array(gt_bboxes_3d)
                    input_dict['ann_info']['gt_labels_3d'] = np.array(gt_labels_3d)
                else:
                    input_dict['ann_info']['gt_bboxes_3d'] = np.zeros((0, 7))
                    input_dict['ann_info']['gt_labels_3d'] = np.array([])
            else:
                input_dict['ann_info']['gt_bboxes_3d'] = np.zeros((0, 7))
                input_dict['ann_info']['gt_labels_3d'] = np.array([])
        else:
            input_dict['ann_info'] = dict()
            input_dict['ann_info']['gt_bboxes_3d'] = np.zeros((0, 7))
            input_dict['ann_info']['gt_labels_3d'] = np.array([])
        
        # Occupancy ground truth path - support both formats
        input_dict['occ_gt_path'] = info.get('occ_gt_path', info.get('occ_path', ''))
        
        # Camera instances
        if 'cam_instances' in info:
            input_dict['cam_instances'] = info['cam_instances']
            
        # Scene token
        if 'scene_token' in info:
            input_dict['scene_token'] = info['scene_token']
        
        return input_dict
        
    def parse_data_info(self, info: dict) -> dict:
        """Parse raw annotation to target format for MMEngine statistics.
        
        This method is called by MMEngine to calculate dataset statistics.
        We need to convert our annotation format to MMEngine's expected format.
        """
        # This receives the raw data from the pkl file, not processed data
        parsed_info = dict()
        parsed_info['sample_idx'] = info.get('sample_idx', 0)
        
        # For statistics calculation, MMEngine expects 'instances' key with bbox info
        if 'instances' in info and len(info['instances']) > 0:
            # The instances are already in the raw data, we just need to make sure
            # they have the correct format for MMEngine statistics
            instances = []
            for instance in info['instances']:
                if instance.get('bbox_3d_isvalid', True):  # Only count valid instances
                    new_instance = dict()
                    new_instance['bbox_label'] = instance.get('bbox_label_3d', instance.get('bbox_label', 0))
                    new_instance['bbox_label_3d'] = instance.get('bbox_label_3d', instance.get('bbox_label', 0))
                    new_instance['bbox_3d'] = instance.get('bbox_3d', [])
                    instances.append(new_instance)
            
            parsed_info['instances'] = instances
        else:
            parsed_info['instances'] = []
            
        return parsed_info
    
    def get_cat_ids(self, idx: int):
        """Get category ids of sample by index.
        
        This method is required for MMEngine to calculate dataset statistics
        including the number of instances per category.
        
        Args:
            idx (int): Index of sample data.
            
        Returns:
            list[int]: List of category ids present in the sample.
        """
        # Ensure data_list is properly loaded and check bounds
        if not hasattr(self, 'data_list') or not self.data_list:
            self.data_list = self.load_data_list()
        
        if idx >= len(self.data_list):
            return []  # Return empty list for out of bounds indices
        
        info = self.data_list[idx]
        cat_ids = []
        
        if 'instances' in info and len(info['instances']) > 0:
            for instance in info['instances']:
                if instance.get('bbox_3d_isvalid', True):  # Only count valid instances
                    cat_id = instance.get('bbox_label_3d', instance.get('bbox_label', 0))
                    if cat_id not in cat_ids:
                        cat_ids.append(cat_id)
        
        return cat_ids
    
    def __len__(self):
        """Return the length of the dataset."""
        # Ensure data_list is loaded
        if not hasattr(self, 'data_list') or not self.data_list:
            self.data_list = self.load_data_list()
        
        return len(self.data_list) if self.data_list else 0
    
    def _calculate_num_ins_per_cat(self):
        """Calculate number of instances per category correctly.
        
        This method manually calculates the statistics that MMEngine BaseDataset
        should calculate but doesn't work properly with our data format.
        """
        # Initialize counters for each class
        if hasattr(self, 'CLASSES'):
            class_names = self.CLASSES
        else:
            class_names = self.metainfo.get('classes', [])
        
        num_ins_per_cat = {name: 0 for name in class_names}
        
        # Ensure data_list is loaded
        if not hasattr(self, 'data_list') or not self.data_list:
            self.data_list = self.load_data_list()
        
        # Count instances for each sample
        data_length = len(self.data_list) if self.data_list else 0
        for idx in range(data_length):
            cat_ids = self.get_cat_ids(idx)
            for cat_id in cat_ids:
                if 0 <= cat_id < len(class_names):
                    class_name = class_names[cat_id]
                    num_ins_per_cat[class_name] += 1
        
        # Override the BaseDataset's num_ins_per_cat
        self.num_ins_per_cat = num_ins_per_cat
    
    def _show_correct_statistics(self):
        """Show dataset statistics using parent class format but with correct values."""
        from mmengine.logging import print_log
        from terminaltables import AsciiTable
        
        # Use the same format as Det3DDataset.__init__ but with correct values
        print_log('-' * 30, 'current')
        print_log(f'The length of the dataset: {len(self)}', 'current')
        content_show = [['category', 'number']]
        for cat_name, num in self.num_ins_per_cat.items():
            content_show.append([cat_name, num])
        
        table = AsciiTable(content_show)
        print_log(
            f'The number of instances per category in the dataset:\n{table.table}',
            'current')
    
    def evaluate(self, occ_results, runner=None, show_dir=None, **eval_kwargs):
        """Evaluate occupancy predictions.
        
        Args:
            occ_results (list): List of occupancy predictions.
            runner: Runner instance (unused).
            show_dir (str): Directory to save visualization results.
            
        Returns:
            dict: Evaluation results.
        """
        try:
            from .occ_metrics import Metric_mIoU
        except ImportError:
            # Create a simple fallback evaluation
            print("Warning: Metric_mIoU not available, using simple evaluation")
            return {'mIoU': 0.0}
            
        self.occ_eval_metrics = Metric_mIoU(
            num_classes=18,
            use_lidar_mask=False,
            use_image_mask=self.use_mask
        )

        print('\nStarting Evaluation...')
        for index, occ_pred in enumerate(tqdm(occ_results)):
            info = self.data_infos[index]

            if 'occ_path' in info:
                occ_gt = np.load(os.path.join(info['occ_path'], 'labels.npz'))
                gt_semantics = occ_gt['semantics']
                mask_lidar = occ_gt['mask_lidar'].astype(bool)
                mask_camera = occ_gt['mask_camera'].astype(bool)
                self.occ_eval_metrics.add_batch(occ_pred, gt_semantics, mask_lidar, mask_camera)

        return self.occ_eval_metrics.count_miou()

    def prepare_data(self, index):
        """Data preparation for both training and testing stage."""
        # breakpoint()
        if self.test_mode:
            return self.prepare_test_data(index)
        else:
            return self.prepare_train_data(index)

    def prepare_train_data(self, index):
        """Training data preparation."""
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None

        # NuScenesDatasetOccpancy는 Det3DDataset을 직접 상속받지 않으므로 pre_pipeline 대신 직접 처리
        # deepcopy here to avoid inplace modification in pipeline.
        input_dict = copy.deepcopy(input_dict)

        # box_type_3d (str): 3D box type.
        input_dict['box_type_3d'] = self.box_type_3d
        # box_mode_3d (str): 3D box mode.
        input_dict['box_mode_3d'] = self.box_mode_3d

        # Add FusionOcc specific data preparation
        if self.modality is None:
            input_dict['modality'] = dict(use_lidar=True, use_camera=True)

        # Add occupancy-specific keys
        input_dict['voxel_semantics'] = None  # Will be loaded by pipeline
        input_dict['mask_camera'] = None  # Will be loaded by pipeline

        try:
            data = self.pipeline(input_dict)
            return data
        except Exception as e:
            # 디버깅을 위한 에러 로깅
            import traceback
            print(f"ERROR in pipeline for index {index}: {e}")
            print(f"input_dict keys: {input_dict.keys()}")
            if 'curr' in input_dict:
                print(f"curr keys: {input_dict['curr'].keys() if isinstance(input_dict['curr'], dict) else 'not dict'}")
            elif 'images' in input_dict:
                print(f"images keys: {list(input_dict['images'].keys()) if isinstance(input_dict['images'], dict) else 'not dict'}")
            traceback.print_exc()
            return None  # None을 반환하면 BaseDataset이 재시도

    def prepare_test_data(self, index):
        """Testing data preparation."""
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None

        # NuScenesDatasetOccpancy는 Det3DDataset을 직접 상속받지 않으므로 pre_pipeline 대신 직접 처리
        # deepcopy here to avoid inplace modification in pipeline.
        input_dict = copy.deepcopy(input_dict)

        # box_type_3d (str): 3D box type.
        input_dict['box_type_3d'] = self.box_type_3d
        # box_mode_3d (str): 3D box mode.
        input_dict['box_mode_3d'] = self.box_mode_3d

        # Add FusionOcc specific data preparation
        if self.modality is None:
            input_dict['modality'] = dict(use_lidar=True, use_camera=True)

        try:
            data = self.pipeline(input_dict)
            return data
        except Exception as e:
            import traceback
            print(f"ERROR in test pipeline for index {index}: {e}")
            traceback.print_exc()
            raise  # 테스트 모드에서는 예외를 다시 발생시킴