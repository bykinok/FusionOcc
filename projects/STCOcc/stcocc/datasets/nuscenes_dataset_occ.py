# Copyright (c) OpenMMLab. All rights reserved.
import os
import mmcv
import torch
import cv2
import numpy as np
from tqdm import tqdm

from nuscenes.nuscenes import NuScenes

from mmdet3d.registry import DATASETS
from mmdet3d.datasets.nuscenes_dataset import NuScenesDataset

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
        
        # Initialize show_ins_var attribute (required by Det3DDataset)
        self.show_ins_var = kwargs.get('show_ins_var', False)
        
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

    def load_data_list(self):
        """Load annotations from file.
        
        Adapts the old format STCOcc annotations to the new mmengine format.
        """
        import mmengine
        
        # Load the original annotation file
        raw_data = mmengine.load(self.ann_file)
        
        # Check if it's already in the new format
        if isinstance(raw_data, dict) and 'data_list' in raw_data and 'metainfo' in raw_data:
            data_list = raw_data['data_list']
            # CRITICAL: Set data_infos for backward compatibility
            self.data_infos = data_list
            return data_list
        
        # Convert old format to new format
        # Assume raw_data is a list of data_infos (old format)
        if isinstance(raw_data, list):
            # Store the data_infos for backward compatibility
            self.data_infos = raw_data
            return raw_data
        elif isinstance(raw_data, dict) and 'infos' in raw_data:
            # Some formats have 'infos' key
            # Store the data_infos for backward compatibility
            self.data_infos = raw_data['infos']
            return raw_data['infos']
        else:
            # Fallback: treat the whole data as data_list
            self.data_infos = raw_data if isinstance(raw_data, list) else [raw_data]
            return raw_data

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
        
        # Super() already provides perfect data structure with cams info
        # Just add curr key pointing to the same data for STCOcc compatibility
        input_dict['curr'] = input_dict.copy()
        
        # Add specific STCOcc fields if they exist in the data
        if 'occ_path' in input_dict:
            input_dict['occ_gt_path'] = input_dict['occ_path']
            
        # Ensure pts_filename is properly set from data_infos
        if 'pts_filename' not in input_dict and hasattr(self, 'data_infos') and index < len(self.data_infos):
            data_info = self.data_infos[index] 
            # STCOcc data uses 'lidar_path' key
            if 'lidar_path' in data_info and data_info['lidar_path']:
                input_dict['pts_filename'] = data_info['lidar_path']
        
        # Handle multi-frame adjacency if needed (STCOcc specific)
        if self.multi_adj_frame_id_cfg and hasattr(self, 'data_infos') and index < len(self.data_infos):
            start_id, end_id, step = self.multi_adj_frame_id_cfg
            adjacent_list = []
            for adj_idx in range(start_id, end_id, step):
                adj_data_idx = index + adj_idx
                if 0 <= adj_data_idx < len(self.data_infos) and adj_data_idx != index:
                    adj_info = self.data_infos[adj_data_idx]
                    adjacent_list.append(adj_info)
            if adjacent_list:
                input_dict['adjacent'] = adjacent_list
        
        input_dict['seg_label_mapping'] = self.SegLabelMapping
        return input_dict
    
    def prepare_data(self, idx):
        """Prepare data for the given index."""
        # Call parent's prepare_data method 
        data = super().prepare_data(idx)
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