# Copyright (c) OpenMMLab. All rights reserved.
import os
import time
import mmcv
import torch
import cv2
import numpy as np
from tqdm import tqdm
from pyquaternion import Quaternion

from mmdet3d.registry import DATASETS
from mmdet3d.datasets.nuscenes_dataset import NuScenesDataset
from .occ_metrics import Metric_mIoU, Metric_FScore
from .ray import generate_rays

nusc_class_nums = torch.Tensor([
    2854504, 7291443, 141614, 4239939, 32248552, 
    1583610, 364372, 2346381, 582961, 4829021, 
    14073691, 191019309, 6249651, 55095657, 
    58484771, 193834360, 131378779
])
dynamic_class = [0, 1, 3, 4, 5, 7, 9, 10]


def load_depth(img_file_path, gt_path):
    file_name = os.path.split(img_file_path)[-1]
    cam_depth = np.fromfile(os.path.join(gt_path, f'{file_name}.bin'),
        dtype=np.float32,
        count=-1).reshape(-1, 3)
    
    coords = cam_depth[:, :2].astype(np.int16)
    depth_label = cam_depth[:,2]
    return coords, depth_label

def load_seg_label(img_file_path, gt_path, img_size=[900,1600], mode='lidarseg'):
    if mode=='lidarseg':  # proj lidarseg to img
        coor, seg_label = load_depth(img_file_path, gt_path)
        seg_map = np.zeros(img_size)
        seg_map[coor[:, 1],coor[:, 0]] = seg_label
    else:
        file_name = os.path.join(gt_path, f'{os.path.split(img_file_path)[-1]}.npy')
        seg_map = np.load(file_name)
    return seg_map

def get_sensor_transforms(cam_info, cam_name):
    w, x, y, z = cam_info['cams'][cam_name]['sensor2ego_rotation']
    # sweep sensor to sweep ego
    sensor2ego_rot = torch.Tensor(
        Quaternion(w, x, y, z).rotation_matrix)
    sensor2ego_tran = torch.Tensor(
        cam_info['cams'][cam_name]['sensor2ego_translation'])
    sensor2ego = sensor2ego_rot.new_zeros((4, 4))
    sensor2ego[3, 3] = 1
    sensor2ego[:3, :3] = sensor2ego_rot
    sensor2ego[:3, -1] = sensor2ego_tran
    # sweep ego to global
    w, x, y, z = cam_info['cams'][cam_name]['ego2global_rotation']
    ego2global_rot = torch.Tensor(
        Quaternion(w, x, y, z).rotation_matrix)
    ego2global_tran = torch.Tensor(
        cam_info['cams'][cam_name]['ego2global_translation'])
    ego2global = ego2global_rot.new_zeros((4, 4))
    ego2global[3, 3] = 1
    ego2global[:3, :3] = ego2global_rot
    ego2global[:3, -1] = ego2global_tran

    return sensor2ego, ego2global


@DATASETS.register_module()
class NuScenesDatasetOccpancy(NuScenesDataset):
    def __init__(self, 
                use_rays=False,
                semantic_gt_path=None,
                depth_gt_path=None,
                aux_frames=[-1,1],
                max_ray_nums=0,
                wrs_use_batch=False,
                load_adj_occ_labels=False,
                hop_target_frame=-1,
                hop_load_all=False,
                classes=None,
                stereo=None,
                **kwargs):
        # Remove incompatible parameters from kwargs for BaseDataset compatibility
        for param in ['classes', 'stereo', 'img_info_prototype', 'load_adj_occ_labels', 
                      'use_valid_flag', 'filter_empty_gt', 'multi_adj_frame_id_cfg', 'modality']:
            if param in kwargs:
                kwargs.pop(param)
        self.CLASSES = classes
        self.stereo = stereo
        super().__init__(**kwargs)
        self.use_rays = use_rays
        self.semantic_gt_path = semantic_gt_path
        self.depth_gt_path = depth_gt_path
        self.aux_frames = aux_frames
        self.max_ray_nums = max_ray_nums
        self.wrs_use_batch = wrs_use_batch
        self.load_adj_occ_labels = load_adj_occ_labels
        self.hop_target_frame = hop_target_frame
        self.hop_load_all = hop_load_all

    def load_data_list(self):
        """Load data list from annotation file.
        
        Convert old format (infos/metadata) to new format (data_list/metainfo).
        """
        import pickle
        from mmengine.fileio import load
        
        # Load annotations
        data = load(self.ann_file)
        
        # Handle old format
        if isinstance(data, dict) and 'infos' in data:
            # Convert old format to new format
            data_list = data['infos']
            # Store metadata separately instead of setting metainfo directly
            self._metadata = data.get('metadata', {})
            
            # Transform each info dict to data_info format expected by mmengine
            processed_data_list = []
            for info in data_list:
                # Keep original structure and just copy all fields
                data_info = dict(info)  # Copy all original fields
                
                # Add specific fields that might be expected by mmengine
                data_info['sample_idx'] = info.get('token', '')
                
                # Map camera information for PrepareImageInputs compatibility
                if 'cams' in info:
                    data_info['images'] = {}
                    for cam_name, cam_data in info['cams'].items():
                        data_info['images'][cam_name] = {
                            'img_path': cam_data.get('data_path', ''),
                            'cam_intrinsic': cam_data.get('cam_intrinsic', []),
                            'timestamp': cam_data.get('timestamp', 0),
                            'sensor2ego_rotation': cam_data.get('sensor2ego_rotation', []),
                            'sensor2ego_translation': cam_data.get('sensor2ego_translation', [])
                        }
                
                # Ensure we have all fields needed by get_data_info method
                if 'with_gt' not in data_info:
                    data_info['with_gt'] = True  # Default value

                processed_data_list.append(data_info)
                
            return processed_data_list
        else:
            # If it's already in new format or different format
            return super().load_data_list()
        
        
    def get_rays(self, index):
        info = self.data_list[index]

        sensor2egos = []
        ego2globals = []
        intrins = []
        coors = []
        label_depths = []
        label_segs = []
        time_ids = {}
        idx = 0

        for time_id in [0] + self.aux_frames:
            time_ids[time_id] = []
            select_id = max(index + time_id, 0)
            
            if select_id>=len(self.data_list) or self.data_list[select_id]['scene_token'] != info['scene_token']:
                select_id = index  # out of sequence
            info = self.data_list[select_id]

            for cam_name in info['cams'].keys():
                intrin = torch.Tensor(info['cams'][cam_name]['cam_intrinsic'])
                sensor2ego, ego2global = get_sensor_transforms(info, cam_name)
                img_file_path = info['cams'][cam_name]['data_path']

                # load seg/depth GT of rays
                seg_map = load_seg_label(img_file_path, self.semantic_gt_path)
                coor, label_depth = load_depth(img_file_path, self.depth_gt_path)
                label_seg = seg_map[coor[:,1], coor[:,0]]

                sensor2egos.append(sensor2ego)
                ego2globals.append(ego2global)
                intrins.append(intrin)
                coors.append(torch.Tensor(coor))
                label_depths.append(torch.Tensor(label_depth))
                label_segs.append(torch.Tensor(label_seg))
                time_ids[time_id].append(idx)
                idx += 1
        
        T, N = len(self.aux_frames)+1, len(info['cams'].keys())
        sensor2egos = torch.stack(sensor2egos)
        ego2globals = torch.stack(ego2globals)
        sensor2egos = sensor2egos.view(T, N, 4, 4)
        ego2globals = ego2globals.view(T, N, 4, 4)

        # calculate the transformation from adjacent_sensor to key_ego
        keyego2global = ego2globals[0, :,  ...].unsqueeze(0)
        global2keyego = torch.inverse(keyego2global.double())
        sensor2keyegos = global2keyego @ ego2globals.double() @ sensor2egos.double()
        sensor2keyegos = sensor2keyegos.float()
        sensor2keyegos = sensor2keyegos.view(T*N, 4, 4)

        # generate rays for all frames
        rays = generate_rays(
            coors, label_depths, label_segs, sensor2keyegos, intrins,
            max_ray_nums=self.max_ray_nums, 
            time_ids=time_ids, 
            dynamic_class=self.dynamic_class, 
            balance_weight=self.WRS_balance_weight)
        return rays

    def __len__(self):
        length = super().__len__()
        return length

    def get_data_info(self, index):
        # Ensure data_list is loaded
        if not hasattr(self, 'data_list') or not self.data_list:
            # Force initialization of data_list
            try:
                self.data_list = self.load_data_list()
            except Exception as e:
                print(f"ERROR: Failed to load data_list: {e}")
                return super().get_data_info(index)
        
        found_data = self.data_list
        
        if index >= len(found_data):
            print(f"ERROR: Index {index} out of range for data length {len(found_data)}")
            input_dict = {
                'with_gt': True,
                'ann_info': {
                    'gt_labels_3d': [],
                    'gt_bboxes_3d': []
                },
                'sample_idx': f'dummy_{index}',
                'timestamp': 0
            }
            return input_dict
        
        # Get the data item
        data_item = found_data[index]
        
        # Convert to the expected format
        input_dict = dict(data_item)  # Copy all original fields
        
        # Map LiDAR information for LoadPointsFromFile compatibility
        if 'lidar_path' in data_item:
            input_dict['lidar_points'] = {
                'lidar_path': data_item['lidar_path'],
                'num_pts_feats': 4  # Default LiDAR point feature dimension
            }
        
        # Set with_gt flag
        input_dict['with_gt'] = data_item.get('with_gt', True)
        
        # Initialize ann_info if not present
        if 'ann_info' not in input_dict:
            input_dict['ann_info'] = {}
            
        # Set default empty annotations to avoid KeyError
        if 'gt_labels_3d' not in input_dict['ann_info']:
            input_dict['ann_info']['gt_labels_3d'] = []
        if 'gt_bboxes_3d' not in input_dict['ann_info']:
            input_dict['ann_info']['gt_bboxes_3d'] = []
        if 'occ_path' in data_item:
            input_dict['occ_gt_path'] = data_item['occ_path']
            
        if self.hop_load_all:    
            adj_occ_path_list=[]
            for i in self.aux_frames:
                # print(i)
                # new_index = index + i
                adj_index = max(0, min(index+i, len(found_data)-1))
                cur_data_info = found_data[adj_index]
                cur_occ_path = cur_data_info.get('occ_path', '')
                adj_occ_path_list.append(cur_occ_path)
            input_dict['hop_load_all']=True
            input_dict['hop_all_path']={"adj_path":adj_occ_path_list}
            
            input_dict['with_target_occ']=False
            return input_dict
            
        if self.load_adj_occ_labels:
            adj_occ_gt_path=self.load_adj_occ_gt_path(index=index,aux_frames=self.aux_frames)
            input_dict['target_occ_gt_path']=adj_occ_gt_path[self.hop_target_frame]
            input_dict['with_target_occ']=True
        # generate rays for rendering supervision
        if self.use_rays:
            rays_info = self.get_rays(index)
            input_dict['rays'] = rays_info
        else:
            input_dict['rays'] = torch.zeros((1))
        return input_dict

    def __getitem__(self, idx):
        """Override __getitem__ to add debug information for pipeline errors."""
        try:
            data_info = self.get_data_info(idx)
            
            # Process data through pipeline
            if hasattr(self, 'pipeline') and self.pipeline:
                for i, transform in enumerate(self.pipeline.transforms):
                    try:
                        data_info = transform(data_info)
                    except Exception as e:
                        print(f"ERROR: Transform {i} ({type(transform).__name__}) failed: {e}")
                        raise e
            
            return data_info
            
        except Exception as e:
            print(f"ERROR: __getitem__({idx}) failed with: {e}")
            raise e

    def evaluate(self, occ_results, runner=None, show_dir=None, **eval_kwargs):
        self.occ_eval_metrics = Metric_mIoU(
            num_classes=18,
            use_lidar_mask=False,
            use_image_mask=True)

        print('\nStarting Evaluation...')
        for index, occ_pred in enumerate(tqdm(occ_results)):
            info = self.data_list[index]
            occ_gt = np.load(os.path.join(info['occ_path'],'labels.npz'))
            gt_semantics = occ_gt['semantics']
            mask_lidar = occ_gt['mask_lidar'].astype(bool)
            mask_camera = occ_gt['mask_camera'].astype(bool)
            # occ_pred = occ_pred
            self.occ_eval_metrics.add_batch(occ_pred, gt_semantics, mask_lidar, mask_camera)


        return self.occ_eval_metrics.count_miou()

    def load_adj_occ_gt_path(self,index=-1,aux_frames=[-3,-2,-1]):
        adj_occ_gt_path=[]
        for i in aux_frames:
            select_id=index+i
            occ_gt_path = self.data_list[select_id]['occ_path']
            adj_occ_gt_path.append(occ_gt_path)
        
        return adj_occ_gt_path
