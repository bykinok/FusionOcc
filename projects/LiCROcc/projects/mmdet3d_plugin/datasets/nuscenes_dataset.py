# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/VoxFormer/blob/main/LICENSE

import os
from os import path as osp
import pickle
from PIL import Image
import glob
import random
import copy
import mmcv
import torch
from torch.utils.data import Dataset
import numpy as np
from numpy.linalg import inv
from torchvision import transforms
try:
    from mmdet3d.registry import DATASETS
except ImportError:
    from mmdet.datasets import DATASETS
# MMEngine 2.x doesn't use DataContainer
# Remove DataContainer usage entirely
# try:
#     from mmcv.parallel import DataContainer as DC
# except ImportError:
#     class DataContainer:
#         def __init__(self, data, *args, **kwargs):
#             self.data = data
#     DC = DataContainer
from ..ssc_rs.utils.ssc_metric import SSCMetrics
from numba import njit, prange
import skimage
import skimage.io
import yaml
import numba as nb
import re
import time
from torch.utils.data import DataLoader  
from nuscenes.nuscenes import NuScenes


normalize_rgb = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

def mmlabNormalize(img):
    from mmcv.image.photometric import imnormalize
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    to_rgb = True
    img = imnormalize(np.array(img), mean, std, to_rgb)
    img = torch.tensor(img).float().permute(2, 0, 1).contiguous()
    return img

def mask_op(data, x_min, x_max):
    mask = (data > x_min) & (data < x_max)
    return mask

def get_mask(pc, lims):
    mask_x = mask_op(pc[:, 0], lims[0][0] + 0.0001, lims[0][1] - 0.0001)
    mask_y = mask_op(pc[:, 1], lims[1][0] + 0.0001, lims[1][1] - 0.0001)
    mask_z = mask_op(pc[:, 2], lims[2][0] + 0.0001, lims[2][1] - 0.0001)
    mask = (mask_x) & (mask_y) & mask_z
    return mask

def augmentation_random_flip(data, flip_type, is_scan=False):
    if flip_type==1:
        if is_scan:
            data[:, 0] = -data[:, 0]
        else:
            data = np.flip(data, axis=0).copy()
    elif flip_type==2:
        if is_scan:
            data[:, 1] = -data[:, 1]
        else:
            data = np.flip(data, axis=1).copy()
    elif flip_type==3:
        if is_scan:
            data[:, 0] = -data[:, 0]
            data[:, 1] = -data[:, 1]
        else:
            data = np.flip(np.flip(data, axis=0), axis=1).copy()
    return data

@nb.jit('u1[:,:,:](u1[:,:,:],i4[:,:])', nopython=True, cache=True, parallel=False)
def nb_process_label(processed_label, sorted_label_voxel_pair):
    label_size = 256
    counter = np.zeros((label_size,), dtype=np.uint16)
    counter[sorted_label_voxel_pair[0, 3]] = 1
    cur_sear_ind = sorted_label_voxel_pair[0, :3]
    for i in range(1, sorted_label_voxel_pair.shape[0]):
        cur_ind = sorted_label_voxel_pair[i, :3]
        if not np.all(np.equal(cur_ind, cur_sear_ind)):
            processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
            counter = np.zeros((label_size,), dtype=np.uint16)
            cur_sear_ind = cur_ind
        counter[sorted_label_voxel_pair[i, 3]] += 1
    processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
    return processed_label


@DATASETS.register_module()
class nuScenesDataset(Dataset):
    def __init__(
        self,
        split,
        test_mode,
        data_root,
        occ_root,
        lims,
        sizes,
        data_config,
        sweeps_num = -1,
        temporal = [],
        augmentation=False,
        img_aug = True,
        shuffle_index=False,
        color_jitter = None,
        downsample =1,
        grid_config=None,
        img_distill=False,
        # occ3d specific parameters
        dataset_name='nuscenes_occ',
        use_semantic=True,
        pc_range=None,
        occ_size=None,
        use_ego_frame=False,
        classes=None,
        use_mask_camera=False,
        use_mask_camera_1_2=False,
        use_mask_camera_1_4=False,
        use_mask_camera_1_8=False,
    ):
        super().__init__()
        
        self.data_root = data_root
        # occ3d: GT and data live under the same root as data_root
        if dataset_name == 'occ3d':
            self.occ_root = data_root
            self.downsample_root = data_root
        else:
            self.occ_root = occ_root
            self.downsample_root = occ_root
        self.sweeps_num = sweeps_num
        self.color_jitter = color_jitter
        self.data_config = data_config
        self.downsample = downsample
        self.grid_config = grid_config
        self.img_aug = img_aug
        self.img_distill = img_distill
        
        # occ3d parameters
        self.dataset_name = dataset_name
        self.use_semantic = use_semantic
        self.use_ego_frame = use_ego_frame
        self.use_occ3d = (dataset_name == 'occ3d')
        
        # Validate occ3d configuration
        if self.use_occ3d:
            if pc_range is None:
                raise ValueError(
                    "pc_range must be specified for occ3d dataset. "
                    "Example: pc_range=[-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]"
                )
            if occ_size is None:
                raise ValueError(
                    "occ_size must be specified for occ3d dataset. "
                    "Example: occ_size=[200, 200, 16]"
                )
            if classes is None:
                raise ValueError(
                    "classes (class_names) must be specified for occ3d dataset. "
                    "Example: classes=['others', 'barrier', ..., 'vegetation', 'free'] (18 classes)"
                )
            if len(classes) != 18:
                raise ValueError(
                    f"occ3d requires 18 classes, but got {len(classes)} classes. "
                    f"Expected: ['others', 'barrier', 'bicycle', 'bus', 'car', "
                    f"'construction_vehicle', 'motorcycle', 'pedestrian', 'traffic_cone', "
                    f"'trailer', 'truck', 'driveable_surface', 'other_flat', 'sidewalk', "
                    f"'terrain', 'manmade', 'vegetation', 'free']"
                )
        
        self.pc_range = pc_range if pc_range is not None else [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
        self.occ_size = occ_size if occ_size is not None else [200, 200, 16]
        self.use_mask_camera = use_mask_camera
        self.use_mask_camera_1_2 = use_mask_camera_1_2
        self.use_mask_camera_1_4 = use_mask_camera_1_4
        self.use_mask_camera_1_8 = use_mask_camera_1_8

        if split == 'train':
            data_path = os.path.join(data_root, 'nuscenes_occ_infos_train.pkl')
        elif split == 'val':
            data_path = os.path.join(data_root, 'nuscenes_occ_infos_val.pkl')
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        self.nusc_infos = data['infos']

        self.lims = lims
        self.sizes = sizes
        self.augmentation = augmentation
        self.shuffle_index = shuffle_index

        self.split = split 
        
        # Set class names based on dataset
        if classes is not None:
            self.class_names = classes
        elif self.use_occ3d:
            # occ3d uses 18 classes (0=others, 1-16=semantic, 17=free)
            self.class_names = ['others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle', 
                               'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck', 
                               'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade', 
                               'vegetation', 'free']
        else:
            # Original 17 classes
            self.class_names =  ['noise',
                   'barrier', 'bicycle', 'bus', 'car', 'construction',
                   'motorcycle', 'pedestrian', 'trafficcone', 'trailer',
                   'truck', 'driveable_surface', 'other', 'sidewalk', 'terrain',
                   'mannade', 'vegetation']
        
        self.num_classes = len(self.class_names)
        
        self.test_mode = test_mode
        self.set_group_flag()
        self.nusc = NuScenes(version='v1.0-trainval', dataroot = data_root, verbose=False)

    def __getitem__(self, index):
        # data_start_ = time.time()
        flip_type = np.random.randint(0, 4) if self.augmentation else -1
        # flip_type = 3
        # Store index for later use
        self._current_index = index
        data = self.prepare_data(index, flip_type)
        # print('!!!!!!!!!!!!!!!!! ={}'.format(data))
        if data == None:
            self.__getitem__(index+1)
        points, target, meta_dict, img_inputs, radar_pc = data
        points = torch.from_numpy(points[:, :4]).float() 
        radar_pc = torch.from_numpy(radar_pc[:, :7]).float()

        # MMEngine 2.x: Don't use DataContainer, return raw data
        # points = DC(points, cpu_only=False, stack=False)
        # radar_pc = DC(radar_pc, cpu_only=False, stack=False)
        # meta_dict = DC(meta_dict, cpu_only=True)
        # Just use the raw data without wrapping

        data_info = dict(
            img_metas = meta_dict,
            points = points,
            radar_pc = radar_pc,
            target = target,
            img_inputs = img_inputs,
        )
        # data_end_ = time.time()
        # print('load data time = {}'.format(data_end_-data_start_))
        return data_info

    def __len__(self):
        return len(self.nusc_infos)

    def set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0. In 3D datasets, they are all the same, thus are all
        zeros.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)

    '''
    获取图像增强参数
    '''
    def sample_augmentation(self, H, W, flip=None, scale=None):
        """
        Args:
            H:
            W:
            flip:
            scale:
        Returns:
            resize: resize比例float.
            resize_dims: (resize_W, resize_H)
            crop: (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip: 0 / 1
            rotate: 随机旋转角度float
        """
        fH, fW = self.data_config['input_size']
        if self.split=="train" and self.img_aug:
            resize = float(fW) / float(W) # 0.44
            resize += np.random.uniform(*self.data_config['resize'])    # resize的比例, 位于[fW/W − 0.06, fW/W + 0.11]之间.
            resize_dims = (int(W * resize), int(H * resize))            # resize后的size
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_config['crop_h'])) *
                         newH) - fH     # s * H - H_in
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))       # max(0, s * W - fW)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = self.data_config['flip'] and np.random.choice([0, 1])
            rotate = np.random.uniform(*self.data_config['rot'])
        else:
            resize = float(fW) / float(W)
            if scale is not None:
                resize += scale
            else:
                resize += self.data_config.get('resize_test', 0.0)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_config['crop_h'])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False if flip is None else flip
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def img_transform_core(self, img, resize_dims, crop, flip, rotate):
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)
        return img

    def get_rot(self, h):
        return torch.Tensor([
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ])

    def img_transform(self, img, post_rot, post_tran, resize, resize_dims,
                      crop, flip, rotate):
        """
        Args:
            img: PIL.Image
            post_rot: torch.eye(2)
            post_tran: torch.eye(2)
            resize: float, resize的比例.
            resize_dims: Tuple(W, H), resize后的图像尺寸
            crop: (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip: bool
            rotate: float 旋转角度
        Returns:
            img: PIL.Image
            post_rot: Tensor (2, 2)
            post_tran: Tensor (2, )
        """
        # adjust image
        img = self.img_transform_core(img, resize_dims, crop, flip, rotate)

        # post-homography transformation
        # 将上述变换以矩阵表示.
        post_rot *= resize
        post_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            post_rot = A.matmul(post_rot)
            post_tran = A.matmul(post_tran) + b
        A = self.get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b

        return img, post_rot, post_tran


    def points2depthmap(self, points, height, width):
        """
        Args:
            points: (N_points, 3):  3: (u, v, d)
            height: int
            width: int

        Returns:
            depth_map：(H, W)
        """
        height, width = height // self.downsample, width // self.downsample
        depth_map = torch.zeros((height, width), dtype=torch.float32)
        coor = torch.round(points[:, :2] / self.downsample)     # (N_points, 2)  2: (u, v)
        depth = points[:, 2]    # (N_points, )哦
        kept1 = (coor[:, 0] >= 0) & (coor[:, 0] < width) & (
            coor[:, 1] >= 0) & (coor[:, 1] < height) & (
                depth < self.grid_config['depth'][1]) & (
                    depth >= self.grid_config['depth'][0])
        # 获取有效投影点.
        coor, depth = coor[kept1], depth[kept1]    # (N, 2), (N, )
        ranks = coor[:, 0] + coor[:, 1] * width
        sort = (ranks + depth / 100.).argsort()
        coor, depth, ranks = coor[sort], depth[sort], ranks[sort]
        kept2 = torch.ones(coor.shape[0], device=coor.device, dtype=torch.bool)
        kept2[1:] = (ranks[1:] != ranks[:-1])
        coor, depth = coor[kept2], depth[kept2]
        coor = coor.to(torch.long)
        depth_map[coor[:, 1], coor[:, 0]] = depth
        return depth_map


    def load_occ3d_gt(self, info, flip_type):
        """
        Load occ3d format GT (dense voxel grid with camera mask).
        occ3d GT: .npz files with 'semantics' and 'mask_camera' keys.
        File naming follows SurroundOcc: info['occ_path'] is the sample directory,
        main file is labels.npz, multi-scale are labels_1_2.npz, labels_1_4.npz, labels_1_8.npz.
        """
        # breakpoint()
        # SurroundOcc occ3d: occ_path = sample directory, GT file = labels.npz
        occ_path_dir = info['occ_path'].rstrip('/')
        occ3d_gt_path = os.path.join(occ_path_dir, 'labels.npz')

        if not os.path.exists(occ3d_gt_path):
            raise FileNotFoundError(
                f"occ3d GT file not found: {occ3d_gt_path}\n"
                f"Expected structure (SurroundOcc): {{occ_path}}/labels.npz\n"
                f"occ_path from info: {info['occ_path']}\n"
                f"Please check that occ_path is set correctly in data infos."
            )
        
        # Load occ3d GT (dense format)
        try:
            occ3d_data = np.load(occ3d_gt_path)
        except Exception as e:
            raise IOError(
                f"Failed to load occ3d GT file: {occ3d_gt_path}\n"
                f"Error: {str(e)}\n"
                f"Please check if the file is corrupted or in the correct format."
            )
        
        # Extract semantics
        if 'semantics' not in occ3d_data:
            raise KeyError(
                f"'semantics' key not found in occ3d GT file: {occ3d_gt_path}\n"
                f"Available keys: {list(occ3d_data.keys())}\n"
                f"Expected occ3d format with 'semantics' and 'mask_camera' keys."
            )
        occ3d_semantic = occ3d_data['semantics']  # (200, 200, 16) or similar
        
        # Extract camera mask
        if 'mask_camera' not in occ3d_data:
            raise KeyError(
                f"'mask_camera' key not found in occ3d GT file: {occ3d_gt_path}\n"
                f"Available keys: {list(occ3d_data.keys())}\n"
                f"Expected occ3d format with 'semantics' and 'mask_camera' keys."
            )
        occ3d_cam_mask = occ3d_data['mask_camera']  # (200, 200, 16), may be bool or uint8 0/1
        
        # Apply camera mask if enabled (SurroundOcc: np.where avoids ~uint8 indexing as 255)
        if self.use_mask_camera:
            target = np.where(occ3d_cam_mask, occ3d_semantic, 255).astype(np.uint8)
        else:
            target = occ3d_semantic.copy().astype(np.uint8)
        # Assert occ3d GT shape (200, 200, 16) to match loss dimensions
        expected_shape = (self.sizes[0], self.sizes[1], self.sizes[2])
        assert target.shape == expected_shape, (
            f"occ3d full-scale GT shape {target.shape} != expected {expected_shape}"
        )
        
        # Apply flip augmentation
        target = augmentation_random_flip(target, flip_type, is_scan=False)
        
        # breakpoint()
        # Multi-scale targets: load from pre-computed files only in training mode
        # In test mode, multi-scale GT may not be available, so return None
        if self.test_mode:
            # Test mode: only return full-scale GT, no multi-scale GT
            return target, None, None, None
        
        # Training mode: load multi-scale GT (SurroundOcc: labels_1_2.npz, labels_1_4.npz, labels_1_8.npz)
        occ3d_gt_path_1_2 = os.path.join(occ_path_dir, 'labels_1_2.npz')
        occ3d_gt_path_1_4 = os.path.join(occ_path_dir, 'labels_1_4.npz')
        occ3d_gt_path_1_8 = os.path.join(occ_path_dir, 'labels_1_8.npz')
        for path in (occ3d_gt_path_1_2, occ3d_gt_path_1_4, occ3d_gt_path_1_8):
            if not os.path.exists(path):
                raise FileNotFoundError(f"occ3d multi-scale GT file not found: {path}")
        occ3d_data_1_2 = np.load(occ3d_gt_path_1_2)
        if self.use_mask_camera_1_2:
            target_1_2 = np.where(occ3d_data_1_2['mask_camera'], occ3d_data_1_2['semantics'], 255).astype(np.uint8)
        else:
            target_1_2 = occ3d_data_1_2['semantics'].astype(np.uint8)
        shape_1_2 = (self.sizes[0] // 2, self.sizes[1] // 2, self.sizes[2] // 2)
        assert target_1_2.shape == shape_1_2, (
            f"occ3d target_1_2 shape {target_1_2.shape} != expected {shape_1_2}"
        )
        target_1_2 = augmentation_random_flip(target_1_2, flip_type, is_scan=False)
        occ3d_data_1_4 = np.load(occ3d_gt_path_1_4)
        if self.use_mask_camera_1_4:
            target_1_4 = np.where(occ3d_data_1_4['mask_camera'], occ3d_data_1_4['semantics'], 255).astype(np.uint8)
        else:
            target_1_4 = occ3d_data_1_4['semantics'].astype(np.uint8)
        shape_1_4 = (self.sizes[0] // 4, self.sizes[1] // 4, self.sizes[2] // 4)
        assert target_1_4.shape == shape_1_4, (
            f"occ3d target_1_4 shape {target_1_4.shape} != expected {shape_1_4}"
        )
        target_1_4 = augmentation_random_flip(target_1_4, flip_type, is_scan=False)
        occ3d_data_1_8 = np.load(occ3d_gt_path_1_8)
        if self.use_mask_camera_1_8:
            target_1_8 = np.where(occ3d_data_1_8['mask_camera'], occ3d_data_1_8['semantics'], 255).astype(np.uint8)
        else:
            target_1_8 = occ3d_data_1_8['semantics'].astype(np.uint8)
        shape_1_8 = (self.sizes[0] // 8, self.sizes[1] // 8, self.sizes[2] // 8)
        assert target_1_8.shape == shape_1_8, (
            f"occ3d target_1_8 shape {target_1_8.shape} != expected {shape_1_8}"
        )
        target_1_8 = augmentation_random_flip(target_1_8, flip_type, is_scan=False)
        return target, target_1_2, target_1_4, target_1_8

    def prepare_data(self, index, flip_type):
        """
        Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
        """
        info = self.nusc_infos[index]
        # 判断是否雨天和夜晚
        # scene_des = self.nusc.get('scene', info['scene_token'])['description']
        if_scene_useful = True #'night' not in scene_des.lower() and 'rain' not in scene_des.lower()
        # if not if_scene_useful:
        #     print(scene_des)
        
        points_path = info['lidar_path']        
        points = np.fromfile(points_path, dtype=np.float32, count=-1).reshape([-1, 5])
        radar_path = info['lidar_path'].replace('samples/LIDAR_TOP', 'radar_bev_filter')
        radar_pc = np.fromfile(radar_path, dtype=np.float32, count=-1).reshape([-1, 7])

        if not os.path.exists(radar_path):
            print('{} is not exists!'.format(radar_path))
            return None
        
    
        if self.img_distill:
            points_depth = radar_pc
        else:
            points_depth = points
        
        if self.sweeps_num > 0:
            sweep_points_list = [points]
            ts = info['timestamp']
            if len(info['sweeps']) <= self.sweeps_num:
                choices = np.arange(len(info['sweeps']))
            else:
                choices = np.random.choice(len(info['sweeps']), self.sweeps_num, replace=False)
            for idx in choices:
                sweep = info['sweeps'][idx]
                points_sweep = np.fromfile(sweep['data_path'], dtype=np.float32, count=-1).reshape([-1, 5])
                sweep_ts = sweep['timestamp'] / 1e6
                points_sweep[:, :3] = points_sweep[:, :3] @ sweep['sensor2lidar_rotation'].T
                points_sweep[:, :3] += sweep['sensor2lidar_translation']
                points_sweep[:, 4] = ts - sweep_ts
                sweep_points_list.append(points_sweep)
            points = np.concatenate(sweep_points_list, axis=0)

        # For occ3d (ego frame): transform lidar points to ego so occupancy is in ego frame
        # IMPORTANT: Transform BEFORE filtering (FusionOcc approach)
        if self.use_ego_frame:
            try:
                from pyquaternion import Quaternion
            except ImportError:
                raise ImportError(
                    "pyquaternion is required for ego frame. pip install pyquaternion"
                )
            sample_record = self.nusc.get('sample', info['token'])
            lidar_data = self.nusc.get('sample_data', sample_record['data']['LIDAR_TOP'])
            calib = self.nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
            lidar2ego_rot = Quaternion(calib['rotation']).rotation_matrix
            lidar2ego_trans = np.array(calib['translation'])
            # p_ego = R @ p_lidar + t
            points[:, :3] = (points[:, :3] @ lidar2ego_rot.T) + lidar2ego_trans
            radar_pc[:, :3] = (radar_pc[:, :3] @ lidar2ego_rot.T) + lidar2ego_trans

        # Filter points by range AFTER ego transformation (FusionOcc approach)
        if self.lims:
            filter_mask = get_mask(points, self.lims)
            points = points[filter_mask]
            # radar
            filter_radar_mask = get_mask(radar_pc, self.lims)
            radar_pc = radar_pc[filter_radar_mask]

        
        min_bound = np.array([self.lims[0][0], self.lims[1][0], self.lims[2][0]])
        max_bound = np.array([self.lims[0][1], self.lims[1][1], self.lims[2][1]])
        intervals = (max_bound - min_bound) / np.array(self.sizes)

        # lidar_start_time = time.time()
        xyz_grid_ind = (np.floor((points[:, :3] - min_bound) / intervals)).astype(np.int32)
        xyz_occ_label = np.ones((xyz_grid_ind.shape[0], 1), dtype=np.uint8)
        label_voxel_pair = np.concatenate([xyz_grid_ind, xyz_occ_label], axis=-1)
        label_voxel_pair = label_voxel_pair[np.lexsort((xyz_grid_ind[:, 0], xyz_grid_ind[:, 1], xyz_grid_ind[:, 2])), :].astype(np.int32)
        occupancy = np.zeros(self.sizes, dtype=np.uint8)
        occupancy = nb_process_label(occupancy, label_voxel_pair)
        # lidar_end_time = time.time()
        # print('lidar occ time = {}'.format(lidar_start_time - lidar_end_time))



        # radar_start_time = time.time()
        radar_xyz_grid_ind = (np.floor((radar_pc[:, :3] - min_bound) / intervals)).astype(np.int32)
        radar_xyz_occ_label = np.ones((radar_xyz_grid_ind.shape[0], 1), dtype=np.uint8)
        radar_label_voxel_pair = np.concatenate([radar_xyz_grid_ind, radar_xyz_occ_label], axis=-1)
        radar_label_voxel_pair = radar_label_voxel_pair[np.lexsort((radar_xyz_grid_ind[:, 0], radar_xyz_grid_ind[:, 1], radar_xyz_grid_ind[:, 2])), :].astype(np.int32)
        radar_occupancy = np.zeros(self.sizes, dtype=np.uint8)
        radar_occupancy = nb_process_label(radar_occupancy, radar_label_voxel_pair)
        # radar_end_time = time.time()
        # print('radar occ time = {}'.format(radar_start_time - radar_end_time))



        # Load GT based on dataset type
        if self.use_occ3d:
            # Load occ3d format GT (dense voxel grid)
            target, target_1_2, target_1_4, target_1_8 = self.load_occ3d_gt(info, flip_type)
        else:
            # nuScenes-Occupancy (original format)
            rel_path = 'scene_{0}/occupancy/{1}.npy'.format(info['scene_token'], info['lidar_token'])
            #  [z y x cls]
            pcd = np.load(os.path.join(self.occ_root, rel_path))
            occ_label = pcd[..., -1:]
            occ_label[occ_label==0] = 255
            occ_xyz_grid = pcd[..., [2,1,0]]


            # 空间换时간
            target_path = rel_path.replace('.npy', '.target')
            target_path = os.path.join(self.downsample_root, target_path)
            save_gt = False
            if os.path.exists(target_path) and save_gt:
                target = np.fromfile(target_path, dtype=np.uint8)  
                target = target.reshape((self.sizes[0], self.sizes[1], self.sizes[2])) 
            else:

                label_voxel_pair = np.concatenate([occ_xyz_grid, occ_label], axis=-1)
                label_voxel_pair = label_voxel_pair[np.lexsort((occ_xyz_grid[:, 0], occ_xyz_grid[:, 1], occ_xyz_grid[:, 2])), :].astype(np.int32)
                target = np.zeros([self.sizes[0], self.sizes[1], self.sizes[2]], dtype=np.uint8)
                target = nb_process_label(target, label_voxel_pair)
                # target.tofile(target_path)  

            target_1_2_path = rel_path.replace('.npy', '.target_1_2')
            target_1_2_path = os.path.join(self.downsample_root, target_1_2_path)
            if os.path.exists(target_1_2_path) and save_gt:
                target_1_2 = np.fromfile(target_1_2_path, dtype=np.uint8)  
                target_1_2 = target_1_2.reshape((self.sizes[0]//2, self.sizes[1]//2, self.sizes[2]//2)) 
            else:
                occ_xyz_grid_1_2 = occ_xyz_grid//2
                label_voxel_pair = np.concatenate([occ_xyz_grid_1_2, occ_label], axis=-1)
                label_voxel_pair = label_voxel_pair[np.lexsort((occ_xyz_grid_1_2[:, 0], occ_xyz_grid_1_2[:, 1], occ_xyz_grid_1_2[:, 2])), :].astype(np.int32)
                target_1_2 = np.zeros([self.sizes[0]//2, self.sizes[1]//2, self.sizes[2]//2], dtype=np.uint8)
                target_1_2 = nb_process_label(target_1_2, label_voxel_pair)
                # target_1_2.tofile(target_1_2_path)  

            target_1_4_path = rel_path.replace('.npy', '.target_1_4')
            target_1_4_path = os.path.join(self.downsample_root, target_1_4_path)
            if os.path.exists(target_1_4_path) and save_gt:
                target_1_4 = np.fromfile(target_1_4_path, dtype=np.uint8)  
                target_1_4 = target_1_4.reshape((self.sizes[0]//4, self.sizes[1]//4, self.sizes[2]//4)) 
            else:
                occ_xyz_grid_1_4 = occ_xyz_grid//4
                label_voxel_pair = np.concatenate([occ_xyz_grid_1_4, occ_label], axis=-1)
                label_voxel_pair = label_voxel_pair[np.lexsort((occ_xyz_grid_1_4[:, 0], occ_xyz_grid_1_4[:, 1], occ_xyz_grid_1_4[:, 2])), :].astype(np.int32)
                target_1_4 = np.zeros([self.sizes[0]//4, self.sizes[1]//4, self.sizes[2]//4], dtype=np.uint8)
                target_1_4 = nb_process_label(target_1_4, label_voxel_pair)
                # target_1_4.tofile(target_1_4_path)  

            target_1_8_path = rel_path.replace('.npy', '.target_1_8')
            target_1_8_path = os.path.join(self.downsample_root, target_1_8_path)
            if os.path.exists(target_1_8_path) and save_gt:
                target_1_8 = np.fromfile(target_1_8_path, dtype=np.uint8)  
                target_1_8 = target_1_8.reshape((self.sizes[0]//8, self.sizes[1]//8, self.sizes[2]//8)) 
            else:
                occ_xyz_grid_1_8 = occ_xyz_grid//8
                label_voxel_pair = np.concatenate([occ_xyz_grid_1_8, occ_label], axis=-1)
                label_voxel_pair = label_voxel_pair[np.lexsort((occ_xyz_grid_1_8[:, 0], occ_xyz_grid_1_8[:, 1], occ_xyz_grid_1_8[:, 2])), :].astype(np.int32)
                target_1_8 = np.zeros([self.sizes[0]//8, self.sizes[1]//8, self.sizes[2]//8], dtype=np.uint8)
                target_1_8 = nb_process_label(target_1_8, label_voxel_pair)
                # target_1_8.tofile(target_1_8_path)  

        
        if self.shuffle_index:
            pt_idx = np.random.permutation(np.arange(0, points.shape[0]))
            points = points[pt_idx]
            # radar
            radar_pt_idx = np.random.permutation(np.arange(0, radar_pc.shape[0]))
            radar_pc = radar_pc[radar_pt_idx]

            
        if True: # 如果使用图片
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []
            lidar2cam_dic = []
            image_list = []
            depth_list = []
            post_rots_list = []
            post_trans_list = []
            depth_map_list = []

            
            # Compute ego2lidar transformation for ego frame (occ3d)
            if self.use_ego_frame:
                # Get lidar2ego transformation from NuScenes API
                sample_record = self.nusc.get('sample', info['token'])
                lidar_data = self.nusc.get('sample_data', sample_record['data']['LIDAR_TOP'])
                calibrated_sensor = self.nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
                
                # Import pyquaternion for rotation
                try:
                    from pyquaternion import Quaternion
                except ImportError:
                    raise ImportError(
                        "pyquaternion is required for ego frame transformation. "
                        "Install it with: pip install pyquaternion"
                    )
                
                # Build lidar2ego transformation matrix
                lidar2ego_rot = Quaternion(calibrated_sensor['rotation']).rotation_matrix
                lidar2ego_trans = np.array(calibrated_sensor['translation'])
                lidar2ego_4x4 = np.eye(4)
                lidar2ego_4x4[:3, :3] = lidar2ego_rot
                lidar2ego_4x4[:3, 3] = lidar2ego_trans
                
                # Compute ego2lidar (inverse)
                ego2lidar_4x4 = np.linalg.inv(lidar2ego_4x4)
                            
            # 添加图像增强
            for cam_type, cam_info in info['cams'].items():

                post_rot = torch.eye(2)
                post_tran = torch.zeros(2)

                image = Image.open(cam_info['data_path']).convert("RGB")

                img_augs = self.sample_augmentation(
                    H=image.height, W=image.width, flip=self.data_config['flip'])
                resize, resize_dims, crop, flip, rotate = img_augs
                image, post_rot2, post_tran2 = \
                self.img_transform(image, post_rot,
                                   post_tran,
                                   resize=resize,
                                   resize_dims=resize_dims,
                                   crop=crop,
                                   flip=flip,
                                   rotate=rotate)

                post_tran = torch.zeros(3)
                post_rot = torch.eye(3)
                post_tran[:2] = post_tran2
                post_rot[:2, :2] = post_rot2

                
               
                image_paths.append(cam_info['data_path'])
                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                lidar2cam_t = cam_info[
                    'sensor2lidar_translation'] @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                intrinsic = cam_info['cam_intrinsic'][:3, :3]
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                
                # For occ3d with ego frame: Convert lidar2img to ego2img
                # This ensures model outputs are in ego frame, matching occ3d GT
                if self.use_ego_frame:
                    # ego2img = lidar2img @ ego2lidar
                    ego2img_rt = (lidar2img_rt.astype(np.float64) @ ego2lidar_4x4.astype(np.float64)).astype(np.float32)
                    lidar2img_rts.append(ego2img_rt)  # Store as lidar2img (encoder reads this key)
                    proj_rt = ego2img_rt  # points in ego; use ego2img for depth map
                else:
                    lidar2img_rts.append(lidar2img_rt)
                    proj_rt = lidar2img_rt
                # print('viewpad = {}'.format(viewpad))

                cam_intrinsics.append(viewpad[:3, :3])
                lidar2cam_rts.append(lidar2cam_rt.T)
                
                lidar2cam_dic.append(lidar2cam_rt.T)
                
                image = normalize_rgb(image) 
                image_list.append(image)


                post_rots_list.append(post_rot)         
                post_trans_list.append(post_tran)      
              
                # proj_rt: ego2img (occ3d) or lidar2img (original) so depth matches BEV frame
                points_img = torch.tensor(points_depth).float()[:, :3].matmul(torch.tensor(proj_rt).float()[:3, :3].T) 
                points_img = points_img + torch.tensor(proj_rt).float()[:3, 3].unsqueeze(0) 

                points_img = torch.cat([points_img[:, :2] / points_img[:, 2:3], points_img[:, 2:3]],1)  # 归一化z
              
                points_img = points_img.matmul(post_rot.float().T) + post_tran.float()     # (N_points, 3):  3: (u, v, d)
                depth_map = self.points2depthmap(points_img,
                                             image.shape[1],     # H
                                             image.shape[2]      # W
                                             )
                depth_map_list.append(depth_map)


    
            _images = torch.stack(image_list, dim=0)  
            _cam_intrinsic = np.array(cam_intrinsics)
            _lidar2cam = np.array(lidar2cam_dic)
            _lidar2img = np.array(lidar2cam_rts)
            _depth = np.array(depth_list)
          
            _post_trans = torch.stack(post_trans_list)
            _post_rots = torch.stack(post_rots_list)
            _depth_map = torch.stack(depth_map_list)


            # numpy2tensor
            _cam_intrinsic = torch.tensor(_cam_intrinsic).float()
            # View transformer expects sensor2ego (cam2keyframe). For ego frame (occ3d): use cam2ego so BEV is in ego
            if self.use_ego_frame:
                # cam2ego = lidar2ego @ cam2lidar = lidar2ego_4x4 @ inv(lidar2cam)
                _cam2ego = np.array([
                    lidar2ego_4x4.astype(np.float64) @ np.linalg.inv(_lidar2cam[i].astype(np.float64))
                    for i in range(len(_lidar2cam))
                ]).astype(np.float32)
                _sensor2egos = torch.tensor(_cam2ego).float()
            else:
                _sensor2egos = torch.tensor(np.linalg.inv(_lidar2cam)).float()
            img_inputs = dict(
                imgs = _images, 
                intrins = _cam_intrinsic,
                sensor2egos = _sensor2egos,
                post_rots = (_post_rots).float(),
                post_trans = (_post_trans).float(),
                flip_type=flip_type,
                gt_depth = _depth_map,
            )
        
        if self.augmentation:
            points = augmentation_random_flip(points, flip_type, is_scan=True)
            occupancy = augmentation_random_flip(occupancy, flip_type)
            target = augmentation_random_flip(target, flip_type)
            target_1_2 = augmentation_random_flip(target_1_2, flip_type)
            target_1_4 = augmentation_random_flip(target_1_4, flip_type)
            target_1_8 = augmentation_random_flip(target_1_8, flip_type)
            # radar
            radar_pc = augmentation_random_flip(radar_pc, flip_type, is_scan=True)
            radar_occupancy = augmentation_random_flip(radar_occupancy, flip_type)

        meta_dict = dict(
            points_paths = [points_path],   
            scene_token = str(info['scene_token']), 
            token = str(info['token']), 
            occupancy=occupancy.astype(np.float32),
            radar_occ = radar_occupancy.astype((np.float32)),
            target_1_2=target_1_2,
            target_1_4=target_1_4,
            target_1_8=target_1_8,
            if_scene_useful = if_scene_useful,
            sample_idx = index,  # Add sample index for evaluation
            )

        return points, target, meta_dict, img_inputs, radar_pc

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 result_name='ssc',
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluation in SemanticKITTI protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            show (bool): Whether to visualize.
                Default: False.
            out_dir (str): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """

        eval_results = {}
        # breakpoint()
        if 'ssc_scores' in results:
            ssc_scores = results['ssc_scores']
            
            class_ssc_iou = ssc_scores['iou_ssc'].tolist()
            res_dic = {
                "SC_Precision": ssc_scores['precision'].item(),
                "SC_Recall": ssc_scores['recall'].item(),
                "SC_IoU": ssc_scores['iou'],
                "SSC_mIoU": ssc_scores['iou_ssc_mean'],
            }
            print(ssc_scores)
        else:
            assert 'ssc_results' in results
            ssc_results = results['ssc_results']
            completion_tp = sum([x[0] for x in ssc_results])
            completion_fp = sum([x[1] for x in ssc_results])
            completion_fn = sum([x[2] for x in ssc_results])
            
            tps = sum([x[3] for x in ssc_results])
            fps = sum([x[4] for x in ssc_results])
            fns = sum([x[5] for x in ssc_results])
            
            precision = completion_tp / (completion_tp + completion_fp)
            recall = completion_tp / (completion_tp + completion_fn)
            iou = completion_tp / \
                    (completion_tp + completion_fp + completion_fn)
            iou_ssc = tps / (tps + fps + fns + 1e-5)
            # breakpoint()
            class_ssc_iou = iou_ssc.tolist()
            res_dic = {
                "SC_Precision": precision,
                "SC_Recall": recall,
                "SC_IoU": iou,
                "SSC_mIoU": iou_ssc[1:].mean(),
            }
            print('res_dic = {}'.format(res_dic))
        
        for name, iou in zip(self.class_names, class_ssc_iou):
            res_dic["SSC_{}_IoU".format(name)] = iou
        
        for key, val in res_dic.items():
            eval_results['nuScenes_{}'.format(key)] = round(val * 100, 2)
        
        # add two main metrics to serve as the sort metric
        eval_results['nuScenes_combined_IoU'] = eval_results['nuScenes_SC_IoU'] + eval_results['nuScenes_SSC_mIoU']
        
        if logger is not None:
            logger.info('NuScenes SSC Evaluation')
            logger.info(eval_results)


        return eval_results


if __name__=="__main__":
    dataset = nuScenesDataset(  
        split='train', 
        data_root='./data/nuscenes/', 
        occ_root='./data/nuScenes-Occupancy', 
        lims=[[-51.2, 51.2], [-51.2, 51.2], [-5, 3.0]], 
        sizes=[512, 512, 40],  
        data_config=dict(
                cams=[
                    'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                    'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
                ],
                Ncams=6,
                input_size=(256, 704),
                src_size=(900, 1600),
                resize=(-0.06, 0.11),
                rot=(-5.4, 5.4),
                flip=True,
                crop_h=(0.0, 0.0),
                resize_test=0.0),  
        sweeps_num=10,  
        temporal=[],  
        augmentation=False,  
        shuffle_index=False, 
        color_jitter=None, 
        downsample=1, 
        grid_config=dict(
                x=[-51.2, 51.2, 0.4],
                y=[-51.2, 51.2, 0.4],
                z=[-1, 5.4, 6.4],
                depth=[1.0, 45.0, 0.5])  
    )  
  
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)  
    
    for data in dataloader:  
        print(data)  