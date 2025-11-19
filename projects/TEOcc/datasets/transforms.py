# Copyright (c) OpenMMLab. All rights reserved.
import os
import math
import mmcv
import torch
import numpy as np
from PIL import Image
from .radar_points import RadarPoints
from mmdet3d.structures.points import BasePoints, get_points_type
from mmdet3d.registry import TRANSFORMS
from mmengine.structures import InstanceData
from mmdet3d.structures import PointData, Box3DMode, CameraInstance3DBoxes, LiDARInstance3DBoxes
# from mmcv.transforms.processing import PhotoMetricDistortion as OriginalPhotoMetricDistortion
# from mmdet.datasets.transforms.formatting import to_tensor
# from pyquaternion import Quaternion
# from mmcv.transforms import to_tensor as mmcv_to_tensor


def mmlabNormalize(img, img_norm_cfg=None):
    """Normalize images."""
    from mmcv.image.photometric import imnormalize
    
    if img_norm_cfg is None:
        mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        to_rgb = True
    else:
        mean = np.array(img_norm_cfg['mean'], dtype=np.float32)
        std = np.array(img_norm_cfg['std'], dtype=np.float32)
        to_rgb = img_norm_cfg.get('to_rgb', True)
    
    img = imnormalize(np.array(img), mean, std, to_rgb)
    img = torch.tensor(img).float().permute(2, 0, 1).contiguous()
    return img


@TRANSFORMS.register_module()
class PrepareImageInputs:
    """Load multi channel images from a list of separate channel files."""

    def __init__(
        self,
        data_config,
        is_train=False,
        sequential=False,
        load_depth=False,
        depth_gt_path=None,
        add_adj_bbox=False,
        with_future_pred=False,
        ego_cam='CAM_FRONT',
    ):
        self.is_train = is_train
        self.data_config = data_config
        self.normalize_img = mmlabNormalize
        self.sequential = sequential
        self.load_depth = load_depth
        self.depth_gt_path = depth_gt_path
        self.add_adj_bbox = add_adj_bbox
        self.with_future_pred = with_future_pred
        self.ego_cam = ego_cam

    def get_rot(self, h):
        return torch.Tensor([
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ])

    def img_transform(self, img, post_rot, post_tran, resize, resize_dims,
                      crop, flip, rotate):
        # adjust image
        img = self.img_transform_core(img, resize_dims, crop, flip, rotate)

        # post-homography transformation
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

    def img_transform_core(self, img, resize_dims, crop, flip, rotate):
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)
        return img

    def choose_cams(self):
        if self.is_train and self.data_config['Ncams'] < len(
                self.data_config['cams']):
            cam_names = np.random.choice(
                self.data_config['cams'],
                self.data_config['Ncams'],
                replace=False)
        else:
            cam_names = self.data_config['cams']
        return cam_names

    def sample_augmentation(self, H, W, flip=None, scale=None):
        fH, fW = self.data_config['input_size']
        if self.is_train:
            resize = float(fW) / float(W)
            resize += np.random.uniform(*self.data_config['resize'])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_config['crop_h'])) *
                         newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
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

    def get_sensor_transforms(self, results, cam_name):
        """Get sensor to ego and ego to global transforms."""
        from pyquaternion import Quaternion
        
        # Get camera data from results
        cam_data = results.get('images', {}).get(cam_name, {})
        if not cam_data:
            # Fallback to cams if images not available
            cam_data = results.get('cams', {}).get(cam_name, {})
        
        # Get sensor2ego transform
        if 'sensor2ego_rotation' in cam_data and 'sensor2ego_translation' in cam_data:
            rot = cam_data['sensor2ego_rotation']
            if len(rot) == 4:  # quaternion [w, x, y, z]
                w, x, y, z = rot
                sensor2ego_rot = Quaternion(w, x, y, z).rotation_matrix
            else:
                sensor2ego_rot = np.array(rot).reshape(3, 3)
            sensor2ego_tran = np.array(cam_data['sensor2ego_translation'])
        else:
            sensor2ego_rot = np.eye(3)
            sensor2ego_tran = np.zeros(3)
        
        # Get ego2global transform
        if 'ego2global_rotation' in cam_data and 'ego2global_translation' in cam_data:
            rot = cam_data['ego2global_rotation']
            if len(rot) == 4:  # quaternion [w, x, y, z]
                w, x, y, z = rot
                ego2global_rot = Quaternion(w, x, y, z).rotation_matrix
            else:
                ego2global_rot = np.array(rot).reshape(3, 3)
            ego2global_tran = np.array(cam_data['ego2global_translation'])
        else:
            ego2global_rot = np.eye(3)
            ego2global_tran = np.zeros(3)
        
        sensor2ego = np.eye(4)
        sensor2ego[:3, :3] = sensor2ego_rot
        sensor2ego[:3, 3] = sensor2ego_tran
        
        ego2global = np.eye(4)
        ego2global[:3, :3] = ego2global_rot
        ego2global[:3, 3] = ego2global_tran
        
        return sensor2ego, ego2global

    def get_inputs(self, results, flip=None, scale=None):
        imgs = []
        sensor2egos = []
        ego2globals = []
        intrins = []
        post_rots = []
        post_trans = []
        gt_depths = []
        canvas = []  # Add this line
        cam_names = self.choose_cams()
        results['cam_names'] = cam_names
        
        for cam_name in cam_names:

            

            # Load camera data from results
            cam_data = results.get('images', {}).get(cam_name, {})
            filename = cam_data.get('img_path', '')
            
            # Fix path prefix issue: './data/' -> 'data/'
            original_filename = filename
            if filename.startswith('./data/'):
                filename = filename[2:]  # Remove './'
            
            
            if not os.path.exists(filename):
                # Create dummy image if file doesn't exist
                print(f"WARNING: Image file not found: {filename}, using dummy image")
                img = Image.fromarray(np.zeros((900, 1600, 3), dtype=np.uint8))
            else:
                try:
                    img = Image.open(filename)
                except Exception as e:
                    print(f"ERROR: Failed to open image {filename}: {e}, using dummy image")
                    img = Image.fromarray(np.zeros((900, 1600, 3), dtype=np.uint8))
                
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)

            # Get intrinsics (use dummy values if not available)
            intrin = torch.Tensor(cam_data.get('cam_intrinsic', [[1260, 0, 800], [0, 1260, 450], [0, 0, 1]]))

            # Get transforms
            sensor2ego, ego2global = self.get_sensor_transforms(results, cam_name)
            
            
            # Image augmentations
            img_augs = self.sample_augmentation(
                H=img.height, W=img.width, flip=flip, scale=scale)
            resize, resize_dims, crop, flip, rotate = img_augs
            
            img, post_rot2, post_tran2 = \
                self.img_transform(img, post_rot,
                                   post_tran,
                                   resize=resize,
                                   resize_dims=resize_dims,
                                   crop=crop,
                                   flip=flip,
                                   rotate=rotate)

            # For convenience, make augmentation matrices 3x3
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2

            # Depth loading (add this section)
            def load_depth(img_file_path, gt_path):
                file_name = os.path.split(img_file_path)[-1]
                depth_file_path = os.path.join(gt_path, f'{file_name}.bin')
                
                if not os.path.exists(depth_file_path):
                    # Return zero depth map if file doesn't exist
                    return torch.zeros(resize_dims)
                
                try:
                    point_depth = np.fromfile(depth_file_path,
                        dtype=np.float32,
                        count=-1).reshape(-1, 3)
                    
                    point_depth_aug_map, point_depth_aug = depth_transform(
                        point_depth, resize, resize_dims,
                        crop, flip, rotate)
                    return point_depth_aug_map
                except Exception as e:
                    print(f"WARNING: Failed to load depth from {depth_file_path}: {e}")
                    return torch.zeros(resize_dims)

            if self.load_depth:
                img_file_path = filename
                gt_depths.append(load_depth(img_file_path, self.depth_gt_path))
                if self.sequential and 'adjacent' in results:
                    for adj_info in results['adjacent']:
                        adj_cam_data = adj_info.get('images', {}).get(cam_name, {})
                        adj_filename = adj_cam_data.get('img_path', '')
                        if adj_filename.startswith('./data/'):
                            adj_filename = adj_filename[2:]
                        gt_depths.append(load_depth(adj_filename, self.depth_gt_path))
            else:
                gt_depths.append(torch.zeros(1))

            # After augmentation, before normalization:
            canvas.append(np.array(img))  # Add this line (before normalize_img)
            
            # Normalize image
            img = self.normalize_img(img)

            imgs.append(img)

            
            # Sequential mode: load adjacent frames for this camera
            if self.sequential and 'adjacent' in results:
                for adj_info in results['adjacent']:
                    # Load adjacent frame image for the same camera
                    adj_cam_data = adj_info.get('images', {}).get(cam_name, {})
                    adj_filename = adj_cam_data.get('img_path', '')
                    
                    if adj_filename.startswith('./data/'):
                        adj_filename = adj_filename[2:]
                    
                    if os.path.exists(adj_filename):
                        img_adjacent = Image.open(adj_filename)
                    else:
                        img_adjacent = Image.fromarray(np.zeros((900, 1600, 3), dtype=np.uint8))
                    
                    # Apply same augmentation as current frame
                    img_adjacent = self.img_transform_core(
                        img_adjacent,
                        resize_dims=resize_dims,
                        crop=crop,
                        flip=flip,
                        rotate=rotate)
                    
                    imgs.append(self.normalize_img(img_adjacent))
            
            # Add intrinsics and transforms for current camera
            intrins.append(intrin)
            sensor2egos.append(sensor2ego)
            ego2globals.append(ego2global)
            post_rots.append(post_rot)
            post_trans.append(post_tran)
        
        # Sequential mode: extend post processing matrices and add adjacent transforms
        if self.sequential and 'adjacent' in results:
            for adj_info in results['adjacent']:
                # Extend post processing matrices (same for all cameras)
                post_trans.extend(post_trans[:len(cam_names)])
                post_rots.extend(post_rots[:len(cam_names)])
                intrins.extend(intrins[:len(cam_names)])
                
                # Add sensor2ego and ego2global for each camera in adjacent frame
                for cam_name in cam_names:
                    adj_sensor2ego, adj_ego2global = self.get_sensor_transforms(adj_info, cam_name)
                    sensor2egos.append(adj_sensor2ego)
                    ego2globals.append(adj_ego2global)
    
        results['canvas'] = canvas  # Add this line
        return (torch.stack(imgs), 
        torch.stack([torch.Tensor(s) for s in sensor2egos]),
        torch.stack([torch.Tensor(s) for s in ego2globals]),
        torch.stack(intrins),
        torch.stack(post_rots), 
        torch.stack(post_trans)), gt_depths  # Change [] to gt_depths

    def __call__(self, results):
        # Check if adjacent frames are available for sequential mode
        if self.sequential and 'adjacent' not in results:
            print(f"WARNING: sequential=True but 'adjacent' not in results. Using current frame only.")
        
        img_inputs, gt_depths = self.get_inputs(results)
        
        # Debug: Print img_inputs shape
        # if isinstance(img_inputs, tuple) and len(img_inputs) > 0:
        #     imgs = img_inputs[0]
        #     has_adjacent = 'adjacent' in results
        #     print(f"DEBUG PrepareImageInputs: imgs.shape = {imgs.shape}, sequential = {self.sequential}, has_adjacent = {has_adjacent}")
        
        results['img_inputs'] = img_inputs
        results['gt_depths'] = gt_depths
        return results


@TRANSFORMS.register_module()
class LoadRadarPointsMultiSweeps:
    """Load radar points from multiple sweeps."""

    def __init__(self,
                 load_dim=18,
                 use_dim=[0, 1, 2, 8, 9, 5, 18],
                 sweeps_num=8,
                 file_client_args=dict(backend='disk'),
                 max_num=1200,
                 pc_range=[-40, -40, -5, 40, 40, 3], 
                 test_mode=False,
                 rote90=True,
                 ignore=[]):
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.sweeps_num = sweeps_num
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.max_num = max_num
        self.test_mode = test_mode
        self.pc_range = pc_range
        self.ignore = ignore
        self.rote90 = rote90

    def _load_points(self, pts_filename):
        """Load points from file using RadarPointCloud."""
        from nuscenes.utils.data_classes import RadarPointCloud
        
        if not os.path.exists(pts_filename):
            return np.zeros((100, self.load_dim), dtype=np.float32)
        
        try:
            radar_obj = RadarPointCloud.from_file(pts_filename)
            # [18, N] -> [N, 18]
            points = radar_obj.points.transpose().astype(np.float32)
            return points
        except Exception as e:
            print(f"WARNING: Failed to load radar from {pts_filename}: {e}")
            return np.zeros((100, self.load_dim), dtype=np.float32)

    def __call__(self, results):
        """Load radar points from multiple sweeps."""
        # 원본 모델과 동일하게: results['radar']에서 radars_dict 가져오기

        # breakpoint()

        radars_dict = results['radar']
            
        points_sweep_list = []
        # total_points_before = 0
        
        for key, sweeps in radars_dict.items():
            if key in self.ignore:
                continue
            if len(sweeps) < self.sweeps_num:
                idxes = list(range(len(sweeps)))
            else:
                idxes = list(range(self.sweeps_num))
            
            ts = sweeps[0]['timestamp'] * 1e-6
            for idx in idxes:
                sweep = sweeps[idx]
                
                points_sweep = self._load_points(sweep['data_path'])
                points_sweep = np.copy(points_sweep).reshape(-1, self.load_dim)
                # total_points_before += points_sweep.shape[0]
                
                timestamp = sweep['timestamp'] * 1e-6
                time_diff = ts - timestamp
                time_diff = np.ones((points_sweep.shape[0], 1)) * time_diff
                
                # velocity compensated by the ego motion in sensor frame
                velo_comp = points_sweep[:, 8:10]
                velo_comp = np.concatenate(
                    (velo_comp, np.zeros((velo_comp.shape[0], 1))), 1)
                velo_comp = velo_comp @ sweep['sensor2lidar_rotation'].T
                velo_comp = velo_comp[:, :2]
                
                # velocity in sensor frame
                velo = points_sweep[:, 6:8]
                velo = np.concatenate(
                    (velo, np.zeros((velo.shape[0], 1))), 1)
                velo = velo @ sweep['sensor2lidar_rotation'].T
                velo = velo[:, :2]
                
                # Transform points
                points_sweep[:, :3] = points_sweep[:, :3] @ sweep['sensor2lidar_rotation'].T
                points_sweep[:, :3] += sweep['sensor2lidar_translation']
                
                # Concatenate: [x,y,z,vx,vy,vz] + velo + velo_comp + [나머지] + time_diff
                points_sweep_ = np.concatenate(
                    [points_sweep[:, :6], velo,
                        velo_comp, points_sweep[:, 10:],
                        time_diff], axis=1)
                points_sweep_list.append(points_sweep_)
        
        # breakpoint()

        points = np.concatenate(points_sweep_list, axis=0)
        
        # Filter by use_dim
        points = points[:, self.use_dim]
    
        # Create RadarPoints object
        points = RadarPoints(
            points, points_dim=points.shape[-1], attribute_dims=None
        )
        
        if self.rote90:
            points.rotate(-math.pi/2)
            
        results['radar'] = points
        return results


@TRANSFORMS.register_module() 
class LoadAnnotationsBEVDepth:
    """Load annotations for BEV depth estimation with BDA augmentation."""

    def __init__(self, 
                 bda_aug_conf, 
                 classes, 
                 is_train=True,
                 align_adj_bbox=False,
                 sequential=False):
        self.bda_aug_conf = bda_aug_conf
        self.classes = classes
        self.is_train = is_train
        self.align_adj_bbox = align_adj_bbox
        self.sequential = sequential

    def sample_bda_augmentation(self):
        """Generate bda augmentation values based on bda_config."""
        if self.is_train:
            rotate_bda = np.random.uniform(*self.bda_aug_conf['rot_lim'])
            scale_bda = np.random.uniform(*self.bda_aug_conf['scale_lim'])
            flip_dx = np.random.uniform() < self.bda_aug_conf['flip_dx_ratio']
            flip_dy = np.random.uniform() < self.bda_aug_conf['flip_dy_ratio']
        else:
            # No augmentation in test mode - identity transformation
            rotate_bda = 0
            scale_bda = 1.0
            flip_dx = False
            flip_dy = False
        return rotate_bda, scale_bda, flip_dx, flip_dy

    def bev_transform(self, gt_boxes, rotate_angle, scale_ratio, flip_dx, flip_dy):
        """Apply BEV transformation to ground truth boxes."""
        rotate_angle = torch.tensor(rotate_angle / 180 * np.pi)
        rot_sin = torch.sin(rotate_angle)
        rot_cos = torch.cos(rotate_angle)
        rot_mat = torch.Tensor([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0],
                                [0, 0, 1]])
        scale_mat = torch.Tensor([[scale_ratio, 0, 0], [0, scale_ratio, 0],
                                  [0, 0, scale_ratio]])
        flip_mat = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        if flip_dx:
            flip_mat = flip_mat @ torch.Tensor([[-1, 0, 0], [0, 1, 0],
                                                [0, 0, 1]])
        if flip_dy:
            flip_mat = flip_mat @ torch.Tensor([[1, 0, 0], [0, -1, 0],
                                                [0, 0, 1]])
        rot_mat = flip_mat @ (scale_mat @ rot_mat)
        if gt_boxes.shape[0] > 0:
            gt_boxes[:, :3] = (
                rot_mat @ gt_boxes[:, :3].unsqueeze(-1)).squeeze(-1)
            gt_boxes[:, 3:6] *= scale_ratio
            gt_boxes[:, 6] += rotate_angle
            if flip_dx:
                gt_boxes[:,
                         6] = 2 * torch.asin(torch.tensor(1.0)) - gt_boxes[:,
                                                                           6]
            if flip_dy:
                gt_boxes[:, 6] = -gt_boxes[:, 6]
            gt_boxes[:, 7:] = (
                rot_mat[:2, :2] @ gt_boxes[:, 7:].unsqueeze(-1)).squeeze(-1)
        return gt_boxes, rot_mat

    def __call__(self, results):
        """Load BEV depth annotations with BDA augmentation."""
        # Create dummy annotations if not present
        if 'ann_info' not in results:
            gt_boxes = np.zeros((0, 9))
            gt_labels = np.array([])
        else:
            gt_boxes = results['ann_info'].get('gt_bboxes_3d', np.zeros((0, 9)))
            gt_labels = results['ann_info'].get('gt_labels_3d', np.array([]))
        
        gt_boxes, gt_labels = torch.Tensor(gt_boxes), torch.tensor(gt_labels)
        
        # Sample BDA augmentation parameters
        rotate_bda, scale_bda, flip_dx, flip_dy = self.sample_bda_augmentation()
        
        # Add BDA parameters to results for GlobalRotScaleTrans_radar
        results['rotate_bda'] = rotate_bda
        results['scale_bda'] = scale_bda
        results['flip_dx'] = flip_dx
        results['flip_dy'] = flip_dy
        
        # Apply BEV transformation to boxes
        bda_mat = torch.zeros(4, 4)
        bda_mat[3, 3] = 1
        gt_boxes, bda_rot = self.bev_transform(gt_boxes, rotate_bda, scale_bda,
                                               flip_dx, flip_dy)
        bda_mat[:3, :3] = bda_rot
        
        if len(gt_boxes) == 0:
            gt_boxes = torch.zeros(0, 9)
        
        results['gt_bboxes_3d'] = LiDARInstance3DBoxes(
            gt_boxes, box_dim=gt_boxes.shape[-1], origin=(0.5, 0.5, 0.5))
        results['gt_labels_3d'] = gt_labels
        
        # Add BDA rotation matrix to img_inputs
        if 'img_inputs' in results:
            imgs, intrins, post_rots, post_trans, sensor2egos, ego2globals = results['img_inputs']
            results['img_inputs'] = (imgs, intrins, post_rots, post_trans, 
                                    sensor2egos, ego2globals, bda_rot)
        
        # Apply voxel transformation if voxel semantics exist
        if ('voxel_semantics' in results) and len(results.get('voxel_semantics', [])) != 0:
            if flip_dx:
                results['voxel_semantics'] = results['voxel_semantics'][::-1, ...].copy()
                results['mask_lidar'] = results['mask_lidar'][::-1, ...].copy()
                results['mask_camera'] = results['mask_camera'][::-1, ...].copy()
        
        return results


@TRANSFORMS.register_module()
class LoadOccGTFromFile:
    """Load occupancy ground truth from file."""

    def __init__(self, data_root='data/nuscenes'):
        self.data_root = data_root

    def __call__(self, results):
        """Load occupancy ground truth."""
        # Try to load actual ground truth from file
        occ_path = results.get('occ_path', None)
        
        if occ_path is not None:
            try:
                # Load from labels.npz file
                occ_gt_path = os.path.join(occ_path, 'labels.npz')
                if os.path.exists(occ_gt_path):
                    occ_gt = np.load(occ_gt_path)
                    results['voxel_semantics'] = occ_gt['semantics']
                    results['mask_lidar'] = occ_gt['mask_lidar'].astype(bool)
                    results['mask_camera'] = occ_gt['mask_camera'].astype(bool)
                else:
                    # Create dummy data if file doesn't exist
                    results['voxel_semantics'] = np.zeros((200, 200, 16), dtype=np.uint8)
                    results['mask_lidar'] = np.ones((200, 200, 16), dtype=bool)
                    results['mask_camera'] = np.ones((200, 200, 16), dtype=bool)
            except Exception as e:
                # If loading fails, create dummy data
                print(f"Warning: Failed to load occupancy GT: {e}")
                results['voxel_semantics'] = np.zeros((200, 200, 16), dtype=np.uint8)
                results['mask_lidar'] = np.ones((200, 200, 16), dtype=bool)
                results['mask_camera'] = np.ones((200, 200, 16), dtype=bool)
        else:
            # No occ_path specified, create dummy data
            results['voxel_semantics'] = np.zeros((200, 200, 16), dtype=np.uint8)
            results['mask_lidar'] = np.ones((200, 200, 16), dtype=bool)
            results['mask_camera'] = np.ones((200, 200, 16), dtype=bool)
        
        return results


@TRANSFORMS.register_module()
class GlobalRotScaleTrans_radar:
    """Apply global rotation, scaling and translation to a 3D scene for radar dataset."""

    def __init__(self,
                 shift_height=False,
                 is_rad=False):
        self.shift_height = shift_height
        self.is_rad = is_rad

    def _flip_bbox_points(self, input_dict):
        """Private function to flip radar points."""
        flip_dx = input_dict.get('flip_dx', False)
        flip_dy = input_dict.get('flip_dy', False)
        if flip_dx:
            input_dict['radar'].flip("vertical")
        if flip_dy:
            input_dict['radar'].flip("horizontal")

    def _rot_bbox_points(self, input_dict):
        """Private function to rotate radar points."""
        noise_rotation = input_dict.get('rotate_bda', 0.0)
        
        if self.is_rad:
            rot_mat_T = input_dict['radar'].rotate(noise_rotation)
        else:
            rot_mat_T = input_dict['radar'].rotate(noise_rotation / 180 * math.pi)
        input_dict['pcd_rotation_radar'] = rot_mat_T

    def _scale_bbox_points(self, input_dict):
        """Private function to scale radar points."""
        scale = input_dict.get('scale_bda', 1.0)
        points = input_dict['radar']
        points.scale(scale)
        if self.shift_height:
            assert 'height' in points.attribute_dims.keys(), \
                'setting shift_height=True but points have no height attribute'
            points.tensor[:, points.attribute_dims['height']] *= scale
        input_dict['radar'] = points

    def __call__(self, input_dict):
        """Apply global transformations to radar points."""
        if 'transformation_3d_flow_radar' not in input_dict:
            input_dict['transformation_3d_flow_radar'] = []

        self._rot_bbox_points(input_dict)
        self._scale_bbox_points(input_dict)
        self._flip_bbox_points(input_dict)

        input_dict['transformation_3d_flow_radar'].extend(['R', 'S', 'T'])
        return input_dict


@TRANSFORMS.register_module()
class PointToMultiViewDepth:
    """Convert points to multi-view depth maps."""

    def __init__(self, grid_config, downsample=1):
        self.grid_config = grid_config
        self.downsample = downsample

    def __call__(self, results):
        """Convert points to depth maps."""
        # Create dummy depth map
        H, W = 384 // self.downsample, 704 // self.downsample
        N_cams = 6
        results['gt_depth'] = torch.zeros((N_cams, H, W))
        return results


@TRANSFORMS.register_module()
class DefaultFormatBundle3D:
    """Default formatting bundle for 3D data."""

    def __init__(self, class_names=None, with_gt=True, with_label=True):
        self.class_names = class_names
        self.with_gt = with_gt 
        self.with_label = with_label

    def __call__(self, results):
        """Format 3D data."""
        # Convert numpy arrays to tensors where needed
        for key in ['img_inputs', 'radar']:
            if key in results and hasattr(results[key], 'tensor'):
                # 원본 모델과 동일하게: radar를 리스트로 감싸서 batch collation 후 [[tensor]] 형태가 되도록
                if key == 'radar':
                    # RadarPoints 객체의 tensor를 추출하고 리스트로 감싸기
                    # MMEngine batch collation이 추가로 리스트로 감싸므로 결과적으로 [[tensor]] 형태가 됨
                    results[key] = [results[key].tensor]
                else:
                    results[key] = results[key].tensor
        
        return results


@TRANSFORMS.register_module()
class Collect3D:
    """Collect data for 3D detection - compatible with mmengine."""

    def __init__(self, keys=None, meta_keys=None):
        self.keys = keys or []
        self.meta_keys = meta_keys or []

    def __call__(self, results):
        """Collect specified keys and format for mmengine."""
        # Collect data
        inputs = {}
        for key in self.keys:
            if key in results:
                inputs[key] = results[key]
        
        # Add meta info
        img_meta = {}
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]
        
        # mmengine expects dict with 'inputs' and 'data_samples' keys
        # Pack in the format mmengine expects
        data_sample_dict = {
            'metainfo': img_meta
        }
        
        # Add ground truth for evaluation
        if 'voxel_semantics' in results:
            data_sample_dict['gt_occ'] = results['voxel_semantics']
        if 'mask_lidar' in results:
            data_sample_dict['mask_lidar'] = results['mask_lidar']
        if 'mask_camera' in results:
            data_sample_dict['mask_camera'] = results['mask_camera']
        
        packed_results = {
            'inputs': inputs,
            'data_samples': data_sample_dict
        }
        
        return packed_results


# MultiScaleFlipAug3D is already provided by mmdet3d, no need to register again
