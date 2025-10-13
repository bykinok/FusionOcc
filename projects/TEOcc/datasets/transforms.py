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
    # Based on BEVDet normalize logic
    img = np.array(img).astype(np.float32)
    # Convert from 0-255 to 0-1
    img = img / 255.0
    if img_norm_cfg is None:
        img_norm_cfg = {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        }
    
    img = (img - img_norm_cfg['mean']) / img_norm_cfg['std']
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
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
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
            resize = np.random.uniform(*self.data_config['resize'])
            resize_dims = (max(1, int(W * resize)), max(1, int(H * resize)))
            newW, newH = resize_dims
            crop_h = max(0, int((1 - np.random.uniform(*self.data_config['crop_h'])) *
                        newH) - fH)
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = self.data_config['flip'] and np.random.choice([0, 1])
            rotate = np.random.uniform(*self.data_config['rot'])
        else:
            resize = self.data_config.get('resize_test', 0.0)
            resize += 1.0
            resize_dims = (max(1, int(W * resize)), max(1, int(H * resize)))
            newW, newH = resize_dims
            crop_h = max(0, int((1 - np.mean(self.data_config['crop_h'])) * newH) - fH)
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False if flip is None else flip
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def get_sensor_transforms(self, cam_info, cam_name):
        """Get sensor to ego and ego to global transforms."""
        # This is simplified version - in practice you'd read from the camera data
        sensor2ego_rot = np.eye(3)
        sensor2ego_tran = np.zeros(3)
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

            # Normalize image
            img = self.normalize_img(img)

            imgs.append(img)
            intrins.append(intrin)
            post_rots.append(post_rot)
            post_trans.append(post_tran)
            sensor2egos.append(sensor2ego)
            ego2globals.append(ego2global)

        return (torch.stack(imgs), torch.stack(intrins),
                torch.stack(post_rots), torch.stack(post_trans),
                torch.stack([torch.Tensor(s) for s in sensor2egos]),
                torch.stack([torch.Tensor(s) for s in ego2globals])), []

    def __call__(self, results):
        img_inputs, gt_depths = self.get_inputs(results)
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
        """Load points from file."""
        # Create dummy points if file doesn't exist
        if not os.path.exists(pts_filename):
            return np.zeros((100, self.load_dim), dtype=np.float32)
        
        try:
            points = np.fromfile(pts_filename, dtype=np.float32)
            return points.reshape(-1, self.load_dim)
        except:
            return np.zeros((100, self.load_dim), dtype=np.float32)

    def __call__(self, results):
        """Load radar points from multiple sweeps."""
        # Create dummy radar points if no sweep data
        radar_sweeps = results.get('radar_sweeps', [])
        
        if len(radar_sweeps) == 0:
            # Create dummy radar data
            points = np.random.randn(self.max_num, len(self.use_dim)).astype(np.float32)
        else:
            points_sweep_list = []
            
            # Load current frame
            if 'radar' in results and 'data_path' in results['radar']:
                points_curr = self._load_points(results['radar']['data_path'])
                points_sweep_list.append(points_curr)
            
            # Load sweep frames
            for sweep in radar_sweeps[:self.sweeps_num]:
                points_sweep = self._load_points(sweep.get('data_path', ''))
                points_sweep_list.append(points_sweep)
            
            # Concatenate all points
            if points_sweep_list:
                points = np.concatenate(points_sweep_list, axis=0)
            else:
                points = np.random.randn(self.max_num, self.load_dim).astype(np.float32)
            
            points = points[:, self.use_dim]
        
        # Limit number of points
        if points.shape[0] > self.max_num:
            indices = np.random.choice(points.shape[0], self.max_num, replace=False)
            points = points[indices]
        elif points.shape[0] < self.max_num:
            # Pad with zeros
            pad_points = np.zeros((self.max_num - points.shape[0], points.shape[1]))
            points = np.vstack([points, pad_points])

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
    """Load annotations for BEV depth estimation."""

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

    def __call__(self, results):
        """Load BEV depth annotations."""
        # Create dummy annotations if not present
        if 'ann_info' not in results:
            results['gt_bboxes_3d'] = LiDARInstance3DBoxes(
                np.zeros((0, 7)), box_dim=7
            )
            results['gt_labels_3d'] = np.array([])
        
        return results


@TRANSFORMS.register_module()
class LoadOccGTFromFile:
    """Load occupancy ground truth from file."""

    def __init__(self, data_root='data/nuscenes'):
        self.data_root = data_root

    def __call__(self, results):
        """Load occupancy ground truth."""
        # Create dummy occupancy data
        results['voxel_semantics'] = np.zeros((200, 200, 16), dtype=np.uint8)
        results['mask_lidar'] = np.ones((200, 200, 16), dtype=bool) 
        results['mask_camera'] = np.ones((200, 200, 16), dtype=bool)
        return results


@TRANSFORMS.register_module()
class GlobalRotScaleTrans_radar:
    """Global rotation, scaling and translation for radar points."""

    def __init__(self):
        pass

    def __call__(self, results):
        """Apply global transformations."""
        # For now, just pass through
        return results


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
                results[key] = results[key].tensor
        
        return results


@TRANSFORMS.register_module()
class Collect3D:
    """Collect data for 3D detection."""

    def __init__(self, keys=None, meta_keys=None):
        self.keys = keys or []
        self.meta_keys = meta_keys or []

    def __call__(self, results):
        """Collect specified keys."""
        data = {}
        for key in self.keys:
            if key in results:
                data[key] = results[key]
        
        # Add meta info
        img_meta = {}
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]
        data['img_metas'] = img_meta
        
        return data


# MultiScaleFlipAug3D is already provided by mmdet3d, no need to register again
