# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
from typing import Optional, Union

import mmcv
import numpy as np
import torch
import torch.nn as nn
from mmcv.transforms.base import BaseTransform
from mmengine.fileio import get
import numba as nb

from mmdet3d.datasets.transforms import LoadMultiViewImageFromFiles
from mmdet3d.registry import TRANSFORMS

Number = Union[int, float]


@TRANSFORMS.register_module()
class SegLabelMapping(BaseTransform):
    """Map semantic segmentation labels according to learning map.
    
    This is equivalent to the label mapping functionality in the original TPVFormer.
    Maps NuScenes lidarseg labels (0-31) to 16 occupancy classes (0-16).
    """
    
    def __init__(self, label_mapping_file: Optional[str] = None):
        """Initialize SegLabelMapping.
        
        Args:
            label_mapping_file (str): Path to label mapping YAML file.
        """
        super().__init__()
        self.label_mapping_file = label_mapping_file
        
        if label_mapping_file is not None:
            try:
                import yaml
                with open(label_mapping_file, 'r') as stream:
                    nuscenesyaml = yaml.safe_load(stream)
                    self.learning_map = nuscenesyaml['learning_map']
            except FileNotFoundError:
                print(f"Warning: Label mapping file {label_mapping_file} not found. Using default NuScenes mapping.")
                self.learning_map = self._get_default_nuscenes_mapping()
            except Exception as e:
                print(f"Warning: Error loading label mapping file: {e}. Using default NuScenes mapping.")
                self.learning_map = self._get_default_nuscenes_mapping()
        else:
            # Use default NuScenes occupancy mapping (33 lidarseg classes [0-32] -> 18 occupancy classes [0-17])
            self.learning_map = self._get_default_nuscenes_mapping()
    
    def _get_default_nuscenes_mapping(self):
        """Get default NuScenes lidarseg to occupancy mapping.
        
        Maps NuScenes lidarseg labels (0-32) to 18 occupancy classes (0-17).
        Based on TPVFormer_ori/config/label_mapping/nuscenes-noIgnore.yaml
        
        Class mapping:
        - Class 0: noise (ignore during training/evaluation)
        - Class 1-16: actual occupancy classes (evaluated)
        - Class 17: empty (used for filling, not evaluated)
        """
        return {
            1: 0, 5: 0, 7: 0, 8: 0, 10: 0, 11: 0, 13: 0, 19: 0, 20: 0, 0: 0, 29: 0, 31: 0,  # noise/ignore -> 0
            9: 1,      # barrier -> 1
            14: 2,     # bicycle -> 2
            15: 3, 16: 3,  # bus -> 3
            17: 4,     # car -> 4
            18: 5,     # construction_vehicle -> 5
            21: 6,     # motorcycle -> 6
            2: 7, 3: 7, 4: 7, 6: 7,  # pedestrian -> 7
            12: 8,     # traffic_cone -> 8
            22: 9,     # trailer -> 9
            23: 10,    # truck -> 10
            24: 11,    # driveable_surface -> 11
            25: 12,    # other_flat -> 12
            26: 13,    # sidewalk -> 13
            27: 14,    # terrain -> 14
            28: 15,    # manmade -> 15
            30: 16,    # vegetation -> 16
            32: 17,    # empty -> 17 ✅ 추가 (원본과 동일)
        }
    
    def transform(self, results: dict) -> dict:
        """Transform function - NO-OP because mapping is done in LoadOccupancyAnnotations.
        
        Label mapping is already performed in LoadOccupancyAnnotations.transform() at line 313.
        This transform is kept for backward compatibility but does nothing.
        Applying label mapping twice would corrupt the labels!
        
        Args:
            results (dict): Result dict containing 'pts_semantic_mask'.
            
        Returns:
            dict: Result dict unchanged.
        """
        # Do nothing - label mapping already done in LoadOccupancyAnnotations
        return results


def _lidar2ego_to_ego2lidar(results: dict) -> Optional[np.ndarray]:
    """Build ego2lidar 4x4 from results['lidar_points']['lidar2ego']."""
    if 'lidar_points' not in results or not isinstance(results.get('lidar_points'), dict):
        return None
    lidar2ego = results['lidar_points'].get('lidar2ego')
    if lidar2ego is None:
        return None
    lidar2ego = np.array(lidar2ego, dtype=np.float64)
    if lidar2ego.shape != (4, 4):
        return None
    return np.linalg.inv(lidar2ego)


@TRANSFORMS.register_module()
class BEVLoadMultiViewImageFromFiles(LoadMultiViewImageFromFiles):
    """Load multi channel images from a list of separate channel files.

    ``BEVLoadMultiViewImageFromFiles`` adds the following keys for the
    convenience of view transforms in the forward:
        - 'cam2lidar'
        - 'lidar2img'

    When use_ego_frame=True (e.g. for Occ3D Ego-frame GT), stores ego2img
    in 'lidar2img' so that the model output is in Ego frame (lidar2img @ ego2lidar).

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
        num_views (int): Number of view in a frame. Defaults to 5.
        num_ref_frames (int): Number of frame in loading. Defaults to -1.
        test_mode (bool): Whether is test mode in loading. Defaults to False.
        set_default_scale (bool): Whether to set default scale.
            Defaults to True.
        use_ego_frame (bool): If True, set lidar2img = lidar2img @ ego2lidar
            so that 3D volume is Ego frame (for Occ3D). Defaults to False.
    """

    def __init__(self, use_ego_frame: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.use_ego_frame = use_ego_frame

    def transform(self, results: dict) -> Optional[dict]:
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data.
            Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        filename, cam2img, lidar2cam, lidar2img = [], [], [], []
        for _, cam_item in results['images'].items():
            filename.append(cam_item['img_path'])
            lidar2cam.append(cam_item['lidar2cam'])

            lidar2cam_array = np.array(cam_item['lidar2cam'])
            cam2img_array = np.eye(4).astype(np.float64)
            cam2img_array[:3, :3] = np.array(cam_item['cam2img'])
            cam2img.append(cam2img_array)
            lidar2img.append(cam2img_array @ lidar2cam_array)

        results['img_path'] = filename
        results['cam2img'] = np.stack(cam2img, axis=0)
        results['lidar2cam'] = np.stack(lidar2cam, axis=0)
        lidar2img_stack = np.stack(lidar2img, axis=0)
        if self.use_ego_frame:
            ego2lidar = _lidar2ego_to_ego2lidar(results)
            if ego2lidar is not None:
                # ego2img = lidar2img @ ego2lidar so that 3D in Ego projects to image
                lidar2img_stack = (lidar2img_stack.astype(np.float64) @ ego2lidar).astype(np.float32)
            else:
                raise ValueError("Ego2lidar matrix not found in results['lidar_points']['lidar2ego']")
        results['lidar2img'] = lidar2img_stack

        results['ori_cam2img'] = copy.deepcopy(results['cam2img'])

        # img is of shape (h, w, c, num_views)
        # h and w can be different for different views
        img_bytes = [
            get(name, backend_args=self.backend_args) for name in filename
        ]
        # gbr follow tpvformer
        imgs = [
            mmcv.imfrombytes(img_byte, flag=self.color_type)
            for img_byte in img_bytes
        ]
        # handle the image with different shape
        img_shapes = np.stack([img.shape for img in imgs], axis=0)
        img_shape_max = np.max(img_shapes, axis=0)
        img_shape_min = np.min(img_shapes, axis=0)
        assert img_shape_min[-1] == img_shape_max[-1]
        if not np.all(img_shape_max == img_shape_min):
            pad_shape = img_shape_max[:2]
        else:
            pad_shape = None
        if pad_shape is not None:
            imgs = [
                mmcv.impad(img, shape=pad_shape, pad_val=0) for img in imgs
            ]
        # Stack images along channel dimension (original TPVFormer behavior)
        img = np.stack(imgs, axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)

        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        results['pad_shape'] = img.shape[:2] if pad_shape is None else pad_shape
        results['scale_factor'] = 1.0
        results['img_norm_cfg'] = dict(
            mean=np.array([0, 0, 0], dtype=np.float32),
            std=np.array([1, 1, 1], dtype=np.float32),
            to_rgb=False)

        return results


@TRANSFORMS.register_module()
class LoadOccupancyAnnotations(BaseTransform):
    """Load occupancy annotations for 3D occupancy prediction.
    
    This transform loads occupancy grid annotations and converts them
    to the format required by the occupancy prediction model.
    
    Args:
        with_bbox_3d (bool): Whether to load 3D bounding boxes.
            Defaults to False.
        with_label_3d (bool): Whether to load 3D labels.
            Defaults to False.
        with_seg_3d (bool): Whether to load 3D segmentation.
            Defaults to True.
        with_attr_label (bool): Whether to load attribute labels.
            Defaults to False.
        seg_3d_dtype (str): Data type of 3D segmentation.
            Defaults to 'np.uint8'.
    """
    
    def __init__(self,
                 with_bbox_3d: bool = False,
                 with_label_3d: bool = False,
                 with_seg_3d: bool = True,
                 with_attr_label: bool = False,
                 seg_3d_dtype: str = 'np.uint8'):
        self.with_bbox_3d = with_bbox_3d
        self.with_label_3d = with_label_3d
        self.with_seg_3d = with_seg_3d
        self.with_attr_label = with_attr_label
        self.seg_3d_dtype = seg_3d_dtype
        
        # Initialize label mapping (원본과 동일: voxelization 전에 매핑)
        # NuScenes lidarseg (0-32) → Occupancy (0-17) mapping
        self.learning_map = self._get_default_nuscenes_mapping()
    
    def _get_default_nuscenes_mapping(self):
        """Get default NuScenes lidarseg to occupancy mapping (원본과 동일)."""
        return {
            1: 0, 5: 0, 7: 0, 8: 0, 10: 0, 11: 0, 13: 0, 19: 0, 20: 0, 0: 0, 29: 0, 31: 0,
            9: 1, 14: 2, 15: 3, 16: 3, 17: 4, 18: 5, 21: 6, 2: 7, 3: 7, 4: 7, 6: 7,
            12: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 30: 16, 32: 17,
        }
    
    def transform(self, results: dict) -> dict:
        """Transform function to load occupancy annotations.
        
        Args:
            results (dict): Result dict from previous transforms.
            
        Returns:
            dict: Result dict with loaded annotations.
        """
        # breakpoint()

        # print(f"[DEBUG LoadOccupancy] Transform called, with_seg_3d: {self.with_seg_3d}, has points: {'points' in results}")
        if self.with_seg_3d:
            
            # Load 3D segmentation labels for occupancy
            if 'points' in results:
                points = results['points']
                pts_semantic_mask = None
                
                # Try to load pts_semantic_mask from different sources
                if 'pts_semantic_mask' in results:
                    pts_semantic_mask = results['pts_semantic_mask']
                    # print(f"[DEBUG LoadOccupancy] Found pts_semantic_mask directly")
                elif 'pts_semantic_mask_path' in results:
                    # Load semantic mask from file path
                    mask_path = results['pts_semantic_mask_path']
                    # print(f"[DEBUG LoadOccupancy] Loading pts_semantic_mask from path: {mask_path}")
                    try:
                        if isinstance(mask_path, str):
                            # Try to construct full path
                            if not mask_path.startswith('/'):
                                # Relative path, convert to absolute
                                # mask_path might be like "./data/nuscenes/lidarseg/..."
                                if mask_path.startswith('./'):
                                    mask_path = mask_path[2:]  # Remove "./"
                                
                                # Construct absolute path from workspace root
                                workspace_root = os.getcwd()
                                full_path = os.path.join(workspace_root, mask_path)
                                # print(f"[DEBUG LoadOccupancy] Trying full path: {full_path}")
                                if os.path.exists(full_path):
                                    pts_semantic_mask = np.fromfile(full_path, dtype=np.uint8)
                                    # print(f"[DEBUG LoadOccupancy] Successfully loaded {len(pts_semantic_mask)} labels")
                                else:
                                    raise FileNotFoundError(
                                        f"[TPVFormer LoadOccupancyAnnotations] Semantic mask file not found: {full_path}\n"
                                        f"Original path: {mask_path}\n"
                                        f"Please check if the lidarseg data exists."
                                    )
                            else:
                                pts_semantic_mask = np.fromfile(mask_path, dtype=np.uint8)
                        else:
                            raise ValueError(
                                f"[TPVFormer LoadOccupancyAnnotations] Invalid mask_path type: {type(mask_path)}\n"
                                f"Expected str, got: {mask_path}\n"
                                f"Please check the dataset configuration."
                            )
                    except FileNotFoundError:
                        raise
                    except ValueError:
                        raise
                    except Exception as e:
                        raise FileNotFoundError(
                            f"[TPVFormer LoadOccupancyAnnotations] Failed to load semantic mask from {mask_path}\n"
                            f"Error: {e}\n"
                            f"Please check if the lidarseg data file exists and is not corrupted."
                        ) from e
                
                if pts_semantic_mask is not None:
                    # print(f"[DEBUG LoadOccupancy] Points shape: {points.shape}, Labels shape: {pts_semantic_mask.shape}")
                    
                    # ★ 원본과 동일: Voxelization 전에 label mapping 수행
                    # Raw lidarseg labels (0-32) → Occupancy labels (0-17)
                    # 이렇게 해야 majority voting이 올바르게 작동함
                    pts_semantic_mask = pts_semantic_mask.reshape(-1)  # Ensure 1D
                    pts_semantic_mask = np.vectorize(lambda x: self.learning_map.get(x, 0))(pts_semantic_mask)
                    pts_semantic_mask = pts_semantic_mask.astype(np.uint8)

                    # breakpoint()
                    
                    # Create voxel grid from point cloud (following tpv04 approach)
                    voxel_semantic_mask, voxel_coords = self._points_to_voxel_grid(
                        points, pts_semantic_mask, results)
                    
                    results['voxel_semantic_mask'] = voxel_semantic_mask
                    results['voxel_coords'] = voxel_coords  # Store voxel coordinates for evaluation
                    # Store original point-level labels as well
                    results['pts_semantic_mask'] = pts_semantic_mask
                    # print(f"[DEBUG LoadOccupancy] Stored voxel_coords with shape: {voxel_coords.shape}")
                    
                    # Ensure the data type is correct
                    if hasattr(np, self.seg_3d_dtype.split('.')[-1]):
                        dtype = getattr(np, self.seg_3d_dtype.split('.')[-1])
                        results['voxel_semantic_mask'] = results['voxel_semantic_mask'].astype(dtype)
                        
                    # print(f"[DEBUG LoadOccupancy] Created voxel_semantic_mask with shape: {results['voxel_semantic_mask'].shape}")
                # else:
                    # print(f"[DEBUG LoadOccupancy] Could not load pts_semantic_mask")
            # else:
                # print(f"[DEBUG LoadOccupancy] No points found in results")
        
        if self.with_bbox_3d:
            # Load 3D bounding boxes if needed
            if 'instances_3d' in results:
                bboxes_3d = []
                labels_3d = []
                for instance in results['instances_3d']:
                    if 'bbox_3d' in instance:
                        bboxes_3d.append(instance['bbox_3d'])
                    if 'label_3d' in instance:
                        labels_3d.append(instance['label_3d'])
                
                if bboxes_3d:
                    results['gt_bboxes_3d'] = np.array(bboxes_3d)
                if labels_3d:
                    results['gt_labels_3d'] = np.array(labels_3d)
        
        if self.with_label_3d:
            # Load 3D labels if needed
            if 'gt_labels_3d' not in results:
                # Create dummy labels if none available
                results['gt_labels_3d'] = np.array([])
        
        if self.with_attr_label:
            # Load attribute labels if needed
            if 'gt_attr_labels' not in results:
                # Create dummy attribute labels if none available
                results['gt_attr_labels'] = np.array([])
        
        return results
    
    def _points_to_voxel_grid(self, points, labels, results):
        """Convert point cloud to voxel grid following tpv04 implementation.
        
        Args:
            points: Point cloud coordinates [N, 3] in (x, y, z) order (numpy array or torch Tensor)
            labels: Point-wise semantic labels [N,] (numpy array)
            results: Results dict containing metadata
            
        Returns:
            tuple: (voxel_grid [W, H, Z], voxel_coords [N, 3])
                - voxel_grid: Voxel grid with semantic labels in (W, H, Z) order
                - voxel_coords: Voxel coordinates for each point in (w, h, z) order (for evaluation)
        """
        # 원본과 동일한 파라미터
        min_bound = np.array([-51.2, -51.2, -5.0])
        max_bound = np.array([51.2, 51.2, 3.0])
        grid_size = np.array([100, 100, 8])  # (W, H, Z)
        fill_label = 17  # Empty voxel label
        
        # Convert points to numpy if it's a torch Tensor
        if isinstance(points, torch.Tensor):
            points = points.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        
        # Calculate voxel indices (원본과 동일)
        crop_range = max_bound - min_bound
        intervals = crop_range / (grid_size - 1)
        
        # Extract xyz coordinates
        if points.shape[1] > 3:
            xyz = points[:, :3]
        else:
            xyz = points
        
        # Clip and convert to grid indices (원본과 동일)
        grid_ind_float = (np.clip(xyz, min_bound, max_bound) - min_bound) / intervals
        grid_ind = np.floor(grid_ind_float).astype(np.int64)  # int64로 변경 (Numba 시그니처와 일치)
        
        # Ensure labels is 1D array
        labels = labels.squeeze() if labels.ndim > 1 else labels
        
        # Initialize dense voxel grid (원본과 동일)
        processed_label = np.ones(grid_size, dtype=np.uint8) * fill_label  # (100, 100, 8) dense!
        
        # Create [x, y, z, label] pairs and sort (원본과 동일)
        label_voxel_pair = np.concatenate([grid_ind, labels.reshape(-1, 1)], axis=1)
        label_voxel_pair = label_voxel_pair[np.lexsort((grid_ind[:, 0], grid_ind[:, 1], grid_ind[:, 2])), :]
        
        # Apply majority voting using Numba-optimized function (원본과 동일)
        processed_label = nb_process_label(np.copy(processed_label), label_voxel_pair)
        
        # Return dense voxel grid and sparse coordinates
        # processed_label: (100, 100, 8) ✅ Dense voxel grid
        # grid_ind: (N, 3) Sparse coordinates
        return processed_label, grid_ind


@nb.jit('u1[:,:,:](u1[:,:,:],i8[:,:])', nopython=True, cache=True, parallel=False)
def nb_process_label(processed_label, sorted_label_voxel_pair):
    """Numba-optimized function for majority voting in voxels (원본과 동일).
    
    Args:
        processed_label: (W, H, Z) dense grid initialized with fill_label
        sorted_label_voxel_pair: (N, 4) array of [x, y, z, label], sorted by voxel coords
        
    Returns:
        processed_label: (W, H, Z) dense grid with labels filled by majority voting
    """
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

@TRANSFORMS.register_module()
class PadMultiViewImage(BaseTransform):
    """Pad the multi-view image.
    
    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    
    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value, 0 by default.
    """

    def __init__(self, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        if self.size is not None:
            padded_img = [mmcv.impad(
                img, shape=self.size, pad_val=self.pad_val) for img in results['img']]
        elif self.size_divisor is not None:
            padded_img = [mmcv.impad_to_multiple(
                img, self.size_divisor, pad_val=self.pad_val) for img in results['img']]
        
        results['ori_shape'] = [img.shape for img in results['img']]
        results['img'] = padded_img
        results['img_shape'] = [img.shape for img in padded_img]
        results['pad_shape'] = [img.shape for img in padded_img]
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def transform(self, results):
        """Call function to pad images.
        
        Args:
            results (dict): Result dict from loading pipeline.
            
        Returns:
            dict: Updated result dict.
        """
        self._pad_img(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'size_divisor={self.size_divisor}, '
        repr_str += f'pad_val={self.pad_val})'
        return repr_str


@TRANSFORMS.register_module()
class GridMask(nn.Module):
    """GridMask data augmentation.

        Modified from https://github.com/dvlab-research/GridMask.

    Args:
        use_h (bool): Whether to mask on height dimension. Defaults to True.
        use_w (bool): Whether to mask on width dimension. Defaults to True.
        rotate (int): Rotation degree. Defaults to 1.
        offset (bool): Whether to mask offset. Defaults to False.
        ratio (float): Mask ratio. Defaults to 0.5.
        mode (int): Mask mode. if mode == 0, mask with square grid.
            if mode == 1, mask the rest. Defaults to 0
        prob (float): Probability of applying the augmentation.
            Defaults to 1.0.
    """

    def __init__(self,
                 use_h: bool = True,
                 use_w: bool = True,
                 rotate: int = 1,
                 offset: bool = False,
                 ratio: float = 0.5,
                 mode: int = 0,
                 prob: float = 1.0):
        super().__init__()
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.prob = prob

    def forward(self, img):
        """Forward function for GridMask augmentation.
        
        Args:
            img (torch.Tensor): Input image tensor.
            
        Returns:
            torch.Tensor: Augmented image tensor.
        """
        if np.random.random() > self.prob:
            return img
        
        # This is a placeholder implementation
        # In practice, you would implement the actual GridMask augmentation
        return img


@TRANSFORMS.register_module()
class BEVAug(BaseTransform):
    """BEV augmentation (Flip X/Y) for occupancy, same as STCOcc/BEVFormer BEVAug.

    Applies random horizontal (flip_dx) and vertical (flip_dy) flips to
    voxel_semantics and mask tensors. Computes bda_mat (BEV Data Augmentation matrix)
    to be used in view transformation with inverse_bda.
    """

    def __init__(self, bda_aug_conf, is_train=True):
        super().__init__()
        self.bda_aug_conf = bda_aug_conf
        self.is_train = is_train

    def sample_bda_augmentation(self):
        """Sample augmentation parameters from bda_aug_conf (same as STCOcc)."""
        if self.is_train:
            rotate_bda = np.random.uniform(*self.bda_aug_conf.get('rot_lim', (0, 0)))
            scale_bda = np.random.uniform(*self.bda_aug_conf.get('scale_lim', (1., 1.)))
            flip_dx = np.random.uniform() < self.bda_aug_conf.get('flip_dx_ratio', 0.5)
            flip_dy = np.random.uniform() < self.bda_aug_conf.get('flip_dy_ratio', 0.5)
            translation_std = self.bda_aug_conf.get('tran_lim', [0.0, 0.0, 0.0])
            tran_bda = np.random.normal(scale=translation_std, size=3)
        else:
            rotate_bda = 0
            scale_bda = 1.0
            flip_dx = False
            flip_dy = False
            tran_bda = np.zeros(3, dtype=np.float32)
        return rotate_bda, scale_bda, flip_dx, flip_dy, tran_bda

    def bev_transform(self, rotate_angle, scale_ratio, flip_dx, flip_dy, tran_bda):
        """Get BEV transformation matrix (same as STCOcc)."""
        # Rotation matrix
        rotate_angle = torch.tensor(rotate_angle / 180 * np.pi)
        rot_sin = torch.sin(rotate_angle)
        rot_cos = torch.cos(rotate_angle)
        rot_mat = torch.Tensor([
            [rot_cos, -rot_sin, 0],
            [rot_sin, rot_cos, 0],
            [0, 0, 1]])
        
        # Scale matrix
        scale_mat = torch.Tensor([
            [scale_ratio, 0, 0],
            [0, scale_ratio, 0],
            [0, 0, scale_ratio]])
        
        # Flip matrix
        flip_mat = torch.Tensor([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]])

        if flip_dx:
            flip_mat = flip_mat @ torch.Tensor([
                [-1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]
            ])
        if flip_dy:
            flip_mat = flip_mat @ torch.Tensor([
                [1, 0, 0],
                [0, -1, 0],
                [0, 0, 1]
            ])

        rot_mat = flip_mat @ (scale_mat @ rot_mat)
        return rot_mat

    def _flip_voxel_x(self, arr):
        """Flip along first axis (X)."""
        return arr[::-1, ...].copy()

    def _flip_voxel_y(self, arr):
        """Flip along second axis (Y)."""
        return arr[:, ::-1, ...].copy()

    def transform(self, results):
        """Transform function to apply BEV augmentation."""
        # Sample augmentation parameters
        rotate_bda, scale_bda, flip_dx, flip_dy, tran_bda = self.sample_bda_augmentation()
        if 'bda_aug' in results:
            flip_dx = results['bda_aug'].get('flip_dx', flip_dx)
            flip_dy = results['bda_aug'].get('flip_dy', flip_dy)

        # Get bda rotation matrix (3x3)
        bda_rot = self.bev_transform(rotate_bda, scale_bda, flip_dx, flip_dy, tran_bda)
        
        # Build 4x4 bda_mat
        bda_mat = torch.zeros(4, 4)
        bda_mat[3, 3] = 1
        bda_mat[:3, :3] = bda_rot
        bda_mat[:3, 3] = torch.from_numpy(tran_bda if isinstance(tran_bda, np.ndarray) else np.array(tran_bda))

        # Apply voxel transformations (TPVFormer uses voxel_semantic_mask)
        if flip_dx:
            if 'voxel_semantic_mask' in results:
                results['voxel_semantic_mask'] = self._flip_voxel_x(results['voxel_semantic_mask'])
            # Flip occ_3d and occ_3d_masked if they exist (after LoadOccupancy)
            if 'occ_3d' in results and isinstance(results['occ_3d'], torch.Tensor):
                # occ_3d: (N, 4) with [x, y, z, label]
                # Flip X: x_new = grid_size[0] - 1 - x_old
                occ_3d = results['occ_3d'].clone()
                occ_3d[:, 0] = 199 - occ_3d[:, 0]  # 200-1 = 199
                results['occ_3d'] = occ_3d
            if 'occ_3d_masked' in results and isinstance(results['occ_3d_masked'], torch.Tensor):
                occ_3d_masked = results['occ_3d_masked'].clone()
                occ_3d_masked[:, 0] = 199 - occ_3d_masked[:, 0]
                results['occ_3d_masked'] = occ_3d_masked

        if flip_dy:
            if 'voxel_semantic_mask' in results:
                results['voxel_semantic_mask'] = self._flip_voxel_y(results['voxel_semantic_mask'])
            if 'occ_3d' in results and isinstance(results['occ_3d'], torch.Tensor):
                occ_3d = results['occ_3d'].clone()
                occ_3d[:, 1] = 199 - occ_3d[:, 1]
                results['occ_3d'] = occ_3d
            if 'occ_3d_masked' in results and isinstance(results['occ_3d_masked'], torch.Tensor):
                occ_3d_masked = results['occ_3d_masked'].clone()
                occ_3d_masked[:, 1] = 199 - occ_3d_masked[:, 1]
                results['occ_3d_masked'] = occ_3d_masked

        # Store flip flags for meta
        results['pcd_horizontal_flip'] = flip_dx
        results['pcd_vertical_flip'] = flip_dy
        
        # Store bda_mat for model to use in view transformation
        results['bda_mat'] = bda_mat.numpy()
        
        return results


@TRANSFORMS.register_module()
class LoadOccupancy(BaseTransform):
    """Load occupancy ground truth data for occ3d format.
    
    This transform loads occupancy ground truth from occ3d format (labels.npz)
    and converts it to the format required by the model.
    
    Args:
        use_occ3d (bool): Whether to use occ3d format. Defaults to False.
        pc_range (list): Point cloud range [x_min, y_min, z_min, x_max, y_max, z_max].
        use_camera_mask (bool): Whether to apply camera mask (set invisible voxels to 255).
            If True, voxels not visible from camera are marked as ignore (255).
            If False, all voxels keep their original labels. Defaults to True.
    """
    
    def __init__(self,
                 use_occ3d: bool = False,
                 pc_range: Optional[list] = None,
                 use_camera_mask: bool = True):
        super().__init__()
        self.use_occ3d = use_occ3d
        self.pc_range = pc_range
        self.use_camera_mask = use_camera_mask
    
    def transform(self, results: dict) -> dict:
        """Transform function to load occupancy data.
        
        Args:
            results (dict): Result dict from previous transforms.
            
        Returns:
            dict: Result dict with loaded occupancy data.
        """
        # breakpoint()

        if self.use_occ3d:
            # Filter points by point cloud range
            if self.pc_range is not None and 'points' in results:
                lidar2ego_rotation = np.array(results['lidar_points']['lidar2ego'])[:3, :3]
                lidar2ego_translation = np.array(results['lidar_points']['lidar2ego'])[:3, -1]
                points = results['points'].numpy()
                points[:, :3] = points[:, :3] @ lidar2ego_rotation.T
                points[:, :3] += lidar2ego_translation
                points = torch.from_numpy(points)
                idx = torch.where(
                    (points[:, 0] > self.pc_range[0]) &
                    (points[:, 1] > self.pc_range[1]) &
                    (points[:, 2] > self.pc_range[2]) &
                    (points[:, 0] < self.pc_range[3]) &
                    (points[:, 1] < self.pc_range[4]) &
                    (points[:, 2] < self.pc_range[5])
                )
                points = points[idx]
                results['points'] = points
                
                # Filter pts_semantic_mask if it exists (to match filtered points)
                if 'pts_semantic_mask' in results:
                    pts_semantic_mask = results['pts_semantic_mask']
                    if isinstance(pts_semantic_mask, np.ndarray):
                        if len(idx[0]) == len(pts_semantic_mask):
                            # idx is a tuple of tensors, extract first element
                            idx_np = idx[0].cpu().numpy() if hasattr(idx[0], 'cpu') else idx[0].numpy()
                            results['pts_semantic_mask'] = pts_semantic_mask[idx_np]
                        elif len(idx[0]) < len(pts_semantic_mask):
                            # Points were filtered, filter semantic mask accordingly
                            idx_np = idx[0].cpu().numpy() if hasattr(idx[0], 'cpu') else idx[0].numpy()
                            results['pts_semantic_mask'] = pts_semantic_mask[idx_np]
            
            # Determine occ3d path from ann_file (occ_path is already provided)
            if 'occ_path' in results:
                # Use pre-computed path from pkl file
                occ_3d_path = os.path.join(results['occ_path'], 'labels.npz')
            else:
                # Fallback: try to construct path from token
                occ_3d_folder = results['lidar_points']['lidar_path'].split('samples')[0] + 'Occ3D'
                occ_3d_path = os.path.join(occ_3d_folder, results['token'], 'labels.npz')
            
            # Load occ3d data
            if not os.path.exists(occ_3d_path):
                raise FileNotFoundError(
                    f"[TPVFormer LoadOccupancy] Occ3D file not found: {occ_3d_path}\n"
                    f"Expected occ_path: {results.get('occ_path', 'N/A')}\n"
                    f"Token: {results.get('token', 'N/A')}\n"
                    f"Please check if the Occ3D ground truth data exists."
                )
            
            occ_3d = np.load(occ_3d_path)
            occ_3d_semantic = occ_3d['semantics']  # (200, 200, 16)
            occ_3d_cam_mask = occ_3d['mask_camera']
            
            # ✅ Use Occ3D standard format (same as STCOcc/FusionOcc)
            # Occ3D standard format:
            # - Class 0: others (occupied but not classified)
            # - Class 1-16: semantic classes (barrier, bicycle, bus, ...)
            # - Class 17: free (empty space)
            # 
            # NO label mapping - use original format for consistency
            gt_occ = occ_3d_semantic.astype(np.int32)
            
            # Create voxel_semantic_mask for TPVFormer loss calculation
            # Optionally apply camera mask based on config
            if self.use_camera_mask:
                # Apply camera mask: invisible voxels → 255 (ignore label)
                voxel_semantic_mask = np.where(occ_3d_cam_mask, gt_occ, 255).astype(np.uint8)
                occ_3d_gt_masked = np.where(occ_3d_cam_mask, gt_occ, 255).astype(np.uint8)
            else:
                # No camera mask: use all voxels
                voxel_semantic_mask = gt_occ.astype(np.uint8)
                occ_3d_gt_masked = gt_occ.astype(np.uint8)
            
            results['voxel_semantic_mask'] = voxel_semantic_mask
            occ_3d_gt_masked = torch.from_numpy(occ_3d_gt_masked)
            # ✅ Exclude only 255 (ignore), include all valid classes (0-17)
            # Class 0 (others) is a valid class in Occ3D standard
            idx_masked = torch.where(occ_3d_gt_masked != 255)
            if len(idx_masked[0]) > 0:
                label_masked = occ_3d_gt_masked[idx_masked[0], idx_masked[1], idx_masked[2]]
                occ_3d_masked = torch.stack([
                    idx_masked[0], idx_masked[1], idx_masked[2], label_masked
                ], dim=1).long()
            else:
                occ_3d_masked = torch.zeros((0, 4), dtype=torch.long)
            
            # Create unmasked version
            occ_3d_gt = torch.from_numpy(gt_occ)
            # ✅ Include all valid classes (0-17)
            # In Occ3D standard: 0=others, 1-16=semantic classes, 17=free
            idx = torch.where(occ_3d_gt >= 0)
            if len(idx[0]) > 0:
                label = occ_3d_gt[idx[0], idx[1], idx[2]]
                occ3d = torch.stack([idx[0], idx[1], idx[2], label], dim=1).long()
            else:
                occ3d = torch.zeros((0, 4), dtype=torch.long)
            
            results['occ_3d_masked'] = occ_3d_masked
            results['occ_3d'] = occ3d
            
            # Create pts_semantic_mask from voxel_semantic_mask for point-level loss
            if 'points' in results:
                points = results['points']
                if isinstance(points, torch.Tensor):
                    points_np = points.cpu().numpy()
                else:
                    points_np = points
                
                # occ3d range and grid size
                occ3d_range = np.array([-40.0, -40.0, -1.0, 40.0, 40.0, 5.4])
                occ3d_grid_size = np.array([200, 200, 16])
                
                # Convert points to voxel indices
                xyz = points_np[:, :3] if points_np.shape[1] > 3 else points_np
                min_bound = occ3d_range[:3]
                max_bound = occ3d_range[3:]
                crop_range = max_bound - min_bound
                intervals = crop_range / occ3d_grid_size
                
                # Clip and convert to grid indices
                grid_ind_float = (np.clip(xyz, min_bound, max_bound) - min_bound) / intervals
                grid_ind = np.floor(grid_ind_float).astype(np.int32)
                
                # Clip indices to valid range
                grid_ind[:, 0] = np.clip(grid_ind[:, 0], 0, occ3d_grid_size[0] - 1)
                grid_ind[:, 1] = np.clip(grid_ind[:, 1], 0, occ3d_grid_size[1] - 1)
                grid_ind[:, 2] = np.clip(grid_ind[:, 2], 0, occ3d_grid_size[2] - 1)
                
                # Get semantic labels from voxel grid
                pts_semantic_mask = voxel_semantic_mask[grid_ind[:, 0], grid_ind[:, 1], grid_ind[:, 2]]
                results['pts_semantic_mask'] = pts_semantic_mask.astype(np.uint8)
        else:
            # Load traditional format (occ_200)
            occ_file_name = results['lidar_points']['lidar_path'].split('/')[-1] + '.npy'
            occ_200_folder = results['lidar_points']['lidar_path'].split('samples')[0] + 'occ_samples'
            occ_200_path = os.path.join(occ_200_folder, occ_file_name)
            
            if not os.path.exists(occ_200_path):
                raise FileNotFoundError(
                    f"[TPVFormer LoadOccupancy] Occ_200 file not found: {occ_200_path}\n"
                    f"LiDAR path: {results['lidar_points']['lidar_path']}\n"
                    f"Please check if the occ_200 ground truth data exists."
                )
            
            occ_200 = np.load(occ_200_path)
            occ_200[:, 3][occ_200[:, 3] == 0] = 255
            occ_200 = torch.from_numpy(occ_200)
            results['occ_200'] = occ_200
        
        return results
    
    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(use_occ3d={self.use_occ3d}, '
        repr_str += f'pc_range={self.pc_range}, '
        repr_str += f'use_camera_mask={self.use_camera_mask})'
        return repr_str
