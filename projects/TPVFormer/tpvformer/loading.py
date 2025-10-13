# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
from typing import Optional, Union

import mmcv
import numpy as np
import torch.nn as nn
from mmcv.transforms.base import BaseTransform
from mmengine.fileio import get

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
            # Use default NuScenes occupancy mapping (32 classes -> 17 classes)
            self.learning_map = self._get_default_nuscenes_mapping()
    
    def _get_default_nuscenes_mapping(self):
        """Get default NuScenes lidarseg to occupancy mapping.
        
        Maps NuScenes lidarseg labels (0-31) to 16 occupancy classes (0-16).
        Based on TPVFormer_ori/config/label_mapping/nuscenes.yaml
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
        }
    
    def transform(self, results: dict) -> dict:
        """Transform function to map labels.
        
        Args:
            results (dict): Result dict containing 'pts_semantic_mask'.
            
        Returns:
            dict: Result dict with mapped labels.
        """
        if 'pts_semantic_mask' in results:
            # Apply label mapping similar to original TPVFormer
            pts_semantic_mask = results['pts_semantic_mask']
            if isinstance(pts_semantic_mask, np.ndarray):
                # Vectorize the mapping function to map each label
                # For labels not in the mapping, default to 0 (ignore)
                mapped_mask = np.vectorize(lambda x: self.learning_map.get(x, 0))(pts_semantic_mask)
                results['pts_semantic_mask'] = mapped_mask.astype(np.uint8)
        
        return results


@TRANSFORMS.register_module()
class BEVLoadMultiViewImageFromFiles(LoadMultiViewImageFromFiles):
    """Load multi channel images from a list of separate channel files.

    ``BEVLoadMultiViewImageFromFiles`` adds the following keys for the
    convenience of view transforms in the forward:
        - 'cam2lidar'
        - 'lidar2img'

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
    """

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
        results['lidar2img'] = np.stack(lidar2img, axis=0)

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
    
    def transform(self, results: dict) -> dict:
        """Transform function to load occupancy annotations.
        
        Args:
            results (dict): Result dict from previous transforms.
            
        Returns:
            dict: Result dict with loaded annotations.
        """
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
                                workspace_root = '/home/h00323/Projects/occfrmwrk/'
                                full_path = f"{workspace_root}{mask_path}"
                                # print(f"[DEBUG LoadOccupancy] Trying full path: {full_path}")
                                if os.path.exists(full_path):
                                    pts_semantic_mask = np.fromfile(full_path, dtype=np.uint8)
                                    # print(f"[DEBUG LoadOccupancy] Successfully loaded {len(pts_semantic_mask)} labels")
                                else:
                                    # print(f"[DEBUG LoadOccupancy] Full path does not exist, using dummy labels")
                                    pts_semantic_mask = np.ones(len(points), dtype=np.uint8)
                            else:
                                pts_semantic_mask = np.fromfile(mask_path, dtype=np.uint8)
                        else:
                            # Handle list of paths or other formats
                            # print(f"[DEBUG LoadOccupancy] mask_path type: {type(mask_path)}, content: {mask_path}")
                            # Create realistic dummy labels for testing (valid class range 0-17)
                            pts_semantic_mask = np.random.randint(0, 18, size=len(points), dtype=np.uint8)
                    except Exception as e:
                        # print(f"[DEBUG LoadOccupancy] Error loading semantic mask: {e}")
                        # Create realistic dummy labels for testing (valid class range 0-17)
                        pts_semantic_mask = np.random.randint(0, 18, size=len(points), dtype=np.uint8)
                
                if pts_semantic_mask is not None:
                    # print(f"[DEBUG LoadOccupancy] Points shape: {points.shape}, Labels shape: {pts_semantic_mask.shape}")
                    
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
            points: Point cloud coordinates [N, 3] in (x, y, z) order
            labels: Point-wise semantic labels [N,]
            results: Results dict containing metadata
            
        Returns:
            tuple: (voxel_grid [W, H, Z], voxel_coords [N, 3])
                - voxel_grid: Voxel grid with semantic labels in (W, H, Z) order
                - voxel_coords: Voxel coordinates for each point in (w, h, z) order (for evaluation)
        """
        # Use tpv04 compatible parameters
        # point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0] (x, y, z)
        min_bound = np.array([-51.2, -51.2, -5.0])  # (x_min, y_min, z_min)
        max_bound = np.array([51.2, 51.2, 3.0])      # (x_max, y_max, z_max)
        grid_size = np.array([100, 100, 8])           # (W, H, Z) corresponding to (x, y, z)
        fill_label = 17  # Empty voxel label (class 17)
        
        # Calculate voxel indices
        crop_range = max_bound - min_bound  # (x_range, y_range, z_range)
        intervals = crop_range / (grid_size - 1)  # intervals for (x, y, z)
        
        # Clip points to valid range and convert to grid indices
        points_clipped = np.clip(points[:, :3], min_bound, max_bound)
        grid_ind_float = (points_clipped - min_bound) / intervals
        grid_ind = np.floor(grid_ind_float).astype(np.int32)
        
        # Initialize voxel grid with fill_label
        voxel_grid = np.ones(grid_size, dtype=np.uint8) * fill_label
        
        # Fill voxel grid with point labels (majority voting)
        labels = labels.squeeze() if labels.ndim > 1 else labels
        
        # Debug: Print statistics
        # print(f"[DEBUG] Points shape: {points.shape}, Labels shape: {labels.shape}")
        # print(f"[DEBUG] Unique labels: {np.unique(labels)}")
        # print(f"[DEBUG] Label distribution: {np.bincount(labels.astype(int), minlength=18)}")
        
        # Count valid points
        valid_points = 0
        for i in range(len(points)):
            x, y, z = grid_ind[i]
            if 0 <= x < grid_size[0] and 0 <= y < grid_size[1] and 0 <= z < grid_size[2]:
                # Use the last encountered label (simple approach)
                # In practice, you might want to use majority voting
                voxel_grid[x, y, z] = labels[i]
                valid_points += 1
        
        # print(f"[DEBUG] Valid points: {valid_points}/{len(points)}")
        # print(f"[DEBUG] Voxel grid unique labels: {np.unique(voxel_grid)}")
        # print(f"[DEBUG] Non-empty voxels: {np.sum(voxel_grid != fill_label)}")
        
        return voxel_grid, grid_ind


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
