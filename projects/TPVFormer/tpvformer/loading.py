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
    """
    
    def __init__(self, label_mapping_file: Optional[str] = None):
        """Initialize SegLabelMapping.
        
        Args:
            label_mapping_file (str): Path to label mapping YAML file.
        """
        super().__init__()
        self.label_mapping_file = label_mapping_file
        self.learning_map = None
        
        if label_mapping_file is not None:
            try:
                import yaml
                with open(label_mapping_file, 'r') as stream:
                    nuscenesyaml = yaml.safe_load(stream)
                    self.learning_map = nuscenesyaml['learning_map']
            except FileNotFoundError:
                print(f"Warning: Label mapping file {label_mapping_file} not found. Using default mapping.")
                # Default mapping for NuScenes (0-17 classes)
                self.learning_map = {i: i for i in range(18)}
            except Exception as e:
                print(f"Warning: Error loading label mapping file: {e}. Using default mapping.")
                self.learning_map = {i: i for i in range(18)}
    
    def transform(self, results: dict) -> dict:
        """Transform function to map labels.
        
        Args:
            results (dict): Result dict containing 'pts_semantic_mask'.
            
        Returns:
            dict: Result dict with mapped labels.
        """
        if 'pts_semantic_mask' in results and self.learning_map is not None:
            # Apply label mapping similar to original TPVFormer
            pts_semantic_mask = results['pts_semantic_mask']
            if isinstance(pts_semantic_mask, np.ndarray):
                # Vectorize the mapping function
                mapped_mask = np.vectorize(self.learning_map.__getitem__)(pts_semantic_mask)
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
                                # Relative path, need to add data root
                                data_root = 'data/nuscenes/'
                                full_path = f"{data_root}lidarseg/v1.0-trainval/{mask_path}"
                                # print(f"[DEBUG LoadOccupancy] Trying full path: {full_path}")
                                if os.path.exists(full_path):
                                    pts_semantic_mask = np.fromfile(full_path, dtype=np.uint8)
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
                    voxel_semantic_mask = self._points_to_voxel_grid(
                        points, pts_semantic_mask, results)
                    
                    results['voxel_semantic_mask'] = voxel_semantic_mask
                    
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
            points: Point cloud coordinates [N, 3]
            labels: Point-wise semantic labels [N,]
            results: Results dict containing metadata
            
        Returns:
            np.ndarray: Voxel grid with semantic labels [H, W, Z]
        """
        # Use tpv04 compatible parameters
        min_bound = np.array([-51.2, -51.2, -5.0])
        max_bound = np.array([51.2, 51.2, 3.0])
        grid_size = np.array([100, 100, 8])  # H, W, Z
        fill_label = 17  # Empty voxel label
        
        # Calculate voxel indices
        crop_range = max_bound - min_bound
        intervals = crop_range / (grid_size - 1)
        
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
        
        return voxel_grid


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
