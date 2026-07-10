import os
import numpy as np
from numpy import random
import mmcv
# mmdet3d와 mmengine 모두의 TRANSFORMS에 등록해야 합니다
from mmdet3d.registry import TRANSFORMS as TRANSFORMS_MMDET3D
from mmengine.registry import TRANSFORMS as TRANSFORMS_MMENGINE
try:
    from mmcv.parallel import DataContainer as DC
except ImportError:
    # DataContainer is deprecated in newer versions, create a simple wrapper
    class DC:
        def __init__(self, data, **kwargs):
            self.data = data
            self._kwargs = kwargs
        def __repr__(self):
            return f'DC({self.data})'
import os

def _build_distance_mask(
    semantics: np.ndarray,
    mask_camera: np.ndarray,
    mode: str,
    free_class_id: int = 17,
    dist_threshold_c: float = 35.0,
    dist_threshold_d: float = 20.0,
    dist_threshold_d_prime: float = 35.0,
    pc_range_x: float = 80.0,
) -> np.ndarray:
    """Build a modified mask_camera for distance-aware ablation study.

    BEVFormer uses mask_camera as a per-voxel binary weight in the loss, so
    conditions are implemented by forcing mask_camera=1 for the voxels that
    should always be supervised.

    condition_C (occupied-only distance mask):
        - Occupied voxels within dist_threshold_c metres: force mask_camera=1.
        - Everything else: obey original mask_camera.

    condition_D (free-only distance mask):
        - Free voxels within dist_threshold_d metres: force mask_camera=1.
        - Everything else: obey original mask_camera.

    condition_D_prime (free-only extended distance mask):
        - Free voxels within dist_threshold_d_prime metres: force mask_camera=1.
        - Everything else: obey original mask_camera.

    condition_D_full (free-only full-range supervision):
        - ALL free voxels at every distance: force mask_camera=1.
        - Occupied voxels: obey original mask_camera.

    condition_C_full (occupied-only full-range supervision):
        - ALL occupied voxels at every distance: force mask_camera=1.
        - Free voxels: obey original mask_camera.
        - Symmetric counterpart of condition_D_full.

    Returns a modified copy of mask_camera (uint8, same shape as semantics).
    """
    mask_camera = mask_camera.copy().astype(np.uint8)
    X, Y, Z = semantics.shape

    voxel_size = pc_range_x / X
    ego_i = (X - 1) / 2.0
    ego_j = (Y - 1) / 2.0
    ix = np.arange(X, dtype=np.float32)
    jy = np.arange(Y, dtype=np.float32)
    dx = (ix - ego_i) * voxel_size
    dy = (jy - ego_j) * voxel_size
    dist_xy = np.sqrt(dx[:, None] ** 2 + dy[None, :] ** 2)
    dist_map = np.broadcast_to(dist_xy[:, :, None], (X, Y, Z))

    is_free = semantics == free_class_id
    is_occupied = ~is_free

    if mode == 'condition_C':
        force_supervised = is_occupied & (dist_map < dist_threshold_c)
    elif mode == 'condition_D':
        force_supervised = is_free & (dist_map < dist_threshold_d)
    elif mode == 'condition_D_prime':
        force_supervised = is_free & (dist_map < dist_threshold_d_prime)
    elif mode == 'condition_D_full':
        force_supervised = is_free
    elif mode == 'condition_C_full':
        force_supervised = is_occupied
    else:
        raise ValueError(
            f"_build_distance_mask: unknown mode='{mode}'. "
            "Expected 'condition_C', 'condition_D', 'condition_D_prime', "
            "'condition_D_full', or 'condition_C_full'."
        )

    mask_camera[force_supervised] = 1
    return mask_camera


@TRANSFORMS_MMDET3D.register_module()
@TRANSFORMS_MMENGINE.register_module()
class LoadOccGTFromFile(object):
    """Load occupancy GT labels with optional training-time mask modification.

    mask_mode controls how mask_camera is modified during training:
        'baseline_with_mask'    – mask_camera unchanged (default).
        'baseline_without_mask' – mask_camera forced to all-ones.
        'condition_C'           – Occupied voxels within dist_threshold_c metres
                                  force mask_camera=1; others unchanged.
        'condition_D'           – Free voxels within dist_threshold_d metres
                                  force mask_camera=1; others unchanged.
        'condition_D_prime'     – Free voxels within dist_threshold_d_prime metres
                                  force mask_camera=1; others unchanged.
        'condition_D_full'      – ALL free voxels force mask_camera=1.
        'condition_C_full'      – ALL occupied voxels force mask_camera=1.

    Eval is unaffected: the evaluator loads mask_camera independently.
    """

    _VALID_MASK_MODES = frozenset([
        'baseline_with_mask',
        'baseline_without_mask',
        'condition_C',
        'condition_D',
        'condition_D_prime',
        'condition_D_full',
        'condition_C_full',
    ])

    def __init__(
            self,
            data_root,
            mask_mode='baseline_with_mask',
            free_class_id=17,
            dist_threshold_c=35.0,
            dist_threshold_d=20.0,
            dist_threshold_d_prime=35.0,
            pc_range_x=80.0,
        ):
        self.data_root = data_root
        if mask_mode not in self._VALID_MASK_MODES:
            raise ValueError(
                f"mask_mode must be one of {sorted(self._VALID_MASK_MODES)}, "
                f"got '{mask_mode}'."
            )
        self.mask_mode = mask_mode
        self.free_class_id = free_class_id
        self.dist_threshold_c = dist_threshold_c
        self.dist_threshold_d = dist_threshold_d
        self.dist_threshold_d_prime = dist_threshold_d_prime
        self.pc_range_x = pc_range_x

    def _apply_mask(self, semantics, mask_camera):
        """Return (potentially modified) mask_camera based on mask_mode."""
        if self.mask_mode == 'baseline_with_mask':
            pass
        elif self.mask_mode == 'baseline_without_mask':
            mask_camera = np.ones_like(mask_camera, dtype=np.uint8)
        else:
            mask_camera = _build_distance_mask(
                semantics=semantics,
                mask_camera=mask_camera,
                mode=self.mask_mode,
                free_class_id=self.free_class_id,
                dist_threshold_c=self.dist_threshold_c,
                dist_threshold_d=self.dist_threshold_d,
                dist_threshold_d_prime=self.dist_threshold_d_prime,
                pc_range_x=self.pc_range_x,
            )
        return mask_camera

    def __call__(self, results):
        if 'occ_gt_path' in results:
            occ_gt_path = results['occ_gt_path']
            occ_gt_path = os.path.join(self.data_root, occ_gt_path)

            occ_labels = np.load(occ_gt_path)
            semantics = occ_labels['semantics']
            mask_lidar = occ_labels['mask_lidar']
            mask_camera = occ_labels['mask_camera']
        else:
            semantics = np.zeros((200, 200, 16), dtype=np.uint8)
            mask_lidar = np.zeros((200, 200, 16), dtype=np.uint8)
            mask_camera = np.zeros((200, 200, 16), dtype=np.uint8)

        mask_camera = self._apply_mask(semantics, mask_camera)

        results['voxel_semantics'] = semantics
        results['mask_lidar'] = mask_lidar
        results['mask_camera'] = mask_camera

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return "{} (data_root={}, mask_mode={})".format(
            self.__class__.__name__, self.data_root, self.mask_mode)


@TRANSFORMS_MMDET3D.register_module()
@TRANSFORMS_MMENGINE.register_module()
class BEVAug(object):
    """BEV augmentation (Flip X/Y) for occupancy, same as STCOcc BEVAug.

    Applies random horizontal (flip_dx) and vertical (flip_dy) flips to
    voxel_semantics and mask tensors. Computes bda_mat (BEV Data Augmentation matrix)
    to be used in view transformation with inverse_bda.
    """

    def __init__(self, bda_aug_conf, is_train=True):
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
            tran_bda = np.random.normal(scale=translation_std, size=3)  # shape: (3,)
        else:
            rotate_bda = 0
            scale_bda = 1.0
            flip_dx = False
            flip_dy = False
            tran_bda = np.zeros(3, dtype=np.float32)  # Fixed: shape (3,) instead of (1, 3)
        return rotate_bda, scale_bda, flip_dx, flip_dy, tran_bda

    def bev_transform(self, rotate_angle, scale_ratio, flip_dx, flip_dy, tran_bda):
        """Get BEV transformation matrix (same as STCOcc)."""
        import torch
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

    def __call__(self, results):
        # Sample augmentation parameters
        rotate_bda, scale_bda, flip_dx, flip_dy, tran_bda = self.sample_bda_augmentation()
        if 'bda_aug' in results:
            flip_dx = results['bda_aug'].get('flip_dx', flip_dx)
            flip_dy = results['bda_aug'].get('flip_dy', flip_dy)

        # Get bda rotation matrix (3x3)
        import torch
        bda_rot = self.bev_transform(rotate_bda, scale_bda, flip_dx, flip_dy, tran_bda)
        
        # Build 4x4 bda_mat
        bda_mat = torch.zeros(4, 4)
        bda_mat[3, 3] = 1
        bda_mat[:3, :3] = bda_rot
        bda_mat[:3, 3] = torch.from_numpy(tran_bda if isinstance(tran_bda, np.ndarray) else np.array(tran_bda))

        # Apply voxel transformations
        if flip_dx:
            if 'voxel_semantics' in results:
                results['voxel_semantics'] = self._flip_voxel_x(results['voxel_semantics'])
            if 'mask_camera' in results:
                results['mask_camera'] = self._flip_voxel_x(results['mask_camera'])
            if 'mask_lidar' in results:
                results['mask_lidar'] = self._flip_voxel_x(results['mask_lidar'])

        if flip_dy:
            if 'voxel_semantics' in results:
                results['voxel_semantics'] = self._flip_voxel_y(results['voxel_semantics'])
            if 'mask_camera' in results:
                results['mask_camera'] = self._flip_voxel_y(results['mask_camera'])
            if 'mask_lidar' in results:
                results['mask_lidar'] = self._flip_voxel_y(results['mask_lidar'])

        # Store flip flags for meta
        results['pcd_horizontal_flip'] = flip_dx
        results['pcd_vertical_flip'] = flip_dy
        
        # Store bda_mat for model to use in view transformation
        results['bda_mat'] = bda_mat.numpy()  # Store as numpy for compatibility
        
        return results


@TRANSFORMS_MMDET3D.register_module()
@TRANSFORMS_MMENGINE.register_module()
class BEVFormerPointToMultiViewDepth(object):
    """LiDAR points -> multi-view depth maps for auxiliary depth supervision (BEVFormer).

    Uses results['lidar2img'], results['ego2lidar'] from NuSceneOcc.
    Projects ego-frame points with ego2img = lidar2img @ ego2lidar so gt_depth
    is consistent with ego-frame view (same convention as TPVFormer use_ego_frame=True).

    Args:
        selected_classes (list[int] | None): If set, only LiDAR points whose
            semantic class (Occ3D IDs 0-16) is in this list contribute to the
            depth map. Requires lidarseg_root to be specified.
            Competing-pair selection (Option 1) example:
              selected_classes=[0,1,4,7,8,11,12,13,14,15,16]
              (excludes bicycle=2, bus=3, CV=5, motorcycle=6, trailer=9, truck=10)
        lidarseg_root (str | None): Root directory of NuScenes lidarseg files
            (e.g. 'data/nuscenes/lidarseg/v1.0-trainval').
            Required when selected_classes is not None.
    """

    # NuScenes lidarseg (0-32) → Occ3D (0-17) mapping
    _LIDARSEG_MAPPING = {
        1: 0, 5: 0, 7: 0, 8: 0, 10: 0, 11: 0, 13: 0, 19: 0, 20: 0, 0: 0, 29: 0, 31: 0,
        9: 1, 14: 2, 15: 3, 16: 3, 17: 4, 18: 5, 21: 6, 2: 7, 3: 7, 4: 7, 6: 7,
        12: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 30: 16, 32: 17,
    }

    def __init__(self, grid_config, downsample=16,
                 selected_classes=None, lidarseg_root=None):
        self.downsample = downsample
        self.grid_config = grid_config
        self.selected_classes = (
            np.array(selected_classes, dtype=np.int32)
            if selected_classes is not None else None
        )
        self.lidarseg_root = lidarseg_root

    def _load_pts_semantic_mask(self, results):
        """Load lidarseg labels and map to Occ3D class IDs."""
        pts_filename = results.get('pts_filename') or \
                       results.get('lidar_points', {}).get('lidar_path', '')
        if not pts_filename:
            return None
        # Derive sample token from filename:
        # e.g. ".../LIDAR_TOP/{token}.pcd.bin" -> "{token}"
        token = os.path.basename(pts_filename).replace('.pcd.bin', '')
        seg_path = os.path.join(self.lidarseg_root, f'{token}_lidarseg.bin')
        if not os.path.exists(seg_path):
            return None
        raw = np.fromfile(seg_path, dtype=np.uint8)
        mapping = self._LIDARSEG_MAPPING
        mapped = np.vectorize(lambda x: mapping.get(int(x), 0))(raw)
        return mapped.astype(np.int32)

    def _points_to_ego(self, results):
        """Points in ego frame (N, 3). lidar2ego = inv(ego2lidar)."""
        ego2lidar = results['ego2lidar']
        ego2lidar = np.asarray(ego2lidar, dtype=np.float64)
        if ego2lidar.shape != (4, 4):
            ego2lidar = np.eye(4, dtype=np.float64)
            ego2lidar[:3, :] = np.asarray(results['ego2lidar']).reshape(4, 4)[:3, :]
        lidar2ego = np.linalg.inv(ego2lidar)
        pts = results['points']
        if hasattr(pts, 'tensor'):
            p = pts.tensor[:, :3].numpy()
        else:
            p = np.asarray(pts)[:, :3]
        ones = np.ones((p.shape[0], 1), dtype=np.float64)
        p_homo = np.hstack([p, ones])  # (N, 4)
        p_ego = (lidar2ego @ p_homo.T).T[:, :3]
        return p_ego

    def _points2depthmap(self, points_2d_z, height, width):
        """points_2d_z: (N, 3) (u, v, z). Output (height//downsample, width//downsample)."""
        height = height // self.downsample
        width = width // self.downsample
        depth_map = np.zeros((height, width), dtype=np.float32)
        d_min, d_max = self.grid_config['depth'][0], self.grid_config['depth'][1]
        coor = np.round(points_2d_z[:, :2] / self.downsample).astype(np.int32)
        depth = points_2d_z[:, 2]
        kept = (
            (coor[:, 0] >= 0) & (coor[:, 0] < width) &
            (coor[:, 1] >= 0) & (coor[:, 1] < height) &
            (depth >= d_min) & (depth < d_max)
        )
        coor = coor[kept]
        depth = depth[kept]
        if coor.size == 0:
            return depth_map
        ranks = coor[:, 0] + coor[:, 1] * width
        order = np.argsort(ranks + depth / 100.0)
        coor = coor[order]
        depth = depth[order]
        ranks = ranks[order]
        keep_first = np.ones(len(ranks), dtype=bool)
        keep_first[1:] = ranks[1:] != ranks[:-1]
        coor = coor[keep_first]
        depth = depth[keep_first]
        depth_map[coor[:, 1], coor[:, 0]] = depth
        return depth_map

    def __call__(self, results):
        import torch
        points_ego = self._points_to_ego(results)

        # Class-selective depth supervision: filter by semantic class
        if self.selected_classes is not None and self.lidarseg_root is not None:
            pts_mask = results.get('pts_semantic_mask')
            if pts_mask is None:
                pts_mask = self._load_pts_semantic_mask(results)
            if pts_mask is not None:
                pts_mask = np.asarray(pts_mask).reshape(-1)
                if len(pts_mask) == len(points_ego):
                    keep = np.isin(pts_mask, self.selected_classes)
                    points_ego = points_ego[keep]

        lidar2img_list = results['lidar2img']
        ego2lidar = np.asarray(results['ego2lidar'], dtype=np.float64)
        if ego2lidar.shape != (4, 4):
            ego2lidar = np.eye(4, dtype=np.float64)
            ego2lidar[:3, :] = np.asarray(results['ego2lidar']).reshape(4, 4)[:3, :]
        if 'img' in results and len(results['img']) > 0:
            img0 = results['img'][0]
            h, w = img0.shape[0], img0.shape[1]
        elif 'img_shape' in results and len(results['img_shape']) > 0:
            sh = results['img_shape'][0]
            h, w = sh[0], sh[1]
        else:
            raise KeyError("Need 'img' or 'img_shape' in results for depth map size")
        num_cams = len(lidar2img_list)
        ones = np.ones((points_ego.shape[0], 1), dtype=np.float64)
        p_homo = np.hstack([points_ego, ones]).T  # (4, N)
        depth_maps = []
        for cid in range(num_cams):
            l2i = np.asarray(lidar2img_list[cid], dtype=np.float64)
            if l2i.shape != (4, 4):
                l2i = np.eye(4, dtype=np.float64)
                l2i[:3, :] = np.asarray(lidar2img_list[cid]).reshape(4, 4)[:3, :]
            ego2img = l2i @ ego2lidar
            pts_cam = (ego2img @ p_homo).T  # (N, 4)
            u = pts_cam[:, 0] / (pts_cam[:, 2] + 1e-6)
            v = pts_cam[:, 1] / (pts_cam[:, 2] + 1e-6)
            z = pts_cam[:, 2]
            points_2d_z = np.stack([u, v, z], axis=1)
            depth_map = self._points2depthmap(points_2d_z, h, w)
            depth_maps.append(depth_map)
        results['gt_depth'] = torch.from_numpy(np.stack(depth_maps, axis=0).astype(np.float32))
        return results

    def __repr__(self):
        return "{} (downsample={}, grid_config={}, selected_classes={})".format(
            self.__class__.__name__, self.downsample, self.grid_config,
            list(self.selected_classes) if self.selected_classes is not None else None)