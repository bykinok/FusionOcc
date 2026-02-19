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

@TRANSFORMS_MMDET3D.register_module()
@TRANSFORMS_MMENGINE.register_module()
class LoadOccGTFromFile(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.
    note that we read image in BGR style to align with opencv.imread
    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(
            self,
            data_root,
        ):
        self.data_root = data_root

    def __call__(self, results):
        # print(results.keys())
        if 'occ_gt_path' in results:
            occ_gt_path = results['occ_gt_path']
            occ_gt_path = os.path.join(self.data_root,occ_gt_path)

            occ_labels = np.load(occ_gt_path)
            semantics = occ_labels['semantics']
            mask_lidar = occ_labels['mask_lidar']
            mask_camera = occ_labels['mask_camera']
        else:
            semantics = np.zeros((200,200,16),dtype=np.uint8)
            mask_lidar = np.zeros((200,200,16),dtype=np.uint8)
            mask_camera = np.zeros((200, 200, 16), dtype=np.uint8)

        results['voxel_semantics'] = semantics
        results['mask_lidar'] = mask_lidar
        results['mask_camera'] = mask_camera


        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return "{} (data_root={}')".format(
            self.__class__.__name__, self.data_root)


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
    """

    def __init__(self, grid_config, downsample=16):
        self.downsample = downsample
        self.grid_config = grid_config

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
        return "{} (downsample={}, grid_config={})".format(
            self.__class__.__name__, self.downsample, self.grid_config)