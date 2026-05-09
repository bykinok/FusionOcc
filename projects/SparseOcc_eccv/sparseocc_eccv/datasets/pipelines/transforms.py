import mmcv
import torch
import numpy as np
from PIL import Image
from numpy import random
# 구버전 mmdet.datasets.builder.PIPELINES → compat 경유
from ...compat import PIPELINES


@PIPELINES.register_module()
class PadMultiViewImage(object):
    """Pad the multi-view image."""

    def __init__(self, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad_img(self, img):
        if self.size_divisor is not None:
            pad_h = int(np.ceil(img.shape[0] / self.size_divisor)) * self.size_divisor
            pad_w = int(np.ceil(img.shape[1] / self.size_divisor)) * self.size_divisor
        else:
            pad_h, pad_w = self.size

        pad_width = ((0, pad_h - img.shape[0]), (0, pad_w - img.shape[1]), (0, 0))
        img = np.pad(img, pad_width, constant_values=self.pad_val)
        return img

    def _pad_imgs(self, results):
        padded_img = [self._pad_img(img) for img in results['img']]
        
        results['ori_shape'] = [img.shape for img in results['img']]
        results['img'] = padded_img
        results['img_shape'] = [img.shape for img in padded_img]
        results['pad_shape'] = [img.shape for img in padded_img]
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def __call__(self, results):
        self._pad_imgs(results)
        return results


@PIPELINES.register_module()
class NormalizeMultiviewImage(object):
    """Normalize the image."""

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32).reshape(-1)
        self.std = 1 / np.array(std, dtype=np.float32).reshape(-1)
        self.to_rgb = to_rgb

    def __call__(self, results):
        normalized_imgs = []

        for img in results['img']:
            img = img.astype(np.float32)
            if self.to_rgb:
                img = img[..., ::-1]
            img = img - self.mean
            img = img * self.std
            normalized_imgs.append(img)

        results['img'] = normalized_imgs
        results['img_norm_cfg'] = dict(
            mean=self.mean,
            std=self.std,
            to_rgb=self.to_rgb
        )
        return results


@PIPELINES.register_module()
class PhotoMetricDistortionMultiViewImage:
    """Apply photometric distortion to image sequentially."""

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, results):
        imgs = results['img']
        new_imgs = []
        for img in imgs:
            ori_dtype = img.dtype
            img = img.astype(np.float32)

            if random.randint(2):
                delta = random.uniform(-self.brightness_delta, self.brightness_delta)
                img += delta

            mode = random.randint(2)
            if mode == 1:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower, self.contrast_upper)
                    img *= alpha

            img = mmcv.bgr2hsv(img)

            if random.randint(2):
                img[..., 1] *= random.uniform(self.saturation_lower, self.saturation_upper)

            if random.randint(2):
                img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
                img[..., 0][img[..., 0] > 360] -= 360
                img[..., 0][img[..., 0] < 0] += 360

            img = mmcv.hsv2bgr(img)

            if mode == 0:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower, self.contrast_upper)
                    img *= alpha

            if random.randint(2):
                img = img[..., random.permutation(3)]

            new_imgs.append(img.astype(ori_dtype))

        results['img'] = new_imgs
        return results


@PIPELINES.register_module()
class RandomTransformImage(object):
    def __init__(self, ida_aug_conf=None, training=True):
        self.ida_aug_conf = ida_aug_conf
        self.training = training

    def __call__(self, results):
        resize, resize_dims, crop, flip, rotate = self.sample_augmentation()
        
        if len(results['lidar2img']) == len(results['img']):
            for i in range(len(results['img'])):
                img = Image.fromarray(np.uint8(results['img'][i]))
                img, ida_mat = self.img_transform(
                    img,
                    resize=resize,
                    resize_dims=resize_dims,
                    crop=crop,
                    flip=flip,
                    rotate=rotate,
                )
                results['img'][i] = np.array(img).astype(np.uint8)
                results['lidar2img'][i] = ida_mat @ results['lidar2img'][i]

        elif len(results['img']) == 6:
            for i in range(len(results['img'])):
                img = Image.fromarray(np.uint8(results['img'][i]))
                img, ida_mat = self.img_transform(
                    img,
                    resize=resize,
                    resize_dims=resize_dims,
                    crop=crop,
                    flip=flip,
                    rotate=rotate,
                )
                results['img'][i] = np.array(img).astype(np.uint8)

            for i in range(len(results['lidar2img'])):
                results['lidar2img'][i] = ida_mat @ results['lidar2img'][i]

        else:
            raise ValueError()

        results['ori_shape'] = [img.shape for img in results['img']]
        results['img_shape'] = [img.shape for img in results['img']]
        results['pad_shape'] = [img.shape for img in results['img']]

        return results

    def img_transform(self, img, resize, resize_dims, crop, flip, rotate):
        def get_rot(h):
            return torch.Tensor([
                [np.cos(h), np.sin(h)],
                [-np.sin(h), np.cos(h)],
            ])

        ida_rot = torch.eye(2)
        ida_tran = torch.zeros(2)

        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)

        ida_rot *= resize
        ida_tran -= torch.Tensor(crop[:2])
        
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            ida_rot = A.matmul(ida_rot)
            ida_tran = A.matmul(ida_tran) + b
        
        A = get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b

        ida_rot = A.matmul(ida_rot)
        ida_tran = A.matmul(ida_tran) + b

        ida_mat = torch.eye(4)
        ida_mat[:2, :2] = ida_rot
        ida_mat[:2, 2] = ida_tran

        return img, ida_mat.numpy()

    def sample_augmentation(self):
        H, W = self.ida_aug_conf['H'], self.ida_aug_conf['W']
        fH, fW = self.ida_aug_conf['final_dim']

        if self.training:
            resize = np.random.uniform(*self.ida_aug_conf['resize_lim'])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.ida_aug_conf['bot_pct_lim'])) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.ida_aug_conf['rand_flip'] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.ida_aug_conf['rot_lim'])
        else:
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.ida_aug_conf['bot_pct_lim'])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0

        return resize, resize_dims, crop, flip, rotate


@PIPELINES.register_module()
class BEVAug(object):
    """BEV 공간 flip 증강 (X/Y 축 반전).

    voxel_semantics와 voxel_instances를 flip하고, ego2lidar 행렬을
    BDA 변환 행렬로 우측 곱하여 카메라 투영의 일관성을 유지한다.

    SparseOcc 좌표계:
      - sample_points: ego frame 기준 (pc_range = [-40,-40,-1, 40,40,5.4])
      - 모델 내 투영: pixel = lidar2img @ ego2lidar @ p_ego
      - BDA flip 후 보정: new_ego2lidar = ego2lidar @ bda_inv
        (flip 행렬은 자기역원이므로 bda_inv = bda)

    Args:
        bda_aug_conf (dict): 'flip_dx_ratio', 'flip_dy_ratio' 포함.
        is_train (bool): True이면 무작위 flip, False이면 항등 변환.
    """

    def __init__(self, bda_aug_conf, is_train=True):
        self.bda_aug_conf = bda_aug_conf
        self.is_train = is_train

    def _sample_flip(self):
        if self.is_train:
            flip_dx = np.random.uniform() < self.bda_aug_conf.get('flip_dx_ratio', 0.5)
            flip_dy = np.random.uniform() < self.bda_aug_conf.get('flip_dy_ratio', 0.5)
        else:
            flip_dx = False
            flip_dy = False
        return flip_dx, flip_dy

    def _build_bda_mat(self, flip_dx, flip_dy):
        """BDA 변환 행렬 (4×4 float64 numpy array) 생성.

        flip 행렬은 자기역원이므로 bda_mat = bda_inv.
        ego2lidar 보정: new_ego2lidar = ego2lidar @ bda_mat.
        """
        bda = np.eye(4, dtype=np.float64)
        if flip_dx:
            bda[0, 0] = -1.0  # X 축 반전
        if flip_dy:
            bda[1, 1] = -1.0  # Y 축 반전
        return bda

    def __call__(self, results):
        flip_dx, flip_dy = self._sample_flip()

        # ── voxel GT flip ──────────────────────────────────────────────────
        if flip_dx:
            if 'voxel_semantics' in results:
                results['voxel_semantics'] = results['voxel_semantics'][::-1, :, :].copy()
            if 'voxel_instances' in results:
                results['voxel_instances'] = results['voxel_instances'][::-1, :, :].copy()
            if 'mask_camera' in results:
                results['mask_camera'] = results['mask_camera'][::-1, :, :].copy()

        if flip_dy:
            if 'voxel_semantics' in results:
                results['voxel_semantics'] = results['voxel_semantics'][:, ::-1, :].copy()
            if 'voxel_instances' in results:
                results['voxel_instances'] = results['voxel_instances'][:, ::-1, :].copy()
            if 'mask_camera' in results:
                results['mask_camera'] = results['mask_camera'][:, ::-1, :].copy()

        # ── ego2lidar 보정: new_ego2lidar = ego2lidar @ bda_mat ────────────
        # SparseOcc transformer: final_proj = lidar2img @ ego2lidar @ p_ego
        # BDA flip 후: p_ego_new = bda @ p_ego_old
        # → pixel = lidar2img @ ego2lidar @ bda_inv @ p_ego_new
        # → new_ego2lidar = ego2lidar @ bda_mat (bda_inv = bda, flip은 자기역원)
        if flip_dx or flip_dy:
            bda_mat = self._build_bda_mat(flip_dx, flip_dy)
            if 'ego2lidar' in results:
                results['ego2lidar'] = [
                    m @ bda_mat for m in results['ego2lidar']
                ]

        # flip 상태 저장 (디버깅/후처리용)
        results['bda_flip_dx'] = flip_dx
        results['bda_flip_dy'] = flip_dy

        return results


@PIPELINES.register_module()
class GlobalRotScaleTransImage(object):
    def __init__(self,
                 rot_range=[-0.3925, 0.3925],
                 scale_ratio_range=[0.95, 1.05],
                 translation_std=[0, 0, 0]):
        self.rot_range = rot_range
        self.scale_ratio_range = scale_ratio_range
        self.translation_std = translation_std

    def __call__(self, results):
        rot_angle = np.random.uniform(*self.rot_range)
        self.rotate_z(results, rot_angle)
        results["gt_bboxes_3d"].rotate(np.array(rot_angle))

        scale_ratio = np.random.uniform(*self.scale_ratio_range)
        self.scale_xyz(results, scale_ratio)
        results["gt_bboxes_3d"].scale(scale_ratio)

        return results

    def rotate_z(self, results, rot_angle):
        rot_cos = torch.cos(torch.tensor(rot_angle))
        rot_sin = torch.sin(torch.tensor(rot_angle))

        rot_mat = torch.tensor([
            [rot_cos, -rot_sin, 0, 0],
            [rot_sin, rot_cos, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        rot_mat_inv = torch.inverse(rot_mat)

        for view in range(len(results['lidar2img'])):
            results['lidar2img'][view] = (torch.tensor(results['lidar2img'][view]).float() @ rot_mat_inv).numpy()

    def scale_xyz(self, results, scale_ratio):
        scale_mat = torch.tensor([
            [scale_ratio, 0, 0, 0],
            [0, scale_ratio, 0, 0],
            [0, 0, scale_ratio, 0],
            [0, 0, 0, 1],
        ])
        scale_mat_inv = torch.inverse(scale_mat)

        for view in range(len(results['lidar2img'])):
            results['lidar2img'][view] = (torch.tensor(results['lidar2img'][view]).float() @ scale_mat_inv).numpy()


@PIPELINES.register_module()
class PointToMultiViewDepth(object):
    """LiDAR points를 각 카메라 이미지의 depth map으로 변환한다.

    LoadPointsFromFile로 로드된 results['points']를 ego frame으로 변환한 후,
    각 카메라의 lidar2img/ego2lidar 행렬로 투영하여
    results['gt_depth'] = Tensor[N_cams, H_feat, W_feat]를 생성한다.
    단, 이미지 크기는 랜덤 크롭/리사이즈 전 원본 기준이므로
    RandomTransformImage 이전에 배치해야 한다.

    Args:
        grid_config (dict): depth 범위 설정, 예 ``dict(depth=[1.0, 45.0, 0.5])``.
        downsample (int): 이미지 대비 depth map 해상도 배율 (기본 16).
    """

    def __init__(self, grid_config: dict, downsample: int = 16):
        self.downsample = downsample
        self.grid_config = grid_config

    # ------------------------------------------------------------------
    # 내부 메서드
    # ------------------------------------------------------------------

    def _points_to_ego(self, results):
        """LiDAR points를 ego frame으로 변환. shape: (N, 3)."""
        ego2lidar = np.asarray(results['ego2lidar'], dtype=np.float64)
        if ego2lidar.ndim == 3:          # list of 6 identical matrices
            ego2lidar = ego2lidar[0]
        if ego2lidar.shape == (3, 4):
            tmp = np.eye(4, dtype=np.float64)
            tmp[:3, :] = ego2lidar
            ego2lidar = tmp
        lidar2ego = np.linalg.inv(ego2lidar)

        pts = results['points']
        p = pts.tensor[:, :3].numpy() if hasattr(pts, 'tensor') else np.asarray(pts)[:, :3]
        ones = np.ones((p.shape[0], 1), dtype=np.float64)
        return (lidar2ego @ np.hstack([p, ones]).T).T[:, :3]

    def _points2depthmap(self, points_2d_z, height, width):
        """(u, v, z) 배열을 downsampled depth map으로 변환."""
        h_out = height // self.downsample
        w_out = width // self.downsample
        depth_map = np.zeros((h_out, w_out), dtype=np.float32)
        d_min = self.grid_config['depth'][0]
        d_max = self.grid_config['depth'][1]

        coor = np.round(points_2d_z[:, :2] / self.downsample).astype(np.int32)
        depth = points_2d_z[:, 2]
        kept = (
            (coor[:, 0] >= 0) & (coor[:, 0] < w_out) &
            (coor[:, 1] >= 0) & (coor[:, 1] < h_out) &
            (depth >= d_min) & (depth < d_max)
        )
        coor, depth = coor[kept], depth[kept]
        if coor.size == 0:
            return depth_map

        ranks = coor[:, 0] + coor[:, 1] * w_out
        order = np.argsort(ranks + depth / 100.0)
        coor, depth, ranks = coor[order], depth[order], ranks[order]
        keep_first = np.ones(len(ranks), dtype=bool)
        keep_first[1:] = ranks[1:] != ranks[:-1]
        coor, depth = coor[keep_first], depth[keep_first]
        depth_map[coor[:, 1], coor[:, 0]] = depth
        return depth_map

    def __call__(self, results):
        points_ego = self._points_to_ego(results)   # (N, 3)

        # 이미지 크기 확인
        if 'img' in results and len(results['img']) > 0:
            img0 = results['img'][0]
            h, w = img0.shape[0], img0.shape[1]
        elif 'img_shape' in results:
            sh = results['img_shape'][0]
            h, w = sh[0], sh[1]
        else:
            raise KeyError("results에 'img' 또는 'img_shape'가 없습니다.")

        # ego2lidar: 현재 프레임 (리스트라면 첫 번째)
        ego2lidar = np.asarray(results['ego2lidar'], dtype=np.float64)
        if ego2lidar.ndim == 3:
            ego2lidar = ego2lidar[0]
        if ego2lidar.shape == (3, 4):
            tmp = np.eye(4, dtype=np.float64)
            tmp[:3, :] = ego2lidar
            ego2lidar = tmp

        ones = np.ones((points_ego.shape[0], 1), dtype=np.float64)
        p_homo = np.hstack([points_ego, ones]).T   # (4, N)

        lidar2img_list = results['lidar2img']
        num_cams = len(lidar2img_list)
        depth_maps = []
        for cid in range(num_cams):
            l2i = np.asarray(lidar2img_list[cid], dtype=np.float64)
            if l2i.shape == (3, 4):
                tmp = np.eye(4, dtype=np.float64)
                tmp[:3, :] = l2i
                l2i = tmp
            # ego frame → image: ego2img = lidar2img @ ego2lidar
            ego2img = l2i @ ego2lidar
            pts_cam = (ego2img @ p_homo).T       # (N, 4)
            z = pts_cam[:, 2]
            u = pts_cam[:, 0] / (z + 1e-6)
            v = pts_cam[:, 1] / (z + 1e-6)
            depth_maps.append(self._points2depthmap(
                np.stack([u, v, z], axis=1), h, w))

        results['gt_depth'] = torch.from_numpy(
            np.stack(depth_maps, axis=0).astype(np.float32))   # [N_cams, H_feat, W_feat]
        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'downsample={self.downsample}, '
                f'grid_config={self.grid_config})')
