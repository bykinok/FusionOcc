import os
import mmcv
import glob
import numpy as np
# 구버전 mmdet.datasets.builder.PIPELINES → compat 경유
from ...compat import PIPELINES, get_dist_info, DC, to_tensor
from numpy.linalg import inv

# ---------------------------------------------------------------------------
# LoadMultiViewImageFromFiles: 구버전 img_filename 기반 다중 뷰 이미지 로더
# 새 mmdet3d는 'images' dict 키를 사용하지만, 본 데이터셋은 'img_filename' 리스트를 사용
# 주의: 이 클래스는 데코레이터 등록 없이 정의만 하고,
#       sparseocc_eccv/__init__.py에서 mmdet3d 전체 import 후 force=True로 등록한다.
# ---------------------------------------------------------------------------
class LoadMultiViewImageFromFiles:
    """구버전 스타일 다중 뷰 이미지 로더.

    results['img_filename'] (list of str)에서 이미지를 읽어
    results['img'], results['filename'], results['img_shape'] 등을 채운다.
    """

    def __init__(self, to_float32=False, color_type='color', **kwargs):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):
        filenames = results.get('img_filename', [])
        imgs = []
        for fname in filenames:
            img = mmcv.imread(fname, channel_order='rgb'
                              if self.color_type == 'color' else 'grayscale')
            if self.to_float32:
                img = img.astype(np.float32)
            imgs.append(img)

        results['filename'] = filenames
        results['img'] = imgs
        results['img_shape'] = [img.shape for img in imgs]
        results['ori_shape'] = [img.shape for img in imgs]
        results['pad_shape'] = [img.shape for img in imgs]
        results['scale_factor'] = 1.0
        results['img_norm_cfg'] = dict(mean=np.zeros(3), std=np.ones(3), to_rgb=False)
        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'to_float32={self.to_float32}, '
                f'color_type={self.color_type!r})')


def compose_lidar2img(ego2global_translation_curr,
                      ego2global_rotation_curr,
                      lidar2ego_translation_curr,
                      lidar2ego_rotation_curr,
                      sensor2global_translation_past,
                      sensor2global_rotation_past,
                      cam_intrinsic_past):
    
    R = sensor2global_rotation_past @ (inv(ego2global_rotation_curr).T @ inv(lidar2ego_rotation_curr).T)
    T = sensor2global_translation_past @ (inv(ego2global_rotation_curr).T @ inv(lidar2ego_rotation_curr).T)
    T -= ego2global_translation_curr @ (inv(ego2global_rotation_curr).T @ inv(lidar2ego_rotation_curr).T) + lidar2ego_translation_curr @ inv(lidar2ego_rotation_curr).T

    lidar2cam_r = inv(R.T)
    lidar2cam_t = T @ lidar2cam_r.T

    lidar2cam_rt = np.eye(4)
    lidar2cam_rt[:3, :3] = lidar2cam_r.T
    lidar2cam_rt[3, :3] = -lidar2cam_t

    viewpad = np.eye(4)
    viewpad[:cam_intrinsic_past.shape[0], :cam_intrinsic_past.shape[1]] = cam_intrinsic_past
    lidar2img = (viewpad @ lidar2cam_rt.T).astype(np.float32)

    return lidar2img


@PIPELINES.register_module()
class LoadMultiViewImageFromMultiSweeps(object):
    def __init__(self,
                 sweeps_num=5,
                 color_type='color',
                 test_mode=False,
                 online_mode=False):
        self.sweeps_num = sweeps_num
        self.color_type = color_type
        self.test_mode = test_mode
        # online_mode=True 일 때만 load_online() 경로를 사용한다.
        # 기존 코드는 world_size==1 이면 자동으로 load_online()을 호출했으나,
        # 이는 FPS 측정 전용 경로로 실제 이미지를 로드하지 않아 1-GPU 추론 시
        # IndexError 를 유발한다. 명시적 플래그로 대체한다.
        self.online_mode = online_mode

        self.train_interval = [4, 8]
        self.test_interval = 6

        try:
            mmcv.use_backend('turbojpeg')
        except ImportError:
            mmcv.use_backend('cv2')

    def load_offline(self, results):
        cam_types = [
            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
        ]

        if len(results['sweeps']['prev']) == 0:
            for _ in range(self.sweeps_num):
                for j in range(len(cam_types)):
                    results['img'].append(results['img'][j])
                    results['img_timestamp'].append(results['img_timestamp'][j])
                    results['filename'].append(results['filename'][j])
                    results['lidar2img'].append(np.copy(results['lidar2img'][j]))
                    if 'ego2lidar' in results:
                        results['ego2lidar'].append(results['ego2lidar'][0])
        else:
            if self.test_mode:
                interval = self.test_interval
                choices = [(k + 1) * interval - 1 for k in range(self.sweeps_num)]
            elif len(results['sweeps']['prev']) <= self.sweeps_num:
                pad_len = self.sweeps_num - len(results['sweeps']['prev'])
                choices = list(range(len(results['sweeps']['prev']))) + [len(results['sweeps']['prev']) - 1] * pad_len
            else:
                max_interval = len(results['sweeps']['prev']) // self.sweeps_num
                max_interval = min(max_interval, self.train_interval[1])
                min_interval = min(max_interval, self.train_interval[0])
                interval = np.random.randint(min_interval, max_interval + 1)
                choices = [(k + 1) * interval - 1 for k in range(self.sweeps_num)]

            for idx in sorted(list(choices)):
                sweep_idx = min(idx, len(results['sweeps']['prev']) - 1)
                sweep = results['sweeps']['prev'][sweep_idx]

                if len(sweep.keys()) < len(cam_types):
                    sweep = results['sweeps']['prev'][sweep_idx - 1]

                for sensor in cam_types:
                    results['img'].append(mmcv.imread(sweep[sensor]['data_path'], self.color_type))
                    results['img_timestamp'].append(sweep[sensor]['timestamp'] / 1e6)
                    results['filename'].append(os.path.relpath(sweep[sensor]['data_path']))
                    results['lidar2img'].append(compose_lidar2img(
                        results['ego2global_translation'],
                        results['ego2global_rotation'],
                        results['lidar2ego_translation'],
                        results['lidar2ego_rotation'],
                        sweep[sensor]['sensor2global_translation'],
                        sweep[sensor]['sensor2global_rotation'],
                        sweep[sensor]['cam_intrinsic'],
                    ))
                    if 'ego2lidar' in results:
                        results['ego2lidar'].append(results['ego2lidar'][0])

        return results

    def load_online(self, results):
        # only used when measuring FPS
        assert self.test_mode
        assert self.test_interval % 6 == 0

        cam_types = [
            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
        ]

        if len(results['sweeps']['prev']) == 0:
            for _ in range(self.sweeps_num):
                for j in range(len(cam_types)):
                    results['img_timestamp'].append(results['img_timestamp'][j])
                    results['filename'].append(results['filename'][j])
                    results['lidar2img'].append(np.copy(results['lidar2img'][j]))
                    if 'ego2lidar' in results:
                        results['ego2lidar'].append(results['ego2lidar'][0])
        else:
            interval = self.test_interval
            choices = [(k + 1) * interval - 1 for k in range(self.sweeps_num)]

            for idx in sorted(list(choices)):
                sweep_idx = min(idx, len(results['sweeps']['prev']) - 1)
                sweep = results['sweeps']['prev'][sweep_idx]

                if len(sweep.keys()) < len(cam_types):
                    sweep = results['sweeps']['prev'][sweep_idx - 1]

                for sensor in cam_types:
                    results['img_timestamp'].append(sweep[sensor]['timestamp'] / 1e6)
                    results['filename'].append(os.path.relpath(sweep[sensor]['data_path']))
                    results['lidar2img'].append(compose_lidar2img(
                        results['ego2global_translation'],
                        results['ego2global_rotation'],
                        results['lidar2ego_translation'],
                        results['lidar2ego_rotation'],
                        sweep[sensor]['sensor2global_translation'],
                        sweep[sensor]['sensor2global_rotation'],
                        sweep[sensor]['cam_intrinsic'],
                    ))
                    if 'ego2lidar' in results:
                        results['ego2lidar'].append(results['ego2lidar'][0])

        return results

    def __call__(self, results):
        if self.sweeps_num == 0:
            return results

        if self.online_mode and self.test_mode:
            # FPS 측정 전용: 이미지를 실제로 로드하지 않고 메타데이터만 처리.
            # 일반 추론(1-GPU 포함)에서는 online_mode=False(기본값)를 사용한다.
            return self.load_online(results)
        else:
            return self.load_offline(results)


@PIPELINES.register_module()
class LoadOccGTFromFile(object):
    def __init__(self, num_classes=18, inst_class_ids=[], load_mask_camera=False):
        self.num_classes = num_classes
        self.inst_class_ids = inst_class_ids
        self.load_mask_camera = load_mask_camera

    def __call__(self, results):
        occ_labels = np.load(results['occ_path'])
        semantics = occ_labels['semantics']  # [200, 200, 16]

        # instance GT
        if 'instances' in occ_labels.keys():
            instances = occ_labels['instances']
            instance_class_ids = [self.num_classes - 1]  # the 0-th class is always free class
            for i in range(1, instances.max() + 1):
                class_id = np.unique(semantics[instances == i])
                assert class_id.shape[0] == 1, "each instance must belong to only one class"
                instance_class_ids.append(class_id[0])
            instance_class_ids = np.array(instance_class_ids)
        else:
            instances = None
            instance_class_ids = None

        instance_count = 0
        final_instance_class_ids = []
        final_instances = np.ones_like(semantics) * 255  # empty space has instance id "255"

        for class_id in range(self.num_classes - 1):
            if np.sum(semantics == class_id) == 0:
                continue

            if class_id in self.inst_class_ids:
                assert instances is not None, 'instance annotation not found'
                for instance_id in range(len(instance_class_ids)):
                    if instance_class_ids[instance_id] != class_id:
                        continue
                    final_instances[instances == instance_id] = instance_count
                    instance_count += 1
                    final_instance_class_ids.append(class_id)
            else:
                final_instances[semantics == class_id] = instance_count
                instance_count += 1
                final_instance_class_ids.append(class_id)

        results['voxel_semantics'] = semantics
        results['voxel_instances'] = final_instances
        results['instance_class_ids'] = DC(to_tensor(final_instance_class_ids))

        # camera mask 로딩 (선택적)
        if self.load_mask_camera:
            if 'mask_camera' in occ_labels.files:
                results['mask_camera'] = occ_labels['mask_camera'].astype(bool)
            else:
                # mask_camera가 없으면 전체 True (모든 voxel 포함)
                results['mask_camera'] = np.ones(semantics.shape, dtype=bool)

        return results
