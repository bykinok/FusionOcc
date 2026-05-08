"""
SparseOcc_ori Compatibility Layer
===================================
구버전 OpenMMLab (mmcv 1.x / mmdet 2.x / mmdet3d 0.x) API를
새 버전 (mmengine / mmdet 3.x / mmdet3d 1.x) 에 맞게 매핑하는 호환성 레이어.

원본 코드의 로직을 최대한 그대로 유지하면서, import 시점에만 호환성을 제공한다.
"""

import math
import warnings
import functools
from contextlib import contextmanager
from typing import Optional, Callable

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# mmengine 기반 import
# ---------------------------------------------------------------------------
try:
    from mmengine.model import BaseModule  # noqa: F401
    from mmengine.dist import get_dist_info  # noqa: F401
    _HAS_MMENGINE = True
except ImportError:
    from torch.nn import Module as BaseModule  # type: ignore[assignment]
    def get_dist_info():
        return 0, 1
    _HAS_MMENGINE = False

# ---------------------------------------------------------------------------
# Registry shims: 구버전 이름 → 새 mmdet3d.registry.MODELS
# ---------------------------------------------------------------------------
try:
    from mmdet3d.registry import MODELS as _MMDET3D_MODELS
    from mmdet3d.registry import DATASETS as _MMDET3D_DATASETS
    from mmdet3d.registry import TRANSFORMS as _MMDET3D_TRANSFORMS
    from mmdet3d.registry import TASK_UTILS as _MMDET3D_TASK_UTILS
    _HAS_MMDET3D = True
except ImportError:
    _MMDET3D_MODELS = None
    _MMDET3D_DATASETS = None
    _MMDET3D_TRANSFORMS = None
    _MMDET3D_TASK_UTILS = None
    _HAS_MMDET3D = False

try:
    import mmdet.models  # noqa: F401
    from mmdet.registry import MODELS as _MMDET_MODELS
    _HAS_MMDET = True
except ImportError:
    _MMDET_MODELS = None
    _HAS_MMDET = False


class _RegistryShim:
    """새 mmdet3d.registry.MODELS 를 구버전 레지스트리 이름으로 접근하게 해주는 래퍼."""

    def __init__(self, registry, fallback=None):
        self._registry = registry
        self._fallback = fallback

    def register_module(self, name=None, force=False, module=None):
        if self._registry is not None:
            return self._registry.register_module(name=name, force=force, module=module)
        def _noop(cls):
            return cls
        if module is not None:
            return module
        return _noop

    def build(self, cfg, **kwargs):
        if self._registry is not None:
            return self._registry.build(cfg, **kwargs)
        raise RuntimeError("Registry not available")

    def get(self, key):
        if self._registry is not None:
            return self._registry.get(key)
        return None

    @property
    def _module_dict(self):
        if self._registry is not None:
            return self._registry._module_dict
        return {}


DETECTORS = _RegistryShim(_MMDET3D_MODELS)
HEADS = _RegistryShim(_MMDET3D_MODELS)
LOSSES = _RegistryShim(_MMDET3D_MODELS)
TRANSFORMER = _RegistryShim(_MMDET3D_MODELS)
DATASETS = _RegistryShim(_MMDET3D_DATASETS)
PIPELINES = _RegistryShim(_MMDET3D_TRANSFORMS)

# ---------------------------------------------------------------------------
# mmdet 모델/task_utils 누락 클래스를 mmdet3d에 등록
# ---------------------------------------------------------------------------
def _register_missing_to_mmdet3d():
    if _MMDET3D_MODELS is None:
        return

    import importlib
    _candidates = [
        ('CrossEntropyLoss', 'mmdet.models.losses.cross_entropy_loss', 'CrossEntropyLoss'),
        ('FocalLoss', 'mmdet.models.losses.focal_loss', 'FocalLoss'),
        ('L1Loss', 'mmdet.models.losses.l1_loss', 'L1Loss'),
        ('GaussianFocalLoss', 'mmdet.models.losses.gaussian_focal_loss', 'GaussianFocalLoss'),
    ]
    for reg_name, mod_path, cls_name in _candidates:
        if reg_name in _MMDET3D_MODELS._module_dict:
            continue
        try:
            mod = importlib.import_module(mod_path)
            cls = getattr(mod, cls_name)
            _MMDET3D_MODELS.register_module(name=reg_name, module=cls, force=True)
        except Exception:
            pass

_register_missing_to_mmdet3d()


def _register_mmdet_task_utils():
    if _MMDET3D_TASK_UTILS is None:
        return
    import importlib
    _task_candidates = [
        ('FocalLossCost', 'mmdet.models.task_modules.assigners.match_cost', 'FocalLossCost'),
        ('ClassificationCost', 'mmdet.models.task_modules.assigners.match_cost', 'ClassificationCost'),
        ('IoUCost', 'mmdet.models.task_modules.assigners.match_cost', 'IoUCost'),
        ('BBoxL1Cost', 'mmdet.models.task_modules.assigners.match_cost', 'BBoxL1Cost'),
    ]
    for reg_name, mod_path, cls_name in _task_candidates:
        if reg_name in _MMDET3D_TASK_UTILS._module_dict:
            continue
        try:
            mod = importlib.import_module(mod_path)
            cls = getattr(mod, cls_name)
            _MMDET3D_TASK_UTILS.register_module(name=reg_name, module=cls, force=False)
        except Exception:
            pass

_register_mmdet_task_utils()

# ---------------------------------------------------------------------------
# force_fp32 / auto_fp16 shims
# (구버전 mmcv.runner 데코레이터 호환 — 새 환경에서는 AMP를 torch.cuda.amp 로 처리)
# ---------------------------------------------------------------------------

def force_fp32(apply_to=None, out_fp32=False):
    """mmcv.runner.force_fp32 호환 데코레이터."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if apply_to:
                new_args = list(args)
                for i, arg_val in enumerate(new_args):
                    pass  # args positional 처리는 복잡, 그냥 통과
                new_kwargs = {
                    k: v.float() if k in apply_to and isinstance(v, torch.Tensor) else v
                    for k, v in kwargs.items()
                }
                return func(*new_args, **new_kwargs)
            return func(*args, **kwargs)
        return wrapper
    # force_fp32(apply_to=(...)) 형식과 @force_fp32 형식 모두 지원
    if callable(apply_to):
        # @force_fp32 (인자 없이 사용된 경우)
        f = apply_to
        apply_to = None
        return decorator(f)
    return decorator


def auto_fp16(apply_to=None, out_fp32=False):
    """mmcv.runner.auto_fp16 호환 데코레이터 — 새 환경에서는 no-op."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    if callable(apply_to):
        f = apply_to
        apply_to = None
        return decorator(f)
    return decorator


# ---------------------------------------------------------------------------
# cast_tensor_type — mmcv.runner.fp16_utils.cast_tensor_type 호환
# ---------------------------------------------------------------------------
def cast_tensor_type(inputs, src_type, dst_type):
    """재귀적으로 tensor를 dst_type으로 변환."""
    if isinstance(inputs, torch.Tensor):
        if inputs.dtype == src_type:
            return inputs.to(dst_type)
        return inputs
    elif isinstance(inputs, (list, tuple)):
        converted = [cast_tensor_type(x, src_type, dst_type) for x in inputs]
        return type(inputs)(converted)
    elif isinstance(inputs, dict):
        return {k: cast_tensor_type(v, src_type, dst_type) for k, v in inputs.items()}
    return inputs


# ---------------------------------------------------------------------------
# build_loss / build_transformer (구버전 mmdet.models.builder 호환)
# ---------------------------------------------------------------------------
def build_loss(cfg):
    """mmdet.models.builder.build_loss 호환."""
    if _MMDET3D_MODELS is not None:
        return _MMDET3D_MODELS.build(cfg)
    raise RuntimeError("mmdet3d not available")


def build_transformer(cfg):
    """mmdet.models.utils.build_transformer 호환."""
    if _MMDET3D_MODELS is not None:
        return _MMDET3D_MODELS.build(cfg)
    raise RuntimeError("mmdet3d not available")


# ---------------------------------------------------------------------------
# FFN (mmcv.cnn.bricks.transformer.FFN 호환)
# ---------------------------------------------------------------------------
try:
    from mmdet.models.utils.ffn import FFN  # mmdet 3.x
    _ffn_source = "mmdet3"
except ImportError:
    try:
        from mmcv.cnn.bricks.transformer import FFN  # mmcv 1.x
        _ffn_source = "mmcv1"
    except ImportError:
        # fallback: 간단한 자체 구현
        class FFN(nn.Module):
            def __init__(self, embed_dims=256, feedforward_channels=1024,
                         num_fcs=2, act_cfg=dict(type='ReLU'),
                         ffn_drop=0.0, dropout_layer=None,
                         add_identity=True, **kwargs):
                super().__init__()
                self.embed_dims = embed_dims
                self.feedforward_channels = feedforward_channels
                self.num_fcs = num_fcs
                self.add_identity = add_identity

                layers = []
                in_channels = embed_dims
                for _ in range(num_fcs - 1):
                    layers.append(nn.Linear(in_channels, feedforward_channels))
                    layers.append(nn.ReLU(inplace=True))
                    layers.append(nn.Dropout(ffn_drop))
                    in_channels = feedforward_channels
                layers.append(nn.Linear(in_channels, embed_dims))
                layers.append(nn.Dropout(ffn_drop))
                self.layers = nn.Sequential(*layers)
                self.norm = nn.LayerNorm(embed_dims)
                self.dropout = nn.Dropout(ffn_drop)

            def forward(self, x, identity=None):
                out = self.layers(x)
                if not self.add_identity:
                    return self.norm(out)
                if identity is None:
                    identity = x
                return self.norm(identity + self.dropout(out))

            def init_weights(self):
                for m in self.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
        _ffn_source = "fallback"


# ---------------------------------------------------------------------------
# DataContainer shim (mmcv.parallel.DataContainer 호환)
# ---------------------------------------------------------------------------
try:
    from mmcv.parallel import DataContainer as DC
except ImportError:
    class DC:
        """mmcv.parallel.DataContainer 최소 호환 shim."""
        def __init__(self, data, stack=False, padding_value=0,
                     cpu_only=False, pad_dims=2):
            self.data = data
            self.stack = stack
            self.padding_value = padding_value
            self.cpu_only = cpu_only
            self.pad_dims = pad_dims

        def __repr__(self):
            return f'DC({self.data})'


# ---------------------------------------------------------------------------
# to_tensor (mmdet.datasets.pipelines 호환)
# ---------------------------------------------------------------------------
try:
    from mmdet.datasets.pipelines import to_tensor
except ImportError:
    try:
        from mmengine.utils.misc import to_tensor  # type: ignore[no-redef]
    except ImportError:
        def to_tensor(data):
            if isinstance(data, torch.Tensor):
                return data
            elif isinstance(data, (int, float)):
                return torch.tensor(data)
            import numpy as np
            if isinstance(data, np.ndarray):
                return torch.from_numpy(data)
            return torch.tensor(data)


# ---------------------------------------------------------------------------
# reduce_mean (mmdet.core 호환)
# ---------------------------------------------------------------------------
try:
    from mmdet.utils import reduce_mean
except ImportError:
    try:
        from mmdet.core import reduce_mean
    except ImportError:
        def reduce_mean(tensor):
            """분산 환경에서 tensor의 평균을 구하는 fallback."""
            if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
                return tensor
            clone = tensor.clone()
            torch.distributed.all_reduce(clone.div_(torch.distributed.get_world_size()))
            return clone


# ---------------------------------------------------------------------------
# build_match_cost (mmdet.core.bbox 호환)
# ---------------------------------------------------------------------------
def build_match_cost(cfg):
    """mmdet.core.bbox.match_costs.build_match_cost 호환."""
    if _MMDET3D_TASK_UTILS is not None:
        try:
            return _MMDET3D_TASK_UTILS.build(cfg)
        except Exception:
            pass
    if _MMDET_MODELS is not None:
        try:
            import mmdet.models.task_modules.assigners.match_cost as mc_module
            cls_name = cfg.get('type', cfg['type'])
            cls = getattr(mc_module, cls_name)
            cfg_copy = {k: v for k, v in cfg.items() if k != 'type'}
            return cls(**cfg_copy)
        except Exception:
            pass
    raise RuntimeError(f"Cannot build match cost: {cfg}")


# ---------------------------------------------------------------------------
# MultiheadAttention (mmcv.cnn.bricks.transformer 호환)
# ---------------------------------------------------------------------------
try:
    from mmcv.cnn.bricks.transformer import MultiheadAttention
except ImportError:
    try:
        from mmdet.models.layers.transformer import MultiheadAttention
    except ImportError:
        MultiheadAttention = nn.MultiheadAttention


# ---------------------------------------------------------------------------
# bias_init_with_prob (mmcv.cnn 호환)
# ---------------------------------------------------------------------------
try:
    from mmcv.cnn import bias_init_with_prob
except ImportError:
    def bias_init_with_prob(prior_prob: float) -> float:
        """bias 초기화 값을 prior probability로부터 계산."""
        bias_init = float(-math.log((1 - prior_prob) / prior_prob))
        return bias_init


# ---------------------------------------------------------------------------
# Conv3d / ConvTranspose3d (mmcv.cnn.bricks 호환 → torch.nn)
# ---------------------------------------------------------------------------
try:
    from mmcv.cnn.bricks import Conv3d, ConvTranspose3d
except ImportError:
    from torch.nn import Conv3d, ConvTranspose3d


# ---------------------------------------------------------------------------
# NuScenesDataset 구버전 API 호환 베이스 (loaders에서 사용)
# 새 mmdet3d NuScenesDataset은 API가 달라서 standalone 구현을 사용
# ---------------------------------------------------------------------------
class LegacyNuScenesDataset:
    """구버전 mmdet3d NuScenesDataset의 핵심 기능을 재구현한 standalone 클래스.

    새 mmdet3d 1.x의 NuScenesDataset은 Det3DDataset 기반으로 API가 크게 변경되어
    원본 NuSceneOcc와 직접 호환되지 않는다.
    이 클래스는 원본 코드가 사용하는 핵심 메서드들을 제공한다.
    """

    CLASSES = None

    def __init__(self,
                 ann_file,
                 pipeline,
                 data_root,
                 classes=None,
                 modality=None,
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 test_mode=False,
                 use_valid_flag=False,
                 **kwargs):
        import pickle
        import os

        self.ann_file = ann_file
        self.data_root = data_root
        self.classes = classes
        self.CLASSES = classes
        self.modality = modality if modality is not None else dict(
            use_lidar=False, use_camera=True, use_radar=False,
            use_map=False, use_external=False)
        self.filter_empty_gt = filter_empty_gt
        self.test_mode = test_mode
        self.use_valid_flag = use_valid_flag
        self.box_type_3d = box_type_3d

        # 파이프라인 구성
        self.pipeline = self._build_pipeline(pipeline)

        # 어노테이션 로드
        self.data_infos = self.load_annotations(ann_file)

        # flag 설정 (GroupSampler 호환)
        self.flag = self._get_flag()

    def _build_pipeline(self, pipeline_cfg):
        from mmdet3d.registry import TRANSFORMS
        from mmengine.registry import TRANSFORMS as ENGINE_TRANSFORMS

        composed = []
        for p in pipeline_cfg:
            p = dict(p)
            t_type = p.pop('type')
            # 먼저 mmdet3d, 그 다음 mmengine에서 찾기
            cls = TRANSFORMS.get(t_type)
            if cls is None:
                cls = ENGINE_TRANSFORMS.get(t_type)
            if cls is None:
                raise ValueError(f"Transform '{t_type}' not found in registry.")
            composed.append(cls(**p))
        return composed

    def _call_pipeline(self, results):
        for transform in self.pipeline:
            results = transform(results)
            if results is None:
                return None
        return results

    def load_annotations(self, ann_file):
        import pickle
        import os
        # ann_file이 절대경로가 아니고, data_root를 포함하지 않은 경우에만 data_root를 붙임
        # (설정에서 ann_file이 이미 'data/nuscenes/xxx.pkl' 형태인 경우 이중 경로 방지)
        if not os.path.isabs(ann_file) and not os.path.exists(ann_file):
            candidate = os.path.join(self.data_root, ann_file)
            # 이미 data_root로 시작하는 경우엔 그대로 사용
            data_root_norm = self.data_root.rstrip('/')
            ann_norm = ann_file.replace('\\', '/')
            if not ann_norm.startswith(data_root_norm):
                ann_file = candidate

        with open(ann_file, 'rb') as f:
            data = pickle.load(f)

        # 구버전 pkl: {'infos': [...], 'metadata': {...}}
        # 또는 새버전 pkl: {'data_list': [...], 'metainfo': {...}}
        if isinstance(data, dict):
            if 'infos' in data:
                # 구버전 mmdet3d NuScenesDataset.load_annotations()와 동일하게
                # timestamp 순으로 정렬 (원본 동작 재현)
                return list(sorted(data['infos'], key=lambda e: e['timestamp']))
            elif 'data_list' in data:
                return data['data_list']
            else:
                return list(data.values())[0]
        elif isinstance(data, list):
            return data
        return []

    def get_ann_info(self, index):
        """3D annotation 정보 반환 (detection용, occ에서는 거의 사용 안 함)."""
        info = self.data_infos[index]
        return self._parse_ann_info(info)

    def _parse_ann_info(self, info):
        """기본 ann_info 파싱."""
        ann_info = dict(
            gt_bboxes_3d=[],
            gt_labels_3d=[],
        )
        return ann_info

    def _get_flag(self):
        """DistributedGroupSampler를 위한 flag 배열 생성."""
        import numpy as np
        return np.zeros(len(self.data_infos), dtype=np.uint8)

    def get_data_info(self, index):
        raise NotImplementedError

    def prepare_train_data(self, index):
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        return self._call_pipeline(input_dict)

    def prepare_test_data(self, index):
        input_dict = self.get_data_info(index)
        return self._call_pipeline(input_dict)

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        if self.test_mode:
            data = self.prepare_test_data(idx)
            if data is None:
                raise Exception(f"Test time pipeline failed for idx {idx}")
            return data

        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def _rand_another(self, idx):
        import numpy as np
        pool = [i for i in range(len(self)) if i != idx]
        return np.random.choice(pool)

    def evaluate(self, results, **kwargs):
        raise NotImplementedError

    def format_results(self, results, **kwargs):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 구버전 mmdet3d Transform 호환 클래스들 (새 버전에서 제거됨)
# mmdet3d.registry.TRANSFORMS에 등록
# ---------------------------------------------------------------------------
try:
    from mmdet3d.registry import TRANSFORMS as _MMDET3D_TRANSFORMS_REG

    @_MMDET3D_TRANSFORMS_REG.register_module()
    class DefaultFormatBundle3D:
        """구버전 mmdet3d DefaultFormatBundle3D 호환 클래스.

        신버전 mmdet3d에서 Pack3DDetInputs으로 대체되었으나,
        기존 dict 기반 결과 포맷 및 class_names 파라미터를 유지한다.
        다중 뷰 이미지를 [N, C, H, W] 텐서로 변환하고 DC(stack=True)로 래핑한다.
        """

        def __init__(self, class_names=None, with_label=True, **kwargs):
            self.class_names = class_names
            self.with_label = with_label

        def __call__(self, results):
            import numpy as np

            if 'img' in results:
                imgs = results['img']
                if isinstance(imgs, list):
                    # 각 이미지 [H, W, C] → [C, H, W] 로 transpose 후 스택
                    imgs_transposed = []
                    for img in imgs:
                        if isinstance(img, np.ndarray):
                            imgs_transposed.append(np.ascontiguousarray(img.transpose(2, 0, 1)))
                        elif hasattr(img, 'permute'):
                            imgs_transposed.append(img)
                        else:
                            imgs_transposed.append(img)
                    try:
                        imgs_arr = np.stack(imgs_transposed, axis=0)  # [N, C, H, W]
                        results['img'] = DC(to_tensor(imgs_arr), stack=True)
                    except Exception:
                        results['img'] = DC(imgs_transposed, stack=False)
                elif isinstance(imgs, np.ndarray) and imgs.ndim == 4:
                    results['img'] = DC(to_tensor(imgs), stack=True)

            return results

        def __repr__(self):
            return (f'{self.__class__.__name__}('
                    f'class_names={self.class_names}, '
                    f'with_label={self.with_label})')

    @_MMDET3D_TRANSFORMS_REG.register_module()
    class Collect3D:
        """구버전 mmdet3d Collect3D 호환 클래스.

        지정된 keys만 결과 dict에 남기고, meta_keys는 img_metas에 묶는다.
        """

        def __init__(self,
                     keys,
                     meta_keys=('filename', 'ori_shape', 'img_shape',
                                'lidar2img', 'depth2img', 'cam2img',
                                'pad_shape', 'scale_factor', 'flip',
                                'pcd_horizontal_flip', 'pcd_vertical_flip',
                                'box_mode_3d', 'box_type_3d', 'img_norm_cfg',
                                'pcd_trans', 'sample_idx', 'pcd_scale_factor',
                                'pcd_rotation', 'pcd_rotation_angle',
                                'pts_filename', 'transformation_3d_flow',
                                'trans_mat', 'affine_aug', 'occ_size',
                                'pc_range', 'scene_token', 'lidar_token',
                                'img_timestamp', 'ego2lidar')):
            self.keys = keys
            self.meta_keys = meta_keys

        def __call__(self, results):
            data = {}
            img_metas = {}
            for key in self.meta_keys:
                if key in results:
                    img_metas[key] = results[key]
            data['img_metas'] = DC(img_metas, cpu_only=True)
            for key in self.keys:
                data[key] = results[key]
            return data

        def __repr__(self):
            return (f'{self.__class__.__name__}('
                    f'keys={self.keys}, meta_keys={self.meta_keys})')

except Exception:
    pass


# ---------------------------------------------------------------------------
# 구버전 mmdet3d Sampler 호환 클래스 (새 버전 mmengine DefaultSampler로 매핑)
# mmdet3d.registry.DATA_SAMPLERS에 등록
# ---------------------------------------------------------------------------
try:
    from mmdet3d.registry import DATA_SAMPLERS as _MMDET3D_DATA_SAMPLERS
    from mmengine.dataset import DefaultSampler

    @_MMDET3D_DATA_SAMPLERS.register_module()
    class DistributedSampler(DefaultSampler):
        """구버전 mmdet3d DistributedSampler 호환 클래스.

        mmengine DefaultSampler를 래핑하여 기존 설정과 호환성 제공.
        """
        def __init__(self, dataset, shuffle=True, seed=0, round_up=True, **kwargs):
            super().__init__(dataset=dataset, shuffle=shuffle, seed=seed, round_up=round_up)

    @_MMDET3D_DATA_SAMPLERS.register_module()
    class DistributedGroupSampler(DefaultSampler):
        """구버전 mmdet3d DistributedGroupSampler 호환 클래스.

        그룹별 샘플링은 무시하고 DefaultSampler(shuffle=True)로 동작.
        """
        def __init__(self, dataset, samples_per_gpu=1, shuffle=True, seed=0, **kwargs):
            super().__init__(dataset=dataset, shuffle=shuffle, seed=seed)

except Exception:
    pass
