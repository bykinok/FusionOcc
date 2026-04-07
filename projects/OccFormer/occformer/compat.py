"""
OccFormer Compatibility Layer
=============================
구버전 OpenMMLab (mmcv 1.x / mmdet 2.x / mmdet3d 0.x) API를
새 버전 (mmengine / mmdet 3.x / mmdet3d 1.x) 에 맞게 매핑하는 호환성 레이어.

SparseOcc 프로젝트의 compat.py를 기반으로 OccFormer에 필요한
추가 shim을 포함합니다.
"""

# ---------------------------------------------------------------------------
# Registry shims
# ---------------------------------------------------------------------------
_MMDET3D_MODELS = _MMDET3D_DATASETS = _MMDET3D_TRANSFORMS = _MMDET3D_TASK_UTILS = None
_ENGINE_MODELS = _ENGINE_DATASETS = _ENGINE_TRANSFORMS = None
_MMDET_MODELS = _MMDET_TASK_UTILS = None

try:
    from mmdet3d.registry import MODELS as _MMDET3D_MODELS
    from mmdet3d.registry import DATASETS as _MMDET3D_DATASETS
    from mmdet3d.registry import TRANSFORMS as _MMDET3D_TRANSFORMS
    from mmdet3d.registry import TASK_UTILS as _MMDET3D_TASK_UTILS
    _HAS_MMDET3D = True
except ImportError:
    _HAS_MMDET3D = False

try:
    from mmengine.registry import MODELS as _ENGINE_MODELS
    from mmengine.registry import DATASETS as _ENGINE_DATASETS
    from mmengine.registry import TRANSFORMS as _ENGINE_TRANSFORMS
    _HAS_MMENGINE = True
except ImportError:
    _HAS_MMENGINE = False

try:
    from mmdet.registry import MODELS as _MMDET_MODELS
    from mmdet.registry import TASK_UTILS as _MMDET_TASK_UTILS
    import mmdet.models  # noqa: F401
    _HAS_MMDET = True
except ImportError:
    _HAS_MMDET = False

# ---------------------------------------------------------------------------
# DetrTransformerDecoder 구버전 API 호환 래퍼
# mmdet 2.x:  transformerlayers=dict(attn_cfgs=..., feedforward_channels=..., operation_order=...)
# mmdet 3.x:  layer_cfg=dict(self_attn_cfg=..., cross_attn_cfg=..., ffn_cfg=...)
# ---------------------------------------------------------------------------
def _build_legacy_detr_layer_cfg(transformerlayers_dict):
    """구버전 transformerlayers dict를 신버전 layer_cfg dict로 변환."""
    import copy
    td = copy.deepcopy(transformerlayers_dict) if transformerlayers_dict else {}
    td.pop('type', None)

    attn_cfgs = td.pop('attn_cfgs', {})
    if isinstance(attn_cfgs, (list, tuple)):
        attn_cfgs = attn_cfgs[0] if attn_cfgs else {}
    attn_cfgs = dict(attn_cfgs)
    attn_cfgs.pop('type', None)
    attn_cfgs['batch_first'] = True

    feedforward_channels = td.pop('feedforward_channels', 2048)
    ffn_dropout = td.pop('ffn_dropout', 0.0)
    td.pop('operation_order', None)
    td.pop('ffn_drop', None)

    embed_dims = attn_cfgs.get('embed_dims', 256)
    layer_cfg = dict(
        self_attn_cfg=dict(**attn_cfgs),
        cross_attn_cfg=dict(**attn_cfgs),
        ffn_cfg=dict(embed_dims=embed_dims, feedforward_channels=feedforward_channels, ffn_drop=ffn_dropout),
        norm_cfg=dict(type='LN'),
    )
    return layer_cfg


def _register_legacy_detr_wrappers():
    """구버전 transformerlayers API를 허용하는 래퍼 클래스를 mmdet3d에 등록."""
    try:
        from mmdet.models.layers.transformer.detr_layers import (
            DetrTransformerDecoder as _NewDecoder)

        class LegacyDetrTransformerDecoder(_NewDecoder):
            """mmdet 2.x 스타일의 `transformerlayers` 파라미터를 수용하는 래퍼."""
            def __init__(self, transformerlayers=None, layer_cfg=None, **kwargs):
                if layer_cfg is None and transformerlayers is not None:
                    layer_cfg = _build_legacy_detr_layer_cfg(transformerlayers)
                super().__init__(layer_cfg=layer_cfg, **kwargs)

        if _MMDET3D_MODELS is not None:
            _MMDET3D_MODELS.register_module(
                name='DetrTransformerDecoder',
                module=LegacyDetrTransformerDecoder,
                force=True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# mmdet3d 레지스트리에 미등록된 mmdet transformer 클래스 수동 등록
# ---------------------------------------------------------------------------
def _register_missing_to_mmdet3d():
    if _MMDET3D_MODELS is None:
        return

    _candidates = [
        ('DetrTransformerDecoder', 'mmdet.models.layers.transformer.detr_layers', 'DetrTransformerDecoder'),
        ('DetrTransformerDecoderLayer', 'mmdet.models.layers.transformer.detr_layers', 'DetrTransformerDecoderLayer'),
        ('DetrTransformerEncoder', 'mmdet.models.layers.transformer.detr_layers', 'DetrTransformerEncoder'),
        ('DetrTransformerEncoderLayer', 'mmdet.models.layers.transformer.detr_layers', 'DetrTransformerEncoderLayer'),
        ('ConditionalDetrTransformerDecoder', 'mmdet.models.layers.transformer.conditional_detr_layers', 'ConditionalDetrTransformerDecoder'),
        ('ConditionalDetrTransformerDecoderLayer', 'mmdet.models.layers.transformer.conditional_detr_layers', 'ConditionalDetrTransformerDecoderLayer'),
        ('CrossEntropyLoss', 'mmdet.models.losses.cross_entropy_loss', 'CrossEntropyLoss'),
        ('FocalLoss', 'mmdet.models.losses.focal_loss', 'FocalLoss'),
        ('L1Loss', 'mmdet.models.losses.l1_loss', 'L1Loss'),
    ]

    _register_legacy_detr_wrappers()

    import importlib
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
    """mmdet TASK_UTILS의 미등록 match cost 클래스를 mmdet3d TASK_UTILS에 등록."""
    if _MMDET3D_TASK_UTILS is None:
        return

    import importlib
    _task_candidates = [
        ('ClassificationCost', 'mmdet.models.task_modules.assigners.match_cost', 'ClassificationCost'),
        ('FocalLossCost', 'mmdet.models.task_modules.assigners.match_cost', 'FocalLossCost'),
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


class _RegistryShim:
    """새 mmdet3d.registry.MODELS 를 구버전 레지스트리 이름으로 접근할 수 있게 해 주는 래퍼."""

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
        raise RuntimeError("Registry not available (mmdet3d not installed)")

    def get(self, key):
        if self._registry is not None:
            return self._registry.get(key)
        return None


_models_registry = _MMDET3D_MODELS if _HAS_MMDET3D else (
    _ENGINE_MODELS if _HAS_MMENGINE else None)
_datasets_registry = _MMDET3D_DATASETS if _HAS_MMDET3D else (
    _ENGINE_DATASETS if _HAS_MMENGINE else None)
_transforms_registry = _MMDET3D_TRANSFORMS if _HAS_MMDET3D else (
    _ENGINE_TRANSFORMS if _HAS_MMENGINE else None)
_task_utils_registry = _MMDET3D_TASK_UTILS if _HAS_MMDET3D else (
    _MMDET_TASK_UTILS if _HAS_MMDET else None)

DETECTORS = _RegistryShim(_models_registry)
BACKBONES = _RegistryShim(_models_registry)
NECKS = _RegistryShim(_models_registry)
HEADS = _RegistryShim(_models_registry)
LOSSES = _RegistryShim(_models_registry)
DATASETS = _RegistryShim(_datasets_registry)
PIPELINES = _RegistryShim(_transforms_registry)  # 구버전 PIPELINES → TRANSFORMS
POSITIONAL_ENCODING = _RegistryShim(_models_registry)
BBOX_ASSIGNERS = _RegistryShim(_task_utils_registry)
BBOX_SAMPLERS = _RegistryShim(_task_utils_registry)
MATCH_COST = _RegistryShim(_task_utils_registry)

# ---------------------------------------------------------------------------
# Runner / FP16 shims  (mmcv.runner → mmengine)
# ---------------------------------------------------------------------------

def force_fp32(apply_to=None, out_fp16=False):
    """mmcv.runner.force_fp32 no-op 호환 데코레이터."""
    def decorator(func):
        return func
    if callable(apply_to):
        _func = apply_to
        return _func
    return decorator


def auto_fp16(apply_to=None, out_fp32=False):
    """mmcv.runner.auto_fp16 no-op 호환 데코레이터."""
    def decorator(func):
        return func
    if callable(apply_to):
        return apply_to
    return decorator


# BaseModule / ModuleList / Sequential
try:
    from mmengine.model import BaseModule, ModuleList, Sequential
except ImportError:
    try:
        from mmcv.runner import BaseModule, ModuleList, Sequential
    except ImportError:
        import torch.nn as nn
        BaseModule = nn.Module
        ModuleList = nn.ModuleList
        Sequential = nn.Sequential

# ---------------------------------------------------------------------------
# Builder shim  (mmdet3d.models.builder → MODELS.build)
# ---------------------------------------------------------------------------

def _multi_registry_build(cfg):
    """mmdet3d, mmdet, mmengine 레지스트리를 순서대로 시도하는 build 함수."""
    if cfg is None:
        return None
    import copy
    cfg_copy = copy.deepcopy(cfg) if isinstance(cfg, dict) else cfg.to_dict()
    obj_type = cfg_copy.pop('type')

    for reg in [_models_registry, _MMDET_MODELS]:
        if reg is None:
            continue
        obj_cls = reg._module_dict.get(obj_type)
        if obj_cls is not None:
            return obj_cls(**cfg_copy)

    raise RuntimeError(
        f"Cannot build '{obj_type}': not found in any registry "
        f"(mmdet3d, mmdet). Check the type name or registration.")


class _BuilderShim:
    """mmdet3d.models.builder 의 build_* 함수들을 새 API로 제공."""

    def _build(self, cfg):
        return _multi_registry_build(cfg)

    def build_backbone(self, cfg):
        return self._build(cfg)

    def build_neck(self, cfg):
        return self._build(cfg)

    def build_head(self, cfg):
        return self._build(cfg)

    def build_loss(self, cfg):
        return self._build(cfg)

    def build_roi_extractor(self, cfg):
        return self._build(cfg)

    def build_shared_head(self, cfg):
        return self._build(cfg)

    def build_detector(self, cfg, train_cfg=None, test_cfg=None):
        return self._build(cfg)


builder = _BuilderShim()


def build_loss(cfg):
    return builder.build_loss(cfg)


def build_match_cost(cfg):
    if _task_utils_registry is not None:
        return _task_utils_registry.build(cfg)
    raise RuntimeError("TASK_UTILS registry not available")


# ---------------------------------------------------------------------------
# mmdet.core 호환 함수들
# ---------------------------------------------------------------------------

try:
    from mmengine.utils import is_list_of
except ImportError:
    def is_list_of(obj, expected_type):
        return isinstance(obj, list) and all(isinstance(x, expected_type) for x in obj)

try:
    from mmdet.utils import reduce_mean
except ImportError:
    import torch
    def reduce_mean(tensor):
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(tensor)
                tensor.div_(dist.get_world_size())
        except Exception:
            pass
        return tensor

try:
    from mmdet.utils import multi_apply
except ImportError:
    try:
        from mmengine.utils import multi_apply
    except ImportError:
        def multi_apply(func, *args, **kwargs):
            pfunc = func
            map_results = map(lambda x: pfunc(*x, **kwargs) if isinstance(x, tuple) else pfunc(x, **kwargs),
                              zip(*args) if len(args) > 1 else args[0])
            results = tuple(map(list, zip(*list(map_results))))
            return results

try:
    from mmdet.models.utils import build_assigner
except ImportError:
    def build_assigner(cfg):
        return build_match_cost(cfg)

try:
    from mmdet.models.utils import build_sampler
except ImportError:
    def build_sampler(cfg, **kwargs):
        if _task_utils_registry is not None:
            return _task_utils_registry.build(cfg)
        raise RuntimeError("TASK_UTILS registry not available")

try:
    from mmdet.core import build_bbox_coder
except ImportError:
    def build_bbox_coder(cfg):
        raise NotImplementedError("build_bbox_coder not available")

# ---------------------------------------------------------------------------
# mmcv.cnn 호환
# ---------------------------------------------------------------------------

try:
    from mmcv.cnn import caffe2_xavier_init
except ImportError:
    try:
        from mmengine.model import caffe2_xavier_init
    except ImportError:
        import torch.nn as _nn
        def caffe2_xavier_init(module, bias=0):
            _nn.init.xavier_uniform_(module.weight, gain=1)
            if hasattr(module, 'bias') and module.bias is not None:
                _nn.init.constant_(module.bias, bias)

try:
    from mmcv.cnn.utils.weight_init import constant_init
except ImportError:
    try:
        from mmengine.model import constant_init
    except ImportError:
        import torch.nn as _nn
        def constant_init(module, val, bias=0):
            if hasattr(module, 'weight') and module.weight is not None:
                _nn.init.constant_(module.weight, val)
            if hasattr(module, 'bias') and module.bias is not None:
                _nn.init.constant_(module.bias, bias)

try:
    from mmcv.cnn.utils.weight_init import normal_init
except ImportError:
    try:
        from mmengine.model import normal_init
    except ImportError:
        import torch.nn as _nn
        def normal_init(module, mean=0, std=1, bias=0):
            if hasattr(module, 'weight') and module.weight is not None:
                _nn.init.normal_(module.weight, mean, std)
            if hasattr(module, 'bias') and module.bias is not None:
                _nn.init.constant_(module.bias, bias)

try:
    from mmcv.cnn import build_plugin_layer
except ImportError:
    def build_plugin_layer(cfg, postfix='', **kwargs):
        raise NotImplementedError("build_plugin_layer not available")

# ---------------------------------------------------------------------------
# mmdet.models.utils / mmdet.models.layers 호환
# ---------------------------------------------------------------------------

try:
    from mmdet.models.layers import SELayer
except ImportError:
    try:
        from mmdet.models.utils import SELayer
    except ImportError:
        SELayer = None

try:
    from mmdet.models.utils import make_divisible
except ImportError:
    def make_divisible(value, divisor, min_value=None, min_ratio=0.9):
        if min_value is None:
            min_value = divisor
        new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
        if new_value < min_ratio * value:
            new_value += divisor
        return new_value

# ---------------------------------------------------------------------------
# mmdet.core.bbox 호환 클래스들
# ---------------------------------------------------------------------------

try:
    from mmdet.models.task_modules.assigners import AssignResult, BaseAssigner
except ImportError:
    try:
        from mmdet.core.bbox.assigners import AssignResult, BaseAssigner
    except ImportError:
        import torch

        class AssignResult:
            def __init__(self, num_gts, gt_inds, max_overlaps, labels=None):
                self.num_gts = num_gts
                self.gt_inds = gt_inds
                self.max_overlaps = max_overlaps
                self.labels = labels

        class BaseAssigner:
            def assign(self, *args, **kwargs):
                raise NotImplementedError

try:
    from mmdet.models.task_modules.samplers import BaseSampler, SamplingResult
except ImportError:
    try:
        from mmdet.core.bbox.samplers import BaseSampler, SamplingResult
    except ImportError:
        class SamplingResult:
            pass

        class BaseSampler:
            def sample(self, *args, **kwargs):
                raise NotImplementedError

# ---------------------------------------------------------------------------
# bbox utility 함수들 (mmdet 2.x → 3.x 경로 변경)
# ---------------------------------------------------------------------------

try:
    from mmdet.structures.bbox import bbox_overlaps
except ImportError:
    try:
        from mmdet.core.bbox.iou_calculators import bbox_overlaps
    except ImportError:
        bbox_overlaps = None

try:
    from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh
except ImportError:
    try:
        from mmdet.core.bbox.transforms import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh
    except ImportError:
        import torch as _torch

        def bbox_cxcywh_to_xyxy(bbox):
            """(cx, cy, w, h) → (x1, y1, x2, y2)"""
            _cx, _cy, _w, _h = bbox.unbind(-1)
            x1 = _cx - 0.5 * _w
            y1 = _cy - 0.5 * _h
            x2 = _cx + 0.5 * _w
            y2 = _cy + 0.5 * _h
            return _torch.stack([x1, y1, x2, y2], dim=-1)

        def bbox_xyxy_to_cxcywh(bbox):
            """(x1, y1, x2, y2) → (cx, cy, w, h)"""
            x1, y1, x2, y2 = bbox.unbind(-1)
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1
            return _torch.stack([cx, cy, w, h], dim=-1)

# merge_aug_proposals, bbox_mapping_back (dense_test_mixins 에서 사용)
try:
    from mmdet.core import merge_aug_proposals, bbox_mapping_back
except ImportError:
    merge_aug_proposals = None
    bbox_mapping_back = None

# MlvlPointGenerator (anchor_free_head 에서 사용)
try:
    from mmdet.models.task_modules.prior_generators import MlvlPointGenerator
except ImportError:
    try:
        from mmdet.core.anchor.point_generator import MlvlPointGenerator
    except ImportError:
        MlvlPointGenerator = None

# ---------------------------------------------------------------------------
# CenterPoint 호환 베이스 클래스
# ---------------------------------------------------------------------------

try:
    from mmdet3d.models.detectors import Base3DDetector as _Base3DDetector
    _HAS_BASE3D = True
except ImportError:
    try:
        import torch.nn as nn
        _Base3DDetector = nn.Module
    except ImportError:
        _Base3DDetector = object
    _HAS_BASE3D = False


class CenterPoint(_Base3DDetector):
    """
    구버전 mmdet3d CenterPoint를 대체하는 호환 베이스 클래스.
    BEVDet / BEVDepth 가 이 클래스를 상속합니다.
    """

    def __init__(self,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 data_preprocessor=None,
                 **kwargs):

        if _HAS_BASE3D:
            super().__init__(
                data_preprocessor=data_preprocessor,
                init_cfg=init_cfg)
        else:
            super().__init__()

        if img_backbone is not None:
            self.img_backbone = _multi_registry_build(img_backbone)

        if img_neck is not None:
            self.img_neck = _multi_registry_build(img_neck)

        if pts_bbox_head is not None:
            pts_train_cfg = train_cfg.pts if train_cfg else None
            pts_bbox_head.update(train_cfg=pts_train_cfg)
            pts_test_cfg = test_cfg.pts if test_cfg else None
            pts_bbox_head.update(test_cfg=pts_test_cfg)
            self.pts_bbox_head = _multi_registry_build(pts_bbox_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    @property
    def with_img_neck(self):
        return hasattr(self, 'img_neck') and self.img_neck is not None

    @property
    def with_pts_bbox(self):
        return hasattr(self, 'pts_bbox_head') and self.pts_bbox_head is not None

    @property
    def with_img_backbone(self):
        return hasattr(self, 'img_backbone') and self.img_backbone is not None

    @property
    def with_pts_backbone(self):
        return hasattr(self, 'pts_backbone') and self.pts_backbone is not None

    def extract_feat(self, points, img, img_metas):
        raise NotImplementedError

    def forward_pts_train(self, pts_feats, gt_bboxes_3d, gt_labels_3d,
                          img_metas, gt_bboxes_ignore=None):
        outs = self.pts_bbox_head(pts_feats, img_metas)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def simple_test_pts(self, x, img_metas, rescale=False):
        outs = self.pts_bbox_head(x, img_metas)
        bbox_list = self.pts_bbox_head.get_bboxes(outs, img_metas, rescale=rescale)
        bbox_results = [
            dict(boxes_3d=det_bboxes, scores_3d=det_scores, labels_3d=det_labels)
            for det_bboxes, det_scores, det_labels in bbox_list
        ]
        return bbox_results

    # ------------------------------------------------------------------
    # mmengine BaseModel / Base3DDetector 추상 메서드 — 구버전 API 브리지
    # ------------------------------------------------------------------

    @staticmethod
    def _unpack_dc_batch(data_batch):
        """pseudo_collate로 묶인 list-of-dict 배치를 구버전 API용 인수로 변환."""
        import torch

        def _unwrap(x):
            if hasattr(x, 'data'):
                return x.data
            return x

        keys = data_batch[0].keys()
        collated = {}
        for key in keys:
            vals = [_unwrap(sample[key]) for sample in data_batch]
            first = vals[0]

            if isinstance(first, torch.Tensor):
                try:
                    collated[key] = torch.stack(vals, dim=0)
                except Exception:
                    collated[key] = vals
            elif isinstance(first, list):
                if len(vals) == 1:
                    collated[key] = vals[0]
                else:
                    collated[key] = vals
            elif isinstance(first, dict):
                collated[key] = vals
            else:
                collated[key] = vals

        return collated

    def train_step(self, data, optim_wrapper):
        """mmengine Runner의 train_step — 구버전 forward_train으로 브리지."""
        with optim_wrapper.optim_context(self):
            if isinstance(data, (list, tuple)):
                kwargs = self._unpack_dc_batch(data)
            else:
                kwargs = data
            losses = self.forward_train(**kwargs)
        parsed_losses, log_vars = self.parse_losses(losses)
        optim_wrapper.update_params(parsed_losses)
        return log_vars

    def val_step(self, data):
        """mmengine Runner의 val_step — 구버전 forward_test로 브리지."""
        if isinstance(data, (list, tuple)):
            kwargs = self._unpack_dc_batch(data)
        else:
            kwargs = data
        return self.forward_test(**kwargs)

    def forward(self, mode='loss', **kwargs):
        """mmengine DDP 호환 forward 메서드."""
        unwrapped = {}
        for key, val in kwargs.items():
            if isinstance(val, list):
                unwrapped[key] = [v.data if hasattr(v, 'data') else v
                                  for v in val]
            elif hasattr(val, 'data'):
                unwrapped[key] = val.data
            else:
                unwrapped[key] = val

        if mode == 'loss':
            return self.forward_train(**unwrapped)
        elif mode in ('predict', 'tensor'):
            return self.forward_test(**unwrapped)
        else:
            raise RuntimeError(f'Unknown forward mode: {mode}')

    def loss(self, batch_inputs_dict, batch_data_samples, **kwargs):
        raise NotImplementedError(
            "loss()는 직접 호출되지 않아야 합니다. train_step()을 통해 호출하세요.")

    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
        raise NotImplementedError(
            "predict()는 직접 호출되지 않아야 합니다. val_step()을 통해 호출하세요.")

    def _forward(self, batch_inputs_dict, batch_data_samples=None, **kwargs):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# DataContainer shim  (mmcv.parallel.DataContainer → 간단 래퍼)
# ---------------------------------------------------------------------------

try:
    from mmcv.parallel import DataContainer as DC
except ImportError:
    class DC:
        """mmcv.parallel.DataContainer 호환 클래스 (최소 구현)."""
        def __init__(self, data, stack=False, padding_value=0, cpu_only=False,
                     pad_dims=2):
            self.data = data
            self.stack = stack
            self.padding_value = padding_value
            self.cpu_only = cpu_only
            self.pad_dims = pad_dims

        def __repr__(self):
            return f'DC({self.data})'


# ---------------------------------------------------------------------------
# mmdet.datasets.pipelines.to_tensor 호환
# ---------------------------------------------------------------------------

try:
    from mmdet.datasets.pipelines import to_tensor
except ImportError:
    try:
        from mmdet.structures.utils import to_tensor
    except ImportError:
        import torch
        import numpy as np

        def to_tensor(data):
            if isinstance(data, torch.Tensor):
                return data
            elif isinstance(data, np.ndarray):
                return torch.from_numpy(data)
            elif isinstance(data, (list, tuple)):
                return torch.tensor(data)
            else:
                return torch.tensor(data)


# ---------------------------------------------------------------------------
# DefaultFormatBundle3D 호환
# ---------------------------------------------------------------------------

class DefaultFormatBundle3D:
    """구버전 mmdet3d DefaultFormatBundle3D 호환 클래스."""

    def __init__(self, class_names=None, with_label=True, **kwargs):
        self.class_names = class_names
        self.with_label = with_label

    def __call__(self, results):
        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'class_names={self.class_names}, '
                f'with_label={self.with_label})')


# ---------------------------------------------------------------------------
# mmcv.cnn 호환
# ---------------------------------------------------------------------------

try:
    from mmcv.cnn import build_conv_layer, build_norm_layer, ConvModule
    from mmcv.cnn import Conv2d, Conv3d
except ImportError:
    build_conv_layer = None
    build_norm_layer = None

try:
    from mmcv.cnn.bricks.transformer import (
        build_positional_encoding,
        build_transformer_layer_sequence,
        build_transformer_layer,
        POSITIONAL_ENCODING as _PE_REGISTRY
    )
    _HAS_MMCV_TRANSFORMER = True
except ImportError:
    _HAS_MMCV_TRANSFORMER = False
    build_positional_encoding = None
    build_transformer_layer_sequence = None
    build_transformer_layer = None

# ---------------------------------------------------------------------------
# mmdet3d.core.bbox 호환
# ---------------------------------------------------------------------------

try:
    from mmdet3d.structures.bbox_3d import LiDARInstance3DBoxes
except ImportError:
    try:
        from mmdet3d.core.bbox import LiDARInstance3DBoxes
    except ImportError:
        LiDARInstance3DBoxes = None


# ---------------------------------------------------------------------------
# get_dist_info 호환
# ---------------------------------------------------------------------------

try:
    from mmengine.dist import get_dist_info
except ImportError:
    try:
        from mmcv.runner import get_dist_info
    except ImportError:
        def get_dist_info():
            return 0, 1


# ---------------------------------------------------------------------------
# NuScenes / SemanticKITTI Dataset 호환
# ---------------------------------------------------------------------------

try:
    from mmdet3d.datasets import NuScenesDataset
except ImportError:
    NuScenesDataset = None

try:
    from mmdet3d.datasets import SemanticKITTIDataset
except ImportError:
    SemanticKITTIDataset = None

# ---------------------------------------------------------------------------
# mmcv collate / Registry 호환 (datasets/builder.py 에서 사용)
# ---------------------------------------------------------------------------

try:
    from mmcv.parallel import collate
except ImportError:
    try:
        from mmengine.utils import collate
    except ImportError:
        collate = None

try:
    from mmcv.utils import Registry, build_from_cfg
except ImportError:
    try:
        from mmengine.utils import Registry
        build_from_cfg = None
    except ImportError:
        Registry = None
        build_from_cfg = None

# ---------------------------------------------------------------------------
# DistributedGroupSampler / DistributedSampler 조기 등록
# ---------------------------------------------------------------------------

try:
    import math as _math
    import numpy as _np_sampler
    import torch as _torch_sampler
    from torch.utils.data import (DistributedSampler as _TorchDistSampler,
                                   Sampler as _TorchSampler)
    from mmdet3d.registry import DATA_SAMPLERS as _DATA_SAMPLERS

    @_DATA_SAMPLERS.register_module(force=True)
    class DistributedGroupSampler(_TorchSampler):
        """그룹 단위 분산 Sampler (학습용). dataset.flag 기반."""

        def __init__(self, dataset, samples_per_gpu=1, num_replicas=None,
                     rank=None, seed=0):
            _rank, _num_replicas = get_dist_info()
            if num_replicas is None:
                num_replicas = _num_replicas
            if rank is None:
                rank = _rank
            self.dataset = dataset
            self.samples_per_gpu = samples_per_gpu
            self.num_replicas = num_replicas
            self.rank = rank
            self.epoch = 0
            self.seed = seed if seed is not None else 0
            assert hasattr(self.dataset, 'flag'), \
                'DistributedGroupSampler requires dataset.flag attribute.'
            self.flag = self.dataset.flag
            self.group_sizes = _np_sampler.bincount(self.flag)
            self.num_samples = 0
            for i, j in enumerate(self.group_sizes):
                self.num_samples += int(
                    _math.ceil(self.group_sizes[i] * 1.0 / self.samples_per_gpu
                               / self.num_replicas)) * self.samples_per_gpu
            self.total_size = self.num_samples * self.num_replicas

        def __iter__(self):
            g = _torch_sampler.Generator()
            g.manual_seed(self.epoch + self.seed)
            indices = []
            for i, size in enumerate(self.group_sizes):
                if size > 0:
                    indice = _np_sampler.where(self.flag == i)[0]
                    indice = indice[list(
                        _torch_sampler.randperm(int(size), generator=g
                                                ).numpy())].tolist()
                    extra = int(
                        _math.ceil(size * 1.0 / self.samples_per_gpu
                                   / self.num_replicas)
                    ) * self.samples_per_gpu * self.num_replicas - len(indice)
                    tmp = indice.copy()
                    for _ in range(extra // size):
                        indice.extend(tmp)
                    indice.extend(tmp[:extra % size])
                    indices.extend(indice)
            assert len(indices) == self.total_size
            indices = [
                indices[j] for i in list(
                    _torch_sampler.randperm(
                        len(indices) // self.samples_per_gpu, generator=g))
                for j in range(i * self.samples_per_gpu,
                               (i + 1) * self.samples_per_gpu)
            ]
            offset = self.num_samples * self.rank
            indices = indices[offset:offset + self.num_samples]
            assert len(indices) == self.num_samples
            return iter(indices)

        def __len__(self):
            return self.num_samples

        def set_epoch(self, epoch):
            self.epoch = epoch

    @_DATA_SAMPLERS.register_module(force=True)
    class DistributedSampler(_TorchDistSampler):
        """평가/테스트용 비셔플 분산 Sampler."""

        def __init__(self, dataset=None, num_replicas=None, rank=None,
                     shuffle=False, seed=0):
            super().__init__(dataset, num_replicas=num_replicas, rank=rank,
                             shuffle=shuffle)
            self.seed = seed if seed is not None else 0

        def __iter__(self):
            if self.shuffle:
                g = _torch_sampler.Generator()
                g.manual_seed(self.epoch + self.seed)
                indices = _torch_sampler.randperm(len(self.dataset),
                                                  generator=g).tolist()
            else:
                indices = _torch_sampler.arange(len(self.dataset)).tolist()
            indices = (indices * _math.ceil(
                self.total_size / len(indices)))[:self.total_size]
            assert len(indices) == self.total_size
            per_replica = self.total_size // self.num_replicas
            indices = indices[self.rank * per_replica:
                              (self.rank + 1) * per_replica]
            assert len(indices) == self.num_samples
            return iter(indices)

except Exception as _e:
    import warnings as _w
    _w.warn(f'OccFormer: DistributedGroupSampler/DistributedSampler 등록 실패: {_e}')
