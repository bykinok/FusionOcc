# Copyright (c) OpenMMLab. All rights reserved.

# mmdet3d의 transforms를 mmengine 레지스트리에도 등록
# 이것은 mmengine의 BaseDataset이 mmengine 레지스트리를 사용하기 때문에 필요합니다
from mmdet3d.registry import TRANSFORMS as TRANSFORMS_MMDET3D
from mmengine.registry import TRANSFORMS as TRANSFORMS_MMENGINE
from mmdet3d.registry import MODELS as MODELS_MMDET3D
from mmengine.registry import MODELS as MODELS_MMENGINE

# mmdet3d의 주요 transforms를 mmengine 레지스트리에 복사
_mmdet3d_transforms = [
    'LoadAnnotations3D',
    'LoadImageFromFileMono3D', 
    'LoadPointsFromDict',
    'LoadPointsFromFile',
    'LoadPointsFromMultiSweeps',
    'ObjectRangeFilter',
    'ObjectNameFilter',
    'MultiScaleFlipAug3D',
]

for transform_name in _mmdet3d_transforms:
    if transform_name in TRANSFORMS_MMDET3D and transform_name not in TRANSFORMS_MMENGINE:
        TRANSFORMS_MMENGINE.register_module(
            name=transform_name,
            module=TRANSFORMS_MMDET3D.get(transform_name)
        )

# Positional encoding 모듈들을 mmengine MODELS 레지스트리에 등록
# build_positional_encoding이 mmengine MODELS를 사용하기 때문
_positional_encodings = [
    'LearnedPositionalEncoding',
    'LearnedPositionalEncoding3D',
]

for enc_name in _positional_encodings:
    if enc_name in MODELS_MMDET3D and enc_name not in MODELS_MMENGINE:
        MODELS_MMENGINE.register_module(
            name=enc_name,
            module=MODELS_MMDET3D.get(enc_name)
        )

from .datasets import *  # noqa: F401, F403
from .dense_heads import *  # noqa: F401, F403
from .detectors import *  # noqa: F401, F403
from .hooks import *  # noqa: F401, F403
from .modules import *  # noqa: F401, F403
from .utils import *  # noqa: F401, F403

__all__ = []
