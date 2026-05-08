"""SparseOcc_ori 플러그인 패키지.

새 mmdet3d (mmengine 기반) API 로 마이그레이션된 SparseOcc_ori 구현.
원본: Ref/SparseOcc_ori

마이그레이션 변경 요약:
  - mmcv.runner.{BaseModule, force_fp32, auto_fp16, get_dist_info} → compat.py 경유
  - mmdet.models.{DETECTORS, HEADS, LOSSES, TRANSFORMER} → mmdet3d.registry.MODELS 매핑
  - mmdet.datasets.builder.PIPELINES → mmdet3d.registry.TRANSFORMS 매핑
  - mmdet.datasets.DATASETS → mmdet3d.registry.DATASETS 매핑
  - mmcv.cnn.bricks.{Conv3d, ConvTranspose3d, FFN} → torch.nn / mmdet 경유
  - mmcv.parallel.DataContainer → compat.py 내 shim
  - MVXTwoStageDetector super() 호출 시그니처 업데이트
  - SparseOcc에 loss() / predict() 어댑터 추가 (새 mmdet3d 1.x 인터페이스)
  - NuSceneOcc를 LegacyNuScenesDataset 기반으로 재구현 (새 Det3DDataset API 우회)
"""

from .models import *
from .datasets import *
from .datasets.pipelines import *

# mmdet3d 레지스트리에 mmdet backbone/neck 모델 등록
# (mmdet와 mmdet3d는 mmengine의 형제 레지스트리라 상호 탐색 불가)
try:
    from mmdet3d.registry import MODELS as _MMDET3D_MODELS
    from mmdet.models.backbones import ResNet, ResNetV1d, ResNeXt, HRNet, SSDVGG
    from mmdet.models.necks import FPN
    for _cls in [ResNet, ResNetV1d, ResNeXt, HRNet, SSDVGG, FPN]:
        if _MMDET3D_MODELS.get(_cls.__name__) is None:
            _MMDET3D_MODELS.register_module(module=_cls)
except Exception:
    pass

# mmdet3d의 LoadMultiViewImageFromFiles를 구버전 img_filename 기반 구현으로 교체
# (새 mmdet3d는 'images' dict 형식을 기대하지만, 본 데이터셋은 'img_filename' 리스트를 사용)
# mmdet3d 모델 빌드 후 TRANSFORMS가 완전히 채워진 다음 force=True로 등록
try:
    import mmdet3d.datasets.transforms  # mmdet3d transforms 강제 import (먼저 등록되도록)
    from mmdet3d.registry import TRANSFORMS as _MMDET3D_TRANSFORMS_FINAL
    from .datasets.pipelines.loading import LoadMultiViewImageFromFiles as _LMVIFromFiles
    _MMDET3D_TRANSFORMS_FINAL.register_module(
        name='LoadMultiViewImageFromFiles', module=_LMVIFromFiles, force=True)
except Exception:
    pass
