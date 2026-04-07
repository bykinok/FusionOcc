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
