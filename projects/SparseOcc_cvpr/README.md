# SparseOcc (새 mmdet3d API 마이그레이션)

**원본**: [`Ref/SparseOcc_cvpr_ori`](../../Ref/SparseOcc_cvpr_ori) (CVPR 2024)  
**마이그레이션**: 구버전 OpenMMLab (mmcv 1.x / mmdet 2.x / mmdet3d 0.x) → 새 버전 (mmengine / mmdet 3.x / mmdet3d 1.x)

## 변경 요약

| 항목 | 구버전 (원본) | 새 버전 (이 프로젝트) |
|------|--------------|-------------------|
| Registry | `mmdet.models.DETECTORS` | `mmdet3d.registry.MODELS` |
| Runner | `mmcv.runner.EpochBasedRunner` | `mmengine.runner.Runner` |
| BaseModule | `mmcv.runner.BaseModule` | `mmengine.model.BaseModule` |
| Dataset | `mmdet3d.datasets.NuScenesDataset` 상속 | `mmengine.dataset.BaseDataset` 기반 |
| Config | `plugin = True` + `plugin_dir` | `custom_imports` |
| DataLoader | `data.train` dict | `train_dataloader` dict |
| Optimizer | `optimizer` + `optimizer_config` | `optim_wrapper` |
| Scheduler | `lr_config` | `param_scheduler` |
| CenterPoint | `mmdet3d.models.detectors.CenterPoint` 상속 | `compat.CenterPoint` 호환 shim |

## 원칙

- **원본 모델 코드 최소 변경**: 모델 아키텍처 (detectors, backbones, necks, heads, image2bev, mask2former) 내부 로직은 그대로 유지
- **`compat.py` 호환 레이어**: 구버전 API 이름을 새 API로 매핑하는 shim 모듈 제공
- **각 파일의 import 만 수정**: 코드 로직은 원본 그대로

## 프로젝트 구조

```
SparseOcc/
├── __init__.py
├── README.md
├── setup.py              # CUDA extension 빌드
├── configs/
│   ├── _base_/
│   │   └── default_runtime.py   # mmengine 런타임 설정
│   └── sparseocc/
│       ├── sparseocc_nusc_256.py  # NuScenes (mmengine 형식)
│       └── sparseocc_kitti.py     # SemanticKITTI
├── tools/
│   ├── dist_train.sh
│   └── dist_test.sh
└── sparseocc/
    ├── __init__.py
    ├── compat.py         # ★ 핵심 호환성 레이어
    ├── apis/             # 학습/테스트 API (mmengine runner 기반)
    ├── backbones/        # SparseLatentDiffuser, CustomEfficientNet
    ├── core/             # 평가 hooks
    ├── datasets/         # CustomNuScenesOccLSSDataset (BaseDataset 기반)
    │   └── pipelines/    # BEVDet 로딩, Occupancy 로딩, Formatting
    ├── detectors/        # BEVDet, BEVDepth, SparseOcc
    ├── image2bev/        # ViewTransformerLSS*
    ├── mask2former/      # SparseMask2Former* heads
    ├── necks/            # SparseFeaturePyramid
    ├── ops/              # occ_pooling CUDA extension
    └── utils/            # 평가 유틸리티
```

## 설치

```bash
# 1. 환경 (원본 requirements 동일)
pip install timm einops fvcore torchmetrics
pip install spconv-cu{CUDA_VERSION}
pip install torch_scatter

# 2. CUDA extension 빌드
cd projects/SparseOcc
python setup.py develop
```

## 학습

워크스페이스 루트의 공용 `tools/` 스크립트를 사용합니다.

```bash
# 단일 GPU
python tools/train.py projects/SparseOcc/configs/sparseocc/sparseocc_nusc_256.py

# 멀티 GPU (8장)
bash tools/dist_train.sh \
    projects/SparseOcc/configs/sparseocc/sparseocc_nusc_256.py 8
```

## 평가

```bash
bash tools/dist_test.sh \
    projects/SparseOcc/configs/sparseocc/sparseocc_nusc_256.py \
    work_dirs/sparseocc_nusc_256/epoch_24.pth 8
```

> **참고**: `tools/dist_train.sh`는 PYTHONPATH에 워크스페이스 루트를 자동으로 추가하며,
> `tools/train.py`가 config의 `custom_imports`를 읽어 SparseOcc 모듈을 자동으로 로드합니다.
> 별도의 프로젝트 전용 셸 스크립트는 필요하지 않습니다.

## 호환성 레이어 (`compat.py`)

`sparseocc/compat.py`는 다음 구버전 API를 새 API로 매핑합니다:

- `DETECTORS`, `BACKBONES`, `NECKS`, `HEADS`, `LOSSES` → `mmdet3d.registry.MODELS`
- `DATASETS` → `mmdet3d.registry.DATASETS`
- `PIPELINES` → `mmdet3d.registry.TRANSFORMS`
- `BBOX_ASSIGNERS`, `BBOX_SAMPLERS`, `MATCH_COST` → `mmdet3d.registry.TASK_UTILS`
- `force_fp32`, `auto_fp16` → no-op decorators
- `BaseModule`, `ModuleList` → `mmengine.model.*`
- `builder` → `MODELS.build()` 기반 shim
- `CenterPoint` → `Base3DDetector` 기반 호환 shim
- `DataContainer` → `mmcv.parallel.DataContainer` (없으면 최소 구현)

## 원본 논문

> SparseOcc: Rethinking Sparse Latent Representation for Vision-Based Semantic Occupancy Prediction  
> CVPR 2024
