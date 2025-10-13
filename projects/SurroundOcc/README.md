# SurroundOcc for MMDetection3D

새로운 MMDetection3D 라이브러리 기반으로 마이그레이션된 SurroundOcc 프로젝트입니다.

## 개요

이 프로젝트는 기존 SurroundOcc_ori를 새로운 MMDetection3D 구조에 맞게 마이그레이션한 버전입니다. 멀티 카메라 이미지를 이용한 3D occupancy prediction을 지원합니다.

## 주요 변경사항

### 1. 새로운 MMDetection3D 구조 적용
- `MODELS.register_module()` 사용 (기존 `DETECTORS`, `HEADS` 대신)
- `custom_imports` 기반 모듈 등록
- 새로운 data preprocessor 구조
- 새로운 config 구조 (`optim_wrapper`, `param_scheduler` 등)

### 2. 모듈 구조
```
projects/SurroundOcc/
├── configs/
│   └── surroundocc.py          # 메인 설정 파일
├── surroundocc/
│   ├── __init__.py
│   ├── detectors/              # 검출기 모듈
│   │   ├── __init__.py
│   │   └── surroundocc.py
│   ├── dense_heads/            # 헤드 모듈
│   │   ├── __init__.py
│   │   └── occ_head.py
│   ├── modules/                # 핵심 모듈
│   │   ├── __init__.py
│   │   ├── transformer.py
│   │   ├── spatial_cross_attention.py
│   │   ├── encoder.py
│   │   └── grid_mask.py
│   ├── datasets/               # 데이터셋
│   │   ├── __init__.py
│   │   └── custom_nuscenes_occ_dataset.py
│   ├── transforms/             # 데이터 변환
│   │   ├── __init__.py
│   │   └── loading.py
│   └── evaluation/             # 평가 메트릭
│       ├── __init__.py
│       └── occupancy_metric.py
└── README.md
```

### 3. 주요 기능
- **SurroundOcc**: 메인 검출기 (Base3DDetector 기반)
- **OccHead**: Occupancy prediction 헤드
- **PerceptionTransformer**: 2D→3D feature lifting을 위한 transformer
- **SpatialCrossAttention**: 공간 교차 어텐션 모듈
- **CustomNuScenesOccDataset**: NuScenes occupancy 데이터셋
- **OccupancyMetric**: Occupancy 평가 메트릭

## 사용방법

### 1. 학습
```bash
python tools/train.py projects/SurroundOcc/configs/surroundocc.py
```

### 2. 테스트
```bash
python tools/test.py projects/SurroundOcc/configs/surroundocc.py [CHECKPOINT_FILE]
```

### 3. 추론
```bash
python tools/test.py projects/SurroundOcc/configs/surroundocc.py [CHECKPOINT_FILE] --show-dir [OUTPUT_DIR]
```

## 설정 파일

주요 설정 매개변수:

- `point_cloud_range`: Point cloud 범위 `[-50, -50, -5.0, 50, 50, 3.0]`
- `occ_size`: Occupancy grid 크기 `[200, 200, 16]`
- `use_semantic`: 시맨틱 occupancy 사용 여부 `True`
- `class_names`: 16개 클래스 명

## 데이터 형식

### 입력
- 멀티뷰 카메라 이미지 (6개 카메라)
- 카메라 내/외부 파라미터
- LiDAR-to-image transformation

### 출력
- 3D occupancy grid (200×200×16)
- 시맨틱 또는 바이너리 occupancy

## 평가 메트릭

### 시맨틱 Occupancy
- mIoU (mean Intersection over Union)
- Accuracy
- 클래스별 IoU

### 바이너리 Occupancy
- Precision
- Recall
- F1-score
- IoU

## 호환성

- MMDetection3D >= 1.4.0
- PyTorch >= 1.9.0
- CUDA >= 11.0

## 참고사항

1. **데이터 준비**: NuScenes 데이터셋과 occupancy annotation이 필요합니다.
2. **메모리 요구사항**: 최소 24GB GPU 메모리 권장
3. **훈련 시간**: 8x RTX 3090에서 약 2.5일 소요

## 문제해결

### 1. Import 오류
```bash
# 모듈이 제대로 등록되었는지 확인
python -c "from projects.SurroundOcc.surroundocc import *"
```

### 2. CUDA 오류
```bash
# GPU 메모리 확인
nvidia-smi
```

### 3. 데이터 로딩 오류
- `data_root` 경로 확인
- annotation 파일 존재 확인
- 권한 설정 확인

## 원본 논문

```bibtex
@article{wei2023surroundocc,
  title={SurroundOcc: Multi-Camera 3D Occupancy Prediction for Autonomous Driving},
  author={Wei, Yi and Zhao, Linqing and Zheng, Wenzhao and Zhu, Zheng and Lu, Jiwen and Zhou, Jie},
  journal={arXiv preprint arXiv:2303.09551},
  year={2023}
}
```

## 라이선스

이 프로젝트는 Apache 2.0 라이선스 하에 배포됩니다.
