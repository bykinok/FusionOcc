# TPVFormer Occupancy Prediction Task Migration

이 문서는 TPVFormer의 segmentation task를 기반으로 occupancy prediction task를 migration하는 과정을 설명합니다.

## 개요

기존 `tpvformer_8xb1-2x_nus-seg.py` 설정 파일을 기반으로 하여, 새로운 occupancy prediction task를 지원하는 설정 파일과 모델을 생성했습니다. 기존 코드는 최대한 수정하지 않고 새로운 파일들을 추가하는 방식으로 진행했습니다.

## 새로 생성된 파일들

### 1. 설정 파일
- `configs/tpvformer_8xb1-2x_nus-occupancy.py`: Occupancy prediction task용 설정 파일

### 2. 모델 클래스들
- `tpvformer/tpv_aggregator.py`: TPVAggregator 클래스 (occupancy prediction용)
- `tpvformer/tpvformer_occupancy.py`: TPVFormerOccupancy 클래스 (occupancy task 지원)
- `tpvformer/nuscenes_occupancy_dataset.py`: NuScenesOccupancyDataset 클래스
- `tpvformer/metrics.py`: OccupancyMetric 클래스

### 3. 데이터 처리
- `tpvformer/loading.py`: LoadOccupancyAnnotations 클래스 추가

## 주요 변경사항

### A. 모델 구조 변경
- **기존**: `decode_head` (TPVFormerDecoder) - segmentation용
- **새로**: `tpv_aggregator` (TPVAggregator) - occupancy prediction용

### B. 파라미터 조정
- `_dim_`: 128 → 256 (occupancy task에 맞게)
- `tpv_h_`, `tpv_w_`: 200 → 100 (occupancy task에 맞게)
- `tpv_z_`: 16 → 8 (occupancy task에 맞게)
- `nbr_classes`: 17 → 18 (occupied 클래스 추가)

### C. Task 설정
- `occupancy = True`: Occupancy task 활성화
- `lovasz_input = 'voxel'`: Voxel 기반 손실 계산
- `ce_input = 'voxel'`: Voxel 기반 Cross-entropy 손실

## 사용 방법

### 1. 학습 실행
```bash
python tools/train.py projects/TPVFormer/configs/tpvformer_8xb1-2x_nus-occupancy.py
```

### 2. 테스트 실행
```bash
python tools/test.py projects/TPVFormer/configs/tpvformer_8xb1-2x_nus-occupancy.py
```

## 구현된 기능

### 1. TPVAggregator
- 3개 TPV 평면 (H-W, Z-H, W-Z)의 특징을 융합
- 3D occupancy grid 예측
- Cross-entropy 및 Lovasz 손실 지원

### 2. NuScenesOccupancyDataset
- NuScenes 데이터셋의 occupancy annotation 처리
- 18개 클래스 지원 (기존 17개 + occupied)
- 3D bounding box를 occupancy grid로 변환

### 3. OccupancyMetric
- IoU, Accuracy 등 occupancy 성능 측정
- 클래스별 성능 분석
- Confusion matrix 기반 평가

## 주의사항

1. **기존 코드 보존**: 원본 segmentation 코드는 전혀 수정하지 않았습니다.
2. **새로운 파일들**: 모든 occupancy 관련 기능은 새로운 파일에 구현되었습니다.
3. **호환성**: 기존 segmentation 모델과 새로운 occupancy 모델은 독립적으로 동작합니다.

## 향후 개선사항

1. **실제 occupancy annotation**: 현재는 placeholder 구현이므로 실제 데이터에 맞게 수정 필요
2. **성능 최적화**: TPVAggregator의 3D 특징 융합 로직 최적화
3. **추가 메트릭**: Occupancy task에 특화된 추가 평가 지표 구현
4. **데이터 증강**: Occupancy task에 적합한 데이터 증강 기법 추가

## 문제 해결

### 1. Import 오류
```bash
# __init__.py 파일이 제대로 업데이트되었는지 확인
python -c "from projects.TPVFormer.tpvformer import *"
```

### 2. 모델 등록 오류
```bash
# 모델이 제대로 등록되었는지 확인
python -c "from mmdet3d.registry import MODELS; print(MODELS.module_dict.keys())"
```

### 3. 데이터셋 오류
```bash
# 데이터셋이 제대로 등록되었는지 확인
python -c "from mmdet3d.registry import DATASETS; print(DATASETS.module_dict.keys())"
```

## 참고 자료

- 원본 TPVFormer 논문
- NuScenes 데이터셋 문서
- mmdet3d 프레임워크 문서
- Occupancy prediction 관련 연구 논문들
