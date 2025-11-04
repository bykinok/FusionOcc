# CONNet (Cascade Occupancy Network)

이 프로젝트는 원본 CONet 코드를 최신 MMDetection3D 라이브러리에 맞게 마이그레이션한 버전입니다.

## 설정 파일들

### 카메라 기반 모델
- `configs/cam_r50_img1600_cascade_x4.py`: ResNet-50 백본을 사용한 카메라 기반 occupancy prediction

### LiDAR 기반 모델  
- `configs/lidar_cascade_x4.py`: LiDAR 데이터만을 사용한 occupancy prediction

### 멀티모달 모델
- `configs/multimodal_r50_img1600_cascade_x4.py`: 카메라와 LiDAR를 모두 사용한 융합 모델

## 주요 변경사항

기존 CONet 설정 파일에서 최신 MMDetection3D v1.4+ 형식으로 마이그레이션:

1. **데이터 로더**: `data` → `train_dataloader`, `val_dataloader`, `test_dataloader`
2. **옵티마이저**: `optimizer` + `optimizer_config` → `optim_wrapper`
3. **학습률 스케줄러**: `lr_config` → `param_scheduler`
4. **평가**: `evaluation` → `val_evaluator`, `test_evaluator`
5. **실행기**: `runner` → `train_cfg`, `val_cfg`, `test_cfg`
6. **플러그인**: `plugin_dir` → `custom_imports`

## 사용 방법

```bash
# 카메라 기반 모델 훈련
python tools/train.py projects/CONNet/configs/cam_r50_img1600_cascade_x4.py

# LiDAR 기반 모델 훈련  
python tools/train.py projects/CONNet/configs/lidar_cascade_x4.py

# 멀티모달 모델 훈련
python tools/train.py projects/CONNet/configs/multimodal_r50_img1600_cascade_x4.py
```

## 요구사항

- MMDetection3D >= 1.4.0
- PyTorch >= 1.8.0
- CUDA >= 11.0

## 데이터셋

nuScenes Occupancy 데이터셋이 필요합니다:
- `./data/nuScenes-Occupancy/`: occupancy ground truth
- `./data/nuscenes/`: nuScenes 원본 데이터
- `./data/depth_gt/`: depth ground truth (카메라 모델용)
