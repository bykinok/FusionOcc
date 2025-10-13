# FusionOCC

FusionOCC는 LiDAR와 카메라 데이터를 융합하여 3D 공간 점유율(occupancy)을 예측하는 모델입니다.

## 모델 구조

FusionOCC는 다음과 같은 주요 컴포넌트로 구성됩니다:

1. **CustomSparseEncoder**: LiDAR 포인트 클라우드를 처리하는 스파스 컨볼루션 인코더
2. **Image Encoder**: 카메라 이미지를 처리하는 백본 네트워크
3. **View Transformer**: 이미지 특징을 BEV(Bird's Eye View)로 변환
4. **Fusion Module**: LiDAR와 이미지 특징을 융합
5. **Occupancy Head**: 3D 공간 점유율을 예측하는 헤드

## 설치

### 의존성 설치

```bash
pip install torch-scatter
```

### 프로젝트 등록

MMDetection3D에서 FusionOCC 프로젝트를 사용하려면 다음과 같이 설정하세요:

```python
# configs/your_config.py
custom_imports = dict(
    imports=['projects.FusionOcc.fusionocc'],
    allow_failed_imports=False)
```

## 설정 파일

### 기본 설정

```python
# projects/FusionOcc/configs/fusion_occ.py
model = dict(
    type='FusionOCC',
    lidar_in_channel=5,
    point_cloud_range=[-40, -40, -1, 40, 40, 5.4],
    voxel_size=[0.05, 0.05, 0.05],
    lidar_out_channel=32,
    out_dim=64,
    num_classes=18,
    use_mask=True,
    loss_occ=dict(
        type='CrossEntropyLoss',
        use_sigmoid=False,
        loss_weight=1.0),
)
```

### 데이터 설정

```python
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        data_root='data/nuscenes/',
        ann_file='data/nuscenes/fusionocc-nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        test_mode=False,
        use_valid_flag=True,
        box_type_3d='LiDAR'),
    val=test_data_config,
    test=test_data_config
)
```

## 훈련

### 단일 GPU 훈련

```bash
python tools/train.py projects/FusionOcc/configs/fusion_occ.py
```

### 다중 GPU 훈련

```bash
bash tools/dist_train.sh projects/FusionOcc/configs/fusion_occ.py 8
```

## 테스트

### 단일 GPU 테스트

```bash
python tools/test.py projects/FusionOcc/configs/fusion_occ.py work_dirs/fusion_occ/latest.pth --eval bbox
```

### 다중 GPU 테스트

```bash
bash tools/dist_test.sh projects/FusionOcc/configs/fusion_occ.py work_dirs/fusion_occ/latest.pth 8 --eval bbox
```

## 데이터셋 준비

### nuScenes 데이터셋

1. nuScenes 데이터셋을 다운로드하고 `data/nuscenes/` 디렉토리에 배치
2. FusionOcc용 데이터 전처리 실행:

```bash
python tools/create_data.py fusionocc --root-path ./data/nuscenes/ --out-dir ./data/nuscenes/ --extra-tag fusionocc-nuscenes --version v1.0-trainval
```

## 모델 성능

현재 구현은 기본적인 구조만 포함되어 있으며, 실제 성능을 위해서는 다음 컴포넌트들을 추가로 구현해야 합니다:

1. **Image Backbone**: SwinTransformer 또는 ResNet
2. **View Transformer**: CrossModalLSS
3. **3D Encoder**: CustomResNet3D
4. **Neck**: LSSFPN3D

## 주의사항

1. **의존성**: `torch-scatter` 패키지가 필요합니다.
2. **메모리**: 3D 컨볼루션 연산으로 인해 많은 GPU 메모리가 필요할 수 있습니다.
3. **데이터**: nuScenes 데이터셋의 전체 버전이 필요합니다.

## 커스터마이징

### 새로운 백본 네트워크 추가

```python
# projects/FusionOcc/fusionocc/your_backbone.py
@MODELS.register_module()
class YourBackbone(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        # 구현
        
    def forward(self, x):
        # 구현
        return x
```

### 새로운 손실 함수 추가

```python
# projects/FusionOcc/fusionocc/your_loss.py
@MODELS.register_module()
class YourLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        # 구현
        
    def forward(self, pred, target):
        # 구현
        return loss
```

## 문제 해결

### 메모리 부족 오류

- `samples_per_gpu`를 줄이거나
- `voxel_size`를 늘리거나
- 모델의 채널 수를 줄이세요

### CUDA 오류

- PyTorch와 CUDA 버전이 호환되는지 확인
- `torch-scatter`가 올바르게 설치되었는지 확인

## 프로젝트 구조

```
projects/FusionOcc/
├── fusionocc/
│   ├── __init__.py
│   ├── fusion_occ.py
│   ├── lidar_encoder.py
│   └── datasets/
│       └── fusionocc_dataset.py
├── configs/
│   └── fusion_occ.py
├── __init__.py
└── README.md
```

## 참고 자료

- [FusionOCC 원본 구현](https://github.com/your-repo/FusionOcc)
- [nuScenes 데이터셋](https://www.nuscenes.org/)
- [MMDetection3D 문서](https://mmdetection3d.readthedocs.io/) 