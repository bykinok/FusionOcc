# TEOcc: Temporal-Efficient Occupancy Prediction via Radar-Camera Fusion

## Introduction

TEOcc는 radar와 camera fusion을 통한 시간적으로 효율적인 점유 예측(occupancy prediction) 방법입니다. 이 프로젝트는 새로운 mmdetection3d 라이브러리 구조에 맞게 마이그레이션되었습니다.

## Architecture

TEOcc는 다음과 같은 주요 컴포넌트로 구성됩니다:

- **BEVStereo4DOCCRC**: Radar-Camera fusion을 통한 3D 점유 예측 detector
- **Multi-frame processing**: 시간적 정보를 활용한 효율적인 점유 예측
- **Radar feature extraction**: Radar 포인트 클라우드 처리 및 BEV 특징 추출
- **Camera-Radar fusion**: 이미지와 radar 특징의 효과적인 융합

## Installation

```bash
# mmdetection3d 환경 설정 후
cd /path/to/mmdetection3d/projects/TEOcc
```

## Data Preparation

nuScenes 데이터셋과 점유 라벨을 준비해주세요:

```bash
# nuScenes 데이터셋 다운로드 및 설정
# 점유 라벨 데이터 준비
```

## Training

```bash
# 단일 GPU 학습
python tools/train.py projects/TEOcc/configs/teocc_rc.py

# 멀티 GPU 학습 (8 GPUs)
./tools/dist_train.sh projects/TEOcc/configs/teocc_rc.py 8
```

## Testing

```bash
# 테스트
python tools/test.py projects/TEOcc/configs/teocc_rc.py /path/to/checkpoint.pth
```

## Main Results

### nuScenes Occupancy Prediction

| Method | mIoU | 
|--------|------|
| TEOcc  | 17.5 |

## Citation

TEOcc를 사용하시는 경우 다음을 인용해주세요:

```bibtex
@article{teocc2023,
  title={TEOcc: Temporal-Efficient Occupancy Prediction via Radar-Camera Fusion},
  author={Authors},
  journal={arXiv preprint arXiv:2309.08693},
  year={2023}
}
```

## License

이 프로젝트는 Apache 2.0 라이센스를 따릅니다.
