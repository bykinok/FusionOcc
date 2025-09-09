# BEVFormer for Occupancy Prediction

## Description

BEVFormer는 Bird's-Eye-View에서의 3D 점유 예측을 위한 Transformer 기반 프레임워크입니다. 이 프로젝트는 원본 BEVFormer를 새로운 mmdetection3d 라이브러리 구조에 마이그레이션한 버전입니다.

## 주요 특징

- BEV 관점에서의 3D occupancy 예측
- Temporal self-attention과 spatial cross-attention을 활용한 Transformer 아키텍처
- Multi-camera 입력 지원
- NuScenes 데이터셋 지원

## 설치 및 사용법

### 학습 명령어

MMDet3D 루트 디렉토리에서 다음 명령어를 실행하여 모델을 학습시킵니다:

```bash
python tools/train.py projects/BEVFormer/configs/bevformer_base_occ.py
```

### 테스트 명령어

MMDet3D 루트 디렉토리에서 다음 명령어를 실행하여 모델을 테스트합니다:

```bash
python tools/test.py projects/BEVFormer/configs/bevformer_base_occ.py ${CHECKPOINT_PATH}
```

## 모델 구조

- **Detector**: `BEVFormerOcc` - 메인 occupancy 예측 모델
- **Head**: `BEVFormerOccHead` - Occupancy 예측을 위한 dense head
- **Transformer**: `TransformerOcc` - Occupancy 예측을 위한 transformer 모듈
- **Encoder**: `BEVFormerEncoder` - BEV feature 인코딩
- **Attention**: Temporal self-attention과 spatial cross-attention

## 데이터셋

현재 지원하는 데이터셋:
- NuScenes Occupancy Dataset

## Citation

```latex
@article{li2022bevformer,
  title={BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers},
  author={Li, Zhiqi and Wang, Wenhai and Li, Hongyang and Xie, Enze and Sima, Chonghao and Lu, Tong and Qiao, Yu and Dai, Jifeng},
  journal={arXiv preprint arXiv:2203.17270},
  year={2022}
}
```

## Checklist

- [x] Milestone 1: PR-ready, and acceptable to be one of the `projects/`.
  - [x] Finish the code
  - [x] Basic docstrings & proper citation
  - [x] A full README

- [ ] Milestone 2: Indicates a successful model implementation.
  - [ ] Training-time correctness

- [ ] Milestone 3: Good to be a part of our core package!
  - [ ] Type hints and docstrings
  - [ ] Unit tests
  - [ ] Code polishing
  - [ ] Metafile.yml
