# 추가 GT / Dataset / Model 검증 전략 — 선택적 Free-Space Supervision 연구
조사 기준일: 2026-10-01

이 메모는 현재 연구의 최종 검증 전략을 정리한 것이다.
핵심 원칙은 **dataset을 바꾸기 위해 모델 architecture를 크게 변경하지 않는다**는 것이다.
첫 번째 연구에서 사용한 occupancy 모델 일부를 다시 활용해야 하므로,
입력 view와 모델 구조가 크게 달라지는 benchmark는 주 검증 대상에서 제외한다.

## 1. 최종 권장 구성

### A. Method development
- **STCOcc + Occ3D-nuScenes**
- 현재 진행 중인 핵심 개발 환경
- invisible-free supervision, mIoU-RayIoU trade-off, reliability 분석의 기준 환경

### B. GT generalization
- **STCOcc + OpenOcc-nuScenes**
- 원본 sensor dataset과 모델 architecture는 유지
- occupancy GT 생성 체계만 변경
- 목적: proposed supervision method가 Occ3D-specific GT와 제공 mask에만 맞춘 것이 아닌지 확인

### C. Dataset generalization
- **STCOcc + Occ3D-Waymo**
- 가능한 한 STCOcc architecture를 유지
- 원본 sensor dataset을 nuScenes -> Waymo로 변경
- Occ3D GT family를 유지하여 dataset/domain 변경 효과를 비교적 분리

### D. Model generalization
- 첫 번째 연구에서 사용한 기존 occupancy 모델 중 2~3개를 선택
- 각 모델의 기존 architecture와 native input formulation을 유지
- 동일 proposed supervision method를 training/loss level에서 적용
- 목적: 특정 STCOcc 구조에 의존하지 않는지 확인

권장 논문 구조:

```text
STCOcc + Occ3D-nuScenes  -> method development
STCOcc + OpenOcc         -> GT generalization
STCOcc + Occ3D-Waymo     -> dataset generalization
Other existing models    -> model generalization
```

이 구성이 가장 중요한 이유는 각 축의 변화가 비교적 분리되기 때문이다.

## 2. SSCBench-KITTI-360을 주 검증 대상에서 제외하는 이유

SSCBench-KITTI-360은 3D semantic scene completion 연구에서 널리 사용되는 benchmark이며
hidden-space supervision 연구와 개념적으로 잘 맞는다.
그러나 실제 사용 protocol이 전방 monocular/stereo SSC 중심이고,
현재 첫 번째 연구에서 사용한 nuScenes surround occupancy 모델들을 그대로 적용하기 어렵다.

이 경우 다음 요소가 동시에 바뀔 수 있다.
- camera configuration
- view transformation
- voxel volume definition
- temporal input
- model architecture / head
- evaluator

따라서 성능 변화가 proposed supervision method 때문인지,
KITTI-360 adaptation을 위한 architecture 변경 때문인지 분리하기 어려워진다.

결론:
- SSCBench-KITTI-360은 주 논문의 필수 dataset generalization 대상에서 제외한다.
- 향후 supplementary 또는 별도 연구에서 architecture adaptation이 명확히 통제될 때만 고려한다.

## 3. Occ3D-Waymo를 우선하는 이유

Occ3D-Waymo는 Waymo sensor data 위에 Occ3D occupancy GT를 제공한다.
공식 coarse setting은 현재 Occ3D-nuScenes와 동일한 공간 범주를 사용한다.
- point cloud range: [-40, -40, -1, 40, 40, 5.4]
- voxel grid: [200, 200, 16]
- voxel size: 0.4 m

Waymo는 5-camera 구성이며 nuScenes의 6-camera와 다르므로 data adapter 수정은 필요하다.
하지만 목표는 **STCOcc architecture 자체를 재설계하지 않고 dataset interface만 바꾸는 것**이다.

필수 확인 항목:
- camera count가 코드에 hard-coded되어 있는지
- camera ordering / intrinsic / extrinsic
- temporal sample interval과 history window
- class mapping과 free index
- FOV / valid-region 정의
- depth supervision 생성 방식
- RayIoU evaluator의 origin / class order / no-hit 처리

가능한 변경:
- dataset loader / metadata adapter
- camera list / ordering config
- class mapping / head output channels
- GT resolver / evaluator config
- temporal sampling config

가능하면 피할 변경:
- backbone 구조 변경
- view transformer 구조 변경
- temporal fusion 구조 변경
- occupancy representation 변경
- 새로운 dataset 전용 head 설계

Architecture 변경이 불가피하면 그 실험은 순수 dataset generalization이 아니라
`architecture-adapted transfer`로 별도 분류해야 한다.

## 4. OpenOcc의 역할

OpenOcc는 nuScenes sensor input을 그대로 두고 GT를 변경할 수 있다는 점이 가장 중요하다.
따라서 OpenOcc 결과는 **다른 dataset 일반화**가 아니라 **다른 GT pipeline 일반화**로 해석한다.

이번 연구에서 확인할 핵심:
- Occ3D camera visibility mask 없이도 baseline이 정상적으로 학습되는가
- proposed method가 제공 visibility annotation 없이 동작하는가
- OpenOcc의 free/occupied/unknown 정의에 맞게 같은 method를 적용할 수 있는가
- mIoU/RayIoU 개선 방향이 Occ3D에서와 일관되는가

## 5. Model generalization 전략

첫 번째 연구에서 사용했던 모든 모델을 다시 돌릴 필요는 없다.
대표성이 다른 2~3개 모델을 선택한다.

선정 원칙 예시:
- camera-only temporal/dense representation 1개
- 다른 representation 또는 sparse/transformer 계열 1개
- 필요하면 sensor-fusion 계열 1개

중요 조건:
1. 모델 architecture는 첫 번째 연구와 동일하게 유지한다.
2. proposed method는 loss/supervision level에서 최소 수정으로 연결한다.
3. dataset/model별로 별도 method logic을 만들지 않는다.
4. hyperparameter는 STCOcc/Occ3D에서 결정한 뒤 가능한 한 zero-retuning으로 적용한다.
5. dataset/model별 재튜닝 결과가 필요하면 zero-retuning 결과와 분리해 보고한다.

최종적으로 다음을 보여주는 것이 목적이다.

```text
same supervision principle
+ different model
+ different GT
+ different dataset
```

## 6. 최종 방법이 만족해야 할 portability 조건

최종 supervision method는 최소한 다음 공통 입력으로 작동하는 것을 목표로 한다.
- model input / features 또는 model prediction
- occupancy GT
- valid/ignore information
- geometry metadata (voxel grid, pose, calibration 등)

필수적으로 요구하지 않아야 하는 것:
- Occ3D의 제공 camera visibility mask
- 특정 STCOcc depth head
- 특정 model의 attention map
- 특정 dataset의 class ID hard-code
- 특정 camera 수
- 20m 같은 dataset-specific 고정 경계

Dataset/GT별 차이는 adapter에서 처리하고,
method core는 같은 interface를 사용해야 한다.

예시:

```text
DatasetAdapter
  -> canonical GT / validity / geometry metadata

ModelAdapter
  -> prediction / optional support features

SupervisionWeightProvider
  -> per-voxel weight

Loss
  -> native occupancy losses with provided weights
```

이 구조는 구현 예시이며, 실제 proposed method의 novelty를 대신하지 않는다.

## 7. 학습 비용 관점의 우선순위

GPU budget을 고려해 모든 조합을 full training하지 않는다.

권장 단계:
1. STCOcc + Occ3D-nuScenes에서 method 개발
2. STCOcc + OpenOcc에서 GT transfer 확인
3. STCOcc + Occ3D-Waymo에서 dataset transfer 확인
4. 기존 모델 2~3개에서 model transfer 확인

각 단계에서 먼저 short/screening protocol을 사용하고,
방법과 baseline 차이가 확인된 뒤 최종 후보만 full protocol / multi-seed로 확증한다.

Occ3D-Waymo의 실제 비용은 공개 sequence 수만 보고 판단하지 않는다.
반드시 실제 info file에서 train sample 수와 temporal sampling 후 optimizer update 수를 계산한다.
STCOcc의 16-frame history가 dataset fps에 따라 실제 시간 범위를 다르게 만들 수 있으므로
`frame count`와 `temporal duration`을 모두 기록한다.

## 8. 보조 후보

### CarlaOcc
CVPR 2026의 synthetic/mesh-based occupancy benchmark다.
LiDAR aggregation 기반 GT와 다른 GT 생성 메커니즘에서 supervision 현상을 분석하는 데 유용할 수 있다.
다만 synthetic data이며 grid와 sensor configuration이 다르므로 주 generalization 결과를 대체하지 않는다.
필요하면 mechanism analysis용 supplementary 후보로만 사용한다.

### UniOcc
여러 occupancy source의 annotation/API를 통합하는 기반이다.
새 독립 dataset으로 세기보다 향후 multi-dataset adapter 설계에 참고한다.

### URScenes
다양한 비정형 도로 환경의 실제 dataset이지만,
현재 연구에서 바로 사용하려면 추가적인 loader/evaluator adaptation 비용이 크다.
우선순위는 낮춘다.

### SemanticKITTI / SSCBench-KITTI-360
SSC 연구에서는 여전히 중요하지만,
현재 연구의 핵심 조건인 '기존 occupancy architecture를 유지'하기 어렵기 때문에
주 검증 대상에서는 제외한다.

## 9. 공통 평가 원칙

1. 각 GT/dataset의 native metric을 보존한다.
2. mIoU의 evaluation population을 명시한다.
3. RayIoU가 native metric이 아니면 임의로 동일 지표라고 부르지 않는다.
4. baseline/proposed는 같은 input, GT, protocol, seed, update budget에서 비교한다.
5. training 시 visibility annotation 미사용과 evaluation 시 native validity/FOV 사용을 구분한다.
6. checkpoint, dataset version, GT version, class mapping, evaluator version을 기록한다.
7. dataset/model마다 method logic을 바꾸지 않는다.
8. architecture 변경이 발생하면 dataset transfer와 분리해 보고한다.

## 10. 주요 1차 출처

Occ3D official repository / Waymo support:
https://github.com/Tsinghua-MARS-Lab/Occ3D

STCOcc:
https://arxiv.org/abs/2504.19749

OpenOcc / Scene as Occupancy 계열은 STCOcc 공식 구현과 원 논문에서 사용하는 GT 정의를 우선 확인한다.

SSCBench (참고용, 주 검증 대상 제외):
https://github.com/ai4ce/SSCBench

CarlaOcc, CVPR 2026:
https://github.com/fengyi233/carlaocc

UniOcc, ICCV 2025:
https://github.com/tasl-lab/UniOcc
