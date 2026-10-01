# Claude Code 시작 지시문

아래를 `CLAUDE_CODE_OPENOCC_REQUIREMENTS_KO.md`와 함께 전달한다.

```text
현재 FusionOcc의 stcocc-selective-free-supervision 브랜치에서
STCOcc가 OpenOcc GT로 occupancy-only 학습·평가를 할 수 있게 확장해줘.
상세 요구사항은 CLAUDE_CODE_OPENOCC_REQUIREMENTS_KO.md를 따라줘.

이번 연구의 검증 전략은 다음과 같다.
- Method development: STCOcc + Occ3D-nuScenes
- GT generalization: STCOcc + OpenOcc-nuScenes
- Dataset generalization: 가능한 한 동일 architecture를 유지한 STCOcc + Occ3D-Waymo
- Model generalization: 첫 번째 연구에서 사용한 기존 occupancy 모델 일부에 동일 supervision method 적용

따라서 이번 OpenOcc 작업에서는 모델 architecture를 바꾸는 방향으로 확장하지 마.
SSCBench-KITTI-360처럼 입력 구성 때문에 모델 architecture 변경이 필요한 dataset 지원도 이번 범위에서 제외한다.
향후 다른 GT/dataset/model로 이식할 수 있도록 dataset/GT-specific 처리는 adapter/config 쪽에 격리하고,
proposed supervision method가 model architecture와 dataset-specific visibility annotation에 의존하지 않는 구조를 준비해줘.

새 모델을 만들거나 기존 Occ3D 실험을 덮어쓰지 마.
기존 openocc_12e config/loader/evaluator를 감사하고 재사용하되,
17개 클래스/free=16, GT 경로, flow optional 처리, temporal sampler,
실제 update 수, mIoU/RayIoU 평가 protocol을 검증해줘.
공통 GT resolver를 훈련·평가·logit export가 함께 사용하도록 해줘.

OpenOcc에 camera mask가 없다고 Occ3D mask를 복사하거나
all-true/all-false mask를 만들지 마.
현재 selective lambda가 필요한 mask 없이 no-op이 되면 실패로 처리해줘.
이번 단계는 baseline과 공통 supervision interface까지야.
아직 없는 proposed method를 임의로 구현 완료라고 하지 마.

향후 Occ3D-Waymo와 기존 occupancy 모델들에 같은 method를 이식할 수 있도록
다음 요소는 hard-code하지 말고 config/adapter에서 공급 가능하게 해줘.
- num_classes / class names / free index / ignore index
- point cloud range / voxel size / grid shape
- camera count / camera ordering / calibration metadata
- temporal sampling metadata
- GT resolver / valid-region definition / evaluation protocol

중요: 모델 backbone, view transformer, temporal fusion, occupancy head 구조는
GT나 dataset을 바꾸기 위해 임의로 재설계하지 마.
아키텍처 변경이 필요한 경우 자동으로 수정하지 말고 BLOCKED 또는 별도 adaptation 항목으로 보고해줘.

우선 데이터 감사, 구현, unit test, config, 실행 명령을 만들고,
승인된 GPU가 있으면 1 iteration backward와 소규모 평가까지만 수행해줘.
장시간 학습이나 대규모 다운로드는 시작하지 말고
research_openocc/에 결과와 다음 실행 계획을 보고해줘.
```
