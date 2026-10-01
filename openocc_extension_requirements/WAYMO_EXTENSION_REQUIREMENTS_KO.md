# WAYMO_EXTENSION_REQUIREMENTS_KO.md

## 목적

현재 진행 중인 OpenOcc integration 작업을 유지하면서, 이후 동일한 supervision method를 **Occ3D-Waymo**에 적용할 수 있도록 STCOcc 코드베이스를 준비한다.

이번 문서는 기존 OpenOcc 요구사항을 대체하지 않는다.  
기존 작업에 **Waymo dataset generalization 준비 항목만 추가하는 delta requirement**다.

핵심 연구 구조는 다음과 같다.

- **STCOcc + Occ3D-nuScenes**: method development
- **STCOcc + OpenOcc-nuScenes**: GT generalization
- **STCOcc + Occ3D-Waymo**: dataset generalization
- **첫 번째 연구의 기존 모델 일부**: model generalization

가장 중요한 원칙은 **Waymo 적용을 위해 모델 architecture를 재설계하지 않는 것**이다.

---

# 1. 작업 우선순위

현재 진행 중인 OpenOcc 작업을 중단하거나 처음부터 다시 시작하지 않는다.

작업 우선순위:

1. 현재 OpenOcc audit / occupancy-only integration을 완료한다.
2. 기존 Occ3D-nuScenes regression을 유지한다.
3. 그 다음 Occ3D-Waymo integration audit 및 준비를 수행한다.
4. Waymo full training은 별도 승인 전에는 실행하지 않는다.

---

# 2. Architecture 변경 정책

## 2.1 허용되는 변경

Waymo 적용을 위해 다음 변경은 허용한다.

- dataset loader / dataset adapter
- camera metadata handling
- camera name/order configuration
- calibration parser / coordinate conversion
- class mapping
- `num_classes`
- `free_class_idx`
- `ignore_index`
- GT resolver / GT path handling
- evaluator wiring
- mIoU / RayIoU dataset-specific protocol
- temporal sampling config
- point-cloud range / voxel metadata config
- FOV / validity mask handling
- dataloader / sampler
- config files
- data preparation scripts
- sanity / smoke test utilities

이 변경들은 **data interface adaptation**으로 간주한다.

## 2.2 허용하지 않는 변경

Waymo를 지원하기 위해 아래를 직접 변경하거나 새로 설계하지 않는다.

- image backbone architecture
- occupancy head architecture
- temporal fusion architecture
- view transformer architecture
- decoder / encoder 구조
- Waymo 전용 neural module
- dataset별 별도 network branch
- Waymo 성능 개선을 위한 추가 feature extractor

Waymo 적용에 위와 같은 architecture 변경이 필요하다고 판단되면:

1. 자동으로 구현하지 않는다.
2. `research_waymo/adaptation_blockers.md`에 기록한다.
3. 어떤 assumption 때문에 문제가 생기는지 코드 위치와 함께 설명한다.
4. architecture 변경 없이 해결 가능한 adapter/config 대안을 우선 제안한다.

---

# 3. Occ3D-Waymo 데이터 감사

구현 전 실제 데이터와 현재 코드 구조를 먼저 감사한다.

문서나 기존 기억만으로 값을 하드코딩하지 말고, 가능한 경우 실제 GT / info metadata / official format을 확인한다.

반드시 확인할 항목:

## 3.1 Dataset / split

- 실제 train sequence 수
- 실제 val sequence 수
- 실제 train sample/frame 수
- 실제 val sample/frame 수
- sample frequency
- temporal spacing
- scene/sequence boundary
- key frame 여부
- training sampler가 sequence continuity를 보존하는지

실제 `info.pkl` 또는 equivalent metadata가 있다면 직접 세어서 기록한다.

---

## 3.2 Occupancy GT schema

확인할 항목:

- semantic GT key 이름
- GT tensor shape
- class 수
- class names
- occupied class index
- free class index
- unknown / ignore label
- invalid region 표현
- camera visibility 관련 field
- FOV mask
- LiDAR visibility / observability field
- GT voxel resolution
- point-cloud range
- coordinate convention
- axis ordering
- dtype

Occ3D-nuScenes와 다른 점을 표로 정리한다.

예:

| Item | Occ3D-nuScenes | Occ3D-Waymo |
|---|---|---|
| num_classes | audit | audit |
| free index | audit | audit |
| voxel shape | audit | audit |
| pc range | audit | audit |
| mask_camera | audit | audit |
| FOV mask | audit | audit |
| cameras | audit | audit |

---

# 4. Camera interface audit

STCOcc는 현재 nuScenes multi-camera 구성을 전제로 만들어졌을 가능성이 높다.

반드시 다음을 확인한다.

- camera 수가 코드에 `6`으로 hard-coded되어 있는 위치
- camera names가 hard-coded되어 있는 위치
- camera ordering dependency
- tensor reshape에서 camera dimension이 고정되어 있는 위치
- camera embedding이 camera 수에 의존하는지
- image augmentation이 camera 수를 전제로 하는지
- calibration matrix packing order
- temporal frame × camera flattening assumption
- visualization / evaluator가 6-camera를 가정하는지

Waymo는 camera configuration이 다르므로, 가능한 한 아래 형태로 config-driven으로 만든다.

```python
camera_names = [...]
num_cameras = len(camera_names)
```

단, camera 수 변경이 network weight shape 자체를 바꾸거나 architecture modification을 요구한다면 `adaptation_blockers.md`에 기록한다.

---

# 5. Calibration / coordinate system audit

nuScenes와 Waymo 사이의 coordinate convention 차이를 확인한다.

반드시 검증할 것:

- ego frame
- lidar frame
- camera frame
- sensor-to-ego
- ego-to-global
- intrinsics
- extrinsics
- handedness
- rotation convention
- translation unit
- timestamp alignment

단순히 기존 nuScenes transform field 이름을 Waymo field 이름에 매핑해서 끝내지 말고, synthetic point 또는 실제 sample point를 이용해 projection sanity check를 수행한다.

권장 sanity check:

1. LiDAR point를 camera image로 projection
2. image bounds 확인
3. front camera 기준 시각화
4. ego origin / forward axis 확인
5. voxel coordinate와 GT occupancy가 동일 frame인지 확인

---

# 6. Temporal input audit

이 연구에서 매우 중요하다.

STCOcc는 현재 temporal fusion을 사용하므로, `16 frames`라는 숫자만 같다고 동일한 temporal context라고 간주하면 안 된다.

반드시 계산할 것:

- nuScenes에서 16-frame history가 커버하는 실제 시간 길이
- Waymo에서 16-frame history가 커버하는 실제 시간 길이
- 두 dataset의 frame interval
- key frame interval
- available history length
- sequence boundary에서 history reset 방식

보고서에 다음을 명시한다.

```text
nuScenes:
- history frames:
- effective temporal window:

Waymo:
- history frames:
- effective temporal window:
```

두 dataset의 temporal duration이 크게 다르면 다음 두 protocol을 제안한다.

### Protocol A: frame-count matched
- 동일한 frame 수 사용

### Protocol B: temporal-duration matched
- 가능한 범위에서 동일한 실제 시간 길이가 되도록 frame 수 조정

단, 어느 protocol을 사용할지는 자동 결정하지 말고 비교표를 작성해 연구자가 선택할 수 있게 한다.

---

# 7. Dataset specification abstraction

가능하면 Occ3D-nuScenes / OpenOcc / Occ3D-Waymo가 공통 interface를 사용하도록 한다.

예시 개념:

```python
DatasetSpec(
    dataset_name=...,
    gt_family=...,
    num_classes=...,
    class_names=...,
    free_class_idx=...,
    ignore_index=...,
    point_cloud_range=...,
    voxel_shape=...,
    camera_names=...,
    valid_mask_policy=...,
    gt_resolver=...,
    eval_protocol=...,
)
```

목적은 dataset별 차이를 detector 내부의 `if dataset == ...` 형태로 확산시키지 않는 것이다.

특히 아래 값은 detector/loss/evaluator에 직접 hard-code하지 않는다.

- class 수
- free class index
- ignore index
- camera 수
- point-cloud range
- voxel shape

---

# 8. Supervision interface 요구사항

최종 연구 방법은 Occ3D-provided visibility mask에 의존하지 않는 방향을 목표로 한다.

따라서 Waymo 준비 과정에서도:

- `mask_camera` 존재 여부와 관계없이 baseline 학습은 가능해야 한다.
- final method interface가 dataset-provided visibility annotation을 필수 입력으로 요구해서는 안 된다.
- 현재 selective λ 실험은 oracle/reference 용도로 분리한다.
- visibility-dependent oracle mode와 annotation-free proposed mode를 명시적으로 분리한다.

권장 interface 개념:

```python
voxel_weight = supervision_weight_provider(
    gt=...,
    prediction=...,
    model_input=...,
    metadata=...,
    dataset_spec=...,
)
```

이번 단계에서는 새로운 learned weighting method를 만들지 않는다.

---

# 9. Waymo baseline config 준비

Waymo 데이터를 사용할 수 있는 경우 최소 다음 config를 준비한다.

예시 이름:

```text
projects/STCOcc/configs/
  stcocc_r50_704x256_16f_occ3d_waymo_baseline.py
  stcocc_r50_704x256_16f_occ3d_waymo_baseline_rayiou.py
```

실제 이름은 저장소 naming convention에 맞춘다.

Config에서 명시할 항목:

- dataset spec
- camera names
- num classes
- class names
- free index
- ignore index
- voxel range
- voxel shape
- temporal frame config
- sampler
- optimizer
- training iterations
- evaluation protocol
- prediction export path

---

# 10. 학습 budget 비교

nuScenes와 Waymo의 데이터 크기가 다르므로 단순히 epoch 수를 동일하게 두지 않는다.

다음 세 값을 계산한다.

### A. Native budget
각 dataset/model 원래 학습 recipe

### B. Matched-update budget
Occ3D-nuScenes 실험과 optimizer update 수 동일

### C. Screening budget
method sanity 확인용 짧은 budget

예:

```text
Screening:
- 10~25% full updates

Confirmation:
- matched update budget

Final:
- native or pre-defined paper budget
```

실제 숫자는 데이터 audit 후 계산한다.

---

# 11. Evaluation protocol

최소 지원 지표:

- semantic mIoU
- RayIoU@1
- RayIoU@2
- RayIoU@4
- RayIoU mean

가능하면 기존 공통 reliability evaluator를 통해:

- ECE
- NLL
- AUROC

도 연결한다.

반드시 확인할 것:

- mIoU valid region
- FOV mask 적용 여부
- camera mask 적용 여부
- free class 포함/제외
- ignore label 제외
- Ray origin 생성 방식
- RayIoU class mapping
- origin sampling
- voxel-to-metric distance 변환

Waymo의 평가 domain이 Occ3D-nuScenes와 다르면 metric name 또는 metadata에 명시한다.

예:

```text
mIoU_occ3d_waymo_<domain>
RayIoU_occ3d_waymo_mean
```

---

# 12. Smoke test

Waymo 데이터가 서버에 있으면 full training 전에 반드시 아래를 수행한다.

## 12.1 Data sanity

몇 개 sample에 대해 출력:

- image shape
- number of cameras
- camera names
- GT shape
- unique semantic labels
- class histogram
- free ratio
- invalid / ignore ratio
- FOV / visibility mask ratio
- timestamp
- history length

## 12.2 Forward

- dataset load
- dataloader batch
- forward pass
- output tensor shape
- class channel 확인

## 12.3 Loss / backward

- occupancy loss 계산
- 1 iteration backward
- NaN/Inf 검사
- gradient 존재 여부

## 12.4 Evaluation

소수 sample로:

- mIoU
- RayIoU@1/2/4/mean

end-to-end evaluator 실행 확인.

---

# 13. Waymo 데이터가 현재 없을 경우

데이터가 서버에 없다고 작업을 중단하지 않는다.

다음까지 준비한다.

- dataset adapter
- dataset specification
- config
- GT resolver
- evaluator wiring
- data preparation checklist
- expected directory structure
- info generation command
- sanity check command
- training command
- evaluation command

실제 데이터가 없어 검증할 수 없는 항목은 추측해서 PASS 처리하지 않는다.

상태를 명확히:

```text
BLOCKED: actual Waymo / Occ3D-Waymo data unavailable
```

로 표시한다.

---

# 14. Regression requirements

Waymo 지원을 추가한 뒤 기존 실험이 깨지지 않아야 한다.

최소 regression:

- Occ3D-nuScenes config load
- OpenOcc config load
- Occ3D-nuScenes sample load
- OpenOcc sample load
- forward smoke test
- 기존 class/free index 유지 확인

Waymo support를 위해 기존 Occ3D/OpenOcc behavior를 변경하지 않는다.

---

# 15. 결과 문서

`research_waymo/`를 만들고 최소 다음 파일을 생성한다.

```text
research_waymo/
├── implementation_audit.md
├── dataset_schema.md
├── camera_temporal_audit.md
├── smoke_test_results.md
├── run_commands.md
├── adaptation_blockers.md
└── remaining_issues.md
```

## implementation_audit.md

- 기존 Waymo 지원 여부
- 수정 파일
- hard-coded nuScenes assumptions
- 해결 여부

## dataset_schema.md

- class schema
- free / ignore
- voxel geometry
- GT keys
- masks
- split/sample 수

## camera_temporal_audit.md

- camera configuration 비교
- temporal interval 비교
- 16-frame 실제 duration 비교
- architecture 변경 필요 여부

## smoke_test_results.md

- data sanity
- forward
- loss
- backward
- evaluation

## run_commands.md

- data preparation
- smoke test
- screening training
- full training
- evaluation

## adaptation_blockers.md

architecture 수정 없이는 해결할 수 없는 문제가 있는 경우만 기록한다.

각 blocker에:

- file/function
- assumption
- reason
- minimum required change
- architecture-neutral workaround 가능 여부

를 작성한다.

---

# 16. 이번 단계에서 하지 말 것

별도 승인 전에는 다음을 하지 않는다.

- Waymo full training
- Waymo 대규모 download
- hyperparameter sweep
- Waymo 전용 model architecture 개발
- 새로운 supervision algorithm 구현
- dataset별 다른 proposed method 개발
- 결과를 확인하지 않은 상태에서 generalization 성공 주장

---

# 17. 완료 조건

Waymo preparation 단계는 다음 조건을 만족하면 완료로 간주한다.

### 데이터가 있는 경우

- [ ] GT schema audit 완료
- [ ] camera/calibration audit 완료
- [ ] temporal audit 완료
- [ ] architecture 변경 없이 dataset load
- [ ] forward 성공
- [ ] occupancy loss 성공
- [ ] 1 iteration backward 성공
- [ ] small-set mIoU 평가 성공
- [ ] small-set RayIoU 평가 성공
- [ ] Occ3D/OpenOcc regression PASS

### 데이터가 없는 경우

- [ ] 코드상 Waymo integration point audit 완료
- [ ] dataset spec / adapter 구현
- [ ] config 준비
- [ ] evaluator wiring 준비
- [ ] data preparation 문서 준비
- [ ] run command 준비
- [ ] 검증 불가능 항목을 BLOCKED로 명시

---

# 최종 연구 관점

Waymo 실험의 목적은 단순히 dataset 하나를 더 추가하는 것이 아니다.

검증하려는 것은:

> 동일한 supervision principle이  
> **architecture를 변경하지 않은 상태에서**  
> 다른 실제 주행 dataset과 다른 camera configuration에서도 유지되는가?

이다.

따라서 Waymo integration에서 모델 자체를 크게 수정해야 한다면,
그 결과는 본 연구의 clean dataset-generalization evidence로 사용하기 어렵다.

가능한 한 **data adapter / configuration / evaluation adaptation만으로 동일 모델을 유지**하는 것을 최우선으로 한다.
