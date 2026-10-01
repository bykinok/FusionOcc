# Claude Code 요구사항 — FusionOcc/STCOcc의 OpenOcc GT 확장
작성일: 2026-10-01
성격: 구현·검증 요구사항. 구현 또는 학습 완료 보고서가 아니다.
확인한 원격 기준: bykinok/FusionOcc / stcocc-selective-free-supervision
기준 commit: d5a359e4692f7a3aed9a47595430e216560365c2
현재 작업 디렉터리가 기준 commit과 다르면 차이를 기록하고 현재의 사용자 변경을 보존한다.

## 1. 작업 목적과 범위

현재 Occ3D-nuScenes 기반 STCOcc 연구 코드를 보존하면서, 동일한 nuScenes 입력에
STCOcc가 사용한 OpenOcc GT를 연결하여 occupancy-only 학습과 평가를 수행할 수 있게 한다.
OpenOcc는 이 작업에서 OccNet / Scene as Occupancy 계열의 GT 및 STCOcc용 openocc_v2를 뜻한다.
CONet / OpenOccupancy benchmark와 혼동하지 않는다.

이번 작업은 먼저 **GT adapter, baseline, 평가, 향후 방법 연결 지점**을 완성하는 것이다.
새로운 visibility 추정기나 adaptive weighting 방법을 임의로 발명하거나,
현재 Occ3D mask-assisted λ 규칙을 mask 없이 동작하는 방법이라고 이름만 바꾸지 않는다.

원본 데이터가 같은 Occ3D→OpenOcc 비교는 GT-pipeline transfer다.
다른 원본 dataset에 대한 검증이나 zero-shot model transfer라고 부르지 않는다.
각 GT에서 baseline/proposed를 동일 조건으로 학습하는 것은 training-method transfer다.

실행 범위:
1. 감사, 코드 수정, synthetic unit test, config 생성, 실행 계획 생성은 수행한다.
2. 실제 GPU smoke test는 지정·승인된 GPU에서만 수행한다.
3. 전체 학습·대규모 평가·대규모 다운로드는 별도 승인 전 실행하지 않는다.
4. 기존 checkpoint·GT·config·결과를 덮어쓰거나 삭제하지 않는다.
5. git reset/clean, 환경 재설치, 원격 push를 하지 않는다.


## 1.1 최종 검증 전략과 portability 제약

이번 연구의 최종 검증 축은 다음과 같이 고정한다.
- method development: STCOcc + Occ3D-nuScenes
- GT generalization: STCOcc + OpenOcc-nuScenes
- dataset generalization: 가능한 한 동일 STCOcc architecture + Occ3D-Waymo
- model generalization: 첫 번째 연구에서 사용한 기존 occupancy 모델 일부에 동일 supervision method 적용

따라서 OpenOcc 지원을 구현할 때도 향후 Occ3D-Waymo 및 다른 기존 모델에 이식하기 쉬운 구조를 우선한다.
Dataset/GT 차이를 이유로 backbone, view transformer, temporal fusion, occupancy representation을 임의로 재설계하지 않는다.
아키텍처 변경이 필요하면 자동 적용하지 말고 별도 adaptation requirement로 보고한다.

SSCBench-KITTI-360은 이번 주 검증 대상에서 제외한다. 이유는 일반적인 사용 protocol이 전방 SSC 중심이라
현재 첫 번째 연구에서 사용한 surround occupancy 모델들의 architecture/input formulation을 바꿀 가능성이 크기 때문이다.
이 연구에서는 dataset generalization과 architecture modification을 가능한 한 분리한다.

최종 supervision core가 의존하지 않아야 할 항목:
- Occ3D 제공 camera visibility mask
- 특정 dataset의 class ID
- 특정 camera 수
- 특정 voxel grid shape
- STCOcc 전용 depth head / feature tensor
- dataset-specific fixed distance threshold

DatasetAdapter / ModelAdapter / SupervisionWeightProvider와 같이 역할을 분리할 수 있도록 인터페이스를 설계하되,
아직 존재하지 않는 proposed method를 임의로 구현하지 않는다.

## 2. 이미 존재하는 코드와 주의 사항

다음 파일을 먼저 조사하고 재사용한다.
- projects/STCOcc/configs/stcocc_r50_704x256_16f_openocc_12e.py
- projects/STCOcc/stcocc/transforms/pipelines/loading.py
  - STCOccLoadOccGTFromFileOpenOcc
- projects/STCOcc/stcocc/detectors/stcocc.py
- projects/STCOcc/stcocc/losses/focal_loss.py
- projects/STCOcc/stcocc/losses/semkitti.py
- projects/STCOcc/stcocc/evaluation/occupancy_metric.py
- projects/STCOcc/stcocc/datasets/의 OpenOcc/RayIoU 관련 코드
- tools/export_occ_logits.py
- tools/compute_metrics_from_file.py
- projects/STCOcc/tools/eval_all_metrics.py

OpenOcc/RayIoU 관련 코드는 실제 파일명을 찾아 기록한다.
이 목록의 파일이 있다고 해서 실행 가능한 OpenOcc pipeline이 완성됐다고 가정하지 않는다.

원격 기준에서 확인된 내용:
- OpenOcc config: 17 channels, free ID=16. Occ3D의 18 channels/free ID=17과 다르다.
- OpenOcc config에는 flow_head와 voxel_flows 수집이 포함되어 있다.
- loader는 semantics 외에 flow를 각 scale에서 무조건 읽는다.
- GT 경로는 'gts' 문자열을 'openocc_v2'로 바꾸어 생성한다.
- optional ray_mask2 로딩 경로가 있으나 camera visibility mask와 같은 정보라고 가정하면 안 된다.
- OpenOcc config에는 num_iters_per_epoch 계산의 4.554 배수가 있다.
- IterBasedTrainLoop와 by_epoch=True StepLR가 같이 있다.
- train sampler는 DefaultSampler로, 현재 Occ3D의 recurrent/grouped sampler와 다르다.
- val_evaluator는 type만 지정하여 dataset_name/ann_file/eval_metric을 명시하지 않는다.
- 평가기 기본 dataset_name은 occ3d이며, 일부 오류 경로가 0점 결과로 조용히 반환한다.
- model의 raw-logit export GT 경로와 외부 평가기의 GT 경로가 따로 구현되어 있다.
- 현재 selective λ는 camera mask가 없으면 조용히 no-op이 될 수 있다.

## 3. 데이터 계약과 공통 GT resolver

### 3.1 명시적 설정
다음 정보를 config/manifest에 명시한다. 실제 이름은 기존 코드와 맞춰도 된다.
- source_dataset: nuscenes
- gt_type: openocc_v2
- gt_version 및 전처리 출처
- raw_dataset_root
- occupancy_gt_root
- train_info_file, val_info_file
- sample_token / scene_token / timestamp
- semantic class names / raw-to-training mapping / free_index / ignore_indices
- point_cloud_range / voxel_size / grid_shape / axis_order / coordinate_frame
- native_validity_source
- training_visibility_source: none
- evaluation_scope와 evaluator version
- load_flow: false (이번 occupancy-only profile)

GT는 하나의 resolver/adapter로 읽는다.
훈련 loader, inline evaluator, file evaluator, logit exporter가 같은 GT identity를 반환해야 한다.
새 profile에서 임의의 문자열 replace로 GT를 선택하지 않는다.
기존 Occ3D legacy 경로는 호환 wrapper로 유지할 수 있지만, 새 OpenOcc profile은
명시적인 root와 token 기반 경로를 우선한다.
OpenOcc가 없다고 Occ3D labels.npz를 자동 fallback하지 않는다.

### 3.2 실제 데이터 감사
사용자가 허용한 프로젝트·데이터 root 범위에서 파일 존재를 확인한다.
train/val의 여러 scene과 sample을 고정해서 keys, dtype, shape, class histogram,
valid/ignore 처리, 세부 scale, timestamp/pose 대응을 기록한다.
원본 nuScenes split이 같더라도 해당 GT release의 누락 token이 있는지 검사한다.
파일이 없거나 출처가 불명확하면 BLOCKED로 보고하고 필요한 경로/자산을 정확히 적는다.
GT를 실제로 읽지 않은 상태에서 schema가 검증됐다고 쓰지 않는다.

### 3.3 Label identity
현재 STCOcc OpenOcc config의 예상 training class order는 다음과 같다.
0 car
1 truck
2 trailer
3 bus
4 construction_vehicle
5 bicycle
6 motorcycle
7 pedestrian
8 traffic_cone
9 barrier
10 driveable_surface
11 other_flat
12 sidewalk
13 terrain
14 manmade
15 vegetation
16 free

실제 GT가 이 순서로 전처리됐는지 확인한다. 필요 시 명시적인 mapping을 한 곳에 구현한다.
label 0을 ignore나 free로 취급하지 않는다. OpenOcc의 car=0이다.
raw label/unknown/invalid를 임의로 free로 채우지 않는다.
class_weights는 OpenOcc reference 또는 train-only 통계에서 가져오고 출처를 기록한다.
Occ3D의 class order, weights, 18-channel output을 그대로 재사용하지 않는다.
모든 cascade predictor와 final head, loss, decoder, evaluator의 channel 계약을 검사한다.

### 3.4 GT validity와 visibility는 다른 정보
dataset-native label validity는 유지한다. visibility annotation 비사용이
unknown/invalid voxel까지 정답으로 사용하는 것을 뜻하지 않는다.
validity를 결정하는 원본 필드와 의미를 manifest에 기록한다.
모든 voxel이 valid인 전처리라면 그 전제와 원본 정보를 어떻게 처리했는지 기록한다.
mask_camera, mask_lidar, ray_mask2를 서로 바꾸어 쓰지 않는다.

## 4. Occupancy-only 학습 profile

### 4.1 Flow를 완전히 분리
새 primary profile은 occupancy-only로 한다.
- flow_head=None 또는 미생성
- load_flow=False를 loader에서 지원
- voxel_flows 및 scale별 flow를 Collect3D에서 요구하지 않음
- flow loss 계산 안 함
- mAVE, flow 기반 Occ score는 null/NOT_APPLICABLE
- 0-flow를 예측했다고 좋은 mAVE/Occ score가 나온 것처럼 보고하지 않음

train_flow=False만 바꿔 충분하다고 가정하지 않는다.
실제 detector는 flow_head 존재를 기준으로 loss를 계산하는 경로가 있으므로
forward_train/simple_test/metrics의 분기를 함께 검증한다.
기존 occupancy+flow reference config는 보존한다.
새 occupancy-only 실험을 원 논문의 joint-flow 숫자와 직접 동등한 재현으로 부르지 않는다.

### 4.2 Initialization과 recipe
Occ3D의 학습된 occupancy head를 조용히 OpenOcc head에 로딩하지 않는다.
공통 image/stereo pretrain은 사용할 수 있으나 로딩 성공·누락·shape mismatch를 보고한다.
OpenOcc baseline/proposed에는 같은 초기화, split, update 수, optimizer, LR, augmentation,
batch, raw/EMA evaluation 정책을 적용한다.

별도 profile을 만든다.
- openocc_native_reference: 확인된 reference recipe; 필요 시 flow 포함
- openocc_occ_only_screen: 현재 방법 개발과 비교 가능한 12-epoch-equivalent screening
- openocc_occ_only_full: 실제로 확인한 full recipe를 사용하는 확증 profile

기존 OpenOcc config의 '12e'를 현재 Occ3D의 '12e'와 같다고 가정하지 않는다.
4.554 multiplier의 의미를 공식 설정과 실제 sample/update 수로 확인한다.
새 screen profile에서는 총 optimizer update와 sample exposure를 명시하고
reference와 달라진 점을 기록한다.
IterBasedTrainLoop의 scheduler/checkpoint by_epoch 설정을 검증한다.
원본 reference config 자체를 무조건 고쳐 덮어쓰지 않는다.

### 4.3 Temporal input
history [16,8,4], stereo adjacent input, recurrent state reset과 sequence sampler의 계약을 확인한다.
단순 random sampler가 history를 깨뜨리지 않도록 기존 검증된 temporal sampler를 재사용한다.
scene 경계, rank별 temporal stream, evaluation ordering, duplicated sample을 검사한다.
GT의 미래 정보와 실제 모델에 입력하는 이미지 history를 혼동하지 않는다.
dataset별 fps가 달라지면 frame count뿐 아니라 관측 시간 폭도 기록한다.

### 4.4 Model geometry와 기존 loss
현재 OpenOcc config는 Occ3D와 같은 범위를 선언하지만 실제 GT 파일로 재확인한다.
[200,200,16], [100,100,8], [50,50,4], [25,25,2]가 해당 release에 맞는지 검사한다.
multi-scale GT가 없으면 임의 nearest downsample을 하지 말고 원본 aggregation을 확인한다.
변환이 필요하면 별도 출력 경로·version·class-preservation test를 사용한다.

focal의 native radial coefficient와 class weights를 baseline/proposed 모두 유지한다.
Lovász, sem_scal, geo_scal의 free index·분모·excluded class를 명시한다.
이번 GT 연결 작업에서 loss normalization까지 조용히 바꾸지 않는다.
다른 grid를 향후 지원할 수 있도록 geometric metadata를 전달하되,
CUDA renderer를 포함한 완전 범용화를 검증 없이 완료했다고 쓰지 않는다.

## 5. Supervision policy의 분리

최소한 다음 정책을 명시적으로 구분한다.
1. none: 모든 valid voxel을 native objective로 학습
2. global_free: 모든 valid free voxel에 동일 α를 적용하는 단순 대조군
3. legacy_occ3d_camera: 제공된 camera mask를 쓰는 기존 진단용 λ
4. method_plugin: 이후 확정할 annotation-free 방법이 voxel weight를 생성
5. optional gt_raycast_reference: OpenOcc GT로 재계산한 visibility를 쓰는 별도 대조군

이 중 이번 필수 구현은 none과 공통 weight interface다.
global_free는 저비용 대조군으로 구현 가능하지만 최종 proposed method라고 부르지 않는다.
method_plugin의 실제 방법이 아직 없으면 NOT_IMPLEMENTED로 명확히 실패시킨다.

중요:
- OpenOcc mask가 없다고 all-true/all-false camera mask를 만들어서는 안 된다.
- Occ3D mask를 같은 token의 OpenOcc GT에 복사하지 않는다.
- ray_mask2를 camera mask로 rename하지 않는다.
- legacy selective λ<1인데 mask가 없으면 오류를 내고 실행을 막는다.
- gt_raycast_reference는 추가 opt-in이며 input-derived visibility나 mask-free method로 속이지 않는다.
- 없음/누락/미구현으로 인해 proposed가 no-op baseline이 되었는지 테스트한다.
- 기존 λ=.25/거리별 규칙의 바로 이식과 final method transfer는 다른 실험이다.


## 5.1 향후 Occ3D-Waymo와 기존 모델 이식을 위한 구현 제약

이번 OpenOcc 작업에서 아래 항목은 가능한 한 config/adapter에서 공급하게 한다.
- num_classes, class_names, free_index, ignore_index
- point_cloud_range, voxel_size, grid_shape, axis_order
- camera count / camera ordering / calibration metadata
- temporal sampling metadata / frame interval
- GT resolver / native validity / evaluation scope

현재 코드의 6-camera, 18-class, free=17, [200,200,16], nuScenes token/path 가정이
method core나 loss core에 남지 않도록 신규 코드는 generic하게 작성한다.
단, 기존 legacy path를 광범위하게 리팩터링하여 회귀 위험을 만들지 않는다.
새 adapter 경로부터 명시적으로 분리하고 regression test를 둔다.

Occ3D-Waymo 지원은 이번 구현 범위가 아니지만, 향후 동일 STCOcc architecture를 유지한 채
loader/config/class head/evaluator만 적응할 수 있는지 audit note를 남긴다.
architecture 변경이 필요해 보이는 지점은 파일/가정/예상 변경량을 목록화한다.

첫 번째 연구의 다른 모델에도 method를 적용할 예정이므로,
STCOcc 전용 tensor 이름이나 module path를 SupervisionWeightProvider의 필수 입력으로 만들지 않는다.
모델별 adapter가 canonical prediction/geometry/support 정보를 제공할 수 있게 경계를 정의한다.

## 6. 평가 계약

### 6.1 Evaluator config 명시
dataset_name='openocc', GT resolver, ann_file, num_classes=17,
class names, free_index=16, point_cloud_range, timestamp ordering을 명시한다.
OccupancyMetric의 기본값에 의존하지 않는다.
OpenOcc와 Occ3D class list가 다른 RayIoU 구현을 정확히 라우팅한다.

### 6.2 mIoU
OpenOcc의 공식/기존 evaluation protocol을 먼저 확인한다.
추가 연구 지표로 all-valid semantic mIoU를 쓰면 이름을
`mIoU_openocc_all_valid` 등으로 표시하고 평가 population을 문서화한다.
camera-visible Occ3D mIoU와 같은 지표라고 부르거나 원시 점수를 직접 비교하지 않는다.
missing camera mask를 오류 없이 모두 visible인 것처럼 기록하지 않는다.
confusion matrix는 전체 합으로 집계하고 free를 제외한 16개 semantic class 평균을 계산한다.
공식 protocol의 absent-class 및 ignore 규칙을 검증하여 그대로 사용하거나 별도 지표로 명명한다.
SC IoU / free IoU / occupied recall은 필요 시 보조 진단으로 분리한다.

### 6.3 RayIoU
기존 OpenOcc reference와 같은 label order, grid/좌표계, origin sampling,
first-hit/no-hit 처리, threshold 1/2/4m, macro averaging을 확인한다.
@1/@2/@4 및 mean을 모두 full precision JSON에 저장한다.
OpenOcc의 공식 규칙과 다르게 unknown 처리나 origin을 바꾸면
새 protocol ID로 분리하고 published score와 같은 표에서 직접 순위화하지 않는다.
GT invalid를 occupied/free로 임의 변환하지 않는다.
native evaluator가 이 정보를 어떻게 처리하는지 확인하고 필요하면
연구용 valid-prefix ray 평가를 별도로 정의한다.

### 6.4 Reliability 및 예측 저장
같은 checkpoint/prediction과 명시된 valid population에서 AUROC/ECE/NLL을 계산한다.
ECE bins, msp 정의, free 포함 여부, calibration transform을 기록한다.
필요하면 all-valid와 GT-occupied-only 결과를 나누고 서로 같은 지표처럼 섞지 않는다.
streaming prediction 저장과 통계를 사용한다.
17-channel OpenOcc logits에 Occ3D 18-class reshape를 사용하지 않는다.
model.export_occ_logits, file evaluator, inference saver의 GT resolver를 통일한다.

주 metric은 동일 semantic prediction을 사용한다.
서로 다른 test pass를 쓰면 두 pass의 sample별 prediction equality를 확인한다.
원본 sensor input은 동일하게 유지하고 inference에 GT가 들어가지 않는지 검사한다.

### 6.5 실패를 0점으로 숨기지 않음
새 strict_evaluation 모드에서는:
- ann_file 없음, GT 없음, 잘못된 gt_type/shape, label mismatch, renderer 오류는 exception.
- dummy Metric_mIoU fallback을 사용하지 않음.
- 필수 metric 미생성/NaN 또는 sample count 불일치는 FAILED.
- 실패를 mIoU=0,count=0인 정상 결과로 CSV에 적지 않음.
- NULL + reason과 정상 0점을 구분.
- checkpoint/config/GT/split/evaluator hash 없는 저장 예측을 무조건 재사용하지 않음.

## 7. 필수 테스트와 완료 조건

T1. OpenOcc class mapping의 one-hot/perfect prediction:
16 semantic classes 및 free=16을 올바르게 처리하는지 확인.
T2. target/prediction label permutation 및 inverse mapping 회복 검사.
T3. free/occupied/unknown/ignore, all-ignore, all-free, no-free batch 검사.
T4. all-one voxel weight에서 baseline loss와 직접 logit gradient 동등성.
T5. mask 필드 없는 OpenOcc baseline forward/loss/backward 성공.
T6. legacy selective λ가 필요한 mask 없이 실행되면 명확한 오류.
T7. 같은 sample의 loader/evaluator/exporter GT path+hash+class mapping 일치.
T8. flow 없는 NPZ에서 occupancy-only 정상; flow-enabled missing flow는 오류.
T9. multi-scale GT와 augmentation의 축·좌표 정렬.
T10. temporal scene boundary reset, index/token dedup, prediction 재현성.
T11. OpenOcc perfect-volume RayIoU synthetic test 및 native evaluator parity.
T12. 최소 실제 sample에서 saved prediction와 inline evaluation 동일성.
T13. 대표 기존 Occ3D config+고정 입력의 회귀검사; legacy 결과의 계산 범위 보존.
T14. 최종 method가 실제로 연결되면 visibility 필드 제거/변경에 weight·loss가 불변인지 검사.
T15. eval 오류가 성공/0점으로 포장되지 않는지 failure-injection test.

전체 training 없이 다음까지 완료한다:
config load → 실제 dataset sample → forward → loss → 1 iteration backward
→ 짧은 승인된 smoke run → 소규모 동일 prediction 평가.
데이터/GPU가 없으면 수행 범위와 BLOCKED 항목을 구분한다.

## 8. 산출물

기존 research_v2를 덮어쓰지 않고 별도 경로를 사용한다.
research_openocc/
  audit.md
  gt_schema.json
  resolved_paths.json
  protocol_manifest.json
  label_mapping.json
  data_manifest.json
  tests/
  configs/
  commands.md
  implementation_changes.md
  baseline_plan.yaml
  evaluation_smoke.json
  results/summary.csv

summary 필드:
source_dataset,gt_type,gt_version,model,method,profile,
class_schema_hash,grid_schema_hash,split_hash,seed,
uses_supplied_visibility,uses_recomputed_visibility,validity_source,
flow_enabled,updates,raw_or_ema,
miou_scope,miou,ray1,ray2,ray4,raymean,auroc,ece,nll,
code_hash,checkpoint_hash,evaluator_hash,status,notes

보고서에는 다음을 답한다.
- 기존 무엇을 재사용했고 무엇을 고쳤는가?
- 현재 OpenOcc GT 파일이 실제로 있는가? 어떤 version인가?
- mask/flow 없이 occupancy baseline 학습과 평가가 가능한가?
- native reference와 새 screen profile의 차이는 무엇인가?
- Occ3D 회귀검사를 통과했는가?
- final annotation-free method는 구현됐는가, 연결 지점만 있는가?
- 장시간 실행을 위해 추가로 필요한 자원과 승인 항목은 무엇인가?

## 9. 다음 실험은 구현 완료 후 별도로 승인

우선 baseline-none 1개로 end-to-end training pipeline을 확인한다.
그 후 동일 OpenOcc profile에서 strongest-simple-baseline과
이미 확정된 proposed method를 비교한다.
Occ3D에서 튜닝한 핵심 method hyperparameter는 우선 고정해 이식한다.
dataset별 추가 tuning을 했다면 zero-retuning 결과와 분리한다.
아직 proposed가 없으면 baseline 준비까지만 완료하고 성능 향상을 만들어내지 않는다.
다른 GT의 raw mIoU가 높다는 이유만으로 transfer 성공이라 결론 내리지 않는다.

## 참고와 확인 범위
[G1] 사용자 repository의 위 commit에서 OpenOcc config, loader, evaluator를 읽었다.
https://github.com/bykinok/FusionOcc/tree/d5a359e4692f7a3aed9a47595430e216560365c2
[G2] 공식 STCOcc repository — version/data preparation/native recipe의 비교 대상
https://github.com/lzzzzzm/STCOcc
이 사양을 작성한 대화에서는 실제 NAS GT·checkpoint를 로딩하지 않았으며 학습도 실행하지 않았다.
