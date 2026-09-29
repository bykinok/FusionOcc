# STCOcc 후속 연구 실험 사양 v2.0

작성일: 2026-09-27
문서 성격: 구현·판별 실험·방법 선택·확증을 위한 사양. 새로운 학습 결과를 보고하는 문서가 아니다.

## 0. 적용 범위와 우선순위

이 문서는 v1.0의 자동 실행 순서와 방법 확정 전제를 대체한다. 기존 코드·config·checkpoint·결과는 덮어쓰지 않는다. v1의 s/g/h는 보류된 후보이며, 구현 착수를 승인한 최종 방법이 아니다.

**연구 목표는 그대로 유지한다.** STCOcc에서 free-space supervision 방법을 개발하여 w/o-mask의 RayIoU 이점을 최대한 유지하면서 w/mask의 mIoU 이점을 회복한다. Reliability는 별도 평가한다. 방법을 고정한 후 다른 모델·GT로 전이한다. 다양한 모델의 sweep을 다시 반복하는 benchmark 연구로 바꾸지 않는다.

**새 실행 순서**

`P0 → E1/E2 → E3 → E4 → E5 → 방법 선택(M) → full 검증(F1) → 일반화(F2) → 정리(R)`

- P0: 코드·loss·평가 감사, split와 annotation contract 확정.
- E1/E2: 기존 checkpoint의 사후 보정과 평가 영역 분해. 새로운 모델 학습은 0회지만 inference/rendering 비용은 발생한다.
- E3: 작은 λ, matched global weighting, shuffled weighting의 최소 4개 학습.
- E4: temporal visibility를 먼저 검증하고, 타당할 때만 그룹별 개입 학습.
- E5: 재계산 visibility+상수, ray/rendering 접근, AdaOcc식 영역 제한의 강한 비교군.
- M: 앞의 결과로 필요한 방법만 구현한다. s/g/h를 무조건 구현하지 않는다.

### 0.1 사실·가설·제안의 구분

확인된 사전 관찰은 사용자가 보고한 STCOcc 12-epoch, single-seed 결과다. local 원본 검증 전에는 `user_reported` 상태다.

확인되지 않은 가설:
- input–GT mismatch가 주된 원인이다.
- hidden free는 항상 작은 양수 weight만 필요하다.
- 선택적 가중치가 global free balancing보다 낫다.
- s와 g가 함께 필요하다.
- 작은 λ의 결과가 모든 모델·full protocol에서도 유지된다.

어느 가설도 성공이 예정된 결론으로 취급하지 않는다. 반대로 한 실험의 부정적 결과만으로 넓은 mismatch 가설이나 연구 전체를 자동 기각하지 않는다.

### 0.2 기본 실행 안전 규칙

1. 실제 저장소의 config/log/commit을 먼저 조사한다. 문서의 경로·36 epoch·loss 명칭을 코드보다 우선하지 않는다.
2. git 미커밋 변경, checkpoint, 기존 결과를 보존한다. 삭제·환경 재설치·원격 push·공유 GPU job 중단은 하지 않는다.
3. `--execute`와 승인된 resolved-plan hash가 없으면 dry-run이다.
4. GPU 번호, 최대 GPU-hour, 최대 disk, stage job 목록·한도를 설정하기 전에는 학습을 시작하지 않는다. 평가 job에도 자원 제한을 적용한다.
5. stage별 결과 검토·승인 후 다음 stage로 이동한다. stage 내부 승인된 job만 순차 실행한다.
6. OOM 때문에 batch/history/resolution/optimizer를 임의 변경하지 않는다. 수학적으로 동등한 chunking은 검증 후 허용한다.
7. 같은 초기 checkpoint, seed, train split, 업데이트 수, augmentation, loss policy에서 비교한다.
8. 불리한 결과·실패·탐색 이력을 삭제하지 않는다. 비용 부족은 과학적 실패와 구분한다.
9. 로컬 데이터를 확인하거나 학습하지 않은 상태에서 재현 완료·novelty 확보·CVPR 채택 가능성을 확정하지 않는다.
10. 현재 사양은 완성된 학습 코드가 아니다. 새 CLI와 test는 실제 저장소에 맞게 구현해야 한다.

## 1. 기존 자산과 해석

사용자가 제공한 조건: STCOcc-R50, 704×256, 16-frame temporal fusion, effective batch 16, 2 GPU, 12 epoch, Occ3D-nuScenes val 6,019개. 원래 36 epoch에서 축소했다는 보고다.

| 설정 | mIoU (%) | RayIoU@1 | RayIoU@2 | RayIoU@4 | RayIoU mean | AUROC (%) | ECE (%) | NLL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| w/o mask = λ1 | 30.05 | .318 | .377 | .415 | .370 | 85.14 | 11.77 | .731 |
| w/ mask | 37.03 | .237 | .306 | .362 | .302 | 89.46 | 5.71 | .397 |
| λ0 | 36.10 | .235 | .307 | .367 | .303 | 89.37 | 4.29 | .374 |
| λ.25 | 35.52 | .308 | .375 | .420 | .368 | 87.57 | 6.75 | .440 |
| λ.50 | 32.15 | .314 | .379 | .421 | .371 | 85.45 | 10.05 | .596 |
| λ.75 | 31.74 | .310 | .377 | .419 | .369 | 84.97 | 10.93 | .650 |

- λ.25 대 w/o-mask: mIoU +5.47 point, RayIoU mean −0.2 point, @1 −1.0 point. 완전한 geometry 유지라고 쓰지 않는다.
- classification/sem_scal/geo_scal에만 λ가 적용됐다고 보고됐다. Lovász가 남으면 λ0은 완전한 supervision 제거가 아니다.
- λ1은 w/o-mask checkpoint의 alias이며 별도 독립 학습 결과가 아니다.
- oracle λ.25는 upper bound가 아니라 mask-assisted reference다.
- 리뷰의 18.0B/15.2B voxel 통계는 아직 원본 검증 전이다. 실제 train split·validity·augmentation/head scale에 맞춰 재계산하기 전에는 설정 근거로 쓰지 않는다.
- voxel 비중은 loss/gradient 비중과 같지 않다. Σw로 나누는 mean loss에서는 weight 감소가 전체 loss 감소나 Lovász 상대 비중 증가로 자동 연결되지 않는다.

기존 경로는 P0에서 존재 여부를 확인한다.

```text
/NAS/work_dirs/stcocc_e12_eval/summary.csv
/NAS/work_dirs/stcocc_r50_704x256_16f_occ3d_e12_stage*
projects/STCOcc/configs/
projects/STCOcc/stcocc/detectors/stcocc.py
projects/STCOcc/tools/eval_all_metrics.py
projects/BEVFormer/datasets/save_predictions_metric.py
**/stcocc/losses/focal_loss.py
**/stcocc/losses/semkitti.py
```

## 2. Annotation·visibility·protocol 계약

### 2.1 용어를 분리한다

- `dataset_camera_mask`: 배포된 camera visibility metadata.
- `recomputed_visibility`: occupancy GT와 camera calibration/pose로 다시 계산한 기하적 visibility.
- `input_support`: 실제 이미지/feature에서 얻은 관측 근거 proxy.
- `gt_geometry_utility`: GT와 예측으로 계산한 기하적 학습 필요성 proxy.
- `label_validity`: supervised target이 정의돼 있는지 나타내는 규칙. camera visibility와 동일시하지 않는다.

재계산 visibility를 쓰면 **precomputed camera-mask metadata는 불필요할 수 있지만 visibility-free 방법은 아니다.** 이 조건의 baseline은 반드시 포함한다. 최종 방법의 조건을 몰래 약화하지 말고 annotation manifest로 공개한다.

`valid/ignore`의 정확한 출처를 기록한다. `mask_lidar`를 validity로 사용한다면 그 의존성은 그대로 남는다. 이를 `uses_lidar_mask=false`로 표시하면 안 된다. 반대로 dependency를 없애기 위해 unknown/ignore를 free로 바꾸지 않는다. 유지해야 할 label-validity와 선택적 visibility filtering을 구분하여 baseline/candidate 모두 같은 조건으로 고정한다.

필수 manifest 필드:

```text
uses_dataset_camera_mask_for_weight
uses_dataset_lidar_mask_for_weight
uses_lidar_mask_for_label_validity
uses_recomputed_current_visibility
uses_recomputed_temporal_visibility
uses_actual_input_images_or_features
uses_future_images
uses_future_training_poses_as_target_geometry
uses_historical_gt
uses_dynamic_annotations_for_visibility
teacher_or_depth_pretraining_annotation_lineage
```

future validation/test label·ray·image는 금지한다. training GT나 training trajectory에서 얻은 기하 target은 별도 baseline에서 허용할 수 있지만, 이를 실제 input evidence로 표현하지 않는다. 새 supervision metadata를 썼다면 추가 annotation 조건으로 보고한다.

### 2.2 Protocol과 데이터

- `legacy_e12`: 기존 run의 실제 설정. 보존한다.
- `screen_e12_v2`: P0에서 확정한 새 비교 설정. legacy와 동일성이 확인된 경우에만 기존 결과를 재사용한다.
- `native_full_v2`: 원본 full recipe의 실제 optimizer update·repeat dataset·scheduler를 조사해 확정한다. epoch 이름만으로 동등하다고 하지 않는다.
- 새로운 warmup/EMA는 모든 대조군에 일괄 적용하거나 별도 pair로 비교한다. 기존 결과와 섞지 않는다.

기존 val 전체 결과가 개발에 사용됐음을 유지한다. 이미 고정된 calibration/dev/confirmation scene split이 있으면 재사용한다. 없으면 사전에 고정한 scene-level 20/40/40 분할을 사용하되 confirmation은 retrospective라고 명시한다. 새로운 독립 test라고 부르지 않는다.

현재 모델이 학습한 train scene을 뒤늦게 hold-out으로 떼면 그 checkpoint에 대한 독립 평가가 되지 않는다. 새로운 train-holdout 정책을 택할 경우 모든 비교 모델을 그 scene 제외 조건으로 재학습하고 초기화/pretraining provenance도 기록한다. 비용을 숨긴 채 자동 변경하지 않는다.

## 3. P0 — 구현·평가 감사 (새 학습 0회)

### 3.1 Loss inventory

head/scale/cascade별로 실제 classification이 CE/softmax인지 sigmoid focal인지 조사한다. 파일명으로 단정하지 않는다. free/ignore index, label mapping, native decode, class weight, loss coefficient, reduction scope를 기록한다.

각 loss에 대해 아래를 저장한다.

```text
loss_name, head, scale, label_population,
weight_applied, numerator, denominator, coefficient,
free_logit_gradient_norm, occupied_logit_gradient_norm,
inv_free_logit_gradient_norm, native_vs_modified_equivalence
```

- λ1 bypass 대 all-ones weighted 경로: loss·logit gradient 수치 일치.
- λ0: loss별 invisible-free logit gradient 잔존 여부. Lovász 잔존은 문서화하며 자동 bug 판정하지 않는다.
- 별도 `allterm_harddrop`: 모든 관련 loss 입력 집합에서 해당 voxel을 제외했을 때 직접 logit gradient=0인지 test.
- sem_scal/geo_scal은 결합된 통계 loss다. scalar에 평균 weight만 곱한 것은 voxel weighting이 아니다.
- fixed denominator 대 Σw denominator의 차이를 frozen batch에서 확인하고 실제 gradient 상대 비중을 기록한다.
- mask/GT/augmentation/axis 정렬, 모든 ignore·free 없음·occupied 없음·특정 class 없음·AMP·DDP edge case test.
- auxiliary/cascade loss가 빠지지 않았는지 검사한다. 공유 feature/parameter gradient까지 0을 요구하지 않는다.

### 3.2 Evaluator와 입력 window

- 분산 sample 중복, token 정렬, prediction dimension/axis, GT free/ignore 처리 확인.
- streaming evaluator와 저장 prediction evaluator의 일치 확인.
- native RayIoU origin builder, query direction, GT no-hit 제외, TP/FP/FN 규칙을 읽고 provenance 확보.
- 실제 history token·timestamp·pose·crop·augmentation·feature cache를 기록. 반복 padding frame을 별도 관측으로 세지 않는다.
- native depth head와 그 지도학습 출처를 확인. 있다는 논문 설명만으로 local per-view depth 사용 가능성을 가정하지 않는다.

### 3.3 재사용 판정

기존 run을 `reusable / relabel_only / reevaluate / retrain / blocked`로 분류한다. 버그 수정이 필요하면 code/loss/evaluator version을 올리고 구·신 표를 분리한다. 새 학습으로 넘어가기 전 핵심 정확성 문제를 해결한다.

산출물: `P0_audit.md`, `loss_inventory.json`, `gradient_coverage.csv`, `annotation_manifest.json`, `protocol_manifest.json`, `split_manifest.json`, `legacy_reuse.json`.

## 4. E1 — 사후 bias frontier와 temperature scaling (새 모델 학습 0회)

### 4.1 대상과 비용

처음에는 w/o-mask, w/mask, λ.25의 3개 checkpoint만 비교한다. λ0/.5/.75는 조건부 확장이다. 고정된 calibration/dev subset에서 시작하고 선택된 operating point만 confirmation/full val에서 평가한다. eval cost도 ledger에 기록한다.

원 logits가 없으면 checkpoint를 chunked inference한다. argmax label만 저장한 파일로 bias sweep을 수행하지 않는다. 새 STCOcc training은 하지 않지만 δ/T fitting과 evaluation 계산은 발생한다.

### 4.2 Bias 조정과 frontier

native decode에 적합한 한 개의 전역 free bias δ를 정의한다. categorical logit의 경우 `z_free'=z_free+δ`다. sigmoid/threshold head는 그 native 규칙에 맞춰 별도로 정의한다.

초기 grid는 `[-2,-1.5,-1,-.5,0,.5,1,1.5,2]`. 동일 checkpoint·scene·voxel에 하나의 δ를 사용하고 metric별 δ를 따로 고르지 않는다. 최적점이 경계에 있으면 사전 승인된 한 번의 범위 확장/국소 refinement만 모든 관련 모델에 같은 budget으로 허용한다.

다음을 따로 만든다.
- raw point(δ=0).
- 각 checkpoint의 post-hoc frontier(검사한 δ grid의 nondominated set).
- dev/calibration에서 고른 하나의 operating point를 confirmation에 적용한 결과.

x=RayIoU mean, y=mIoU이며 모두 높을수록 좋다. @1 guardrail도 동시에 적용한다.

**판정 방향을 뒤집지 않는다.**
- λ.25 또는 proposed가 w/o-mask의 feasible frontier보다 더 높은 mIoU를 동일 RayIoU 조건에서 보이면, 사후 bias만으로 설명되지 않는 학습 이득의 근거다.
- w/o-mask+bias가 후보와 동등하거나 우세하면, tested family에서는 결정 경계 설명이 강하다. 공간 선택의 추가 필요성은 미확인이다.
- point가 frontier보다 위라는 이유로 학습 효과를 포기하지 않는다.

유한 grid frontier는 이론적 upper bound가 아니다. frontier 밖의 개선도 완전한 geometry representation 변화나 causal mechanism의 단독 증명은 아니다. 가능하면 후보 checkpoint에도 동일한 bias budget을 주어 frontier-to-frontier를 비교한다.

기본 point 비교는 RayIoU mean 감소 ≤.002, @1 감소 ≤.005와 mIoU 증가를 사용하되, 이는 이전 사양의 개발용 허용치이지 공식 기준이 아니다. 새 결과를 보기 전에 유지/변경 여부를 기록하고 이후 유리하게 완화하지 않는다.

### 4.3 Temperature scaling

각 모델별 calibration split에서 NLL로 하나의 양수 T를 fitting하고, dev/confirmation에서 평가한다. ECE를 보고 bin 수나 T를 매번 다시 고르지 않는다.

결과는 `raw / TS only / bias only / bias then TS`로 분리한다. bias 선택 후 T를 fitting하는 순서를 고정한다.

공통 양수 T가 categorical argmax를 바꾸지 않는지 실제 decoder로 test한다. confidence threshold가 decode에 관여하면 TS가 prediction을 바꿀 수 있으므로 calibration-only 결과와 분리한다. AUROC는 재계산하며 TS 후 동일하다고 가정하지 않는다.

Raw ECE/NLL 개선은 TS 비교 전에도 그 평가 집합 안의 관찰로 보고할 수 있다. 다만 TS를 넘어서는 독자적 calibration 기여인지는 별도 비교로 판단한다. ECE가 λ0보다 나쁘다는 이유만으로 w/o-mask 대비 개선을 부정하지 않는다.

산출물: `E1_bias_metrics.csv`, `E1_operating_points.json`, `E1_frontier_mean.*`, `E1_frontier_ray1.*`, `E1_calibration.csv`, `E1_report.md`.

## 5. E2 — RayIoU origin·거리·support 분해 (새 학습 0회)

### 5.1 원래 평가를 그대로 보존한다

SparseOcc의 RayIoU는 ray 집합을 평가하며 current/past/future ego-path origins를 사용할 수 있다 [R1]. 그러나 local evaluator의 실제 origins를 먼저 확인한다. 논문 설명만으로 현재 실험이 동일하다고 단정하지 않는다.

원래 ray와 GT first-hit 규칙을 그대로 둔 채 tagging만 추가한다. 모든 slice sufficient statistics를 합치면 원래 전체 평가가 정확히 복원돼야 한다. origin subset 점수의 단순 평균으로 전체 RayIoU를 만들지 않는다.

현재 origin이 원래 set에 없으면 새 current-only ray set은 보조 진단으로 생성할 수 있으나, 이를 official decomposition에 포함하지 않는다.

### 5.2 필수 ray record

```text
sample_token, ray_id, origin_id, origin_timestamp, target_timestamp,
origin_time_group, origin_in_actual_input_window,
origin_xyz_in_target_coordinates, ray_direction,
gt_hit_exists, gt_hit_depth, gt_hit_class, gt_hit_voxel,
pred_hit_depth, pred_hit_class, is_tp_at_1_2_4,
origin_to_gt_range, current_ego_to_gt_range,
current_gt_support_group, history_support_group
```

origin_time_group은 current/past/future/unknown이다. current 판정 tolerance와 원 timestamp 출처를 명시한다. 실제 입력이 포함하지 않는 과거 origin은 past-in-window와 별도로 나눈다. 동일 pose의 다른 timestamp를 임의 병합하지 말고 origin 중복 자체를 보고한다.

현재 volume을 과거/future 위치에서 조회하는 것과 그 시점의 scene 자체를 평가하는 것을 혼동하지 않는다. 이 분석에서 GT volume은 동일 target time이다.

### 5.3 Slice와 대안 설명 통제

우선 origin group × GT-hit distance(0–20/20–40/40m 이상, 단위·좌표 명시)별 @1/@2/@4를 계산한다. 유효 sample/ray/class 수와 ΔTP/FP/FN도 저장한다.

가능하면 origin group × current-input GT visibility/support를 교차 분석한다. 분류 기준은 GT와 고정된 observation metadata이며 prediction으로 집합을 바꾸지 않는다. GT hit 자체의 support와 ray가 통과한 free-prefix의 support는 다른 진단이므로 구분한다.

특정 origin 이득이 크면 클래스·거리·ray density 구성이 원인일 수 있다. 공통 stratum의 비교와 표준화 결과를 보조로 제시하되, 공식 RayIoU를 바꾼 결과처럼 보고하지 않는다.

평가 origin이 future라는 사실만으로 해당 target이 실제 입력에서 unobservable이라고 판단하지 않는다. RayIoU와 mIoU는 집합뿐 아니라 matching rule도 다르다. 같은 영역으로 제한해도 완전히 같은 metric이 되지 않는다.

**E2는 성능 변화 위치를 설명하는 실험이지 mismatch의 단독 검증/기각 실험이 아니다.** Temporal-origin 이득과 input–GT mismatch는 동시에 성립할 수 있다.

산출물: `ray_manifest.json`, `E2_origin_metrics.csv`, `E2_origin_distance_metrics.csv`, `E2_support_cross_metrics.csv`, `E2_reconstruction_test.json`, `E2_report.md`.

## 6. E3 — 최소 추가 학습으로 strength·selection 분리

P0의 동일 profile/loss policy에서 seed0부터 수행한다. 초기 승인량은 아래 4개 training job이다. 필요한 anchor 재학습은 별도 승인한다.

| ID | 설정 | 질문 |
|---|---|---|
| E3-L005 | 기존 invisible-free λ=.05 | RayIoU 회복 구간이 더 낮은가? |
| E3-L010 | 기존 invisible-free λ=.10 | 작은 양수 가중치로 충분한가? |
| E3-GMATCH | sample별 평균 free weight를 oracle λ.25와 일치시킨 global weighting | 위치가 아니라 free 총량으로 설명되는가? |
| E3-SHUFFLE | oracle λ.25 free weight multiset을 sample의 free 위치 안에서 permutation | 가중치 합·분포가 같아도 위치가 중요한가? |

G-MATCH에서 sample b의 free 평균은
`alpha_b=(N_visible_free + .25*N_invisible_free)/N_free`다.
모든 valid free에 alpha_b를 적용하고 occupied는 1이다. SHUFFLE은 각 sample의 .25/1 weight 개수와 합을 그대로 유지한다. sample token/epoch/seed로 난수를 고정한다.

이 두 대조군은 mask를 이용해 통제량을 계산하므로 diagnostic reference다. mask-free 방법으로 포장하지 않는다. 원 보고의 85% 통계를 alpha 값으로 바로 넣지 않는다.

동일 평균·분포라도 logits와 weight의 상관관계, sem_scal/geo_scal의 비선형성, 실제 gradient norm은 같지 않다. 그 차이도 기록한다. Binary random drop은 continuous .25/1 permutation과 다르므로 기본 대조군으로 대체하지 않는다.

조건부 추가:
- `E3-GFIX`: train-only dataset 평균 alpha로 고정된 global free weighting. GMATCH가 유망하면 1회. 실질적으로 사용할 단순 규칙의 기준점이다.
- `E3-HARDDROP`: Lovász 등을 포함한 invisible-free all-term drop. 'supervision 존재가 필요한가'를 주장하려면 우선 필요하다. 기존 λ0과 혼동하지 않는다.
- normalization/Lovász 대조: P0가 실제로 중요한 confound를 찾았을 때 baseline/candidate pair만 수행한다.

λ.05나 .10이 좋아도 '강도는 중요하지 않고 존재만 중요'라고 하지 않는다. 측정 범위의 포화 가설만 보고한다. 작은 상수 규칙이 충분하면 복잡한 모듈의 필요성을 보류하지만, 단순한 방법이라는 이유만으로 method paper를 자동 포기하지 않는다.

## 7. E4 — Temporal visibility: 측정 검증 후 개입

### 7.1 E4-OFFLINE을 먼저 수행한다

새 학습 전에 현재 checkpoint를 이용해 아래 두 종류를 구별한다.

**G: 현재 GT snapshot을 입력 window pose에서 ray casting한 geometric pose-union.**
현재 scene을 여러 위치에서 보았다고 가정한 기하적 coverage다. 빠르고 재계산 가능하지만, 실제 과거 영상의 가시성을 그대로 나타내지는 않는다. 이동 객체/가림 변화/volume 밖 경로에 민감하다.

**H: 실제 historical observation support의 annotation-derived proxy.**
가능하면 각 입력 frame의 당시 occupancy/validity/pose로 free line-of-sight를 계산한 뒤 target 좌표계로 정합한다. 현재 free가 과거에도 free였는지, 동적 객체가 경로를 막았는지, 관측 범위 밖은 아닌지 구분한다. historical GT나 필요한 정합 정보가 없으면 `H_UNAVAILABLE`이다. G를 H로 이름만 바꾸지 않는다.

과거 mask의 단순 union도 이동 객체 문제를 해결하지 못한다. 동적 객체 annotation을 쓰면 provenance에 추가하고, primary 분석은 신뢰할 수 있는 정적 공간 또는 conservative dynamic-exclusion 구간으로 제한한다. 현재 GT만으로 정적 여부를 확정할 수 없다면 `static-scene proxy`라는 한계를 남긴다.

### 7.2 Visibility 계산 검증

- calibrated FOV, crop/resize/flip, voxel extent, ray sampling 밀도, depth/range 규칙 일치.
- 첫 occupied까지의 known free만 confirmed visible. unknown/ignore 또는 관측되지 않은 경로 뒤를 free로 채우지 않는다.
- volume 밖에서 시작하는 ray가 volume 밖 occluder를 무시한다면 visibility 불확실로 표시한다.
- 보간·splatting·voxel 경계 오차를 검증하고 duplication을 처리한다.
- 미관측/불명확을 visible/hidden 중 하나로 강제하지 않는다. `visible / confirmed_not_visible / unknown`의 세 상태를 유지한다.
- 같은 raw scene의 반복 frame이나 padding을 추가 관측으로 세지 않는다.
- current recomputed visibility를 배포 mask와 비교하되 정확한 동일성을 가정하지 않는다. 구현/voxel_state/GT refinement 차이를 기록한다.

### 7.3 분석 그룹

현재 invisible-free F를 다음으로 분리한다.
- `F_H`: 현재는 안 보이지만 실제 input history에서 free로 지지되는 group.
- `F_N`: 신뢰 가능한 관측/정합 범위에서 어떤 actual input view도 지지하지 않는 group.
- `F_U`: history 자료 부족, dynamic ambiguity, 범위 밖 경로 등으로 판단 불가.

`F_U`를 `F_N`에 넣지 않는다. G만 계산 가능하면 `G_potentially_visible / G_no_support / G_unknown`로 별도 명명하고 actual observability 주장을 보류한다. F_N도 통계적 추론 불가능의 정답이 아니라 직접 관측 근거 부재에 대한 proxy다.

먼저 group 크기, distance/height/class/surface-proximity, checkpoint별 오류, E2 ray support 교차 결과를 분석한다.

### 7.4 조건부 개입 2개

그룹 신뢰도와 크기가 충분할 때만 각각 F_N-only downweight, F_H-only downweight를 학습한다. 초기 target λ는 기존 .25로 고정하고 나머지 free weight는 1이다. U와 occupied는 원래 weight를 유지한다.

두 그룹의 크기가 다르면 native-population 효과와 matched-mass 비교를 구분한다. 최소 비교는 train-only spatial stratum에서 두 그룹의 matched voxel 수를 고정해 동일한 총 weight reduction을 적용한다. 양쪽의 부족한 stratum은 결과를 보기 전에 제외한다. nonoverlap이 크면 causal comparison은 INCONCLUSIVE로 남긴다.

그룹 간 효과 차이가 있으면 history support와 supervision 효과의 관련성을 지지한다. 차이가 작으면 측정 품질·overlap·seed 변동을 먼저 확인한다. 곧바로 모든 mismatch 메커니즘을 기각하지 않는다.

## 8. E5 — 복잡한 방법이 넘어야 할 단순·선행 비교군

이 단계의 목적은 새 모듈을 정당화하기 위한 무제한 benchmark가 아니다. E1–E4에서 남은 질문에 필요한 비교만 승인한다.

### A. Recomputed visibility + 작은 상수

- `E5-VCURRENT`: 현재 GT와 current camera pose에서 재계산한 visibility로 hidden-free에 ε, 나머지에 1.
- `E5-VWINDOW`: 동일 현재 GT를 actual-input window pose에서 조회한 G pose-union으로 hidden-free에 ε.
- 같은 ε를 비교군에 적용한다. E3에서 고른 값 또는 .25를 사전 기록한다.
- G snapshot approximation, unknown 보수 처리, preprocessing 비용을 보고한다.
- E4의 진짜 historical-support split과 구별한다. 두 baseline이 exact historical visibility는 아니다.
- GT geometry-derived visibility를 사용한다고 정직하게 명시하며, inference에서는 필요하지 않다.

### B. Direct ray/rendering supervision

같은 renderer/origin policy를 사용하여 아래 pair를 우선 비교한다.
- w/o-mask + direct ray loss.
- w/mask 또는 명확히 정의한 partial/allterm-free-drop + direct ray loss.

w/mask와 λ0을 서로 대체하지 않는다. 어느 것을 사용하는지 job ID에 명시한다. 이후 가장 강한 global-free weighting + direct ray loss는 필요할 때 추가한다.

현재 camera-only, actual history pose, virtual/multi-origin을 서로 다른 factor로 둔다. origin이 달라지면 supervision coverage와 학습 신호가 바뀌므로 손실 형태의 효과로만 해석하지 않는다. train geometry만 사용하고 평가 set의 query/label을 weight 생성에 사용하지 않는다.

Generic ray loss는 GaussRender 재현이 아니다. 정식 비교는 공식 Gaussian rendering, viewpoint, depth/semantic loss, adaptation을 확인하고 별도 job으로 수행한다 [R4].

### C. AdaOcc ray-visible region baseline

정확한 논문은 **AdaOcc: Adaptive Forward View Transformation and Flow Modeling for 3D Occupancy and Flow Prediction**, arXiv:2407.01436v1, 2024다 [R3]. Adaptive-Resolution Occupancy Prediction(arXiv:2408.13454)과 혼동하지 않는다.

해당 논문은 ego trajectory 여러 LiDAR origin의 ray visibility와 ray-visible occupied 주변 2m critical region, hard-example mining을 기술한다.

- visibility 영역 제한만 이식하면 `AdaOcc-inspired ray-region control`로 부른다.
- 2m 확장·sampling·uncertainty mining까지 확인해 재현한 경우에만 `adapted AdaOcc training component`로 보고한다. 전체 AdaOcc 모델을 재현했다고 하지 않는다.
- baseline은 free뿐 아니라 occupied loss population도 바꿀 수 있다. free-only 방법과 같은 intervention이 아님을 기록한다.
- future training pose를 쓰면 `extra_target_geometry`로 표시한다. 실제 input observability라고 해석하지 않는다.
- 자료/구현 세부가 없으면 `BLOCKED / not reproduced`로 남긴다.

## 9. M — 방법 후보 선택과 최소 구현

E1–E5의 근거로 하나의 방법 family를 선택해 `method_decision.md`를 작성한다. 다음은 자동 성공 조건이 아니라 의사결정 지침이다.

| 결과 | 다음 방향 |
|---|---|
| post-hoc bias가 주요 이득을 설명 | training의 추가 이득 주장을 좁히고 후보 모두 같은 post-hoc budget에서 비교 |
| global/shuffled가 oracle과 비슷 | 위치 특정성의 필요성 미확인. class balancing·loss 구성부터 검토 |
| 작은 상수나 recomputed visibility가 강함 | 이를 실용 baseline으로 채택하고 넘어설 기술적 요구를 특정. 복잡화 자동 금지 |
| history 관련 선택성이 검증됨 | temporal support-aware supervision 후보를 좁혀 개발 |
| geometry/rendering만으로 충분 | support module 없이 geometry 중심으로 단순화 |
| 남은 오류가 early/late hit와 spatial allocation에 연결 | g 또는 s/g 후보의 추가 정보 검증 후 작은 prototype |

### 9.1 s/g/h 재개 조건

s/g/h는 가설로 유지한다. 재개 전에 frozen prediction·train subset에서 다음을 검사한다.
- s와 recomputed visibility의 AUROC/AUPRC/stratified error. 높은 일치율만으로 s가 쓸모없다고 결론 내리지 않는다. uncertainty·stability·target suitability의 추가 가치를 실제 비교한다.
- g와 native p_occ/free loss, ray coverage, depth residual의 관계.
- `s 낮음 & g 높음` 영역의 크기와 실제 오류. thresholds를 성능을 본 뒤 선택하지 않는다.
- current-origin g가 hidden region에서 거의 0인지, origin을 늘릴 때 생기는 변화.
- squared/Huber depth loss, truncated first-hit renderer 등 실제 정의에 따라 sign이 달라질 수 있으므로 analytic/finite-difference test.

초기 구현은 기존 per-view depth를 쓸 수 있는 경우만 작은 STCOcc prototype을 만든다. 새로운 무거운 plane-sweep fallback은 default로 만들지 않는다. Depth 없는 모델 전이에 별도 모듈이 필요하면 annotation·training cost를 포함해 재승인한다.

v1의 구체적 s/g 수식은 archive의 후보 설명으로 남긴다. 그대로 재개하려면 별도 승인을 받는다. output-space gradient를 parameter influence로 주장하지 않으며, score/weight detach와 second-order 금지 원칙을 유지한다.

### 9.2 방법이 결정된 뒤의 핵심 ablation

같은 profile, loss scope, mean free weight, warmup/search budget에서:
`constant / support-only / geometry-only / joint / hardness / shuffled / direct-ray` 중 선택한 주장에 필요한 최소 집합을 비교한다.

동일 평균 α가 같은 gradient budget을 의미하지 않으므로 loss별 gradient 통계도 기록한다. 작은 양수 residual supervision을 weak prior라고 부를 때는 operational hypothesis임을 밝힌다. Bayesian prior 또는 최소 필요량의 이론을 입증한 것처럼 쓰지 않는다.

## 10. F1/F2 — 확증과 일반화

### F1: native full protocol

방법/하이퍼파라미터/annotation scope/선택 규칙을 freeze한 뒤 다음을 같은 native full profile에서 비교한다.
- w/o-mask.
- w/mask.
- 가장 강한 단순 baseline(E1/E3/E5 결과로 사전 선택).
- proposed.

core seeds는 {0,1,2}. 동일 run만 재사용한다. 필요하면 full seed0 oracle reference와 핵심 component pair를 추가 승인한다. 12ep ablation과 full main result는 표를 분리한다.

각 seed의 paired difference, mean±SD, scene-level bootstrap을 보고한다. scene sufficient statistics를 합쳐 IoU를 재계산한다. voxel/frame을 IID 표본으로 취급하지 않는다. 미유의성을 동등성으로 바꾸지 않는다. 동일한 평가 집합·raw/bias/TS 경로를 유지한다.

RayIoU mean과 @1을 함께 보고, @2/@4도 숨기지 않는다. mIoU 회복률과 절대 차이를 함께 보고한다. 작은 차이가 불확실하면 `INCONCLUSIVE`로 남긴다.

### F2: 일반화

방법이 고정된 뒤 대표 모델 2개와 별도 GT/데이터셋 1개를 가용 자산에서 선택한다. depth-head 없는 모델을 최소 하나 포함하는 것을 목표로 하되, 실제 코드·adapter 가용성을 확인한다.

각 모델의 native recipe 안에서 baseline / strongest simple / proposed를 비교한다. 새 모델마다 광범위한 λ 탐색은 하지 않는다. hyperparameter 재조정 결과는 zero-retuning과 분리한다.

같은 nuScenes의 다른 GT는 GT-pipeline transfer, 다른 dataset은 dataset transfer다. annotation 조건 차이·ignore/unknown·grid mapping을 기록한다. 데이터가 없으면 가상의 결과를 만들지 않는다.

## 11. 결과 수집·판정·보고

### 11.1 저장 정책

```text
research_v2/
  resolved_plan.yaml
  protocol_manifest.json
  annotation_manifest.json
  split_manifest.json
  legacy_reuse.json
  experiment_registry.jsonl
  decisions.jsonl
  audits/
  runs/<run_id>/
  diagnostics/E1_bias/
  diagnostics/E2_origins/
  diagnostics/E4_temporal/
  analysis/summary_raw.csv
  analysis/claim_evidence.json
  reports/stage_<id>.md
  reports/method_decision.md
  reports/professor_summary.md
  reports/final_research_report.md
  reports/paper_experiment_outline.md
```

기존 research/를 자동 덮어쓰지 않는다. 매 run에 source kind(user_reported/locally_verified), config/code/checkpoint/evaluator/split hash, seeds, loss scope, annotation access, origin policy, GPU-hour/VRAM/disk, status를 기록한다.

status는 `PLANNED / APPROVED / RUNNING / COMPLETED / REUSED / FAILED / BLOCKED / SKIPPED / INVALIDATED`다. 완료는 checkpoint·exit code·sample count·finite metrics로 확인한다. raw evaluator artifact와 반올림 표가 다르면 원자료를 우선한다.

### 11.2 결론 규칙

- 결과, 가능한 해석, 대안 설명, 다음 행동을 구분한다.
- E2에서 future-origin 이득이 커도 mismatch를 기각하지 않는다.
- E4 차이가 없으면 검증한 proxy와 조건에서 근거가 부족하다고 한다.
- s의 visibility AUROC가 높아도 downstream 이득/비용 비교 없이 s의 필요성을 기각하지 않는다.
- 작은 λ가 좋아도 모든 공간에서 동일 상수가 최적이라는 결론은 아니다.
- w/o-mask+bias가 후보를 따라오면 추가 복잡성을 정당화할 근거가 약해진다. 학습 개입의 관측 효과 자체가 없었다고 하지는 않는다.
- 공식 target을 넘어서는 hidden completion 개선과 observation mismatch는 양립 가능하다.
- raw calibration 개선과 temperature-scaled 성능은 서로 다른 주장이다.
- 원본 camera mask 미사용과 geometry-derived visibility 미사용은 서로 다른 주장이다.
- CVPR novelty/채택 가능성을 수치 규칙으로 자동 판정하지 않는다.

### 11.3 Claim registry

각 claim에 supporting/counterevidence run IDs, metric population, protocol, estimate, uncertainty, limit를 기록한다.

필수 claim:
C01 legacy partial reweighting 효과, C02 post-hoc bias 이상의 효과, C03 global/free-mass 이상의 selection 효과, C04 origin별 성능 변화 위치, C05 history-support와 효과의 관련성, C06 recomputed-visibility baseline 이상의 효과, C07 direct rendering 이상의 효과, C08 각 module의 추가 기여, C09 annotation 의존성, C10 full-protocol trade-off 개선, C11 raw/TS calibration, C12 transfer와 비용.

상태: `OBSERVED_SINGLE_SEED / SUPPORTED_IN_CHECKED_SCOPE / INCONCLUSIVE / NOT_SUPPORTED / CONTRADICTED / NOT_TESTED`.

### 11.4 보고서

stage 보고서는 핵심 표 1개, 관찰, 해석 제한, 다음 선택을 한국어로 작성한다. 전체 대화를 반복하지 않는다.

교수님 요약은 연구 목표 2–3문장, 핵심 결과표, 현재 가능한 스토리, 다음 결정 3개 이하로 작성한다. 미실행 결과를 예상 숫자로 채우지 않는다.

최종 논문 스토리는 결과에 따라 선택한다. method paper를 목표로 하지만 복잡한 방법을 정당화하도록 결과를 강제하지 않는다. first-study 결과를 second-study의 새로운 발견으로 중복 주장하지 않는다.

## 12. CLI와 완료 조건

다음은 새로 구현할 CLI 계약이다. 기존 training/evaluation entrypoint를 내부적으로 재사용한다.

```bash
python tools/research_pipeline.py audit --plan research_v2/plan.yaml
python tools/research_pipeline.py plan --stage E1 --plan research_v2/plan.yaml
python tools/research_pipeline.py run --stage E1 --plan research_v2/plan.yaml --execute --approve-plan PLAN_SHA256
python tools/research_pipeline.py analyze --stage E1 --plan research_v2/plan.yaml
python tools/research_pipeline.py report --plan research_v2/plan.yaml
```

- E1/E2/E4-offline도 expensive job 수와 재평가 scope를 resolved plan에 명시한다.
- P0 실패 시 training은 막되 정적 분석·문서 정리는 계속할 수 있다.
- E4 historical provenance가 부족하면 G proxy 결과만 보고하고 mismatch 확증 주장은 막는다. 다른 방법 가설 개발까지 자동 차단하지 않는다.
- E5 comparison을 구현하지 못한 경우 이름만 같은 근사로 통과 처리하지 않는다.
- 결과에 따라 새 hypothesis를 제안할 수 있지만 budget 초과 학습을 자동 실행하지 않는다.

완료 기준은 코드 통합, unit/smoke test, 승인 범위의 실제 실험과 실패 기록, 근거에 맞는 결론이다. 어떤 방법이 성공하지 않아도 재현 가능한 부정적 결과와 다음 결정은 유효한 산출물이다.

## 13. 원문 근거

이 문서의 상세 확인 내용과 수정 근거는 `SOURCE_CHECK_AND_CHANGES.md`에 있다. 원문은 local 구현을 대신하지 않는다.

[R1] Fully Sparse 3D Occupancy Prediction, ECCV 2024. https://arxiv.org/html/2312.17118v5 — §4.2 및 evaluation setup.
[R2] Occ3D, NeurIPS 2023. https://arxiv.org/html/2304.14365v3 — §3.3.2–3.3.3, Appendix D.
[R3] AdaOcc: Adaptive Forward View Transformation and Flow Modeling for 3D Occupancy and Flow Prediction. https://arxiv.org/html/2407.01436v1 — §2.3 Ray Visible Mask.
[R4] GaussRender, ICCV 2025. https://arxiv.org/html/2502.05040v3 — §3.2, Table 4. https://valeoai.github.io/publications/gaussrender/
[R5] STCOcc, CVPR 2025. https://arxiv.org/html/2504.19749 — §3.1; actual config/depth provenance는 local 확인.
[R6] On Calibration of Modern Neural Networks, ICML 2017. https://arxiv.org/abs/1706.04599 — temperature scaling.

실험 숫자 [U1]: 사용자가 제공한 2026-09-27 STCOcc sweep 요약. 원본 NAS와 GPU 작업은 이 문서 수정 과정에서 실행하지 않았다.
