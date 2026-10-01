# OpenOcc 구현 변경 사항 (2026-10-01)

전제: `research_openocc/audit.md`(사전 감사)를 사용자가 승인한 뒤, 거기서 식별된 항목을 순서대로 구현했다. 구현 과정에서 사용자가 제공한 `Ref/STCOcc_ori`(원저자 STCOcc repo 미러)를 대조한 결과, audit.md의 일부 결론이 수정되었다 — 아래 "audit.md 수정 사항"을 먼저 읽을 것.

## 0. audit.md 수정 사항 (Ref/STCOcc_ori 대조 결과)

| audit.md 원래 결론 | 대조 결과 | 수정된 결론 |
|---|---|---|
| ⑥ `IterBasedTrainLoop`+`by_epoch=True StepLR`가 "LR decay가 사실상 전혀 적용 안 되는 버그" | 원저자 repo의 occ3d/openocc 두 config 모두 `lr_config=dict(step=[num_iters_per_epoch*total_epoch])` — **단일 스텝이 학습 종료 시점과 정확히 일치하도록 설계된 원본 recipe 자체**. mmengine 포팅판 중 Occ3D 쪽(`stage2_invfree_l025.py`)은 `by_epoch=False, step_size=<iteration count>`로 올바르게 이식했고, openocc_12e는 `by_epoch=True, step_size=<epoch count>`로 이식해 메커니즘은 다르지만 **두 경우 모두 "마지막 iteration에서 1회 감쇠"로 동일하게 귀결**된다. | 버그 아님, 원본 설계를 그대로 재현한 것. 새 config는 이미 검증된 Occ3D 쪽 이식 방식(`by_epoch=False`)을 그대로 사용했다(더 견고하다는 이유로 선택했을 뿐, 수정이 필요해서가 아니다). |
| ⑥ 4.554 배수 "출처 불명, BLOCKED" | 원저자 repo의 `stcocc_r50_704x256_16f_openocc_12e.py:86`에 **동일하게** `* 4.554`가 있고 occ3d 쪽 config들에는 없다 — 포팅 중 생긴 값이 아니라 **원저자 본인이 OpenOcc에만 넣은 값**. 정확한 경험적 근거는 원본 repo에도 주석이 없어 여전히 불명. | "포팅 오류"가 아니라 "원저자의 의도적이지만 미설명된 선택"으로 재분류. 새 screen profile에서는 이 배수를 쓰지 않기로 결정했다(이유는 아래 2.6절). native reference profile(미실행)에는 그대로 보존. |
| ③ `ray_mask2`가 "camera mask와 혼동 위험이 있는 불확실한 필드" | `Ref/STCOcc_ori` 전체에서 `ray_mask` 문자열을 grep한 결과, 로더 밖 어디에서도(detector, loss, Collect3D) 전혀 소비되지 않는다 — 원저자 repo에서도 동일. 게다가 `openocc_v2_ray_mask` 디렉터리 자체가 로컬/NAS 어디에도 존재하지 않는다. | 두 repo 모두에서 **미사용(vestigial) 필드로 확정**. camera mask와 혼동할 실질적 위험이 없음(애초에 아무것도 안 읽으므로). 모든 새 config에서 `load_ray_mask=False` 유지. |
| val_evaluator 미설정 원인 | `Ref/STCOcc_ori`의 openocc_12e config는 `dataset_name`/`eval_metric`을 `share_data_config`를 통해 **dataset 객체**(구버전 mmdet3d 방식, `dataset.evaluate()`)에 전달한다 — 애초에 `OccupancyMetric` 같은 분리된 mmengine evaluator가 없었다. | 이 저장소가 mmengine으로 포팅하면서 **별도 evaluator 클래스를 새로 만들었는데, dataset_name/eval_metric을 그 evaluator로 옮기는 과정이 Occ3D config들에는 됐고 openocc_12e에는 누락**된 것으로 확인됨(근본 원인 규명). |

## 1. 코드 변경 (파일별)

### 1.1 `projects/STCOcc/stcocc/utils/gt_resolver.py` (신규)
`resolve_occ_gt_dir(occ_path, dataset_name)` 단일 함수. `'gts'→'openocc_v2'` 문자열 치환이 기존에 4곳(로더 1곳 + evaluator 3곳)에 독립적으로 중복돼 있던 것을 통합(요구사항 3.1). `tools/compute_metrics_from_file_v2.py`의 (현재 도달 불가능한) openocc 분기도 같은 헬퍼를 쓰도록 맞춤.

### 1.2 `projects/STCOcc/stcocc/transforms/pipelines/loading.py`
- `LoadOccGTFromFileOpenOcc.__init__`에 `load_flow=True` 파라미터 추가(기본값 True로 기존 동작 100% 보존). `load_flow=False`일 때 전체 해상도 및 1/2,1/4,1/8 스케일 모두에서 `flow` 키를 아예 읽지 않는다.
- GT 경로 생성을 `resolve_occ_gt_dir` 호출로 교체.
- `ray_mask2` 경로는 의도적으로 공통 resolver에 포함하지 않음(GT identity가 아니라 미사용 보조 필드이므로 분리 유지, 주석으로 명시).

### 1.3 `mmdet3d/datasets/occ_metrics.py`
`Metric_mIoU.__init__`에 `class_names=None` 파라미터 + `num_classes==17`(OpenOcc) 분기 복원(`Ref/STCOcc_ori`와 동일한 분기를 재적용). 기존 18-class 하드코딩은 로컬에서 radius/height 통계를 추가하면서 사라진 것으로 보이며(이번 세션에 git blame으로 특정 커밋까지 추적하지는 않았음), OpenOcc 전용 버그가 아니라 **이 저장소 고유의 회귀**였다. 검증: `Metric_mIoU(num_classes=17).class_names[0]=='car'`, `Metric_mIoU(num_classes=18).class_names[0]=='others'` (둘 다 이번 세션에 직접 실행 확인).

### 1.4 `projects/STCOcc/stcocc/detectors/stcocc.py`
`get_voxel_loss`에 T6 가드 추가: `camera_mask is None and not _lambda_is_noop`이면 `RuntimeError`. 기존에는 이 경우 `voxel_weight=None`(=균등 가중치)으로 조용히 떨어져, `lambda_inv_free=0.25` 같은 설정이 실제로는 `lambda_inv_free=1.0`과 똑같이 학습되는데도 설정값만 0.25로 기록되는 문제가 있었다(요구사항 T6, audit.md 최우선 항목). 기존 Occ3D config 50개 전수 확인 결과, `lambda_inv_free != 1.0`을 쓰는 모든 config(`invfree_l000/025/050/075/100`, `stage3_adaptive_*`)는 이미 Collect3D에 camera mask 키를 포함하고 있어 **이 가드로 인한 회귀는 없다**(research_openocc/tests 및 기존 research_v2 회귀 테스트로 재확인).

### 1.5 `tools/generate_ms_occ_parallel.py` (신규)
`tools/generate_ms_occ.py`의 `downsample_label`/`downsample_mask` 함수를 **알고리즘 변경 없이 그대로 재사용**하면서 `multiprocessing.Pool`로 샘플 단위 병렬화만 추가. 단일 스레드 기준 샘플당 ~0.55초(34149개 전체 약 5.2시간)였던 것을 28 workers로 실측 ~20 samples/s(전체 약 25~30분)까지 단축. 임시파일→rename 방식으로 중단-재개 안전(멱등적, 이미 존재하는 출력은 건너뜀). 기준 함수와의 bit-identical 일치를 샘플 단위로 직접 검증(아래 2절).

### 1.6 `projects/STCOcc/configs/stcocc_r50_704x256_16f_openocc_occ_only_screen.py` (신규)
`stcocc_r50_704x256_16f_occ3d_e12_stage2_invfree_l025.py`(이미 여러 차례 학습·평가된 검증된 config)를 템플릿으로, GT/클래스 관련 부분만 OpenOcc로 교체:
- `flow_head` 키 자체를 제거(occupancy-only), `load_flow=False`, Collect3D에 `voxel_flows` 없음.
- `lambda_inv_free=1.0` 명시(= `none` 정책, OpenOcc에 camera mask가 없으므로 유일하게 합법적인 값 — T6 가드가 이를 강제).
- `num_classes=17`, `occ_class_names`/`class_weights`는 원저자 openocc_12e config와 동일값 재사용.
- sampler를 `InfiniteGroupEachSampleInBatchSampler`(Occ3D 쪽 이식 방식)로 설정 — openocc_12e의 `DefaultSampler`(scene 연속성 보장 안 됨, audit.md ⑤)를 쓰지 않음.
- `num_iters_per_epoch`는 Occ3D screen profile과 동일한 공식(4.554 배수 없음) 사용 → `total_epoch=12 * num_iters_per_epoch=1758 = 21096` optimizer update, **이는 `research_v2/plan.yaml`의 Occ3D `screen_e12_v2` profile의 `total_optimizer_updates: 21096`과 정확히 동일** — GT만 바뀐 통제된 비교가 되도록 의도적으로 맞춤.
- `load_from`을 실제 존재하는 경로(`projects/STCOcc/pretrain/...`, 원본 openocc_12e의 `pretrained/...`는 존재하지 않는 오타)로 수정.
- `val_evaluator`에 `dataset_name='openocc', num_classes=17, ann_file=<val pkl>, use_image_mask=False, compute_uncertainty_metrics=True, sort_by_timestamp=True` 명시.

### 1.7 `projects/STCOcc/configs/stcocc_r50_704x256_16f_openocc_occ_only_screen_miou.py` (신규)
위 config의 평가 전용 companion(기존 Occ3D `_rayiou` companion과 동일 관례, 체크포인트 공유). `eval_metric='miou'`만 다름 — 요구사항 6.2의 `mIoU_openocc_all_valid` population을 얻기 위한 보조 진단용. 지표 자체의 dict key(`mIoU`)는 Occ3D 쪽 도구와 공유되므로 바꾸지 않았고, population 구분은 `research_openocc/results/summary.csv`의 `miou_scope` 컬럼에서 한다(요구사항 8절 산출물 스키마가 애초에 그렇게 설계돼 있음).

모두 `mmengine.config.Config.fromfile`로 로드 검증 완료(본문 수치 확인됨, 아래 2절).

## 2. 검증 (이번 세션에 실행한 것)

- **Config 로드**: 두 config 모두 `Config.fromfile` 성공. `num_classes=17`, `model`에 `flow_head` 키 없음, `lambda_inv_free=1.0`, `val_evaluator`가 의도한 모든 필드를 가짐, `max_iters=21096`, Collect3D 키에 flow/mask 없음 — 전부 확인.
- **멀티스케일 GT 생성 정확성**: `generate_ms_occ_parallel.py`의 출력이 단일 스레드 원본 함수 출력과 bit-identical(`np.array_equal`)임을 실제 샘플로 확인. shape(100,100,8 / 50,50,4 / 25,25,2), dtype(uint8), class-preservation(다운샘플 결과 클래스 집합이 원본의 부분집합) 모두 확인.
- **Unit test**: `research_openocc/tests/test_openocc_baseline.py` 16 PASS / 0 FAIL / 2 KNOWN_GAP(아래 참고). T1(class mapping), T4(all-ones weight 동등성), T5(mask 없이 lambda=1.0 성공), T6(mask 없이 lambda≠1.0이면 RuntimeError), T7(resolver 일관성), T8(load_flow 플래그가 실제로 읽기를 게이트), T9(멀티스케일 shape/class-preservation, 실제 생성 파일 spot-check) 포함.
- **Occ3D 회귀**: 이번 세션에서 건드린 파일(`occ_metrics.py`, `stcocc.py`, `occupancy_metric.py`, `loading.py`)과 **무관한** 기존 `research_v2/tests/`의 3개 테스트를 재실행:
  - `test_adaptive_lambda_radius.py`: 전체 PASS.
  - `test_rayiou_decomposition_regression.py`: PASS (miou/mave/occ_score 수치 decompose 유무 상관없이 동일).
  - `test_loss_gradient_properties_v2.py`: 13 PASS, 2 FAIL — **이 2개 FAIL은 `semkitti.py`(이번 세션에 손대지 않음)의 `sem_scal_loss`가 "all-ignore"/"all-occupied(no-free)" 같은 극단 edge case에서 ZeroDivisionError를 내는 기존(pre-existing) 버그**다. `git status`로 이번 세션이 이 파일을 전혀 수정하지 않았음을 확인했고, 내 자체 unit test(T3)에서도 동일한 ZeroDivisionError가 동일한 edge case에서 재현되어 교차 확인됨. 수정하지 않음(Occ3D와 공유되는 loss 코드를 이 작업 범위 밖에서 고치는 것은 회귀 위험을 키움) — `KNOWN_GAP`으로 기록만 함.

## 2.5 GPU smoke test (ssh mando-h100, 사용자 승인 후 실행 완료)

20-iteration 학습 + 24-sample 1-GPU 평가를 ssh mando-h100의 `occfrmwrk_h100_new` 컨테이너에서 실행해 전체 파이프라인(config load → 실제 dataset sample → forward → loss → 1-iteration backward → 짧은 학습 → checkpoint 저장 → 소규모 RayIoU 평가)이 end-to-end로 동작함을 확인했다. 코드/멀티스케일 GT는 git을 쓰지 않고 tar+scp+docker cp로 직접 전달했다(커밋/푸시 없음). 상세 결과는 `research_openocc/evaluation_smoke.json`, `results/summary.csv` 참고.

핵심 확인 사항:
- loss가 20 iteration 동안 101.19→96.30으로 단조 감소, grad_norm 유한, NaN/Inf 없음.
- `loss_voxel_flow_*` 항목이 로그에 전혀 없음 — flow_head가 실제로 생성되지 않았음을 실행으로 재확인(요구사항 4.1).
- `lambda_inv_free=1.0`(none 정책)에서 T6 가드가 트리거되지 않고 정상 진행 — 의도한 대로 동작.
- 평가 로그가 `ray_metrics_openocc.py`(occ3d 아님)를 참조 — `dataset_name='openocc'` 라우팅이 실제 실행에서도 올바르게 동작함을 확인(val_evaluator 수정의 실증).
- 평가 결과 테이블이 OpenOcc 17-class 순서(car, truck, ... vegetation)로 출력됨.
- `mIoU≈0.0001`(20-iteration 체크포인트이므로 당연히 거의 무의미한 수치, 성능 주장 아님), `mAVE=nan`/`occ_score=nan`(flow 없음 → 요구사항 4.1이 요구하는 NOT_APPLICABLE과 정확히 일치).
- 2-GPU 평가는 24-sample(단일 scene) 슬라이스가 `InfiniteGroupEachSampleInBatchSamplerEval`의 `groups_num(1) < global_batch_size(2)` 조건에 걸려 실패 — OpenOcc 통합 결함이 아니라 테스트 슬라이스가 너무 작고 단일 scene이라 생긴 하네스 제약. 1-GPU로 전환해 성공.

## 3. 아직 하지 않은 것 (의도적으로 미루거나 범위 밖)

- **전체 학습/평가**: smoke test(20 iteration, 24 sample)까지만 실행했다. 전체 21096-iteration 학습과 전체 6019-sample val 평가는 별도 승인 필요.
- **mando-h100_2 동기화**: 이번 smoke test는 mando-h100에서만 실행했다. mando-h100_2에는 아직 코드/멀티스케일 GT가 전달되지 않았다.
- **`strict_evaluation` 모드**: `OccupancyMetric.compute_metrics()`의 포괄적 예외→0점 fallback은 그대로다. smoke 평가 결과를 받을 때 "0점"이 진짜 0점인지 숨겨진 실패인지 로그를 직접 확인해야 한다.
- **`openocc_native_reference`(flow 포함, 4.554 배수) 실행**: config는 원본 그대로 보존했지만, 이 자체도 멀티스케일 GT 부재로 실행 불가능했고(이번 세션에 생성된 멀티스케일 GT로 이제는 파일 자체는 있음) val_evaluator 문제도 안 고쳤으므로 평가는 여전히 불가능 — 별도 작업으로 분리.
- **method_plugin**: 아직 proposed method가 없으므로 구현하지 않음. 연결 지점(`get_voxel_loss`의 `camera_mask`/`voxel_weight` 인터페이스)만 공통으로 유지.
- **eval_all_metrics.py용 openocc `_rayiou` 사본 config**: 안 만듦 — screen.py 자체가 이미 `eval_metric='rayiou'`를 기본으로 하므로 `tools/test.py`로 직접 평가 가능, 이 2-config 관례이 필요한 건 Occ3D 쪽에서 mIoU-전용 학습 config에 rayiou 평가를 붙이고 싶을 때였음.

## 4. 요구사항 문서 8절 질문에 대한 답 (업데이트)

- **기존 무엇을 재사용했고 무엇을 고쳤는가?** 위 1절 참고. 재사용: `STCOccLoadOccGTFromFileOpenOcc` 전체 구조, `ray_metrics_openocc.py`, `tools/generate_ms_occ.py`의 다운샘플 알고리즘, Occ3D `stage2_invfree_l025` config 전체 레시피. 수정: GT resolver 통합, `load_flow` 플래그 추가, `Metric_mIoU` class_names 복원, T6 가드 추가.
- **현재 OpenOcc GT 파일이 실제로 있는가? 어떤 version인가?** 있음, `openocc_v2.1`. 전체 해상도는 처음부터 100% 존재. 멀티스케일은 이번 세션에 처음 생성(§5 완료 시 100%).
- **mask/flow 없이 occupancy baseline 학습과 평가가 가능한가?** 코드/config 상으로는 이제 가능(막혔던 4가지 이유 모두 해결: 멀티스케일 GT 생성, flow 완전 분리, val_evaluator 완비, selective λ 안전장치). **실제로 돌려서 확인한 것은 아직 없음** — 다음 단계가 1-iteration smoke test인 이유.
- **native reference와 새 screen profile의 차이는?** §1.6/baseline_plan.yaml 참고 (4.554 배수 유무, GPU/batch 분배, flow 유무, sampler by_epoch 메커니즘).
- **Occ3D 회귀검사를 통과했는가?** 통과(§2). 2건의 FAIL은 이번 세션과 무관한 기존 버그로 확인됨.
- **final annotation-free method는 구현됐는가, 연결 지점만 있는가?** 연결 지점만. method_plugin 자체는 미구현.
- **장시간 실행을 위해 추가로 필요한 자원과 승인 항목은?** Smoke test(§2.5)는 완료됐다(PASS). 남은 것: (1) 전체 21096-iteration 학습 승인(mando-h100 및/또는 mando-h100_2), (2) 전체 val(6019 sample) 평가 승인, (3) 두 사이트를 병렬로 쓰려면 mando-h100_2에도 코드+멀티스케일 GT 동기화 필요.
