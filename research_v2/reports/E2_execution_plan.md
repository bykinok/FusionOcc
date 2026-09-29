# E2 실행 계획 — origin/거리별 RayIoU 분해

**상태 업데이트 (2026-09-27, 사용자 자원 승인 이후): 아래 evaluator 변경은 구현 완료.** 3개 checkpoint에 대한 실 실행(회귀 테스트 포함)만 아직 남음. 원래 이 문서에서 "새로 만들어야 함"이라고 적었던 거리(radius)/높이(height) 분해는 **구현 도중 이미 존재한다는 것을 발견**했다 (`ray_metrics_occ3d.py`의 `RADIUS_BINS`/`HEIGHT_BINS`/`_print_bin_table`, 아마 이전의 다른 연구에서 추가됨). 실제로 새로 만든 것은 **origin의 시간 오프셋(time_offset_s) 태깅뿐**이며, 그 외에는 이미 있던 `_accumulate_bin_stats`/`_print_bin_table` 인프라를 그대로 재사용했다. 아래 1~2절은 이 발견을 반영해 갱신했다.

## 0. 목적과 P0의 연결

P0(`loss_inventory_v2_addendum.json:rayiou_origin_builder_audit`)에서 이미 코드로 확인한 사실:
RayIoU origin은 같은 scene 안의 과거+미래 pose를 모두 후보로 삼아 8개로 subsample하며, 모델의 실제 16-frame 입력창과 무관하다.

E2는 여기서 한 걸음 더 나아가 **"그래서 λ-스윕의 RayIoU 이득 중 얼마가 future-origin에서 나오는가"**를 답한다. v2 핵심 판정 수정에 따라:
- future-origin 이득만으로 mismatch를 기각하지 않되, **origin별로 이득을 분해해서 보고**해야 그 판단 자체가 가능해진다.
- 지금 evaluator는 8개 origin의 ray를 전부 합쳐 하나의 TP/FP/FN 카운트로 뭉갠다 — 분해가 안 되는 구조.

## 1. 현재 evaluator가 origin 정보를 어디서 잃는지 (코드 확인)

- `nuscenes_ego_pose_loader.py`의 `nuScenesDataset`이 8개 origin의 pose(위치+시간 index)를 만들지만, 이 metadata는 `occupancy_metric.py:_compute_rayiou`가 `nusdata.__getitem__(index)`를 호출하는 순간 **origin별 시간 태그 없이 좌표 텐서만** `ray_metrics_occ3d.py:main()`로 넘어간다.
- **거리(radius)/높이(height) 분해는 이미 구현돼 있었다** (`RADIUS_BINS=[(0,20),(20,35),(35,inf)]`, `HEIGHT_BINS`, `_accumulate_bin_stats`/`_print_bin_table`, `main()` 마지막에 두 테이블 출력) — E2가 새로 만들 필요 없음, 최초 이 문서 작성 시점의 오판이었음.
- 정말 없는 것은 **origin이 reference frame 기준 과거인지 미래인지, 몇 초 떨어졌는지**뿐이었다.

## 2. 구현한 evaluator 변경 (완료, 2026-09-27)

### 변경 A: `nuscenes_ego_pose_loader.py` — origin에 시간 provenance 부착
`nuScenesDataset.__getitem__`의 로직을 `_compute_origins()` 헬퍼로 추출(기존 2-tuple 반환은 완전히 그대로 유지, 회귀 위험 0)하고, 새 메서드 `get_origins_with_time_offsets(idx)`를 추가해 origin별 `time_offset_s`(초 단위 부호 있는 값: reference=0, 과거<0, 미래>0)를 함께 반환하도록 확장했다. 기존 `__getitem__` 호출부(occupancy_metric.py, nuscenes_dataset_occ.py의 legacy 경로)는 코드 한 글자도 바뀌지 않았다.

### 변경 B: `ray_metrics_occ3d.py` — ray별 origin 태그를 TP/FP/FN 집계까지 관통
- `process_one_sample(..., origin_time_offsets=None)`: 주어지면 각 origin(t)이 만든 ray들에 `time_offset_s`를 8번째 컬럼(index 7)으로 추가(append, 기존 0:3=xyz/3=class/4=depth/5:7=flow 레이아웃은 그대로 — 기존 소비 코드가 고정 컬럼 인덱스로 접근하므로 삽입이 아닌 append여야 안전).
- `main(..., lidar_origin_time_offset_list=None)`: `None`이면(기존 모든 호출부) **기존과 완전히 동일한 동작·출력** — `decompose_by_origin = lidar_origin_time_offset_list is not None`으로 전체를 게이트. 값이 주어지면 이미 있던 `_accumulate_bin_stats`/`_print_bin_table`을 그대로 재사용해 `ORIGIN_TIME_BINS = [(-inf,0),(0,1e-6),(1e-6,inf)]` (`past/reference/future`) 테이블을 추가로 출력.

### 변경 C: `occupancy_metric.py` — opt-in 플래그로 노출
`OccupancyMetric.__init__`에 `rayiou_decompose_by_origin: bool = False` 추가(기본값 False로 기존 전체 파이프라인 무변경 보장). `True`일 때만 `_compute_rayiou`가 `get_origins_with_time_offsets`를 호출하고 `lidar_origin_time_offset_list`를 만들어 `main()`에 전달.

### 회귀 안전장치 (구현됨, 아직 GPU로 미검증)
A/B/C 모두 **집계 방식만 추가**하고 판정 룰(threshold, class match)은 건드리지 않았다. `rayiou_decompose_by_origin=False`(기본값) 경로는 코드상 완전히 기존과 동일한 함수 호출 그래프를 타므로 회귀 위험이 사실상 없다. `True` 경로는 mIoU/mAVE/occ_score가 `False`일 때와 **정확히 같은 값**을 내는지(새 origin 테이블만 추가로 출력) 작은 샘플 subset으로 실제 GPU 회귀 테스트가 아직 필요 — 3절 참고.

## 3. 실행 순서 (승인 시, 전부 미실행)

```bash
# (1) 변경 A/B/C를 ray_metrics_occ3d.py, nuscenes_ego_pose_loader.py에 구현
# (2) 회귀 테스트: 기존 6개 legacy checkpoint 중 1개로 old-aggregate == new-aggregate-summed 확인
python research_v2/tests/test_rayiou_decomposition_regression.py \
  --config <기존 config 1개> --checkpoint <기존 ckpt 1개>

# (3) 본 실행: nomask / cameramask / l025 3개 checkpoint에 대해 origin/거리별 분해 RayIoU 산출
python projects/STCOcc/tools/eval_all_metrics.py \
  <config> <checkpoint> --rayiou-decompose-by origin_time_offset,distance_bucket \
  --out research_v2/diagnostics/E2_decomp/<name>.csv
```

## 4. 필요 자원

| 항목 | 추정 |
|---|---|
| evaluator 코드 변경 + 회귀테스트 | GPU 불필요 (1개 checkpoint로 CPU/소량 GPU 검증) — 단, DVR 렌더러 자체가 CUDA 전용이라 회귀 실행 1회는 GPU 필요 |
| 본 실행 (3 checkpoint × 전체 val RayIoU 1회씩) | 기존 RayIoU 평가와 동일 비용 — checkpoint당 ~1 GPU-hour 수준(기존 로그 기준 추정, 재확인 필요), 3개 = ~3 GPU-hour |
| 디스크 | CSV 수십 KB, 무시 가능 |

## 5. 미설정 항목 (실행 전 기록, 이제 전부 해소됨)

1. ~~GPU 번호, evaluation GPU-hour 예산~~ → 승인됨, 실행함
2. ~~회귀 테스트 대상~~ → 실제 checkpoint 대신 synthetic data로 대체(합성 grid + 합성 origin), 실제 체크포인트 불필요, GPU만 필요 → `research_v2/tests/test_rayiou_decomposition_regression.py`, PASS
3. 거리 버킷은 이미 구현돼 있던 `RADIUS_BINS=[(0,20),(20,35),(35,inf)]`를 그대로 사용(새로 정할 필요 없었음)

## 6. 결과 (2026-09-27)

### 회귀 테스트: PASS
`decompose_by_origin=False`(전 호출부 기본값)와 `True`가 miou/mave/occ_score에서 **완전히 동일한 값**을 내고, `True`일 때만 origin 테이블이 추가로 출력됨을 synthetic data로 확인. 기존 evaluator는 전혀 영향받지 않음.

### 전체 val set(6019샘플) origin 분해 — 3개 checkpoint 전부 실행

| | past(과거) IoU@1/2/4 | reference(기준) IoU@1/2/4 | future(미래) IoU@1/2/4 |
|---|---|---|---|
| nomask | 0.330/0.388/0.424 | 0.383/0.455/0.497 | 0.297/0.357/0.397 |
| cameramask | 0.245/0.312/0.366 | 0.296/0.385/0.448 | 0.224/0.295/0.351 |
| l025 | 0.320/0.384/0.425 | 0.376/0.453/0.502 | 0.290/0.359/0.407 |

**핵심 발견**: 3개 checkpoint 전부 **reference > past > future** 순서로 일관됨. future-relative-to-reference origin이 오히려 가장 어려운 조건으로 나타났다 — v2가 우려한 "RayIoU가 future-origin을 통해 부당한 이득을 줄 수 있다"는 가설은 이 3개 checkpoint에서는 지지되지 않는다. v2 판정 규칙에 따라 이 결과는 결론(예: "그러니 mismatch가 아니다")을 내리는 근거가 아니라, 향후 후보 평가 시 함께 보고해야 할 사실적 근거로 취급한다.

원본 로그: `/NAS/work_dirs/research_v2_e2_decomp/{nomask,cameramask,l025}_rayiou_decomp/`
