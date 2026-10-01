# Waymo Smoke Test 결과

작성일: 2026-10-01 (초안), **업데이트: 2026-10-01 같은 날 — 실제 구현 후 smoke test 완료, 전부 성공**

## 상태: 성공 (SUCCESS)

`WAYMO_EXTENSION_REQUIREMENTS_KO.md` 12절이 요구한 data sanity → forward → loss/backward → evaluation을 전부 실행했다. 실행 위치는 **로컬 호스트(2x RTX 3090)**다 — ssh mando-h100/mando-h100_2가 아니다. 이유: Waymo 원본 센서 데이터(5-camera 이미지 ~198,068장×5, LiDAR point cloud ~198,068개, occupancy GT 수십GB)가 로컬에만 있고 원격 사이트의 `/NAS`(OpenOcc 작업에서 겪은 것과 동일한 문제, 호스트별로 분리된 저장소)에는 전혀 동기화되지 않았다 — 이번 smoke test 규모에 맞지 않는 대용량 전송이라 로컬 실행을 선택했다(OpenOcc 때는 230MB라 동기화했지만, Waymo는 최소 수십GB 단위). 전체 학습을 승인받으면 그때 동기화 방식을 다시 결정해야 한다.

## 실행 중 발견하고 고친 버그 (구현 과정의 일부, 전부 config/도구 레벨 — architecture 아님)

실제로 돌려보지 않았다면 몰랐을 버그 4개를 찾아 고쳤다. 전부 "코드를 읽기만 해서는" 발견하지 못했을 것들이다:

1. **`tools/create_data_waymo_occ.py`의 `lidar_path` 필드 누락**: 초안에서 `lidar_path=''`로 비워뒀다가(당시엔 "LiDAR 포인트를 직접 쓸 필요 없다"고 잘못 판단), `STCOccPointToMultiViewDepth`가 `kwargs['gt_depth']`를 **무조건** 요구한다는 것(`stcocc.py:1101`, 조건문 없음)을 소스에서 재확인하고 수정. 실제 `.bin` 경로를 채워 넣음.
2. **`STCOccLoadPointsFromFile`의 `load_dim` 불일치**: nuScenes 기본값 `load_dim=5`를 그대로 썼다가 `FileNotFoundError` 다음 단계에서 포인트 배열이 깨질 뻔했다 — Waymo의 velodyne `.bin`은 **6 floats/point**(`num_features=6`, 파일 크기가 6으로 나누어떨어짐을 직접 확인)임을 실측하고 `load_dim=6`으로 수정.
3. **`BEVFormer`의 `num_cams=6` 기본값**: `use_cams_embeds=False`라 수치적으로는 무해하다고 판단했었으나, 5-camera 입력과의 broadcasting shape 자체가 안 맞아 `RuntimeError`. `num_cams=5` 명시로 해결(`adaptation_blockers.md` 업데이트 참고).
4. **`OA_SpatialCrossAttention`의 독립적인 `num_cams=6` 기본값**: 코드 조사 단계에서 놓쳤던 두 번째 인스턴스. 동일하게 `num_cams=5` 명시로 해결.

이 4개 전부 **config 또는 도구 스크립트 수정**으로 해결됐고, STCOcc architecture 코드 자체(backbone/view transformer/temporal fusion/occupancy head)는 단 한 줄도 바꾸지 않았다 — `WAYMO_EXTENSION_REQUIREMENTS_KO.md`의 핵심 제약을 끝까지 지켰다.

## 12.1 Data sanity — 통과

- 5-camera 이미지 경로(`image_idx` 디코딩, camera 2/3 스왑 보정 포함) 실제 이미지 로드 성공.
- LiDAR point cloud(6 floats/point) 로드 및 `STCOccPointToMultiViewDepth`를 통한 `gt_depth` 생성 성공(`loss_depth` 값이 매 iteration 정상적으로 계산됨, §12.3 참고).
- Occupancy GT(voxel04, raw free=23→15 remap, FOV ignore-마스킹) 로드 성공, 멀티스케일(1/2,1/4,1/8) 포함.

## 12.2 Forward — 통과

20-iteration 학습 전체에서 forward pass가 매 iteration 성공. 5-camera 입력, BEVFormer backward projection, temporal fusion, occupancy head 전부 정상 동작(코드 수정 없이, config 2곳 수정만으로).

## 12.3 Loss / Backward — 통과

```
Iter 1/20:  loss=115.31  loss_depth=2.85  (+ 16개 occupancy loss term, 4 scale x 4 loss type)
Iter 20/20: loss=108.43  loss_depth=2.84
```

20 iteration 동안 total loss가 115.31 → 108.43으로 단조 감소, grad_norm 유한(373→329), NaN/Inf 없음. `loss_depth`가 매 iteration 정상 계산됨 — LiDAR→카메라 calibration 체인(이번 세션에 직접 유도하고 실측 LiDAR 투영으로 검증한 그 체인)이 실제 학습에서도 올바르게 동작함을 재확인. Checkpoint 저장 성공(`iter_20.pth`, 722MB + EMA 241MB). GPU 메모리 사용량 ~10GB(RTX 3090 24GB 중, 여유 충분).

학습 데이터는 scene 549(198 frame) 하나로 제한했다 — 2개 이상 GPU를 쓰려면 최소 2개 이상의 서로 다른 scene(그룹)이 필요하다는 것도 실측으로 재확인했다(OpenOcc 때와 동일한 `groups_num < global_batch_size` assertion, 1-GPU로 우회).

## 12.4 Evaluation(mIoU) — 통과 (RayIoU는 이번 범위 밖, 의도적)

`OccupancyMetric`을 Waymo용으로 배선하지 않기로 한 결정(§ 아래 참고)에 따라, **`tools/eval_waymo_smoke.py`(신규, 이번 세션 작성)**로 24-sample mIoU를 직접 계산했다:
- validation scene 000(완전히 생성된 scene)의 24 frame을 사용.
- 학습에 쓴 것과 **동일한 `LoadOccGTFromFileWaymo`**로 GT를 다시 읽어(train/eval GT identity 일치 보장), 모델 예측과 비교.
- 24 sample 전부 정상 처리, per-class confusion matrix 출력 성공(`building` class IoU 0.26 등 — 20-iteration 체크포인트이므로 전반적으로 낮은 수치는 예상된 것이지 버그가 아님. 성능 주장 아님, 파이프라인이 끝까지 도는 것의 증거).

**RayIoU는 구현하지 않았다** — `ray_metrics_waymo.py` 같은 전용 커널이 존재하지 않고(기존 `ray_metrics_occ3d.py`/`ray_metrics_openocc.py`는 각 GT의 class/좌표 가정이 달라 그대로 재사용 불가), 참고한 CVT-Occ 레퍼런스 구현도 Occ3D-Waymo에 대해 mIoU만 보고하고 RayIoU는 쓰지 않는다(실제 커뮤니티 관행과 일치). `research_waymo/remaining_issues.md`에 향후 과제로 기록.

## `OccupancyMetric`을 Waymo용으로 배선하지 않은 이유 (의도적 결정)

`occupancy_metric.py`의 내부 GT 재로딩 코드(3곳)가 Occ3D/OpenOcc 공유 포맷(`labels.npz` 파일명, `semantics` 키)을 하드코딩하고 있어, Waymo(`_04.npz`, `voxel_label` 키, raw free=23)를 지원하려면 **Occ3D/OpenOcc와 공유하는 파일에 새 분기를 추가**해야 한다 — 이는 회귀 위험이 있는 변경이라, 이번 세션 범위에서는 보류하고 대신 독립된 standalone 스크립트로 소규모 평가 요건을 충족시켰다. `tools/test.py`로 이 config를 그대로 평가하면 (val_evaluator가 의도적으로 비워져 있어) 조용히 0점이 나온다 — 이는 알려진, 문서화된 상태다.

## 최종 요약

| 항목 | 상태 |
|---|---|
| Data sanity | ✅ |
| Forward | ✅ |
| Occupancy loss (4 scale) | ✅ |
| Depth loss (LiDAR 투영 기반) | ✅ |
| 1-iteration backward | ✅ |
| 20-iteration short run | ✅ (loss 단조 감소) |
| Checkpoint save/load | ✅ |
| Small-set mIoU | ✅ (standalone 스크립트) |
| Small-set RayIoU | 범위 밖 (의도적, 커뮤니티 관행과 일치) |
| Occ3D-nuScenes/OpenOcc regression | ✅ (config load + unit test 16 PASS 재확인) |
| Architecture 코드 수정 | **없음** |
