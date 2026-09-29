# E1 실행 계획 — bias frontier + temperature scaling

**상태 업데이트 (2026-09-27, 사용자 자원 승인 이후):**
- **중대 발견**: `tools/export_occ_logits.py`는 기본값으로 `mask_camera==True`인 voxel만 export하도록 설계돼 있었다(`_process_dense_sample`). 이 스크립트를 그대로 썼다면 우리가 정확히 연구하려는 invisible-free 영역이 통째로 사라진 채 bias sweep을 하는, 조용히 틀린 결과를 냈을 것 — 실행 직전에 코드를 읽다가 발견했다. `--keep-invisible` 플래그를 추가해(기본값 False로 기존 동작·다른 모델 호환성 100% 유지) 전체 voxel을 export하고 `mask_camera` 값은 필터가 아니라 별도 배열로 보존하도록 수정했다.
- calibration split(1202 샘플, scene 30개)에 대해 nomask/cameramask/λ=0.25 3개 checkpoint의 raw logit+GT export를 실행 중(로컬 GPU 0/1, 순차: nomask+cameramask 병렬 → l025).
- `research_v2/tools/bias_frontier_sweep.py` 구현 완료(δ grid, overall/invisible mIoU + invisible free-IoU 계산). export 완료 후 바로 실행 가능.
- 아래 원안은 큰 틀에서 유효하며, 위 변경사항만 반영해 갱신했다.

## 0. 큰 발견: 필요한 evaluator 인프라의 상당 부분이 이미 존재한다

코드를 읽어보니 **temperature scaling과 raw-logit export는 이미 완전히 구현돼 있다** — 새로 만들 필요가 없다.

| 구성요소 | 상태 | 위치 |
|---|---|---|
| Temperature scaling (T로 logit 나누기) | ✅ 이미 구현됨 | `stcocc.py:749-750` (`self.temperature`, model 생성자 인자) |
| Raw logit + GT 동시 export | ✅ 이미 구현됨 | `stcocc.py:775-792` (`export_occ_logits=True` 플래그) |
| Logit → NLL 기준 T 자동 fitting | ✅ 이미 구현됨, 다른 모델에도 범용 | `tools/train_temperature.py` (STCOcc 명시 지원, `--free-class 17` 권장) |
| Logit export 커맨드 | ✅ 이미 구현됨 | `tools/export_occ_logits.py` |
| **δ(bias) grid sweep + 재-decode** | ❌ **없음, 새로 만들어야 함** | 아래 4절 설계 참고 |

이건 이전에 다른 calibration 연구(예: `stcocc_r50_704x256_16f_occ3d_36e_miou_unified_calib_eval.py`가 `temperature=1.6136`을 이미 쓰고 있음)에서 만들어둔 인프라로 보인다. **우리 λ-스윕 체크포인트(nomask/cameramask/l025)에는 아직 한 번도 적용된 적이 없다.**

## 1. 대상 (spec 4.1)

우선 3개 checkpoint만: A0(nomask), A1(cameramask), A2(λ=0.25). λ0/.5/.75는 조건부.

## 2. 실행 순서 (승인 시)

```bash
# (1) calibration/development split 생성 (E1 필수 선행, GPU 불필요)
python research_v2/tools/make_scene_split.py --seed 20260927 --out research_v2/splits/
# (아직 미작성 -- 스크립트 자체는 P1 스타일로 준비만 됨, 아래 4절)

# (2) 각 checkpoint의 raw logit + GT export (GPU 필요, chunked)
python tools/export_occ_logits.py \
  projects/STCOcc/configs/stcocc_r50_704x256_16f_occ3d_e12_stage1_nomask.py \
  /NAS/work_dirs/stcocc_r50_704x256_16f_occ3d_e12_stage1_nomask/iter_21096.pth \
  --cfg-options model.export_occ_logits=True \
  --output research_v2/diagnostics/E1_bias/nomask_logits.npz
# (cameramask, invfree_l025도 동일 패턴)

# (3) calibration split에서 T 자동 fitting (기존 스크립트 그대로 재사용, GPU 불필요 -- npz만 읽음)
python tools/train_temperature.py \
  --logits-file research_v2/diagnostics/E1_bias/nomask_logits.npz \
  --free-class 17 \
  --output research_v2/diagnostics/E1_bias/nomask_temperature.pt

# (4) δ grid bias sweep -- 새 스크립트 필요, 아래 4절 설계. 미구현.
python research_v2/tools/bias_frontier_sweep.py \
  --logits-file research_v2/diagnostics/E1_bias/nomask_logits.npz \
  --delta-grid -2 -1.5 -1 -0.5 0 0.5 1 1.5 2 \
  --free-class 17 \
  --out research_v2/diagnostics/E1_bias/nomask_bias_sweep.csv

# (5) bias 적용 후 재-decode된 occ_results로 mIoU 재계산은 (4)의 출력에서 CPU로 가능.
#     RayIoU 재계산은 DVR 렌더러가 CUDA 전용이라 GPU 필요 -- 이 단계부터 "expensive evaluation".
```

## 3. 필요 자원

| 항목 | 추정 |
|---|---|
| logit export 3건 (A0/A1/A2) | 각 ~30분 GPU (mIoU-eval과 비슷한 inference 비용) |
| T fitting 3건 | CPU만, 수분 |
| δ grid bias sweep (mIoU만) | CPU만, 수분 (한 번 export된 logit 재사용) |
| δ grid bias sweep (RayIoU 포함) | **GPU 필요** — δ값 9개 × 3 checkpoint × RayIoU 1회 비용(~30분) = 최대 ~13.5시간 GPU (전부 재계산 시) — 실제로는 raw point(δ=0)만 우선 RayIoU 계산하고 나머지는 mIoU-only로 frontier를 그린 뒤 필요한 지점만 RayIoU 확인하는 식으로 줄일 수 있음 |
| 디스크 | logit npz 1개 ≈ 200×200×16×18×4바이트 ≈ 92MB/sample × 6019 = 너무 큼 → **calibration+development split(전체의 60%)만 export** 권장, 그래도 수십GB |

## 4. 새로 설계해야 하는 것: `bias_frontier_sweep.py` (미구현, 설계만)

```python
def apply_free_bias(logits: np.ndarray, free_class: int, delta: float) -> np.ndarray:
    """logits: (..., num_classes). native decode와 동일하게 softmax 전에 free 채널에만 delta를 더한다.
    categorical(softmax) head 기준 -- STCOcc는 sigmoid 기반 loss를 쓰지만 실제 DECODE는
    stcocc.py:751 `pred_voxel_semantic.softmax(-1).argmax(-1)` 이므로 decode 시점 기준으로는
    softmax+argmax가 native decode다. bias_rule=RESOLVE_NATIVE_DECODE_EQUIVALENCE(template)
    이 값으로 이미 해소됨: softmax 입력에 free 채널만 +delta.
    """
    out = logits.copy()
    out[..., free_class] += delta
    return out

def decode(logits_biased: np.ndarray) -> np.ndarray:
    return logits_biased.argmax(-1).astype(np.uint8)  # softmax는 argmax에 영향 없음 (monotonic)
```

- 각 δ에 대해 `decode()`로 새 `occ_results`를 만들고, 기존 `occupancy_metric.py`의 mIoU 계산 경로(GT와 비교)를 **직접 재사용**(model rerun 불필요, 이미 export된 GT+logit만 있으면 CPU로 충분).
- RayIoU까지 재계산하려면 이 `occ_results`를 `eval_all_metrics.py`가 기대하는 pkl 포맷으로 포장해서 `compute_metrics_from_file.py` 또는 DVR 렌더러 경로에 넣어야 함 — 이 부분만 GPU 필요.
- `bias_rule`은 template이 `RESOLVE_NATIVE_DECODE_EQUIVALENCE`로 뒀는데, 위에서 이미 해소: STCOcc의 실제 decode가 softmax+argmax이므로 categorical bias 규칙(`z_free'=z_free+δ`)이 그대로 맞다. sigmoid-head용 별도 규칙은 불필요.

## 5. 미설정 항목 (실행 전 기록, 이제 전부 해소됨)

1. ~~GPU 번호~~ → 로컬 0,1 + 원격 mando-h100_2 0,1 확인, 실제로는 로컬만 사용
2. ~~calibration/development split 미생성~~ → `research_v2/tools/make_scene_split.py`로 생성 완료
3. ~~`bias_frontier_sweep.py` 미작성~~ → 작성 및 실행 완료

## 6. 결과 (2026-09-27, calibration split 1202샘플 기준)

**주의**: 아래 mIoU는 **invisible 포함 전체 voxel** 기준이며, 기존에 보고된 w/mask 학습 mIoU(camera-visible voxel만)와는 다른 지표다. cameramask가 여기서 크게 낮게 나오는 것은 모순이 아니라 invisible 영역에서 loss를 받지 않은 결과가 그대로 드러난 것.

| checkpoint | δ=0 (native) overall_miou | 탐색된 정점 | invisible_free_iou @δ=0 |
|---|---|---|---|
| nomask (λ=1) | 0.2267 | δ=-0.5, 0.2271 (거의 이미 최적) | 0.956 |
| l025 (λ=0.25) | 0.2182 | δ=+2.0, 0.2287 | 0.923 |
| cameramask (w/mask) | 0.1226 | δ=+5.0, 0.1348 (δ=+3까지는 포화 안 돼 그리드 확장함) | 0.327 (매우 낮음) |

**핵심 해석**: cameramask는 자기 최적 δ에서도 nomask/l025의 native 수준에 한참 못 미친다 — post-hoc bias 보정만으로는 invisible-region 미보정 문제를 해소하지 못하며, 이는 근본적으로 학습 시점에 invisible 영역 supervision이 없었다는 것과 일치하는 결과다.

전체 CSV: `research_v2/diagnostics/E1_bias/{nomask,cameramask,cameramask_extended,l025}_bias_sweep*.csv`

**RayIoU까지의 bias sweep은 이번 패스에서 실행하지 않음** (DVR 렌더러 재실행 비용 때문에 mIoU-only로 스코프를 좁혔다는 원안 그대로 유지) — 필요 시 후속 단계에서 결정.
