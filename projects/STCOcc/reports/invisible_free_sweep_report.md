# STCOcc Invisible-Free Supervision Weight Sweep — 결과 요약

**작성일**: 2026-09-27
**모델**: STCOcc-R50, 704×256, 16 frame temporal fusion
**데이터셋**: Occ3D-nuScenes (val 6019 samples)
**학습**: 12 epoch, 2×GPU 분산 (effective batch 16), 원본 STCOcc protocol 기준 (optimizer/LR schedule/augmentation/architecture/loss 구성 불변, epoch 수만 36→12로 축소)

---

## 1. 배경 및 목적

STCOcc는 학습 시 camera visibility mask를 적용해 카메라로 보이지 않는(invisible) voxel의 loss를 제외할지 여부를 선택할 수 있다. 두 baseline을 비교하면 뚜렷한 trade-off가 있다:

- **w/ camera mask**: mIoU는 높지만 RayIoU(LiDAR ray 기반 평가)는 낮음
- **w/o camera mask**: RayIoU는 높지만 mIoU는 낮고 calibration(ECE/NLL)도 나쁨

이번 실험의 목적은 **새로운 방법을 개발하는 것이 아니라**, invisible & free voxel에 대한 loss weight `λ`(lambda_inv_free) 하나만 0~1 사이에서 조절해서 이 trade-off를 완화할 수 있는지 확인하는 것이다.

- `λ = 1.0` → 기존 w/o mask와 동일 (모든 voxel 동일 가중치)
- `λ = 0.0` → invisible-free voxel supervision만 제거 (invisible-occupied는 유지 — w/ camera mask와는 다름, w/ camera mask는 invisible-free와 invisible-occupied를 모두 제거함)

---

## 2. Stage 1 — Baseline 재현

| Setting | mIoU | RayIoU@1 | RayIoU@2 | RayIoU@4 | RayIoU mean | AUROC(msp) | ECE | NLL |
|---|---|---|---|---|---|---|---|---|
| w/o camera mask | 30.05 | 0.318 | 0.377 | 0.415 | 0.370 | 85.14 | 11.77 | 0.731 |
| w/ camera mask | 37.03 | 0.237 | 0.306 | 0.362 | 0.302 | 89.46 | 5.71 | 0.397 |

w/ mask가 mIoU +6.98pt, ECE −6.06pt(개선), NLL −0.334(개선) 우위인 대신 RayIoU는 −0.068 낮다. 예상된 trade-off가 명확히 재현되었다.

---

## 3. Stage 2 — Invisible-Free Supervision Weight Sweep

w/o mask를 기본으로 하고, invisible & free voxel에만 `λ`를 적용해 5개 지점을 스윕했다.

| λ | mIoU | RayIoU@1 | RayIoU@2 | RayIoU@4 | RayIoU mean | AUROC(msp) | ECE | NLL |
|---|---|---|---|---|---|---|---|---|
| 0.00 | 36.10 | 0.235 | 0.307 | 0.367 | 0.303 | 89.37 | 4.29 | 0.374 |
| **0.25** | **35.52** | 0.308 | 0.375 | 0.420 | **0.368** | 87.57 | 6.75 | 0.440 |
| 0.50 | 32.15 | 0.314 | 0.379 | 0.421 | 0.371 | 85.45 | 10.05 | 0.596 |
| 0.75 | 31.74 | 0.310 | 0.377 | 0.419 | 0.369 | 84.97 | 10.93 | 0.650 |
| 1.00 (=w/o mask) | 30.05 | 0.318 | 0.377 | 0.415 | 0.370 | 85.14 | 11.77 | 0.731 |

(참고: λ=1.00 행은 코드상 λ=1.0이 가중치 생성 분기를 건너뛰어 w/o mask와 완전히 동일한 loss 계산이 되므로, 별도 학습 없이 Stage 1의 `stcocc_nomask` 체크포인트 결과를 그대로 사용했다.)

### 핵심 관찰

1. **RayIoU는 λ=0→0.25 구간에서 급격히 회복된다.** 0.303→0.368로 +0.065pt 상승하며, 이후 λ=0.5~1.0 구간(0.368~0.371)에서는 사실상 평평하다. 즉 RayIoU 회복에는 큰 λ가 필요 없다.
2. **mIoU는 λ에 대해 단조 감소한다.** 36.10 → 35.52 → 32.15 → 31.74 → 30.05.
3. **이 둘을 합치면 λ=0.25가 뚜렷한 최적점이다**: RayIoU는 이미 w/o mask 수준(0.368 vs 0.370, 차이 0.002)까지 회복했는데, mIoU는 아직 35.52로 w/ mask(37.03)에 가까운 수준을 유지한다.
4. **Calibration(ECE/NLL)은 λ에 대해 단조 악화된다** (λ=0의 4.29 → λ=1의 11.77). λ=0.25(6.75)는 w/ mask(5.71)에 가장 가까운 지점이기도 하다.

### Naive interpolation 대비 우위 (Case A 검증)

두 baseline(w/mask, w/o mask)을 단순히 선형으로 잇는 직선을 "naive interpolation"이라 하면, 같은 RayIoU 값에서 그 직선이 예측하는 mIoU와 실제 λ 스윕 결과를 비교할 수 있다.

| λ | 실측 RayIoU | naive 직선상의 예상 mIoU | 실측 mIoU | 우위 |
|---|---|---|---|---|
| 0.25 | 0.368 | ≈30.3 | 35.52 | **+5.2pt** |
| 0.50 | 0.371 | ≈29.9 | 32.15 | +2.2pt |
| 0.75 | 0.369 | ≈30.2 | 31.74 | +1.6pt |

세 지점 모두 naive interpolation보다 위에 있다 — 즉 **λ 조절은 두 baseline을 단순히 섞는 것보다 실질적으로 더 나은 mIoU–RayIoU 절충점을 만든다.**

**→ 원 스펙의 Case A(완화 가능)에 해당하는 결과.** Case B(단순 Pareto 곡선, 해소 어려움)로 보이지 않는다.

---

## 4. 결론 및 다음 단계 제안

- invisible-free voxel에 대한 loss weight `λ`만 조절하는 가장 단순한 방법으로도, w/ mask와 w/o mask 사이의 mIoU–RayIoU trade-off를 상당 부분 완화할 수 있다.
- 테스트한 5개 값 중에서는 **λ ≈ 0.25**가 가장 좋은 절충점이다 (mIoU 35.52 / RayIoU 0.368 / ECE 6.75).
- **제안 1**: λ∈[0.1, 0.35] 구간을 더 촘촘히(예: 0.10 / 0.15 / 0.20 / 0.30) 스윕해서 정확한 최적점을 좁힌다.
- **제안 2**: 이 결과는 "visibility mask 없이 voxel마다 적절한 weight를 학습으로 추정"하는 다음 단계(원 스펙에서 의도적으로 유보했던 adaptive weighting)를 시도해볼 근거가 된다.
- **한계**: 12 epoch, seed 고정 1회 실행 결과이며 반복 실험(다른 seed)은 하지 않았다. RayIoU 급변 구간(λ 0→0.25)의 정확한 위치는 더 촘촘한 스윕 전까지는 확정적이지 않다. Lovasz loss는 λ로 재가중되지 않아(랭크 기반이라 안전한 continuous-weight 일반화가 없음) λ의 효과가 CE/sem_scal/geo_scal 3개 loss에서만 나타나며, 이는 관측된 효과를 다소 과소평가했을 가능성이 있다.

---

## Appendix — 재현 정보

- 코드 변경: `projects/STCOcc/stcocc/detectors/stcocc.py`(`lambda_inv_free`, `build_inv_free_voxel_weight`), `stcocc/losses/focal_loss.py`, `stcocc/losses/semkitti.py`(continuous voxel weight 지원), `projects/BEVFormer/datasets/save_predictions_metric.py`(예측 저장 시 spurious batch-dim 버그 수정)
- Config: `projects/STCOcc/configs/stcocc_r50_704x256_16f_occ3d_e12_stage1_{nomask,cameramask}.py`, `..._stage2_invfree_l{000,025,050,075,100}.py` (각 `_rayiou.py` 평가용 companion 포함)
- 평가 파이프라인: `projects/STCOcc/tools/eval_all_metrics.py` (mIoU/AUROC/ECE/NLL/RayIoU를 한 번에 계산, 대용량 예측 저장으로 인한 OOM 방지)
- Raw 결과 CSV: `/NAS/work_dirs/stcocc_e12_eval/summary.csv`
- 체크포인트: `/NAS/work_dirs/stcocc_r50_704x256_16f_occ3d_e12_stage{1,2}_*`
