# STCOcc 후속 연구 — 지금까지 확인된 사실 종합 (2026-09-29)

이 문서는 v1(P0~λ스윕)부터 v2(P0 감사~E1~E2)까지, 지금까지 실행된 모든 실험·감사에서 **실제로 확인된 사실**만 모은 것이다. 해석·추측·다음 단계 제안은 별도로 표시했다.

---

## 1. 출발점 — w/mask vs w/o mask trade-off (v1 Stage 1, 실측)

12 epoch, 2×GPU, effective batch 16, seed 고정 1회.

| Setting | mIoU (visible voxel만) | RayIoU mean | AUROC(msp) | ECE | NLL |
|---|---|---|---|---|---|
| w/o camera mask (λ=1) | 30.05 | 0.370 | 85.14 | 11.77 | 0.731 |
| w/ camera mask | 37.03 | 0.302 | 89.46 | 5.71 | 0.397 |

**사실**: w/mask가 mIoU +6.98pt 우위, RayIoU −0.068 열위. w/o mask는 그 반대. Trade-off는 실측으로 재현됨.

## 2. λ(invisible-free voxel weight) 스윕 결과 (v1 Stage 2, 실측)

| λ | mIoU | RayIoU mean | ECE | NLL |
|---|---|---|---|---|
| 0.00 | 36.10 | 0.303 | 4.29 | 0.374 |
| 0.25 | 35.52 | 0.368 | 6.75 | 0.440 |
| 0.50 | 32.15 | 0.371 | 10.05 | 0.596 |
| 0.75 | 31.74 | 0.369 | 10.93 | 0.650 |
| 1.00 (=w/o mask) | 30.05 | 0.370 | 11.77 | 0.731 |

**사실**:
- RayIoU는 λ=0→0.25 구간에서 0.303→0.368로 급격히 회복되고, 이후 λ=0.5~1.0(0.368~0.371)에서는 평평하다.
- mIoU는 λ에 대해 단조 감소한다(36.10→30.05).
- λ=0(36.10 mIoU, 0.303 RayIoU)은 w/mask(37.03, 0.302)와 거의 같은 지점에 있다 — 이는 λ=0이 continuous weight=0(하드 드롭이 아님)으로 구현돼 있어 이 세 loss(CE/sem_scal/geo_scal)에서는 사실상 w/mask와 동일한 효과를 내기 때문으로, 메커니즘상 일관된 결과다.
- Calibration(ECE/NLL)은 λ에 대해 단조 악화된다.
- λ=0.25가 두 baseline을 잇는 naive 직선보다 mIoU 기준 +1.6~5.2pt 우위에 있다(naive interpolation 대비 우위 확인, Case A에 해당).

## 3. v2 P0 감사에서 코드로 확인된 사실 (실측 아님, 코드 읽기로 확인)

- **RayIoU origin은 모델의 실제 입력창과 무관하다.** `nuScenesDataset`이 같은 scene 내 과거+미래 pose를 모두 후보로 삼아 8개로 subsample한다. 모델의 실제 16-frame recurrent history와 연결된 코드가 아니다.
- **`mask_lidar`는 원본 GT npz에 존재하지만 코드 어디에서도 사용되지 않는다.**
- **`ignore_index=255`는 데이터셋 고유값이 아니라 코드가 주입하는 값이다.** 현재 w/mask baseline은 label validity와 camera-visibility weighting을 사실상 같은 메커니즘으로 구현하고 있다(둘을 구분하지 않음).
- **λ=1 bypass와 weighted 경로는 bit-identical**(gradient 포함, 단위테스트로 확인).
- **Lovász loss는 어떤 λ에서도 재가중되지 않는다**(의도된 설계). 합성 데이터 기준 invisible-free voxel의 전체 gradient 중 Lovász 비중은 λ=1.0에서 ~0.4%, λ=0.25에서 ~0.5% — 작지만 0은 아니다. (실제 학습 배치로는 재확인 안 됨, 방향만 확인.)
- **6개 legacy run 모두 EMA가 아닌 raw checkpoint로 평가됐다** — MEGVIIEMAHook은 모든 config에서 활성 상태였지만 EMA 가중치는 한 번도 평가에 쓰이지 않았다. (미해결, 여전히 open item.)
- **frozen train/val split이 없다** — 6개 run 전부 retrospective full-val(6019샘플 전체)로 평가됐고, λ=0.25를 "최선"으로 고른 판단 자체가 이미 그 전체 val을 본 뒤의 선택이다.
- **lovasz_softmax.py에 latent bug 존재**: `flatten_probas`가 bare tensor 입력에 대해 `UnboundLocalError`를 낸다. 실제 학습에서는 항상 list로 들어와 발동한 적 없음(재현은 확인됨, 수정은 미적용).

## 4. v2 E1 — Bias frontier (2026-09-27 실행, calibration split 1202샘플)

**주의**: 이 mIoU는 **invisible voxel을 포함한 전체 voxel** 기준이며, 위 1·2절의 mIoU(camera-visible voxel만)와는 다른 지표다. 직접 비교 불가.

| checkpoint | native(δ=0) mIoU | 찾은 정점 | invisible_free_iou @δ=0 |
|---|---|---|---|
| nomask (λ=1) | 0.2267 | δ=-0.5, 0.2271 (native가 이미 정점 근처) | 0.956 |
| l025 (λ=0.25) | 0.2182 | δ=+2.0, 0.2287 (+0.0105 여유) | 0.923 |
| cameramask (w/mask) | 0.1226 | δ=+5.0, 0.1348 (그래도 nomask/l025 native에 한참 못 미침) | 0.327 |

**사실**:
- cameramask는 invisible voxel의 free 여부를 native decode에서 거의 못 맞춘다(invisible_free_iou 0.327 vs nomask/l025의 0.92~0.96). 이는 학습 시 invisible 영역에 loss를 아예 안 준 것과 정확히 일치하는 결과다.
- cameramask는 post-hoc bias 보정(δ 이동)만으로는 이 격차를 못 메운다 — 자기 최적점(δ=+5)에서도 nomask/l025 native 수준에 크게 못 미친다.
- nomask는 이미 native decode가 거의 post-hoc 최적점이다 — 추가로 짜낼 여지가 거의 없다.
- l025는 post-hoc bias만으로 약간의(+0.0105) mIoU 여유가 있다 — 학습을 다시 하지 않고도 결정 경계를 옮기는 것만으로 일부 개선 가능하다는 뜻.

## 5. v2 E2 — RayIoU origin 시간대별 분해 (2026-09-27 실행, 전체 val 6019샘플, 3개 checkpoint)

| | past(과거) IoU@1/2/4 | reference(기준) IoU@1/2/4 | future(미래) IoU@1/2/4 |
|---|---|---|---|
| nomask | 0.330/0.388/0.424 | 0.383/0.455/0.497 | 0.297/0.357/0.397 |
| cameramask | 0.245/0.312/0.366 | 0.296/0.385/0.448 | 0.224/0.295/0.351 |
| l025 | 0.320/0.384/0.425 | 0.376/0.453/0.502 | 0.290/0.359/0.407 |

**사실**: 3개 checkpoint 전부 **reference > past > future** 순서로 일관된다. future-relative-to-reference origin은 가장 어려운 조건이었다.

**해석 (v2 판정 규칙에 따라 결론이 아닌 근거로 취급)**: "RayIoU가 future-origin을 통해 모델에 부당한 이득을 줄 수 있다"는 v2의 우려는 이 3개 checkpoint에서는 지지되지 않는다. 즉 지금까지 보고된 RayIoU 우위(예: λ=0.25의 0.368)가 future-origin 덕분에 부풀려진 것이라는 근거는 없다 — 오히려 반대 방향의 증거다.

## 6. 이번 세션에서 발견해 고친 코드 문제 (연구 결과가 아니라 도구 정합성 문제)

- `export_occ_logits.py`가 기본적으로 camera-visible voxel만 export하도록 돼 있었다 — invisible 영역 분석 도구로 쓰려면 반드시 고쳐야 했음(`--keep-invisible` 추가).
- `InfiniteGroupEachSampleInBatchSamplerEval`이 single-GPU 평가에서 샘플 17개를 중복 export하는 버그가 있었다 — **안전한 중복**(데이터 오염 아님)이며, 기존 mIoU/RayIoU 코드는 이미 index 기준 dedup으로 방어하고 있어 **1·2절의 legacy 숫자는 이 버그의 영향을 받지 않았다.** 새로 만든 export 도구에는 동일한 dedup을 추가함.

## 7. 아직 확인되지 않은 것 (facts로 승격 안 된 항목)

- EMA checkpoint로 평가하면 숫자가 어떻게 달라지는지 — 전혀 실행된 적 없음.
- calibration/development/confirmation 3-way split(20/40/40)은 생성만 됐고, 이 split으로 λ=0.25 선택을 **재확인**하는 실험은 아직 안 함 — 지금까지의 "λ=0.25가 최선"이라는 판단은 여전히 dev-exposed 전체 val 기준이다.
- Lovász의 실제 학습 배치 기준 gradient 비중(합성 데이터 아님) — 미측정.
- RayIoU의 거리(radius)/높이(height) 분해는 인프라는 있으나(이전부터 구현돼 있었음), 이번 세션에서 3개 checkpoint 전체 val 기준 수치를 별도로 보고하지는 않았다(로그에는 존재).
- 더 촘촘한 λ 스윕(0.1/0.15/0.2/0.3)은 아직 실행 안 됨 — v1 보고서의 제안 1이 그대로 남아있다.

---
근거 파일: `projects/STCOcc/reports/invisible_free_sweep_report.md`(1·2절), `research_v2/audits/loss_inventory_v2_addendum.json`(3절), `research_v2/reports/E1_execution_plan.md`(4절), `research_v2/reports/E2_execution_plan.md`(5절), `research_v2/plan.yaml`(전체 상태 스냅샷).
