# P0-v2 Audit Report — STCOcc Selective Free-Space Supervision (v2.0)

**Base**: research/ (v1 P0, untouched) + this pass's new/extended checks.
**Scope**: spec v2 section 3 only. Zero new training, zero new GPU evaluation jobs. CPU-only unit tests.

## 1. What's new in this pass vs. v1

| v2 requirement | Status | Where |
|---|---|---|
| loss별 λ 적용/normalization/Lovász 잔존/cascade 범위 | v1 findings carried forward + gradient NORMS added | `loss_inventory_v2_addendum.json` |
| λ=1 bypass/weighted 동등성 + evaluator 검증 | Re-confirmed (test still PASS) | `gradient_coverage_v2.csv` |
| **actual history tokens/poses·RayIoU origin metadata** | **NEW, done this pass** | `loss_inventory_v2_addendum.json:rayiou_origin_builder_audit` |
| label validity vs. camera/lidar visibility 의존성 분리 | NEW, done this pass | `annotation_manifest.json` |
| 기존 checkpoint 재사용 + calibration/dev/confirmation 정책 | v1 carried forward, v2 vocabulary added | `legacy_reuse.json`, `split_manifest.json` |

## 2. The single most important new finding

**RayIoU's ray origins are not restricted to the model's actual input window.** `nuScenesDataset.__getitem__` (in `nuscenes_ego_pose_loader.py`) builds the origin set for a sample by scanning **every sample in the same scene** — before and after the reference frame — filtering only by spatial extent (within ±39 m) and then subsampling to 8. There is no code connecting this to the model's real 16-frame recurrent history buffer, and no origin metadata (timestamp, time-group) is currently retained anywhere. This is read directly from code, not inferred from the SparseOcc paper — confirms the paper's documented flexibility ("origins can be current/past/future") is **actually exercised** in this local evaluator, unverified until now.

This does not itself prove anything about the λ-sweep's RayIoU gains — that is exactly what **E2** is for. P0's job was only to confirm the mechanism exists, which it does.

## 3. Two new latent (never-triggered) bugs found

1. `sem_scal_loss` division-by-zero on degenerate all-ignore/all-free batches (carried forward from v1, unchanged).
2. **NEW**: `lovasz_softmax.py`'s `flatten_probas` raises `UnboundLocalError` for any non-list `labels` input, due to a conditionally-scoped `import numpy as np`. Never triggered in the 6 real training runs because `target_voxel_semantic` always arrives as a Python list at that call site (confirmed). Reproduced directly in this audit's test script; trivial one-line fix identified, **not applied** (audit-only pass).

## 4. Quantitative Lovász-share check (template said not to assume — checked instead)

On synthetic data: Lovász's share of total invisible-free gradient magnitude is ~0.4% at λ=1.0 (the historical w/o-mask condition), rising mechanically to 100% at λ=0 (since the other three losses are then exactly zero there). Directionally confirms the template's suspicion; the *absolute* magnitude comparison is **not** transferable to real training batches without re-measuring on an actual converged checkpoint's logits — flagged as such, not overclaimed.

## 5. Annotation findings

- `mask_lidar` exists in every raw GT file but is used **nowhere** in this codebase — confirmed by exhaustive grep, not assumed.
- `ignore_index=255` is a purely local, training-policy-injected sentinel; the raw Occ3D `semantics` array never contains it. Label validity and camera-visibility weighting are **the same mechanism** in the existing w/mask baseline (by original design, predating this work) — exactly the conflation spec v2 §2.1 warns against, now explicitly documented rather than left implicit.

## 6. Existing-result reusability (v2 vocabulary)

All 6 legacy runs: **`reusable`**. Nothing found in this pass invalidates any trained model. Two conditions carried over from v1 remain open and HIGH-priority: (a) EMA vs. raw checkpoint never resolved, (b) scene-based calibration/development/confirmation split never generated (still only retrospective full-val).

## 7. Gate decision

**P0-v2: PASS.** No defect blocks proceeding to plan E1/E2 (planning only — see `reports/E1_execution_plan.md`, `reports/E2_execution_plan.md`). GPU ids, training budget, **and now also evaluation budget** (`max_evaluation_gpu_hours`, `max_evaluation_jobs` — new in v2) remain unset; no expensive evaluation or training is authorized by this pass.
