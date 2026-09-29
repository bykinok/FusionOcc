# P0 Audit Report — STCOcc Selective Free-Space Supervision

**Audited commit**: `ad2a0c3153db3f6ec90e4d93d52246cd2f8fbaed` (+ uncommitted working-tree diff, preserved as-is)
**Audit date**: 2026-09-27
**Scope**: spec section 4 (P0) only. No new training executed. GPU usage in this pass was CPU-only by choice (no GPU ids designated yet).

---

## 1. What was actually checked

| Item | Method | Result |
|---|---|---|
| Repo/commit/diff state | `git status`, `git rev-parse` | Captured, see `resolved_paths.json` |
| Env versions | `python -c "import torch..."` | torch 2.4.0+cu124, mmcv 2.1.0, mmdet3d 1.3.0, mmengine 0.10.3 (identical local + remote) |
| Running GPU jobs | `nvidia-smi` | Both local GPUs idle at audit time; not interrupted |
| Loss inventory | direct code reading of `stcocc.py`, `focal_loss.py`, `semkitti.py`, `lovasz_softmax.py` | `research/audits/loss_inventory.json` |
| λ=1 equivalence | CPU synthetic-tensor unit test, real autograd | **PASS** — value + gradient bit-identical for CE/focal primitive, sem_scal, geo_scal |
| λ=0 partial gradient | CPU synthetic-tensor unit test, real autograd | **PASS** — exact-zero gradient at invisible-free voxels for all 3 reweighted losses |
| Edge cases (ignore/empty/etc.) | CPU synthetic-tensor test, 6 cases | 4 PASS, **2 FAIL** (see finding F3) |
| Camera-mask semantics (True=visible) | code reading + this session's prior empirical sanity logs | Consistent, no contradiction found |
| Checkpoint SHA256 | `sha256sum` on all 12 final checkpoints (6 runs × raw+EMA) | Recorded in `resolved_paths.json` |
| Metric-equivalence (live vs. file-based evaluator) | re-derivation from this session's own prior debugging history | **Real evidence exists** (not assumed) — see finding F1 |
| DDP reduction semantics | — | **PENDING** (needs GPU) |
| Resume equivalence | — | **PENDING** (needs GPU, never exercised) |
| Duplicate evaluation (dist. sampler) | inspected all `tools/test.py` invocations used historically | N/A — every eval ran single-process |

---

## 2. Findings, ranked by why they matter

### F1 — EMA checkpoint was never evaluated (HIGH severity)
Every one of the 6 legacy training runs has `MEGVIIEMAHook` active, producing both `iter_21096.pth` (raw) and `iter_21096_ema.pth`. **Every evaluation so far — for every baseline and every λ — used the raw checkpoint.** If the intended/native protocol reports EMA weights (plausible, given the hook exists specifically to produce a reportable averaged model), all absolute numbers already shared with the user's professor could shift once EMA is evaluated. Because the same choice was applied uniformly to all 6 runs, the *relative* λ trend is presumed more robust than the *absolute* numbers, but this presumption is itself untested. See `decisions.jsonl` D03.

### F2 — `screen_e12_v1` and `legacy_e12` are the same profile, not two
The template anticipates a `legacy_partial` policy distinct from `controlled_partial_v1`. In this repository, no such distinct historical policy exists — the single implementation built during the λ-sweep work already satisfies `controlled_partial_v1`'s definition (weighted sufficient statistics for sem_scal/geo_scal, weighted mean for CE, unweighted Lovász) and is bit-identical to "no weighting at all" at λ=1. There is nothing separate to preserve. Recorded explicitly in `plan.yaml` rather than silently merged.

### F3 — Latent `sem_scal_loss` division-by-zero on degenerate batches (MEDIUM severity, LOW practical likelihood)
`sem_scal_loss` divides by an integer `count` that stays 0 when every class's weighted occurrence count is 0 (all-ignore batch, or all-voxels-are-the-excluded-free-class batch). This is a **pre-existing bug**, not introduced by the λ-weighting generalization — the original boolean-mask code has the identical failure mode. Never triggered in the 6 legacy runs (real batches always have valid=5,120,000/5,120,000 with all classes represented). Flagged as a guard to add before any future chunked/streaming re-evaluation or synthetic-scene diagnostic that could hit a degenerate chunk. `geo_scal_loss` and the CE/focal primitive do **not** share this failure mode (both PASS all 6 edge cases).

### F4 — Data-split contamination relative to the new spec (by design of the new spec, not a mistake)
All 6 legacy numbers were computed on the full 6019-sample official val, and the aggregate across all 5 λ points was already used to pick λ=0.25 as "best" in the prior report. The new spec's calibration/development/confirmation scene-split (seed 20260927) did not exist yet when that selection was made. This is exactly the situation spec section 3.2 anticipates and explicitly permits labeling as `retrospective_full_val, dev-exposed` rather than invalid. No retraining is implied by this finding — only relabeling and, later, a fresh confirmation-set number before any noninferiority claim.

### F5 — Metric-equivalence evidence already exists (positive finding)
Not a gap: this session's own prior debugging already produced a real, non-hypothetical before/after comparison of the live evaluator vs. the file-based streaming evaluator (`tools/compute_metrics_from_file.py`), including catching and fixing a real spurious-batch-dimension bug in `SavePredictionsEvaluator` (`projects/BEVFormer/datasets/save_predictions_metric.py`). Post-fix, both paths gave `mIoU=30.05` for `stage1_nomask` with zero GT-load failures. This satisfies spec 4.2 item 9 and 3.3's "저장 버그 수정 전후" requirement for at least this one pair of runs.

### F6 — Lovász is not, and was never claimed to be, reweighted
Confirmed by code reading: `lambda_inv_free` never reaches `lovasz_softmax`. At every λ including λ=0, Lovász fully supervises invisible-free voxels. This was already disclosed to the user before this audit; now it is a structured, checkable fact (`loss_inventory.json`) rather than only a prose caveat.

### F7 — Metric unit convention is inconsistent in the existing summary.csv
`mIoU`/`AUROC`/`ECE` are stored as percent (e.g. `35.52`), `RayIoU` as fraction (e.g. `0.368`), in the same CSV, with no unit metadata. Values are correct; only the bookkeeping needs harmonizing for the new `[0,1]`-fraction internal-storage convention (spec 3.3). See `decisions.jsonl` D02.

### F8 — `native_full_v1` candidate exists locally but is not yet frozen
Corrected during this pass (see `sources.md`): `stcocc_r50_704x256_16f_occ3d_36e_miou_ori_setting.py` (36 epoch, no AMP) is a plausible `native_full_v1` base, but its exact equivalence to `screen_e12_v1` in every other respect (data split, EMA convention, mask policy) is not yet confirmed.

---

## 3. Existing-result reusability classification

See `decisions.jsonl` for the full, structured decision log. Summary:

| Run | mIoU/RayIoU numbers | Classification |
|---|---|---|
| nomask, cameramask, l000, l025, l050, l075 (all 6) | Correct as computed | `reusable` as `screen_e12_v1, seed0, raw-checkpoint, retrospective_full_val` evidence; **not** yet a clean confirmation-set or native-protocol result |
| l100 (λ=1) | N/A (aliased) | `reusable` — code + CPU-test equivalence evidence for the alias-not-a-rerun decision |
| Per-class / radius / height breakdowns | Exist in raw logs, not in summary.csv | `relabel_only` — recoverable without rerunning anything |
| unit convention (%/fraction mixed) | values correct | `relabel_only` |
| EMA-vs-raw checkpoint | untested | `reevaluation_required` (HIGH priority, first P1 action recommended) |
| Data split | full-val, dev-exposed | `reevaluation_required` for any confirmatory claim only (split doesn't exist yet) |

**No run in this audit was classified `retraining_required`.** Nothing found in this pass invalidates the 6 completed training runs themselves — findings are about checkpoint selection, evaluation split, and unit bookkeeping, not about the training having produced wrong models.

---

## 4. Gate P0 — decision

Per spec 4.4: *"core loss, label/mask 정렬, evaluation 오류가 해결되기 전에는 방법 training을 시작하지 않는다."*

- Core loss (CE/sem_scal/geo_scal) λ-application: **verified**, no blocking defect.
- Label/mask alignment (camera_mask True=visible, invisible-free targeting): **verified** by code + prior empirical logs.
- Evaluation correctness: **verified** for the live-vs-file-based path (F5); EMA-checkpoint question (F1) is a real open issue but does not indicate the evaluation MACHINERY is broken — it indicates a checkpoint-selection policy question.

**P0 gate: PASS, with F1 (EMA) and F4 (split) carried forward as mandatory P1 pre-work, not as blockers to starting P1's audit-safe activities (anchors/simple-controls planning).** No full training is authorized by this P0 pass — GPU ids and budgets remain unset (see `reports/stage_P0.md`).
