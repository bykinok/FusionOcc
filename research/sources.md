# Sources used in this P0 pass

## Primary evidence (used directly)
- Actual local repository code at commit `ad2a0c3153db3f6ec90e4d93d52246cd2f8fbaed` + uncommitted working-tree diff (this is the ONLY code basis for loss_inventory.json, protocol_manifest.json, and gradient_coverage.csv).
- Actual local/remote checkpoints and eval logs under `/NAS/work_dirs/stcocc_r50_704x256_16f_occ3d_e12_stage{1,2}_*` and `/NAS/work_dirs/stcocc_e12_eval/`.
- `[U1]` User-reported summary table from the prior conversation (2026-09-27), cross-checked line-by-line against `/NAS/work_dirs/stcocc_e12_eval/summary.csv` — **matches exactly** (all 6 rows, all columns, to the last reported decimal). No discrepancy found between what the user reported and what this repo's own CSV records.

## Reference-only (NOT fetched, NOT executed, NOT used to overwrite anything)
Per spec 1.2 ("공식 repository의 디렉터리는 사용자의 통합 repository와 다를 수 있다... 논문·공식 repository는 참고 자료") and per this being a P0 pass with no training approval yet, the following were **not** downloaded or run in this session:
- [R1] STCOcc official repository (https://github.com/lzzzzzm/STCOcc)
- [R2] STCOcc CVPR 2025 paper page
- [R3] STCOcc official semkitti.py (used only as a *named reference point* in the spec; this audit's loss_inventory.json describes the LOCAL modified file, not R3)
- [R4]-[R9] (SparseOcc/RayIoU, GaussRender, HASSC, Kälble/EvOcc, Learning-to-Reweight)

**Consequence:** any statement in this P0 output about what the "native full" STCOcc recipe does (epoch count, AMP, EMA-vs-raw reporting convention, LR schedule shape) is inferred from the LOCAL config's own naming/comments (e.g. `reported_epoch_target: 36` in the template, `MEGVIIEMAHook` naming) and from the user's own prior description, not from having read R1-R3. Anywhere this matters (see protocol_manifest.json `lr_schedule.interpretation` and `amp_syncbn_ema_freeze.ema_finding`), it is marked PENDING rather than asserted.

## Correction made during this pass
An earlier draft of this file claimed no `native_full_v1` candidate config exists locally. That was wrong and has been corrected: `projects/STCOcc/configs/stcocc_r50_704x256_16f_occ3d_36e_miou_ori_setting.py` (and its `wo_train_cam_mask` sibling) DO exist, with `total_epoch=36`, `OptimWrapper` (no AMP), and are plausible `native_full_v1` candidates. They share the same underlying `stcocc.py`/loss files this audit inspected (so lambda_inv_free defaults to 1.0 = legacy behavior for them, unaffected by this work). **Not yet confirmed** whether their data pipeline / mask_mode / EMA config exactly matches `screen_e12_v1`'s policy in every other respect (e.g. a distinct, PRE-EXISTING and PRE-DATING-this-work bug was found in an earlier session in `stcocc_r50_704x256_16f_occ3d_36e_miou_wo_train_cam_mask_ori_setting.py`'s `mask_mode` default, documented in `projects/STCOcc/reports/invisible_free_sweep_report.md` and left un-fixed per explicit prior user instruction not to fix bugs in configs outside this line of work). Resolving `native_full_v1`'s exact base_config is a named P1 field (`profiles.native_full_v1.base_config`) — this candidate is the leading option but is not yet frozen.
