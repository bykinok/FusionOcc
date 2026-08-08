# GaussianFormer (probabilistic, gs25600)

Port of the probabilistic-Gaussian GaussianFormer occupancy model
(`config/prob/nuscenes_gs25600.py`, mIoU 20.33 in the original repo) from
[`Ref/GaussianFormer_ori`](../../Ref/GaussianFormer_ori) into this
framework's mmdetection3d + mmengine `Runner` pipeline.

Only the `prob/nuscenes_gs25600.py` config path was ported (as requested) --
the non-probabilistic v1 lineage (`GaussianLifter`, `nuscenes_gs144000.py`,
`nuscenes_gs25600_solid.py`, `localagg`/`localagg_prob_fast`) was left
untouched in `Ref/GaussianFormer_ori`; it follows the same module layout and
can be added the same way if needed later.

## Porting principle

All model/loss/dataset/transform math is copied over **unchanged** --
`GaussianLifterV2`, `GaussianOccEncoder` (and its `anchor_encoder`/`ffn`/
`deformable_model`/`refine_layer`/`spconv_layer` sub-modules), `GaussianHead`,
`OccupancyLoss`/`MultiLoss`/`PixelDistributionLoss`, `NuScenesDataset`, and
all pipeline transforms are byte-for-byte the same code as
`Ref/GaussianFormer_ori`, just relocated under `gaussianformer/`. Only the
*scaffolding* around that code was adapted, because GaussianFormer_ori's own
scaffolding depends on things this framework's environment doesn't have
(mmsegmentation) or doesn't use (a bespoke `train.py` loop instead of
mmengine's `Runner`). Verified: the ported model builds with the exact same
parameter count (124,943,205) as `mmseg.models.build_segmentor(cfg.model)`
does in the original repo's own `selfocc` conda env.

## What changed, and why

- **mmseg dependency removed.** GaussianFormer_ori registers its segmentor
  through `mmseg.models.SEGMENTORS`/`builder` and gets `ResNet`/`FPN` from
  `mmseg.models.backbones`/`necks`. mmsegmentation is not installed in this
  repo's `occfrmwrk` conda env (nor used by any other ported project here).
  `model/segmentor/base_segmentor.py` and `bev_segmentor.py` now register
  under `mmdet3d.registry.MODELS` and build submodules via
  `models/utils/build.py::build_model`, which special-cases `ResNet`/`FPN`
  by instantiating them directly from `mmdet.models` -- the exact same
  workaround `projects/SurroundOcc/surroundocc/detectors/surroundocc.py`
  already uses in this repo, since `mmdet3d.registry.MODELS` and
  `mmdet.registry.MODELS` are sibling scopes, not parent/child (mmdet's
  `ResNet`/`FPN` are simply not resolvable by name through mmdet3d's
  registry otherwise). `SECONDFPN`/`DCNv2` are unaffected -- they're already
  registered under mmdet3d/mmcv directly.
- **`DiceLoss` import in `losses/occupancy_loss.py` made lazy.** It's an
  mmseg class, only used when `use_dice_loss=True`, which
  `nuscenes_gs25600.py` never sets. Importing it eagerly would make the
  module fail to import at all in this environment for no functional
  benefit.
- **`BaseModel` instead of `BaseModule`.** `CustomBaseSegmentor`/
  `BEVSegmentor` now derive from `mmengine.model.BaseModel` instead of
  `BaseModule`, and `BEVSegmentor` gained `loss()`/`predict()`/`_forward()`
  plus a `forward(mode=...)` dispatcher, so mmengine's `Runner`
  (`train_step`/`val_step`) can drive it. `forward_bev()` is the original
  `BEVSegmentor.forward()` body, untouched. The `loss()`/`predict()` methods
  reproduce -- inside the model now, instead of in GaussianFormer_ori's
  external `train.py` loop -- exactly what that loop did: build `MultiLoss`
  from `cfg.model.loss`, remap the model's output dict through
  `cfg.model.loss_input_convertion`, and (for checkpoint loading) the
  `img_neck.*`/`lifter.anchor` key-stripping retry from
  `misc/checkpoint_util.py::refine_load_from_sd` (ported unchanged, just
  wired into `load_state_dict` instead of a manual `try/except` in
  `train.py`).
- **Dataset/dataloader.** `NuScenesDataset` (renamed `GaussianFormerNuScenesDataset`
  when dual-registered into `mmdet3d`/`mmengine`'s `DATASETS` registries --
  `mmdet3d` already has its own, differently-shaped `NuScenesDataset`) and
  `custom_collate_fn_temporal` are unchanged; they're just also registered
  under this framework's registries so a standard `train_dataloader`/
  `val_dataloader` config can build them, instead of GaussianFormer_ori's own
  `dataset/__init__.py::get_dataloader()`. The dataset still builds its own
  pipeline internally through its own `OPENOCC_TRANSFORMS` registry
  (unchanged), so no transform classes needed dual-registration.
- **Evaluation.** GaussianFormer_ori's `misc/metric_util.py::MeanIoU`
  accumulates into persistent CUDA buffers and calls `dist.all_reduce`
  itself -- reproduced unchanged in `evaluation/mean_iou.py` (only its
  logger lookup was changed from a hardcoded `MMLogger.get_instance('selfocc')`
  to `get_current_instance()`), but it isn't used directly: it doesn't
  compose safely with mmengine's `BaseMetric.evaluate()`, which already
  gathers per-sample results across ranks itself before calling
  `compute_metrics()` once on the main process. `evaluation/occupancy_metric.py`
  reproduces the exact same per-class seen/correct/positive counting and IoU
  formulas, restructured to fit that `process()`/`compute_metrics()` split
  (same pattern as `projects/SurroundOcc/surroundocc/evaluation/occupancy_metric.py`).
- **Loss logging.** `MultiLoss.forward()` (unchanged) returns per-component
  losses as plain Python floats (`.detach().item()`, for its own
  now-unused tensorboard writer) rather than tensors, so they can't be
  returned as-is from `BEVSegmentor.loss()` (mmengine's `parse_losses`
  requires tensors or errors out). They're surfaced via
  `mmengine.logging.MessageHub` instead.
- **Config format.** `config/prob/nuscenes_gs25600.py` + its 3 `_base_`
  files (mmengine `Config` dicts, GaussianFormer_ori's own format) were
  flattened into one file and re-expressed in this framework's
  `train_dataloader`/`optim_wrapper`/`param_scheduler`/`val_evaluator`
  style. All hyperparameters were carried over exactly as
  `mmengine.Config.fromfile` resolves the original 4-file chain to (verified
  by diffing against `cfg.pretty_text` from the original repo) -- the LR
  schedule is the one approximation: the original iteration-based
  `timm.scheduler.CosineLRScheduler` (500-iter linear warmup, then cosine to
  `lr*0.1` over `len(train_loader)*max_epochs` iters) is expressed as a
  500-iter `LinearLR` warmup followed by an *epoch-wise* `CosineAnnealingLR`
  to `lr*0.1` -- same warmup+cosine-to-`lr*0.1` shape, following the
  convention already used by `projects/CONet`/`projects/SurroundOcc` for the
  same reason (no need to know `len(train_loader)` upfront).

## CUDA extensions

Three CUDA extensions are ported as source (unchanged) and were rebuilt
for this repo's `occfrmwrk` conda env (Python 3.10, torch 2.4+cu121) --
the original repo's prebuilt `.so` files are compiled for Python 3.8 and
are not ABI-compatible here:

| Extension | Path | Used by | Install |
|---|---|---|---|
| `deformable_aggregation_ext` | `gaussianformer/models/encoder/gaussian_encoder/ops/` | `DeformableFeatureAggregation` | `python setup.py build_ext --inplace` (imported as a relative submodule, no separate pip install needed) |
| `local_aggregate_prob` | `gaussianformer/models/head/localagg_prob/` | `GaussianHead` (`use_localaggprob=True`) | `pip install -e .` |
| `pointops` | `gaussianformer/ops/pointops/` | `GaussianLifterV2` (`farthest_point_sampling`, active since this config sets `random_sampling=False`) | `pip install .` (**not** `-e .` -- its `package_dir={'pointops': '.'}` mapping doesn't resolve correctly under a modern pip editable install) |

All three are already built and installed in the `occfrmwrk` conda env as of
this port. To rebuild from scratch (e.g. after a CUDA/torch upgrade):

```bash
conda activate occfrmwrk
export CUDA_HOME=/usr/local/cuda-12.1   # must match the CUDA version torch was built with, not `which nvcc`
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

cd projects/GaussianFormer/gaussianformer/models/encoder/gaussian_encoder/ops && python setup.py build_ext --inplace
cd ../../../head/localagg_prob && pip install -e . --no-build-isolation
cd ../../../../ops/pointops && pip install . --no-build-isolation
```

`jaxtyping` (used only for type annotations in `models/utils/sampler.py`)
was also added to the `occfrmwrk` env -- it isn't a real runtime dependency,
just missing from the environment.

## Data / checkpoints

`pretrain/`, `ckpt/`, and each file under `data/nuscenes_cam/` are symlinks
directly into `/home/h00323/DATA/mmDataset/nuscenes_gaussianformer/` --
following the same per-project symlink convention as
`projects/SurroundOcc`/`projects/CONet`. They point at that shared storage
directly, **not** through `Ref/GaussianFormer_ori` (an earlier version of
this symlink went through `Ref/GaussianFormer_ori/data/nuscenes_cam`, which
would have broken if `Ref/` were ever moved or removed; fixed to point at
the underlying storage instead, so this project has no filesystem
dependency on `Ref/GaussianFormer_ori` at all -- confirmed by
`grep -rn "Ref/GaussianFormer_ori" projects/GaussianFormer --include="*.py"`,
which only turns up comments/docstrings, and by `find projects/GaussianFormer
-type l` resolving no symlink into `Ref/`). `data_root`/`occ_path` in the
config point at this repo's own pre-existing shared `data/nuscenes/` and
`data/nuscenes_occ/samples` (the latter happens to be backed by the same
underlying SurroundOcc GT storage `Ref/GaussianFormer_ori/data/surroundocc`
also symlinks to, but again independently, not through `Ref/`).

## Verified

- `MODELS.build(cfg.model)` builds successfully with **124,943,205**
  parameters -- matching the original repo's own `build_segmentor(cfg.model)`
  parameter count exactly.
- A full forward/backward/predict pass against a real training sample (real
  images + real SurroundOcc-format occupancy GT, no synthetic data) runs
  end-to-end through the rebuilt CUDA ops and produces a finite loss with
  gradients flowing.
- `GaussianFormerOccupancyMetric.process()`/`compute_metrics()` run against
  that same sample's `predict()` output and produce per-class IoU/mIoU.

Not verified (out of scope for this port): an actual multi-epoch training
run, or a numeric mIoU match against the original repo's published 20.33 on
the full validation set -- that requires the released `state_dict_gsf2_nuscenes_gs25600.pth`
checkpoint (present at `ckpt/state_dict_gsf2_nuscenes_gs25600.pth`) and a full
eval pass, which needs a GPU allocation beyond a quick smoke test.

## Usage

```bash
python tools/train.py projects/GaussianFormer/configs/gaussianformer/nuscenes_gs25600.py
python tools/test.py projects/GaussianFormer/configs/gaussianformer/nuscenes_gs25600.py <checkpoint>
```

## Occ3D-nuScenes GT configs

`configs/gaussianformer/nuscenes_gs25600.py` trains against SurroundOcc-format
GT (`LoadOccupancySurroundOcc`). The `nuscenes_gs25600_occ3d*.py` family
trains/evals against Occ3D-nuScenes GT instead, mirroring the experiment axes
`projects/SurroundOcc`/`projects/CONet` already have for their own models
(`*_ori_setting`, `*_unified`, `*_wo_train_cam_mask_*`, `*_rayiou`,
`*_condition_{C,D}_full`, `*_calib_*`).

**No new pkl was needed.** GaussianFormer's existing
`data/nuscenes_cam/nuscenes_infos_{train,val}_sweeps_occ.pkl` already carries
an Occ3D GT path per frame (`info['occ_path']` -> e.g.
`gts/scene-0003/<token>/labels.npz`, resolved against `data_root`) --
confirmed by loading it directly and reading the referenced `.npz`
(`semantics`/`mask_lidar`/`mask_camera`, 200x200x16, unused until now). A new
transform, `gaussianformer/datasets/transform_3d.py::LoadOccupancyOcc3D`,
reads it (see its docstring for the `use_mask_camera`/`mask_mode` params
implementing the mask-related axes below).

Occ3D-nuScenes uses a different physical grid than SurroundOcc GT
(`pc_range=[-40,-40,-1,40,40,5.4]`, voxel 0.4m, vs SurroundOcc's
`[-50,-50,-5,50,50,3]`/0.5m -- both 200x200x16, but a different real-world
volume), so `configs/gaussianformer/_base_/occ3d_model.py` overrides every
model field whose output is interpreted against real-world coordinates
(`pc_range` used by the deformable/refine/spconv layers, the lifter's own
`pc_range`/`voxel_size`, and the head's `cuda_kwargs.pc_min`/`grid_size`) to
match. See that file's docstring for why `spconv_layer.grid_size`'s
z-component is `0.8` rather than `1.0`.

**Coordinate frame (LiDAR vs. ego) -- caught and fixed.** SurroundOcc GT is
natively defined in the **LiDAR** sensor frame; Occ3D-nuScenes GT is defined
in the **ego-vehicle** frame. `NuScenesAdaptor` (in every pipeline) picks
which camera projection matrix `model.forward` actually uses --
`projection_mat = lidar2img` when `use_ego=False`, `ego2img` when
`use_ego=True` -- and `GaussianLifterV2` inverts that same matrix to
project camera rays into 3D and look them up against `occ_label`/
`occ_cam_mask`. These three things (the GT tensor's frame, the projection
matrix's frame, and `pc_range`) all have to agree. The first pass at these
Occ3D configs copied `NuScenesAdaptor(use_ego=False, ...)` verbatim from
`nuscenes_gs25600.py` (correct there, since SurroundOcc GT is LiDAR-frame)
without changing it to `use_ego=True` for the Occ3D grid -- silently
projecting ego-frame GT through a LiDAR-frame camera matrix. Both
SurroundOcc's and CONet's own Occ3D pipelines have an equivalent
`use_ego_frame` toggle specifically for this
(`projects/SurroundOcc/surroundocc/transforms/loading.py:443-445`,
`projects/CONet/mmdet3d_plugin/datasets/pipelines/loading_bevdet.py:92-99,314-316`),
which is what surfaced the mismatch on review. Fixed: all
`nuscenes_gs25600_occ3d*.py` configs now set `NuScenesAdaptor(use_ego=True, ...)`.
Confirmed empirically on a real val sample: the fraction of camera-ray
samples in `GaussianLifterV2`'s pixel-distribution head that land on a real
occupied voxel (`pixel_gt`) went from 53.0% (buggy, `use_ego=False`) to
73.6% (fixed, `use_ego=True`) -- consistent with rays now actually landing
on real scene geometry rather than an offset/rotated copy of it.
`LoadOccupancyOcc3D`'s own `lidar_origin` output (used only by the RayIoU
metric, unaffected by this bug) was already ego-frame-correct, since it's
computed independently from `ego2lidar` rather than via `NuScenesAdaptor`.

| Config | Axis | Model code needed? |
|---|---|---|
| `_occ3d_ori_setting.py` | Occ3D GT, GaussianFormer's own original recipe (same optimizer/schedule as `nuscenes_gs25600.py`) | no |
| `_occ3d_unified.py` | Occ3D GT, cross-project standardized recipe (no PhotoMetricDistortion, `lr=2e-4`, grad accumulation=8, 24 epochs, iteration-based warmup+cosine, `val_interval=9999`) | no |
| `_occ3d_wo_train_cam_mask_{ori_setting,unified}.py` | train without camera-visibility masking (`occ_cam_mask`=all-True at train time only); eval unchanged | no |
| `_occ3d_unified_condition_{C,D}_full.py` | C: all *occupied* voxels always supervised; D: all *free* voxels always supervised (mirror of C); eval unchanged | no |
| `_occ3d_{ori_setting,unified}_rayiou.py` | RayIoU eval instead of per-voxel mIoU (ray-casts from the LiDAR origin; visibility masking irrelevant, so `use_mask_camera=False` at eval) | small: `predict()` already emits a dense `occ_results`/`voxel_semantics` grid + `lidar_origin` for this |
| `_occ3d_calib_train.py` / `_calib_eval_before.py` / `_calib_eval.py` | confidence calibration (temperature scaling), fit on a held-out calib split, evaluated on a disjoint eval split | small: `BEVSegmentor(temperature=...)` |

**"Unified" recipe -- audited against every other project that has one, 3
mismatches found and fixed.** `_unified.py` configs exist in 8 other ported
projects (SurroundOcc, CONet, BEVFormer, TPVFormer, STCOcc, FusionOcc,
LiCROcc, SparseOcc_eccv); comparing all of them against their own
`_ori_setting` sibling shows the recipe is invariant across every one of
them regardless of that model's own paper hyperparameters (e.g. CONet's own
lr 3e-4, FusionOcc's own 5e-5 both become 2e-4 in `_unified`): `AdamW`,
**`lr=2e-4`**, `weight_decay=0.01`, `paramwise_cfg={'img_backbone':
lr_mult=0.1}`, `clip_grad=dict(max_norm=35, norm_type=2)`,
`accumulative_counts=8`, a 500-iter `LinearLR(start_factor=0.05)` warmup
(-> absolute start lr `1e-5`) followed by `CosineAnnealingLR(begin=500,
end=24*num_iters_per_epoch, by_epoch=False, eta_min=1e-6)`, `max_epochs=24`,
`val_interval=9999`, and `randomness=dict(seed=0, deterministic=False)` --
the last one set in every project's `_ori_setting` too, not just
`_unified`. `nuscenes_gs25600_occ3d_unified.py` had drifted from this on 3
points (all fixed now, verified by rebuilding the optimizer/schedulers):
kept GaussianFormer's own `lr=4e-4` instead of the universal `2e-4`; used
`CosineAnnealingLR(by_epoch=True, T_max=24, begin=0)` instead of the
universal iteration-based `by_epoch=False, begin=500, end=24*num_iters_per_epoch`
(computed the same way SurroundOcc's own `_unified.py` does --
`train_samples=28130` there is, not coincidentally, GaussianFormer's own
train split length too); and never set `randomness` at all (now set in
`_base_/occ3d_runtime.py`, so it also applies to `_ori_setting.py`, matching
that convention across projects). `_ori_setting.py` keeping `lr=4e-4` is
correct and unaffected -- it's meant to preserve GaussianFormer's own
recipe, exactly like every other project's own `_ori_setting` keeps its own
paper LR.

**mIoU** (`gaussianformer/evaluation/occ3d_metric.py::GaussianFormerOcc3DMetric`,
used by `val_evaluator`/`test_evaluator` in every non-`_rayiou` config above)
delegates the actual counting/reporting to `mmdet3d.datasets.occ_metrics.Metric_mIoU`
-- the same canonical class `projects/STCOcc`/`projects/FusionOcc` call directly,
and that `projects/SurroundOcc`/`projects/CONet` call transitively through
`OccupancyMetricHybrid` -> STCOcc's `evaluation/occupancy_metric.py`. Its
`count_miou()` prints the exact same per-class IoU/TP/FP/FN table plus
radius-bin and height-bin mIoU breakdown tables those projects' own Occ3D
eval logs show (confirmed by running it against real data -- see
"Verified" below) -- since it's the identical function, the format is
identical by construction, not just matched by hand. `Metric_mIoU` needs
`mask_lidar` too (even with `use_lidar_mask=False`, its `add_batch()`
signature requires it), so `LoadOccupancyOcc3D` now also loads it from the
Occ3D `.npz` and it's carried through `predict()`'s output the same way
`mask_camera` already was. One behavioral difference from the mIoU
evaluator `configs/gaussianformer/nuscenes_gs25600.py` (SurroundOcc GT)
still uses (`GaussianFormerOccupancyMetric`, unchanged, since
`Metric_mIoU` hardcodes Occ3D's own `pc_range` and would mislabel
SurroundOcc GT's different grid in the radius/height breakdown): `Metric_mIoU`
averages over Occ3D's full 17 non-free classes (0='others' through
16='vegetation'), the standard Occ3D-nuScenes benchmark convention, whereas
the previous 16-class list (ported from GaussianFormer_ori's own
SurroundOcc-GT evaluation, which has no 'others' class) excluded class 0
entirely.

**RayIoU** (`gaussianformer/evaluation/rayiou_metric.py`) reuses the shared
DVR ray-casting kernel from `projects/STCOcc/stcocc/datasets/ray_metrics_occ3d.py`
(the same one `projects/SurroundOcc`'s `OccupancyMetricHybrid` delegates to),
but does **not** reuse that class's index-based cross-referencing against a
flat `ann_file` + `nuscenes-devkit`'s own val-sample ordering -- GaussianFormer's
dataset iterates samples in its own order (sorted by
`scene_token + zero-padded frame index`, not nuscenes-devkit's scene-creation
order), so reusing that mechanism as-is would have silently paired
predictions with the wrong ego pose. Instead, each sample's own
lidar-in-ego-frame origin is computed directly in `LoadOccupancyOcc3D` (from
the same `ego2lidar` extrinsic `dataset.py::get_data_info` already computes)
and carried through `predict()` -- no external ordering dependency. This was
checked against `ray_metrics_occ3d.py`'s own reference value (see that
file's docstring for details); running the full eval against a trained
checkpoint to sanity-check the resulting RayIoU score itself was not done.

**Calibration** divides the model's predicted class distribution by a
`temperature` before argmax, same purpose as SurroundOcc/CONet's detectors
dividing their pre-softmax logits by `temperature` in `predict()`. GaussianHead's
CUDA aggregator only exposes a post-aggregation class *probability*
distribution (verified to sum to 1 over classes), not raw logits, so
`BEVSegmentor.predict()` rescales in probability space instead:
`p_T = normalize(p ** (1/T))`, which is mathematically identical to
`softmax(logit/T)` (the shared normalizing constant cancels). `predict()`
also supports `model.export_occ_logits=True` (per `tools/export_occ_logits.py`'s
documented per-model contract: returns `occ_logits`/`voxel_semantics`/
`mask_camera`), needed to fit a temperature with `tools/train_temperature.py`.
The calib/eval split pkls (`nuscenes_infos_val_sweeps_occ_{calib,eval}.pkl`,
75/75 scenes, seed 0) were generated by
`projects/GaussianFormer/tools/split_val_calib_eval.py` -- a project-local
adaptation of the shared `tools/split_val_calib_eval.py` for GaussianFormer's
nested (`{scene_token: [frame, ...]}`) pkl schema, which the shared script's
flat-list assumption doesn't support. **No temperature was actually fit**
(`_calib_eval.py`'s `temperature=1.0` is a placeholder -- fitting needs a
full inference pass over the calib split with a trained checkpoint, out of
scope for this port); see that config's docstring for the exact commands to
run once a checkpoint exists.

`tools/export_occ_logits.py` (the exact same invocation every other project
uses: `python tools/export_occ_logits.py <config> <checkpoint> --output
<out>.npz`, no extra flags) was run end-to-end against `_calib_train.py`
with a dummy checkpoint (the freshly-initialized model's own `state_dict()`,
to exercise the full `Runner.from_cfg` -> `load_checkpoint` -> `test_step`
-> dense-mode extraction path without needing a trained model) and produced
a correctly-shaped `logits (N,18) float32` / `gt (N,) int64` / `mask (N,)
bool` `.npz`, auto-detected as `mode=dense`. This caught a real bug: `predict()`
was returning `occ_logits`/`voxel_semantics`/`mask_camera`/`mask_lidar`/
`lidar_origin` as CUDA tensors, but `export_occ_logits.py`'s
`_process_dense_sample` calls `np.asarray(...)` on them directly with no
`.cpu()` step (matching SurroundOcc/CONet's own detectors, which explicitly
return `.cpu().numpy()` from `predict()` for these same fields) --
`np.asarray()` on a CUDA tensor raises. Fixed by moving those fields to
`.cpu().numpy()` in `predict()` (not `final_occ`/`sampled_label`/`occ_mask`,
which `GaussianFormerOccupancyMetric` -- used by the SurroundOcc-GT baseline
config -- still consumes as tensors via `torch.sum`/boolean-tensor
indexing).

**Uncertainty/calibration metrics via `dist_test.sh --save-predictions` +
`compute_metrics_from_file.py`** (the workflow every other project uses,
e.g. `bash tools/dist_test.sh <config> <ckpt> 1 --save-predictions
<path>.pkl --save-predictions-only --cfg-options
model.compute_uncertainty=True`, then `python
tools/compute_metrics_from_file.py --predictions <path>_rank0.pkl --config
<config> --metric-group ece_nll`/`auroc_fpr95`) needed three more additions,
found and fixed by actually running it end-to-end (dummy checkpoint, real
`dist_test.sh`/`compute_metrics_from_file.py` invocations, not just
unit-testing `predict()` in isolation):

1. **`model.compute_uncertainty=True` support.** `BEVSegmentor.predict()`
   now also computes `softmax_probs` (the temperature-scaled probability
   distribution itself), `uncertainty_msp = 1 - max(softmax_probs)`, and
   `uncertainty_entropy = -(softmax_probs * log(softmax_probs)).sum(-1)`,
   matching CONet's exact formulas
   (`mmdet3d_plugin/occupancy/detectors/occnet.py:883-893`).
2. **`SavePredictionsEvaluator`/STCOcc's `OccupancyMetric.compute_metrics_from_file`
   both key predictions back to GT by an integer `index`**, which
   GaussianFormer_ori's own dataset never produced (`dataset.py::__getitem__`
   only ever returns fields consumers explicitly ask for via
   `return_keys`, and there was no `index` among them, since GaussianFormer_ori's
   own train.py/eval.py never needed one -- they iterate matched
   image/GT pairs together, no separate cross-referencing step). Fixed with
   a small, math-inert addition: `__getitem__` now also threads through the
   sample's position in `self.keyframes` (the *global* dataset index,
   captured before it gets locally shadowed by
   `scene_token, index = self.keyframes[index]`) as `input_dict['index']`,
   picked up via `return_keys=[..., 'index']` and surfaced in `predict()`'s
   output the same way `mask_lidar`/`lidar_origin` already were.
3. **GT-reload-by-index needs a flat `ann_file`.** `SavePredictionsEvaluator`
   only saves `occ_results`/`index`/uncertainty fields, not GT -- both it
   and STCOcc's `OccupancyMetric.compute_metrics_from_file` reload
   `voxel_semantics`/`mask_camera` afterward via `self.data_infos[index]`
   from a flat, list-shaped `ann_file` pkl, which GaussianFormer's own
   nested (`{scene_token: [frame, ...]}`) pkl isn't. Unlike the RayIoU
   metric's lidar-origin lookup (which needed nuscenes-devkit's own
   val-sample ordering and was deliberately built independently instead --
   see `rayiou_metric.py`'s docstring), this index only needs to match
   GaussianFormer's *own* dataset order, which it does by construction (the
   `index` field above *is* that position). So
   `projects/GaussianFormer/tools/build_flat_ann_file.py` (a project-local
   companion to `tools/split_val_calib_eval.py`, for the same reason: that
   shared script's flat-list assumption doesn't fit GaussianFormer's schema)
   replicates `NuScenesDataset.__init__`'s own sort
   and flattens the pkl into one list in that exact order -- verified
   directly, `flat_pkl['data_list'][index]['occ_path']` matches
   `dataset.scene_infos[dataset.keyframes[index][0]][dataset.keyframes[index][1]]['occ_path']`
   for every index checked. Generated for both the standard val split and
   the calib-axis eval split; `GaussianFormerOcc3DMetric` gained
   (unused-internally, carried through only for
   `compute_metrics_from_file.py`'s config-introspection) `ann_file`/
   `data_root` constructor params pointing at these, mirroring how it reads
   them off SurroundOcc/CONet's own `OccupancyMetricHybrid` config.
4. **`occ_logits` was exporting the wrong quantity.** `tools/train_temperature.py`
   computes `softmax(logits / T)` internally; GaussianHead's output is
   already a probability `p` (not a logit), so exporting `p` directly would
   double-softmax. Fixed to export `log(p)` instead: `softmax(log(p) / T) ==
   normalize(p ** (1/T))` exactly (`exp(log(p)) == p`, and `p` already sums
   to 1 so `T=1` recovers `p` unchanged) -- makes `log(p)` behave exactly
   like a true pre-softmax logit for this tool's purposes, without needing
   to touch `train_temperature.py`.

Verified against real data with a dummy checkpoint: `dist_test.sh
--save-predictions --save-predictions-only --cfg-options
model.compute_uncertainty=True` against `_calib_eval_before.py` produced a
predictions pkl with correctly-shaped `occ_results`/`index`/`uncertainty_msp`/
`uncertainty_entropy`/`softmax_probs` per sample; both
`compute_metrics_from_file.py --metric-group ece_nll` and `--metric-group
auroc_fpr95` ran against it end-to-end and printed the same
class/radius-bin/height-bin breakdown format other projects' calibration
runs do (numbers themselves are meaningless for an untrained dummy
checkpoint, as expected).

### Verified (Occ3D configs)

All 11 `nuscenes_gs25600_occ3d*.py` configs parse and resolve their `_base_`
chain correctly. For `_occ3d_ori_setting.py` specifically (representative of
the shared model/dataset code every other variant also exercises):
dataset build + a real sample load (real Occ3D `.npz`, not synthetic),
full forward/loss/backward/predict pass, temperature-scaling code path, and
both the mIoU metric (`GaussianFormerOcc3DMetric`) and the RayIoU metric's
`process()`/`compute_metrics()` all ran end-to-end against real data with a
freshly-initialized (untrained) model -- for the mIoU metric, confirmed the
printed output (per-class IoU/TP/FP/FN, radius-bin and height-bin
verification/summary/detail tables, `use mask: True`) matches
`Metric_mIoU.count_miou()`'s format exactly, since it's the same function
call. The `use_mask_camera=False`/`condition_C_full`/`condition_D_full` mask
logic was verified directly against a real sample's mask arrays. The
calib/eval pkl splits were verified to load (3003/3016 frames). Also caught
and fixed on review: `NuScenesAdaptor` was left at `use_ego=False` in every
occ3d pipeline (copied from the SurroundOcc-GT config), silently projecting
ego-frame Occ3D GT through a LiDAR-frame camera matrix -- see "Coordinate
frame (LiDAR vs. ego)" above.
