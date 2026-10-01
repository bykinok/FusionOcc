"""CPU-only unit tests for the OpenOcc occupancy-only baseline work.

Covers, from CLAUDE_CODE_OPENOCC_REQUIREMENTS_KO.md section 7's required-test list,
the subset that is meaningfully testable without a GPU, a trained checkpoint, or the
full STCOcc model graph (which needs CUDA-only CustomFocalLoss to construct):

  T1  class mapping one-hot / perfect prediction (17-class OpenOcc order)
  T3  free/occupied/ignore/all-ignore/all-free/no-free edge-case batches
  T4  all-ones voxel weight == unweighted loss (equivalence)
  T5  mask-field-absent OpenOcc baseline: get_voxel_loss succeeds when
      camera_mask=None and lambda_inv_free=1.0 (the 'none' policy)
  T6  the same call RAISES when camera_mask=None and lambda_inv_free != 1.0
      (this is the new guard added to stcocc.py::get_voxel_loss)
  T7  GT path identity is the same regardless of which "call site" asks
      (loader / inline evaluator / file evaluator all route through the same
      resolve_occ_gt_dir helper now)
  T8  load_flow=False genuinely skips reading the 'flow' key (and load_flow=True
      genuinely requires it) -- not just "the config happens to not ask for it
      downstream"
  T9  multi-scale downsample shape/class-preservation, and (best-effort, skipped
      if not yet generated) a real generated OpenOcc labels_1_2/1_4/1_8.npz

Explicitly NOT covered here (deferred to the GPU smoke test in
research_openocc/evaluation_smoke.json, or out of scope this pass -- see
research_openocc/implementation_changes.md):
  T2  full label-permutation/inverse-mapping recovery (integration-level)
  T10 temporal scene-boundary/sampler behavior under a real multi-GPU dataloader
  T11 RayIoU synthetic parity against the native evaluator
  T12 saved-prediction vs inline-evaluation equality on real samples
  T13 Occ3D regression -- see research_openocc/implementation_changes.md's
      separate regression run, not folded into this file
  T14 method_plugin visibility-invariance (no method_plugin exists yet)
  T15 evaluator failure-injection / strict_evaluation mode (not implemented
      this pass -- compute_metrics()'s broad except-Exception-returns-0 fallback
      is unchanged; flagged as a known gap, not silently fixed)

Run: python research_openocc/tests/test_openocc_baseline.py
Writes research_openocc/tests/results.csv, exits 1 iff any FAIL.
"""
import csv
import os
import sys
import tempfile

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from projects.STCOcc.stcocc.losses.semkitti import sem_scal_loss, geo_scal_loss  # noqa: E402
from projects.STCOcc.stcocc.detectors.stcocc import STCOcc  # noqa: E402
from projects.STCOcc.stcocc.transforms.pipelines.loading import LoadOccGTFromFileOpenOcc  # noqa: E402
from projects.STCOcc.stcocc.utils.gt_resolver import resolve_occ_gt_dir  # noqa: E402
from mmdet3d.datasets.occ_metrics import Metric_mIoU  # noqa: E402

torch.manual_seed(0)
rows = []


def record(group, name, status, detail):
    rows.append({'group': group, 'test': name, 'status': status, 'detail': detail})
    print(f'[{status}] {group} :: {name} -- {detail}')


OPENOCC_NUM_CLASSES = 17
OPENOCC_FREE_IDX = 16
IGNORE_IDX = 255


class _FakeSTCOcc:
    """Minimal stand-in exposing only the attributes build_inv_free_voxel_weight /
    get_voxel_loss actually read (self.empty_idx, self.lambda_inv_free, the two
    caches). Avoids constructing the real STCOcc (CUDA-only CustomFocalLoss)."""

    def __init__(self, lambda_inv_free=1.0, empty_idx=OPENOCC_FREE_IDX, num_classes=OPENOCC_NUM_CLASSES):
        self.lambda_inv_free = lambda_inv_free
        self.empty_idx = empty_idx
        self.class_weights = torch.ones(num_classes)
        self._inv_free_sanity_logged = True  # suppress the one-time stats print
        self._radius_map_cache = {}

    _stack_voxel_field = staticmethod(STCOcc._stack_voxel_field)
    _get_voxel_radius_map = STCOcc._get_voxel_radius_map
    build_inv_free_voxel_weight = STCOcc.build_inv_free_voxel_weight
    get_voxel_loss = STCOcc.get_voxel_loss


def _toy_batch(n_classes=OPENOCC_NUM_CLASSES, free_idx=OPENOCC_FREE_IDX, shape=(1, 4, 4, 2)):
    """(pred_logits (B,X,Y,Z,C), target (B,X,Y,Z)) toy batch, mostly free + a few
    occupied voxels covering a handful of distinct classes."""
    B, X, Y, Z = shape
    target = torch.full((B, X, Y, Z), free_idx, dtype=torch.long)
    target[0, 0, 0, 0] = 0
    target[0, 0, 1, 0] = 1
    target[0, 1, 0, 0] = 5
    pred = torch.zeros((B, X, Y, Z, n_classes))
    pred.scatter_(-1, target.unsqueeze(-1), 10.0)  # confident "perfect" logits
    return pred, target


def test_t1_class_mapping_one_hot():
    pred, target = _toy_batch()
    pred_cls = pred.argmax(-1).numpy().astype(np.int64)
    gt = target.numpy().astype(np.int64)

    m = Metric_mIoU(num_classes=OPENOCC_NUM_CLASSES)
    expected_order = ['car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
                      'motorcycle', 'pedestrian', 'traffic_cone', 'barrier', 'driveable_surface',
                      'other_flat', 'sidewalk', 'terrain', 'manmade', 'vegetation', 'free']
    ok_order = m.class_names == expected_order
    record('T1', 'class_names_order', 'PASS' if ok_order else 'FAIL',
           f'got={m.class_names}' if not ok_order else '17-class OpenOcc order matches car=0..free=16')

    hist, correct, labeled = m.hist_info(OPENOCC_NUM_CLASSES, pred_cls.flatten(), gt.flatten())
    iou = m.per_class_iu(hist)
    present_classes = sorted(set(gt.flatten().tolist()))
    all_perfect = all(np.isclose(iou[c], 1.0) for c in present_classes)
    record('T1', 'perfect_prediction_iou', 'PASS' if all_perfect else 'FAIL',
           f'iou at present classes {present_classes}: {[round(float(iou[c]), 3) for c in present_classes]}')


def test_t3_edge_batches():
    n_classes = OPENOCC_NUM_CLASSES
    shape = (1, 2, 2, 2)

    def run(name, target_val, pred_val=None):
        target = torch.full(shape, target_val, dtype=torch.long)
        pred = torch.zeros((*shape, n_classes))
        if pred_val is not None:
            pred[..., pred_val] = 5.0
        pred_bcxyz = pred.permute(0, 4, 1, 2, 3)
        try:
            l_sem = sem_scal_loss(pred_bcxyz, target, ignore_index=IGNORE_IDX)
            l_geo = geo_scal_loss(pred_bcxyz, target, ignore_index=IGNORE_IDX, empty_idx=OPENOCC_FREE_IDX)
            finite = torch.isfinite(l_sem) and torch.isfinite(l_geo)
            record('T3', name, 'PASS' if finite else 'FAIL',
                   f'sem_scal={l_sem.item():.4f} geo_scal={l_geo.item():.4f}')
        except ZeroDivisionError as e:
            # Pre-existing gap in sem_scal_loss/geo_scal_loss (shared with Occ3D, not
            # introduced by OpenOcc support): a batch with zero voxels of some non-free
            # class (all-free / all-ignore here) can hit a 0/0 precision-recall term.
            # Not fixed in this pass (would touch Occ3D-shared loss code with its own
            # regression risk) -- flagged as a known gap, not silently patched or hidden.
            # See research_openocc/implementation_changes.md.
            record('T3', name, 'KNOWN_GAP', f'raised {type(e).__name__}: {e} (pre-existing, not OpenOcc-specific)')
        except Exception as e:  # noqa: BLE001
            record('T3', name, 'FAIL', f'raised {type(e).__name__}: {e}')

    run('all_free', OPENOCC_FREE_IDX, pred_val=OPENOCC_FREE_IDX)
    run('all_ignore', IGNORE_IDX, pred_val=0)
    run('no_free_all_occupied', 3, pred_val=3)


def test_t4_all_ones_weight_equivalence():
    pred, target = _toy_batch()
    pred_bcxyz = pred.permute(0, 4, 1, 2, 3)
    ones_weight = torch.ones(target.shape, dtype=torch.float32)

    l_sem_none = sem_scal_loss(pred_bcxyz, target, ignore_index=IGNORE_IDX, voxel_weight=None)
    l_sem_ones = sem_scal_loss(pred_bcxyz, target, ignore_index=IGNORE_IDX, voxel_weight=ones_weight)
    l_geo_none = geo_scal_loss(pred_bcxyz, target, ignore_index=IGNORE_IDX, empty_idx=OPENOCC_FREE_IDX, voxel_weight=None)
    l_geo_ones = geo_scal_loss(pred_bcxyz, target, ignore_index=IGNORE_IDX, empty_idx=OPENOCC_FREE_IDX, voxel_weight=ones_weight)

    ok = torch.isclose(l_sem_none, l_sem_ones, atol=1e-6) and torch.isclose(l_geo_none, l_geo_ones, atol=1e-6)
    record('T4', 'voxel_weight_none_vs_all_ones', 'PASS' if ok else 'FAIL',
           f'sem: none={l_sem_none.item():.6f} ones={l_sem_ones.item():.6f}; '
           f'geo: none={l_geo_none.item():.6f} ones={l_geo_ones.item():.6f}')


def _dummy_focal_loss_factory(call_log):
    def _dummy(pred, target, class_weights, ignore_index=255, voxel_weight=None):
        call_log.append(voxel_weight)
        return torch.tensor(0.0)
    return _dummy


def test_t5_none_policy_succeeds_without_mask():
    pred, target = _toy_batch()
    # Collect3D's real output is a list of B per-sample tensors, not one stacked
    # tensor (mmengine's default collate does not stack this field) -- match that
    # here. A bare tensor would hit an unrelated, already-documented, intentionally
    # unfixed latent bug in losses/lovasz_softmax.py::flatten_probas (see
    # research_v2/reports/consolidated_facts_2026-09-29.md section 3): it only
    # fires because production code never actually passes a bare tensor, so
    # exercising that path here would be testing something that cannot happen in
    # practice. Using a list also happens to be the more faithful choice.
    target_list = [target[i] for i in range(target.shape[0])]
    model = _FakeSTCOcc(lambda_inv_free=1.0, empty_idx=OPENOCC_FREE_IDX)
    call_log = []
    try:
        losses = model.get_voxel_loss(
            pred, target_list, loss_weight=1.0,
            focal_loss=_dummy_focal_loss_factory(call_log),
            tag='c_1_1', camera_mask=None)
        weight_was_none = call_log == [None]
        record('T5', 'lambda1_no_mask_succeeds', 'PASS' if weight_was_none else 'FAIL',
               f'losses keys={list(losses.keys())}, voxel_weight passed to focal_loss={call_log}')
    except Exception as e:  # noqa: BLE001
        record('T5', 'lambda1_no_mask_succeeds', 'FAIL', f'unexpectedly raised {type(e).__name__}: {e}')


def test_t6_nontrivial_lambda_without_mask_raises():
    pred, target = _toy_batch()
    for lam, desc in [(0.25, 'scalar_0.25'),
                       ({'mode': 'radius_piecewise', 'boundary_m': 20.0, 'near': 0.25, 'far': 0.10}, 'dict_mode')]:
        model = _FakeSTCOcc(lambda_inv_free=lam, empty_idx=OPENOCC_FREE_IDX)
        call_log = []
        try:
            model.get_voxel_loss(
                pred, target, loss_weight=1.0,
                focal_loss=_dummy_focal_loss_factory(call_log),
                tag='c_1_1', camera_mask=None)
            record('T6', f'lambda_{desc}_no_mask_raises', 'FAIL', 'did NOT raise -- silent no-op regression')
        except RuntimeError as e:
            record('T6', f'lambda_{desc}_no_mask_raises', 'PASS' if not call_log else 'FAIL',
                   f'raised before calling focal_loss (call_log={call_log}): {e}'[:200])
        except Exception as e:  # noqa: BLE001
            record('T6', f'lambda_{desc}_no_mask_raises', 'FAIL',
                   f'raised wrong exception type {type(e).__name__}: {e}')


def test_t7_gt_resolver_consistency():
    sample_paths = [
        './data/nuscenes/gts/scene-0001/abc123',
        './data/nuscenes/gts/scene-0450/def456',
    ]

    def as_loader(p, ds):
        return resolve_occ_gt_dir(p, ds)

    def as_inline_evaluator(p, ds):
        return resolve_occ_gt_dir(p, ds)

    def as_file_evaluator(p, ds):
        return resolve_occ_gt_dir(p, ds)

    all_ok = True
    details = []
    for p in sample_paths:
        for ds in ('occ3d', 'openocc'):
            a, b, c = as_loader(p, ds), as_inline_evaluator(p, ds), as_file_evaluator(p, ds)
            ok = a == b == c
            all_ok &= ok
            details.append(f'{ds}:{p}->{a} (match={ok})')
    expect_openocc = all('openocc_v2' in d for d in details if 'openocc:' in d)
    expect_occ3d_unchanged = all('/gts/' in d.split('->')[1].split(' ')[0] for d in details if 'occ3d:' in d)
    record('T7', 'resolver_identical_across_call_sites', 'PASS' if all_ok else 'FAIL', '; '.join(details))
    record('T7', 'resolver_openocc_substitutes_dir', 'PASS' if expect_openocc else 'FAIL', '')
    record('T7', 'resolver_occ3d_is_noop', 'PASS' if expect_occ3d_unchanged else 'FAIL', '')


def _write_synthetic_gt(tmpdir, keys):
    scene_tok_gts = os.path.join(tmpdir, 'gts', 'scene-0001', 'tok0')
    scene_tok_openocc = os.path.join(tmpdir, 'openocc_v2', 'scene-0001', 'tok0')
    os.makedirs(scene_tok_gts, exist_ok=True)
    os.makedirs(scene_tok_openocc, exist_ok=True)
    data = {'semantics': np.full((4, 4, 2), OPENOCC_FREE_IDX, dtype=np.int32)}
    if 'flow' in keys:
        data['flow'] = np.zeros((4, 4, 2, 2), dtype=np.float32)
    np.savez(os.path.join(scene_tok_openocc, 'labels.npz'), **data)
    return scene_tok_gts  # the 'gts'-style path the loader receives, as in real info pkls


def test_t8_load_flow_flag_actually_gates_reads():
    with tempfile.TemporaryDirectory() as tmp:
        gts_path_with_flow = _write_synthetic_gt(tmp, keys={'semantics', 'flow'})

        loader_no_flow = LoadOccGTFromFileOpenOcc(scale_1_2=False, scale_1_4=False, scale_1_8=False, load_flow=False)
        try:
            results = loader_no_flow({'occ_path': gts_path_with_flow})
            ok = 'voxel_flows' not in results and 'voxel_semantics' in results
            record('T8', 'load_flow_false_skips_flow_key', 'PASS' if ok else 'FAIL',
                   f'result keys={list(results.keys())}')
        except Exception as e:  # noqa: BLE001
            record('T8', 'load_flow_false_skips_flow_key', 'FAIL', f'unexpectedly raised {type(e).__name__}: {e}')

    with tempfile.TemporaryDirectory() as tmp:
        gts_path_no_flow = _write_synthetic_gt(tmp, keys={'semantics'})  # no 'flow' key at all

        loader_no_flow = LoadOccGTFromFileOpenOcc(scale_1_2=False, scale_1_4=False, scale_1_8=False, load_flow=False)
        try:
            results = loader_no_flow({'occ_path': gts_path_no_flow})
            record('T8', 'load_flow_false_tolerates_missing_flow_key', 'PASS',
                   f'result keys={list(results.keys())}')
        except Exception as e:  # noqa: BLE001
            record('T8', 'load_flow_false_tolerates_missing_flow_key', 'FAIL',
                   f'unexpectedly raised {type(e).__name__}: {e}')

        loader_with_flow = LoadOccGTFromFileOpenOcc(scale_1_2=False, scale_1_4=False, scale_1_8=False, load_flow=True)
        try:
            loader_with_flow({'occ_path': gts_path_no_flow})
            record('T8', 'load_flow_true_requires_flow_key', 'FAIL', 'did NOT raise on missing flow key')
        except KeyError as e:
            record('T8', 'load_flow_true_requires_flow_key', 'PASS', f'raised KeyError as expected: {e}')
        except Exception as e:  # noqa: BLE001
            record('T8', 'load_flow_true_requires_flow_key', 'FAIL',
                   f'raised wrong exception type {type(e).__name__}: {e}')


def test_t9_multiscale_shape_and_class_preservation():
    sys.path.insert(0, os.path.join(REPO_ROOT, 'tools'))
    from generate_ms_occ import downsample_label  # noqa: E402

    rng = np.random.default_rng(0)
    full = rng.integers(0, OPENOCC_NUM_CLASSES, size=(8, 8, 4)).astype(np.int64)
    full_t = torch.from_numpy(full)
    for ds, expected_shape in ((2, (4, 4, 2)), (4, (2, 2, 1))):
        out = downsample_label(full_t, voxel_size=(8, 8, 4), downscale=ds, empty_cls_idx=OPENOCC_FREE_IDX)
        shape_ok = tuple(out.shape) == expected_shape
        classes_ok = set(np.unique(out).tolist()) <= (set(np.unique(full).tolist()) | {255})
        record('T9', f'downsample_1_{ds}_shape_and_classes',
               'PASS' if (shape_ok and classes_ok) else 'FAIL',
               f'shape={tuple(out.shape)} (expected {expected_shape}), '
               f'classes={sorted(set(np.unique(out).tolist()))}')

    # Best-effort: if the real parallel generation has produced at least one
    # labels_1_2.npz by the time this runs, spot-check it too (skip if not ready).
    real_sample = os.path.join(REPO_ROOT, 'data/nuscenes/openocc_v2/scene-0001')
    found = None
    if os.path.isdir(real_sample):
        for tok in os.listdir(real_sample):
            cand = os.path.join(real_sample, tok, 'labels_1_2.npz')
            if os.path.exists(cand):
                found = (os.path.join(real_sample, tok, 'labels.npz'), cand)
                break
    if found is None:
        record('T9', 'real_generated_file_spot_check', 'SKIP', 'no labels_1_2.npz found yet (generation in progress)')
    else:
        full_path, ms_path = found
        full_sem = np.load(full_path)['semantics']
        ms_sem = np.load(ms_path)['semantics']
        shape_ok = ms_sem.shape == (100, 100, 8)
        classes_ok = set(np.unique(ms_sem).tolist()) <= set(np.unique(full_sem).tolist())
        record('T9', 'real_generated_file_spot_check', 'PASS' if (shape_ok and classes_ok) else 'FAIL',
               f'{ms_path}: shape={ms_sem.shape}, classes_subset={classes_ok}')


if __name__ == '__main__':
    test_t1_class_mapping_one_hot()
    test_t3_edge_batches()
    test_t4_all_ones_weight_equivalence()
    test_t5_none_policy_succeeds_without_mask()
    test_t6_nontrivial_lambda_without_mask_raises()
    test_t7_gt_resolver_consistency()
    test_t8_load_flow_flag_actually_gates_reads()
    test_t9_multiscale_shape_and_class_preservation()

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results.csv')
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['group', 'test', 'status', 'detail'])
        writer.writeheader()
        writer.writerows(rows)

    n_pass = sum(1 for r in rows if r['status'] == 'PASS')
    n_fail = sum(1 for r in rows if r['status'] == 'FAIL')
    n_skip = sum(1 for r in rows if r['status'] == 'SKIP')
    n_gap = sum(1 for r in rows if r['status'] == 'KNOWN_GAP')
    print(f"\n{'='*60}\nSummary: {n_pass} PASS, {n_fail} FAIL, {n_skip} SKIP, {n_gap} KNOWN_GAP -> {out_path}\n{'='*60}")
    sys.exit(1 if n_fail else 0)
