#!/usr/bin/env python3
"""
compute_metrics_from_file_v2.py — variant of compute_metrics_from_file.py

Changes vs v1:
  - Radius bins: 0-20m / 20-35m / 35-50m (35m+)  ← 3 bins instead of 7
  - Radius-based summary: per-class breakdown 없음
  - Height-based  summary: per-class breakdown 없음
  - 전체(global) per-class AUROC/ECE/NLL: 'free' class (마지막 클래스) 제외

Usage (single GPU):
  python tools/compute_metrics_from_file_v2.py \\
    --predictions work_dirs/predictions_rank0.pkl \\
    --config projects/BEVFormer/configs/... \\
    [--chunk-size 100] [--ann-file ...] [--data-root ...]

  To reduce memory for very large pkl (e.g. 145GB):
    --passes 3          Run 3 passes (miou | ece_nll | auroc_fpr95) and merge results.
    --metric-group X    Compute only one group: miou | ece_nll | auroc_fpr95.
"""
import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# V2: 3-bin radius scheme  (last bin = 35m+, upper bound is point-cloud max 50m)
V2_RADIUS_BINS = [0, 20, 35, 50]


def main():
    parser = argparse.ArgumentParser(
        description='Compute occupancy metrics from saved predictions file (v2)'
    )
    parser.add_argument('--predictions', required=True, nargs='+',
                        help='Path(s) to .pkl file(s)')
    parser.add_argument('--config', required=True,
                        help='Config file (for ann_file, data_root, etc.)')
    parser.add_argument('--chunk-size', type=int, default=400,
                        help='Process this many predictions per chunk (default: 400)')
    parser.add_argument('--metric-group', choices=['miou', 'ece_nll', 'auroc_fpr95'],
                        default=None,
                        help='Compute only this metric group. Default: all.')
    parser.add_argument('--passes', type=int, choices=[1, 3], default=1,
                        help='If 3: run file 3 times and merge (saves memory). '
                             'Ignored if --metric-group is set.')
    parser.add_argument('--ann-file', default=None,
                        help='Override ann_file')
    parser.add_argument('--data-root', default=None,
                        help='Override data_root')
    parser.add_argument('--cfg-options', nargs='+', default=None,
                        help='Override config options, e.g. test_evaluator.ann_file=other.pkl')
    parser.add_argument('--verbose', action='store_true',
                        help='Print all metric keys. Default: summary only.')
    args = parser.parse_args()

    from mmengine.config import Config
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        opts = {}
        for x in args.cfg_options:
            if '=' in x:
                k, v = x.split('=', 1)
                opts[k] = v
        if opts:
            cfg.merge_from_dict(opts)

    # Resolve ann_file / data_root from config
    ann_file = args.ann_file
    data_root = args.data_root
    if ann_file is None and hasattr(cfg, 'test_evaluator') and isinstance(cfg.test_evaluator, dict):
        ann_file = cfg.test_evaluator.get('ann_file')
    if ann_file is None and hasattr(cfg, 'val_evaluator') and isinstance(cfg.val_evaluator, dict):
        ann_file = cfg.val_evaluator.get('ann_file')
    if data_root is None and hasattr(cfg, 'test_evaluator') and isinstance(cfg.test_evaluator, dict):
        data_root = cfg.test_evaluator.get('data_root', '')
    if data_root is None and hasattr(cfg, 'val_evaluator') and isinstance(cfg.val_evaluator, dict):
        data_root = cfg.val_evaluator.get('data_root', '')
    if not ann_file:
        for key in ('test_dataloader', 'val_dataloader'):
            dl = getattr(cfg, key, None)
            if isinstance(dl, dict) and isinstance(dl.get('dataset'), dict):
                ann_file = dl['dataset'].get('ann_file')
                if ann_file and not data_root and 'data_root' in dl['dataset']:
                    data_root = dl['dataset']['data_root']
                if ann_file:
                    break
    if not ann_file:
        raise SystemExit(
            "Could not determine ann_file. "
            "Pass --ann-file or use a config with test_evaluator.ann_file."
        )
    if data_root and not os.path.isabs(ann_file) and os.path.basename(ann_file) == ann_file:
        ann_file = os.path.join(data_root, ann_file)

    # Load STCOcc evaluation module (avoid registry conflict)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    stcocc_stcocc = os.path.join(root, 'projects', 'STCOcc', 'stcocc')
    if not os.path.isdir(stcocc_stcocc):
        raise SystemExit(f"STCOcc stcocc dir not found at {stcocc_stcocc}")
    if stcocc_stcocc not in sys.path:
        sys.path.insert(0, stcocc_stcocc)

    from mmengine.registry import METRICS as ENGINE_METRICS
    from mmdet3d.registry import METRICS as DET3D_METRICS
    ENGINE_METRICS._module_dict.pop('OccupancyMetric', None)
    DET3D_METRICS._module_dict.pop('OccupancyMetric', None)

    from evaluation.occupancy_metric import OccupancyMetric
    from evaluation.occupancy_metric import (
        _get_height_bins_actual,
        _get_radius_height_grids,
        HEIGHT_BINS_RELATIVE,
        compute_auroc_fpr95,
        _print_auroc_fpr95_summary,
        _print_ece_nll_summary,
        _print_radius_height_uncertainty_summary,
    )

    try:
        from evaluation.occupancy_metric_utils import (
            ece_bin_stats_update,
            ece_from_bin_stats,
            nll_neglog_sum_count,
        )
    except ImportError:
        import importlib.util as _ilu
        _utils_path = os.path.join(stcocc_stcocc, 'evaluation', 'occupancy_metric_utils.py')
        _spec = _ilu.spec_from_file_location('occupancy_metric_utils', _utils_path)
        _mod = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        ece_bin_stats_update = _mod.ece_bin_stats_update
        ece_from_bin_stats = _mod.ece_from_bin_stats
        nll_neglog_sum_count = _mod.nll_neglog_sum_count

    # ------------------------------------------------------------------
    # V2 subclass
    # ------------------------------------------------------------------
    class OccupancyMetricV2(OccupancyMetric):
        """OccupancyMetric with:
        - 3 radius bins: 0-20m, 20-35m, 35-50m (35m+)
        - no per-class breakdown inside radius / height bins
        - 'free' class (index num_classes-1) excluded from per-class global metrics
        """

        @property
        def _free_cls(self):
            return self.num_classes - 1

        # --------------------------------------------------------------
        # compute_metrics_from_file  — overridden to use V2 radius bins
        # and skip per-class radius/height accumulators
        # --------------------------------------------------------------
        def compute_metrics_from_file(self, file_paths, chunk_size=400,
                                      metric_groups=None):
            if not self.data_infos:
                print("Warning: data_infos not loaded. Cannot compute metrics.")
                return {'mIoU': 0.0, 'count': 0}

            groups = metric_groups if metric_groups is not None else [
                'miou', 'ece_nll', 'auroc_fpr95'
            ]
            self._f_metric_groups = set(groups)
            print('\nStarting Evaluation (from file, v2)...')
            print(f'  Radius bins (v2): {V2_RADIUS_BINS}  '
                  f'→ last bin = {V2_RADIUS_BINS[-2]}-{V2_RADIUS_BINS[-1]}m (35m+)')
            if len(self._f_metric_groups) < 3:
                print(f'  Metric groups: {sorted(self._f_metric_groups)} '
                      f'(single-pass mode)')

            from tqdm import tqdm  # noqa: F401 (used in _process_one_chunk)
            self.miou_metric.reset()
            self._f_processed_set = set()
            self._f_height_bins_actual = _get_height_bins_actual(
                self.point_cloud_range
            )

            rk = [
                f'{V2_RADIUS_BINS[i]}-{V2_RADIUS_BINS[i + 1]}m'
                for i in range(len(V2_RADIUS_BINS) - 1)
            ]
            hk = [
                f'{HEIGHT_BINS_RELATIVE[i]}-{HEIGHT_BINS_RELATIVE[i + 1]}m'
                for i in range(len(HEIGHT_BINS_RELATIVE) - 1)
            ]
            self._v2_rk = rk
            self._v2_hk = hk

            def _mk_list(bk):
                return {k: [] for k in bk}

            def _mk_bin_stats(bk):
                return {k: [(0.0, 0.0, 0) for _ in range(10)] for k in bk}

            def _mk_sum_count(bk):
                return {k: 0.0 for k in bk}

            def _mk_count(bk):
                return {k: 0 for k in bk}

            if 'auroc_fpr95' in self._f_metric_groups:
                self._f_pairs_msp = []
                self._f_pairs_entropy = []
                # global (free excluded)
                self._f_pairs_msp_nofree = []
                self._f_pairs_entropy_nofree = []
                # global per-class (free excluded at finalize)
                self._f_class_pairs_msp = {c: [] for c in range(self.num_classes)}
                self._f_class_pairs_entropy = {c: [] for c in range(self.num_classes)}
                # radius/height overall (V2: no per-class)
                self._f_r_msp = _mk_list(rk)
                self._f_r_ent = _mk_list(rk)
                self._f_h_msp = _mk_list(hk)
                self._f_h_ent = _mk_list(hk)

            if 'ece_nll' in self._f_metric_groups:
                self._f_ece_bin_stats = [(0.0, 0.0, 0) for _ in range(10)]
                self._f_nll_neglog_sum = 0.0
                self._f_nll_count = 0
                # global (free excluded)
                self._f_ece_bin_stats_nofree = [(0.0, 0.0, 0) for _ in range(10)]
                self._f_nll_neglog_sum_nofree = 0.0
                self._f_nll_count_nofree = 0
                # global per-class (free excluded at finalize)
                self._f_class_ece_bin_stats = {
                    c: [(0.0, 0.0, 0) for _ in range(10)]
                    for c in range(self.num_classes)
                }
                self._f_class_nll_sum = {c: 0.0 for c in range(self.num_classes)}
                self._f_class_nll_count = {c: 0 for c in range(self.num_classes)}
                # radius/height overall (V2: no per-class)
                self._f_r_ece_bin_stats = _mk_bin_stats(rk)
                self._f_r_nll_sum = _mk_sum_count(rk)
                self._f_r_nll_count = _mk_count(rk)
                self._f_h_ece_bin_stats = _mk_bin_stats(hk)
                self._f_h_nll_sum = _mk_sum_count(hk)
                self._f_h_nll_count = _mk_count(hk)

            buffer = []
            for path in file_paths:
                if not os.path.isfile(path):
                    continue
                with open(path, 'rb') as fh:
                    while True:
                        try:
                            batch = pickle.load(fh)
                        except EOFError:
                            break
                        if isinstance(batch, list):
                            buffer.extend(batch)
                        else:
                            buffer.append(batch)
                        while len(buffer) >= chunk_size:
                            chunk = buffer[:chunk_size]
                            buffer = buffer[chunk_size:]
                            self._process_one_chunk(chunk)
            if buffer:
                self._process_one_chunk(buffer)

            return self._finalize_file_metrics()

        # --------------------------------------------------------------
        # _process_one_chunk  — overridden to:
        #   1) use V2_RADIUS_BINS (3 bins)
        #   2) skip per-class inside radius/height loops
        #   3) skip free class in global per-class accumulation
        # --------------------------------------------------------------
        def _process_one_chunk(self, chunk):
            from tqdm import tqdm
            pred_sems, data_index = [], []
            u_msp_list, u_ent_list, sp_list = [], [], []

            for pred_dict in chunk:
                occ_results = pred_dict.get('occ_results')
                if (occ_results is None
                        or 'index' not in pred_dict
                        or pred_dict['index'] is None):
                    continue
                data_id = pred_dict['index']
                if not isinstance(data_id, (list, tuple, np.ndarray)):
                    data_id = [data_id]
                for i, id_ in enumerate(data_id):
                    if id_ in self._f_processed_set:
                        continue
                    self._f_processed_set.add(id_)
                    if i >= len(occ_results):
                        continue
                    data_index.append(id_)
                    pred_sems.append(occ_results[i])
                    need_auroc = 'auroc_fpr95' in self._f_metric_groups
                    need_ece_nll = 'ece_nll' in self._f_metric_groups
                    if need_auroc or need_ece_nll:
                        u_msp_list.append(
                            np.asarray(pred_dict['uncertainty_msp'][i]).astype(np.float64)
                            if pred_dict.get('uncertainty_msp') and i < len(pred_dict['uncertainty_msp'])
                            else None
                        )
                        u_ent_list.append(
                            np.asarray(pred_dict['uncertainty_entropy'][i]).astype(np.float64)
                            if pred_dict.get('uncertainty_entropy') and i < len(pred_dict['uncertainty_entropy'])
                            else None
                        )
                    else:
                        u_msp_list.append(None)
                        u_ent_list.append(None)
                    if need_ece_nll:
                        sp_list.append(
                            np.asarray(pred_dict['softmax_probs'][i]).astype(np.float64)
                            if pred_dict.get('softmax_probs') and i < len(pred_dict['softmax_probs'])
                            else None
                        )
                    else:
                        sp_list.append(None)

            if not data_index:
                return

            free_cls = self._free_cls

            for index in tqdm(data_index, leave=False, desc='Chunk'):
                if index >= len(self.data_infos):
                    break
                info = self.data_infos[index]
                occ_path = (info.get('occ3d_gt_path')
                            or info.get('occ_path')
                            or info.get('occ_gt_path'))
                if not occ_path:
                    continue
                if self.dataset_name == 'openocc':
                    occ_path = occ_path.replace('gts', 'openocc_v2')
                if not occ_path.endswith('labels.npz'):
                    occ_path = os.path.join(occ_path, 'labels.npz')
                if self.data_root and not os.path.isabs(occ_path):
                    root = os.path.normpath(self.data_root.rstrip(os.sep))
                    path_norm = os.path.normpath(occ_path)
                    if not path_norm.startswith(root + os.sep) and path_norm != root:
                        occ_path = os.path.join(self.data_root, occ_path)

                try:
                    occ_gt = np.load(occ_path, allow_pickle=True)
                    gt_semantics = occ_gt['semantics']
                    idx = data_index.index(index)
                    pr_semantics = pred_sems[idx]
                    mask_camera = (
                        occ_gt['mask_camera'].astype(bool)
                        if (self.dataset_name == 'occ3d' or self.use_image_mask)
                        else None
                    )
                    if 'miou' in self._f_metric_groups:
                        self.miou_metric.add_batch(
                            pr_semantics, gt_semantics, None, mask_camera
                        )

                    pr_flat = np.asarray(pr_semantics).reshape(-1)
                    gt_flat = np.asarray(gt_semantics).reshape(-1)
                    mask_flat = (
                        mask_camera.reshape(-1) if mask_camera is not None
                        else np.ones(pr_flat.shape, dtype=bool)
                    )
                    valid = mask_flat
                    n_valid = valid.sum()

                    if n_valid == 0:
                        continue

                    gt_valid = gt_flat[valid].astype(np.int64)
                    pr_valid = pr_flat[valid].astype(np.int64)
                    y_incorrect = (pr_valid != gt_valid).astype(np.int64)
                    non_free_global = (gt_valid != free_cls)

                    # ---- Global AUROC/FPR95 ----
                    if 'auroc_fpr95' in self._f_metric_groups:
                        if idx < len(u_msp_list) and u_msp_list[idx] is not None:
                            u_msp_flat = np.asarray(
                                u_msp_list[idx]
                            ).reshape(-1)[valid].astype(np.float64)
                            self._f_pairs_msp.append((y_incorrect, u_msp_flat))
                            if non_free_global.sum() > 0:
                                self._f_pairs_msp_nofree.append((
                                    y_incorrect[non_free_global],
                                    u_msp_flat[non_free_global],
                                ))
                            for c in range(self.num_classes):
                                if c == free_cls:
                                    continue
                                mask_c = (gt_valid == c)
                                if mask_c.sum() > 0:
                                    self._f_class_pairs_msp[c].append((
                                        (pr_valid[mask_c] != c).astype(np.int64),
                                        u_msp_flat[mask_c],
                                    ))
                        if idx < len(u_ent_list) and u_ent_list[idx] is not None:
                            u_ent_flat = np.asarray(
                                u_ent_list[idx]
                            ).reshape(-1)[valid].astype(np.float64)
                            self._f_pairs_entropy.append((y_incorrect, u_ent_flat))
                            if non_free_global.sum() > 0:
                                self._f_pairs_entropy_nofree.append((
                                    y_incorrect[non_free_global],
                                    u_ent_flat[non_free_global],
                                ))
                            for c in range(self.num_classes):
                                if c == free_cls:
                                    continue
                                mask_c = (gt_valid == c)
                                if mask_c.sum() > 0:
                                    self._f_class_pairs_entropy[c].append((
                                        (pr_valid[mask_c] != c).astype(np.int64),
                                        u_ent_flat[mask_c],
                                    ))

                    # ---- Global ECE/NLL ----
                    if 'ece_nll' in self._f_metric_groups:
                        if idx < len(u_msp_list) and u_msp_list[idx] is not None:
                            conf = (
                                1.0 - np.asarray(u_msp_list[idx]).reshape(-1)[valid]
                            ).astype(np.float64)
                            acc = (pr_valid == gt_valid).astype(np.float64)
                            ece_bin_stats_update(
                                self._f_ece_bin_stats, conf, acc, n_bins=10
                            )
                            if non_free_global.sum() > 0:
                                ece_bin_stats_update(
                                    self._f_ece_bin_stats_nofree,
                                    conf[non_free_global], acc[non_free_global],
                                    n_bins=10,
                                )
                            for c in range(self.num_classes):
                                if c == free_cls:
                                    continue
                                mask_c = (gt_valid == c)
                                if mask_c.sum() > 0:
                                    ece_bin_stats_update(
                                        self._f_class_ece_bin_stats[c],
                                        conf[mask_c], acc[mask_c], n_bins=10,
                                    )
                        if idx < len(sp_list) and sp_list[idx] is not None:
                            probs_flat = np.asarray(
                                sp_list[idx]
                            ).reshape(-1, self.num_classes)[valid]
                            neglog_sum, cnt = nll_neglog_sum_count(probs_flat, gt_valid)
                            self._f_nll_neglog_sum += neglog_sum
                            self._f_nll_count += cnt
                            if non_free_global.sum() > 0:
                                ns_nf, nc_nf = nll_neglog_sum_count(
                                    probs_flat[non_free_global],
                                    gt_valid[non_free_global],
                                )
                                self._f_nll_neglog_sum_nofree += ns_nf
                                self._f_nll_count_nofree += nc_nf
                            for c in range(self.num_classes):
                                if c == free_cls:
                                    continue
                                mask_c = (gt_valid == c)
                                if mask_c.sum() > 0:
                                    ns_c, nc_c = nll_neglog_sum_count(
                                        probs_flat[mask_c], gt_valid[mask_c]
                                    )
                                    self._f_class_nll_sum[c] += ns_c
                                    self._f_class_nll_count[c] += nc_c

                    # ---- Radius / Height breakdown (V2: no per-class) ----
                    try:
                        shp = np.asarray(pr_semantics).shape
                        if len(shp) < 3:
                            continue
                        radius_grid, z_grid = _get_radius_height_grids(
                            shp, self.point_cloud_range
                        )
                        radius_flat = radius_grid.reshape(-1)[valid]
                        height_flat = z_grid.reshape(-1)[valid]

                        # free class 제외 마스크 (radius/height 공용)
                        non_free = (gt_valid != free_cls)

                        # Radius bins (V2: 3 bins, no per-class, free excluded)
                        for ri in range(len(V2_RADIUS_BINS) - 1):
                            r_min, r_max = V2_RADIUS_BINS[ri], V2_RADIUS_BINS[ri + 1]
                            r_key = f'{r_min}-{r_max}m'
                            if ri == len(V2_RADIUS_BINS) - 2:
                                in_r_raw = (radius_flat >= r_min)
                            else:
                                in_r_raw = (radius_flat >= r_min) & (radius_flat < r_max)
                            in_r = in_r_raw & non_free   # free 제외
                            if in_r.sum() == 0:
                                continue

                            if 'auroc_fpr95' in self._f_metric_groups:
                                if idx < len(u_msp_list) and u_msp_list[idx] is not None:
                                    u_msp_flat = np.asarray(
                                        u_msp_list[idx]
                                    ).reshape(-1)[valid].astype(np.float64)
                                    self._f_r_msp[r_key].append(
                                        (y_incorrect[in_r], u_msp_flat[in_r])
                                    )
                                if idx < len(u_ent_list) and u_ent_list[idx] is not None:
                                    u_ent_flat = np.asarray(
                                        u_ent_list[idx]
                                    ).reshape(-1)[valid].astype(np.float64)
                                    self._f_r_ent[r_key].append(
                                        (y_incorrect[in_r], u_ent_flat[in_r])
                                    )
                            if 'ece_nll' in self._f_metric_groups:
                                if idx < len(u_msp_list) and u_msp_list[idx] is not None:
                                    conf = (
                                        1.0 - np.asarray(u_msp_list[idx]).reshape(-1)[valid]
                                    ).astype(np.float64)
                                    acc = (pr_valid == gt_valid).astype(np.float64)
                                    ece_bin_stats_update(
                                        self._f_r_ece_bin_stats[r_key],
                                        conf[in_r], acc[in_r], n_bins=10,
                                    )
                                if idx < len(sp_list) and sp_list[idx] is not None:
                                    probs_flat = np.asarray(
                                        sp_list[idx]
                                    ).reshape(-1, self.num_classes)[valid]
                                    ns, nc = nll_neglog_sum_count(
                                        probs_flat[in_r], gt_valid[in_r]
                                    )
                                    self._f_r_nll_sum[r_key] += ns
                                    self._f_r_nll_count[r_key] += nc

                        # Height bins (no per-class, free excluded)
                        for hi in range(len(self._f_height_bins_actual) - 1):
                            h_min = self._f_height_bins_actual[hi]
                            h_max = self._f_height_bins_actual[hi + 1]
                            h_key = (
                                f'{HEIGHT_BINS_RELATIVE[hi]}'
                                f'-{HEIGHT_BINS_RELATIVE[hi + 1]}m'
                            )
                            if hi == len(self._f_height_bins_actual) - 2:
                                in_h_raw = (height_flat >= h_min)
                            else:
                                in_h_raw = (height_flat >= h_min) & (height_flat < h_max)
                            in_h = in_h_raw & non_free   # free 제외
                            if in_h.sum() == 0:
                                continue

                            if 'auroc_fpr95' in self._f_metric_groups:
                                if idx < len(u_msp_list) and u_msp_list[idx] is not None:
                                    u_msp_flat = np.asarray(
                                        u_msp_list[idx]
                                    ).reshape(-1)[valid].astype(np.float64)
                                    self._f_h_msp[h_key].append(
                                        (y_incorrect[in_h], u_msp_flat[in_h])
                                    )
                                if idx < len(u_ent_list) and u_ent_list[idx] is not None:
                                    u_ent_flat = np.asarray(
                                        u_ent_list[idx]
                                    ).reshape(-1)[valid].astype(np.float64)
                                    self._f_h_ent[h_key].append(
                                        (y_incorrect[in_h], u_ent_flat[in_h])
                                    )
                            if 'ece_nll' in self._f_metric_groups:
                                if idx < len(u_msp_list) and u_msp_list[idx] is not None:
                                    conf = (
                                        1.0 - np.asarray(u_msp_list[idx]).reshape(-1)[valid]
                                    ).astype(np.float64)
                                    acc = (pr_valid == gt_valid).astype(np.float64)
                                    ece_bin_stats_update(
                                        self._f_h_ece_bin_stats[h_key],
                                        conf[in_h], acc[in_h], n_bins=10,
                                    )
                                if idx < len(sp_list) and sp_list[idx] is not None:
                                    probs_flat = np.asarray(
                                        sp_list[idx]
                                    ).reshape(-1, self.num_classes)[valid]
                                    ns, nc = nll_neglog_sum_count(
                                        probs_flat[in_h], gt_valid[in_h]
                                    )
                                    self._f_h_nll_sum[h_key] += ns
                                    self._f_h_nll_count[h_key] += nc

                    except Exception:
                        pass

                except Exception as e:
                    print(f"Warning: Failed to load GT for index {index}: {e}")

        # --------------------------------------------------------------
        # _finalize_file_metrics  — overridden to:
        #   1) exclude free class from global per-class AUROC/ECE/NLL
        #   2) skip per-class radius/height output
        # --------------------------------------------------------------
        def _finalize_file_metrics(self):
            class_names, miou_array, cnt = self.miou_metric.count_miou()
            metrics = {}
            free_cls = self._free_cls

            if 'miou' in self._f_metric_groups:
                mean_iou = np.nanmean(miou_array[:self.num_classes - 1]) * 100
                metrics['mIoU'] = mean_iou
                metrics['count'] = cnt
                for i, (class_name, iou) in enumerate(zip(class_names, miou_array)):
                    if i < len(class_names):
                        metrics[f'IoU_{class_name}'] = round(iou * 100, 2)
            else:
                metrics['count'] = cnt

            has_auroc = (
                'auroc_fpr95' in self._f_metric_groups
                and getattr(self, '_f_pairs_msp', None) is not None
                and (self._f_pairs_msp or self._f_pairs_entropy)
            )
            has_ece_nll = (
                'ece_nll' in self._f_metric_groups
                and (
                    getattr(self, '_f_nll_count', 0) > 0
                    or any(b[2] > 0 for b in getattr(self, '_f_ece_bin_stats', []))
                )
            )
            if not has_auroc and not has_ece_nll:
                return metrics

            # ---- AUROC / FPR95 ----
            if 'auroc_fpr95' in self._f_metric_groups:
                if self._f_pairs_msp:
                    y_all = np.concatenate([p[0] for p in self._f_pairs_msp], axis=0)
                    s_all = np.concatenate([p[1] for p in self._f_pairs_msp], axis=0)
                    auroc_msp, fpr95_msp = compute_auroc_fpr95(y_all, s_all)
                    metrics['AUROC_uncertainty_msp'] = round(auroc_msp * 100, 2)
                    metrics['FPR95_uncertainty_msp'] = round(fpr95_msp * 100, 2)
                if self._f_pairs_entropy:
                    y_all = np.concatenate([p[0] for p in self._f_pairs_entropy], axis=0)
                    s_all = np.concatenate([p[1] for p in self._f_pairs_entropy], axis=0)
                    auroc_ent, fpr95_ent = compute_auroc_fpr95(y_all, s_all)
                    metrics['AUROC_uncertainty_entropy'] = round(auroc_ent * 100, 2)
                    metrics['FPR95_uncertainty_entropy'] = round(fpr95_ent * 100, 2)
                # free-excluded global
                if self._f_pairs_msp_nofree:
                    y_all = np.concatenate([p[0] for p in self._f_pairs_msp_nofree], axis=0)
                    s_all = np.concatenate([p[1] for p in self._f_pairs_msp_nofree], axis=0)
                    auroc_msp_nf, fpr95_msp_nf = compute_auroc_fpr95(y_all, s_all)
                    metrics['AUROC_nofree_uncertainty_msp'] = round(auroc_msp_nf * 100, 2)
                    metrics['FPR95_nofree_uncertainty_msp'] = round(fpr95_msp_nf * 100, 2)
                if self._f_pairs_entropy_nofree:
                    y_all = np.concatenate([p[0] for p in self._f_pairs_entropy_nofree], axis=0)
                    s_all = np.concatenate([p[1] for p in self._f_pairs_entropy_nofree], axis=0)
                    auroc_ent_nf, fpr95_ent_nf = compute_auroc_fpr95(y_all, s_all)
                    metrics['AUROC_nofree_uncertainty_entropy'] = round(auroc_ent_nf * 100, 2)
                    metrics['FPR95_nofree_uncertainty_entropy'] = round(fpr95_ent_nf * 100, 2)

                def _add_per_class_auroc(class_pairs_dict, prefix):
                    auroc_list, fpr95_list = [], []
                    for c in range(self.num_classes):
                        if c == free_cls:   # V2: skip free
                            continue
                        if not class_pairs_dict[c]:
                            continue
                        y_all = np.concatenate(
                            [p[0] for p in class_pairs_dict[c]], axis=0
                        )
                        s_all = np.concatenate(
                            [p[1] for p in class_pairs_dict[c]], axis=0
                        )
                        auroc_c, fpr95_c = compute_auroc_fpr95(y_all, s_all)
                        name = class_names[c] if c < len(class_names) else f'class_{c}'
                        metrics[f'{prefix}_AUROC_{name}'] = round(auroc_c * 100, 2)
                        metrics[f'{prefix}_FPR95_{name}'] = round(fpr95_c * 100, 2)
                        if not np.isnan(auroc_c):
                            auroc_list.append(auroc_c * 100)
                        if not np.isnan(fpr95_c):
                            fpr95_list.append(fpr95_c * 100)
                    if auroc_list:
                        metrics[f'mAUROC_{prefix}'] = round(float(np.mean(auroc_list)), 2)
                    if fpr95_list:
                        metrics[f'mFPR95_{prefix}'] = round(float(np.mean(fpr95_list)), 2)

                if any(
                    self._f_class_pairs_msp[c]
                    for c in range(self.num_classes) if c != free_cls
                ):
                    _add_per_class_auroc(self._f_class_pairs_msp, 'uncertainty_msp')
                if any(
                    self._f_class_pairs_entropy[c]
                    for c in range(self.num_classes) if c != free_cls
                ):
                    _add_per_class_auroc(self._f_class_pairs_entropy, 'uncertainty_entropy')

                _print_auroc_fpr95_summary(metrics, class_names, self.num_classes)

                # Print free-excluded global
                if ('AUROC_nofree_uncertainty_msp' in metrics
                        or 'AUROC_nofree_uncertainty_entropy' in metrics):
                    print("\n===> Global (free class excluded):")
                    print(f"  {'Measure':>22s} | {'AUROC %':>8s} | {'FPR95 %':>8s}")
                    print("  " + "-" * 44)
                    if 'AUROC_nofree_uncertainty_msp' in metrics:
                        print(
                            f"  {'uncertainty_msp':>22s} | "
                            f"{metrics['AUROC_nofree_uncertainty_msp']:>7.2f} | "
                            f"{metrics.get('FPR95_nofree_uncertainty_msp', 0):>7.2f}"
                        )
                    if 'AUROC_nofree_uncertainty_entropy' in metrics:
                        print(
                            f"  {'uncertainty_entropy':>22s} | "
                            f"{metrics['AUROC_nofree_uncertainty_entropy']:>7.2f} | "
                            f"{metrics.get('FPR95_nofree_uncertainty_entropy', 0):>7.2f}"
                        )

            # ---- ECE / NLL ----
            if 'ece_nll' in self._f_metric_groups:
                if any(b[2] > 0 for b in self._f_ece_bin_stats):
                    metrics['ECE'] = round(
                        ece_from_bin_stats(self._f_ece_bin_stats, n_bins=10) * 100, 2
                    )
                if self._f_nll_count > 0:
                    metrics['NLL'] = round(
                        self._f_nll_neglog_sum / self._f_nll_count, 4
                    )
                # free-excluded global
                if any(b[2] > 0 for b in self._f_ece_bin_stats_nofree):
                    metrics['ECE_nofree'] = round(
                        ece_from_bin_stats(self._f_ece_bin_stats_nofree, n_bins=10) * 100, 2
                    )
                if self._f_nll_count_nofree > 0:
                    metrics['NLL_nofree'] = round(
                        self._f_nll_neglog_sum_nofree / self._f_nll_count_nofree, 4
                    )

                # Per-class ECE (free excluded)
                ece_list = []
                for c in range(self.num_classes):
                    if c == free_cls:
                        continue
                    if not any(b[2] > 0 for b in self._f_class_ece_bin_stats[c]):
                        continue
                    ece_c = ece_from_bin_stats(self._f_class_ece_bin_stats[c], n_bins=10)
                    name = class_names[c] if c < len(class_names) else f'class_{c}'
                    metrics[f'ECE_{name}'] = round(ece_c * 100, 2)
                    if not np.isnan(ece_c):
                        ece_list.append(ece_c * 100)
                if ece_list:
                    metrics['mECE'] = round(float(np.mean(ece_list)), 2)

                # Per-class NLL (free excluded)
                nll_list = []
                for c in range(self.num_classes):
                    if c == free_cls:
                        continue
                    if self._f_class_nll_count[c] <= 0:
                        continue
                    nll_c = self._f_class_nll_sum[c] / self._f_class_nll_count[c]
                    name = class_names[c] if c < len(class_names) else f'class_{c}'
                    metrics[f'NLL_{name}'] = round(nll_c, 4)
                    if not np.isnan(nll_c):
                        nll_list.append(nll_c)
                if nll_list:
                    metrics['mNLL'] = round(float(np.mean(nll_list)), 4)

                _print_ece_nll_summary(metrics, class_names, self.num_classes)

                # Print free-excluded overall
                if 'ECE_nofree' in metrics or 'NLL_nofree' in metrics:
                    print("\n===> Overall (free class excluded):")
                    print(f"  {'Metric':>8s} | {'Value':>10s}")
                    print("  " + "-" * 22)
                    if 'ECE_nofree' in metrics:
                        print(f"  {'ECE %':>8s} | {metrics['ECE_nofree']:>10.2f}")
                    if 'NLL_nofree' in metrics:
                        print(f"  {'NLL':>8s} | {metrics['NLL_nofree']:>10.4f}")

                # Radius ECE/NLL (V2: no per-class inner loop)
                for r_key in sorted(
                    self._f_r_ece_bin_stats.keys(),
                    key=lambda x: float(x.split('-')[0]),
                ):
                    if any(b[2] > 0 for b in self._f_r_ece_bin_stats[r_key]):
                        metrics[f'radius_{r_key}_ECE'] = round(
                            ece_from_bin_stats(
                                self._f_r_ece_bin_stats[r_key], n_bins=10
                            ) * 100, 2
                        )
                    if self._f_r_nll_count[r_key] > 0:
                        metrics[f'radius_{r_key}_NLL'] = round(
                            self._f_r_nll_sum[r_key] / self._f_r_nll_count[r_key], 4
                        )

                # Height ECE/NLL (V2: no per-class inner loop)
                for h_key in sorted(
                    self._f_h_ece_bin_stats.keys(),
                    key=lambda x: float(x.split('-')[0]),
                ):
                    if any(b[2] > 0 for b in self._f_h_ece_bin_stats[h_key]):
                        metrics[f'height_{h_key}_ECE'] = round(
                            ece_from_bin_stats(
                                self._f_h_ece_bin_stats[h_key], n_bins=10
                            ) * 100, 2
                        )
                    if self._f_h_nll_count[h_key] > 0:
                        metrics[f'height_{h_key}_NLL'] = round(
                            self._f_h_nll_sum[h_key] / self._f_h_nll_count[h_key], 4
                        )

            # Radius AUROC (V2: no per-class inner loop)
            if 'auroc_fpr95' in self._f_metric_groups:
                for r_key in sorted(
                    self._f_r_msp.keys(), key=lambda x: float(x.split('-')[0])
                ):
                    if self._f_r_msp[r_key]:
                        y_all = np.concatenate(
                            [p[0] for p in self._f_r_msp[r_key]], axis=0
                        )
                        s_all = np.concatenate(
                            [p[1] for p in self._f_r_msp[r_key]], axis=0
                        )
                        auroc, fpr95 = compute_auroc_fpr95(y_all, s_all)
                        metrics[f'radius_{r_key}_AUROC_msp'] = round(auroc * 100, 2)
                        metrics[f'radius_{r_key}_FPR95_msp'] = round(fpr95 * 100, 2)
                    if self._f_r_ent[r_key]:
                        y_all = np.concatenate(
                            [p[0] for p in self._f_r_ent[r_key]], axis=0
                        )
                        s_all = np.concatenate(
                            [p[1] for p in self._f_r_ent[r_key]], axis=0
                        )
                        auroc, fpr95 = compute_auroc_fpr95(y_all, s_all)
                        metrics[f'radius_{r_key}_AUROC_entropy'] = round(auroc * 100, 2)
                        metrics[f'radius_{r_key}_FPR95_entropy'] = round(fpr95 * 100, 2)

                # Height AUROC (V2: no per-class inner loop)
                for h_key in sorted(
                    self._f_h_msp.keys(), key=lambda x: float(x.split('-')[0])
                ):
                    if self._f_h_msp[h_key]:
                        y_all = np.concatenate(
                            [p[0] for p in self._f_h_msp[h_key]], axis=0
                        )
                        s_all = np.concatenate(
                            [p[1] for p in self._f_h_msp[h_key]], axis=0
                        )
                        auroc, fpr95 = compute_auroc_fpr95(y_all, s_all)
                        metrics[f'height_{h_key}_AUROC_msp'] = round(auroc * 100, 2)
                        metrics[f'height_{h_key}_FPR95_msp'] = round(fpr95 * 100, 2)
                    if self._f_h_ent[h_key]:
                        y_all = np.concatenate(
                            [p[0] for p in self._f_h_ent[h_key]], axis=0
                        )
                        s_all = np.concatenate(
                            [p[1] for p in self._f_h_ent[h_key]], axis=0
                        )
                        auroc, fpr95 = compute_auroc_fpr95(y_all, s_all)
                        metrics[f'height_{h_key}_AUROC_entropy'] = round(auroc * 100, 2)
                        metrics[f'height_{h_key}_FPR95_entropy'] = round(fpr95 * 100, 2)

            if 'ece_nll' in self._f_metric_groups or 'auroc_fpr95' in self._f_metric_groups:
                _print_radius_height_uncertainty_summary(
                    metrics, class_names, self.num_classes
                )

            return metrics

    # ------------------------------------------------------------------
    # Instantiate V2 metric and run
    # ------------------------------------------------------------------
    num_classes = 18
    if hasattr(cfg, 'test_evaluator') and isinstance(cfg.test_evaluator, dict):
        num_classes = cfg.test_evaluator.get('num_classes', 18)

    metric = OccupancyMetricV2(
        num_classes=num_classes,
        use_lidar_mask=False,
        use_image_mask=True,
        ann_file=ann_file,
        data_root=data_root or '',
        dataset_name='occ3d',
        eval_metric='miou',
        sort_by_timestamp=True,
        point_cloud_range=[-40.0, -40.0, -1.0, 40.0, 40.0, 5.4],
        compute_uncertainty_metrics=True,
    )

    if not hasattr(metric, 'compute_metrics_from_file'):
        raise SystemExit(
            "OccupancyMetric does not have compute_metrics_from_file. "
            "Update STCOcc metric."
        )

    if args.metric_group:
        metric_groups = [args.metric_group]
        print(
            f"Reading predictions from {args.predictions}, "
            f"chunk_size={args.chunk_size}, metric_group={args.metric_group}"
        )
        results = metric.compute_metrics_from_file(
            args.predictions, chunk_size=args.chunk_size,
            metric_groups=metric_groups,
        )
    elif args.passes == 3:
        print(
            f"Running 3 passes (miou, ece_nll, auroc_fpr95) to reduce memory. "
            f"Reading from {args.predictions}, chunk_size={args.chunk_size}"
        )
        results = {}
        for group in ['miou', 'ece_nll', 'auroc_fpr95']:
            print(f"\n--- Pass: {group} ---")
            out = metric.compute_metrics_from_file(
                args.predictions, chunk_size=args.chunk_size,
                metric_groups=[group],
            )
            for k, v in out.items():
                if k == 'count' and v == 0 and results.get('count', 0) != 0:
                    continue
                results[k] = v
    else:
        print(
            f"Reading predictions from {args.predictions}, "
            f"chunk_size={args.chunk_size}"
        )
        results = metric.compute_metrics_from_file(
            args.predictions, chunk_size=args.chunk_size,
        )

    print("=" * 60)
    print("=== Metrics Summary (v2) ===")
    print("=== Radius bins: 0-20m / 20-35m / 35-50m(35m+) ===")
    print("=== Per-class: free class excluded ===")
    print("=" * 60)
    print(f"  mIoU:    {results.get('mIoU', 0):.2f}%")
    print(f"  count:   {results.get('count', 0)}")
    if 'AUROC_uncertainty_msp' in results or 'AUROC_uncertainty_entropy' in results:
        print("\n--- Uncertainty (global, all classes incl. free) ---")
        print(f"  {'Measure':>22s} | {'AUROC %':>8s} | {'FPR95 %':>8s}")
        print("  " + "-" * 42)
        if 'AUROC_uncertainty_msp' in results:
            print(
                f"  {'uncertainty_msp':>22s} | "
                f"{results['AUROC_uncertainty_msp']:>7.2f} | "
                f"{results.get('FPR95_uncertainty_msp', 0):>7.2f}"
            )
        if 'AUROC_uncertainty_entropy' in results:
            print(
                f"  {'uncertainty_entropy':>22s} | "
                f"{results['AUROC_uncertainty_entropy']:>7.2f} | "
                f"{results.get('FPR95_uncertainty_entropy', 0):>7.2f}"
            )
    if 'AUROC_nofree_uncertainty_msp' in results or 'AUROC_nofree_uncertainty_entropy' in results:
        print("\n--- Uncertainty (global, free class excluded) ---")
        print(f"  {'Measure':>22s} | {'AUROC %':>8s} | {'FPR95 %':>8s}")
        print("  " + "-" * 42)
        if 'AUROC_nofree_uncertainty_msp' in results:
            print(
                f"  {'uncertainty_msp':>22s} | "
                f"{results['AUROC_nofree_uncertainty_msp']:>7.2f} | "
                f"{results.get('FPR95_nofree_uncertainty_msp', 0):>7.2f}"
            )
        if 'AUROC_nofree_uncertainty_entropy' in results:
            print(
                f"  {'uncertainty_entropy':>22s} | "
                f"{results['AUROC_nofree_uncertainty_entropy']:>7.2f} | "
                f"{results.get('FPR95_nofree_uncertainty_entropy', 0):>7.2f}"
            )
    if 'ECE' in results or 'NLL' in results:
        print("\n--- Calibration (all classes incl. free) ---")
        if 'ECE' in results:
            print(f"  ECE %:   {results['ECE']}")
        if 'NLL' in results:
            print(f"  NLL:     {results['NLL']}")
    if 'ECE_nofree' in results or 'NLL_nofree' in results:
        print("\n--- Calibration (free class excluded) ---")
        if 'ECE_nofree' in results:
            print(f"  ECE %:   {results['ECE_nofree']}")
        if 'NLL_nofree' in results:
            print(f"  NLL:     {results['NLL_nofree']}")
    if args.verbose:
        print("\n--- All metrics (verbose) ---")
        for k, v in sorted(results.items()):
            print(f"  {k}: {v}")
    else:
        print("\n(Use --verbose to print all radius/height/class breakdown keys.)")
    print("=" * 60)
    return results


if __name__ == '__main__':
    main()
