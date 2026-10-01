#!/usr/bin/env python3
"""Standalone small-set mIoU smoke evaluation for the Occ3D-Waymo baseline.

Deliberately bypasses OccupancyMetric (see the NOTE in
stcocc_r50_704x256_16f_occ3d_waymo_baseline.py and
research_waymo/remaining_issues.md): OccupancyMetric's internal GT-reloading
code hardcodes Occ3D/OpenOcc's file convention (always 'labels.npz', always a
'semantics' key) in 3 places shared with Occ3D-nuScenes/OpenOcc evaluation,
and making it Waymo-aware needs real new branches there (regression risk),
not a drop-in config change. This script instead:
  1. builds the model from the config and loads a checkpoint,
  2. runs the test pipeline (images/points only, no GT) through the model,
  3. separately loads GT via the SAME STCOccLoadOccGTFromFileWaymo used by
     training (so GT identity is guaranteed consistent, same principle as
     the shared gt_resolver from the OpenOcc work),
  4. accumulates a confusion matrix via Metric_mIoU directly.

Usage:
  python tools/eval_waymo_smoke.py <config> <checkpoint> --ann-file <val pkl> [--limit N]
"""
import argparse
import sys

import numpy as np
import torch
from mmengine.config import Config
from mmengine.runner import Runner

sys.path.insert(0, '.')
from mmdet3d.datasets.occ_metrics import Metric_mIoU  # noqa: E402
from projects.STCOcc.stcocc.transforms.pipelines.loading import LoadOccGTFromFileWaymo  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    ap.add_argument('checkpoint')
    ap.add_argument('--ann-file', required=True)
    ap.add_argument('--limit', type=int, default=None)
    args = ap.parse_args()

    cfg = Config.fromfile(args.config)
    cfg.test_dataloader.dataset.ann_file = args.ann_file
    cfg.val_dataloader.dataset.ann_file = args.ann_file
    cfg.load_from = args.checkpoint
    cfg.work_dir = './work_dirs/waymo_eval_smoke_tmp'

    runner = Runner.from_cfg(cfg)
    runner.load_checkpoint(args.checkpoint)
    model = runner.model
    model.eval()

    gt_loader = LoadOccGTFromFileWaymo(num_classes=cfg.num_classes, apply_fov_ignore=True)
    metric = Metric_mIoU(num_classes=cfg.num_classes)

    loader = runner.test_dataloader
    n_done = 0
    with torch.no_grad():
        for data_batch in loader:
            if args.limit is not None and n_done >= args.limit:
                break
            data = model.data_preprocessor(data_batch, False) if hasattr(model, 'data_preprocessor') else data_batch
            outputs = model.test_step(data_batch)
            for sample in outputs:
                if 'occ_results' not in sample:
                    print('WARNING: no occ_results in model output, keys=', list(sample.keys()))
                    continue
                pred = np.asarray(sample['occ_results'])
                idx = sample['index']
                if isinstance(idx, (list, tuple)):
                    idx = idx[0]
                if hasattr(idx, 'item'):
                    idx = idx.item()
                idx = int(idx)
                infos = loader.dataset.data_infos if hasattr(loader.dataset, 'data_infos') and loader.dataset.data_infos else loader.dataset.data_list
                info = infos[idx]
                gt_results = gt_loader({'occ_path': info['occ_path']})
                gt = gt_results['voxel_semantics']
                metric.add_batch(pred.reshape(-1), gt.reshape(-1), None, None)
                n_done += 1
                print(f'[eval_waymo_smoke] sample {n_done} (token={info.get("token")}) done')

    print(f'[eval_waymo_smoke] evaluated {n_done} samples')
    metric.count_miou()


if __name__ == '__main__':
    main()
