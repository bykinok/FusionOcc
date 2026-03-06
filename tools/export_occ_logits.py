# Copyright (c) OpenMMLab. All rights reserved.
"""
Export occupancy logits and ground truth from a trained model for temperature scaling.

Runs the model on the test/val set with export_occ_logits=True and saves (logits, gt, mask)
to a single .npz file that tools/train_temperature.py can consume.

Usage (BEVFormer example):
  python tools/export_occ_logits.py config.py checkpoint.pth --output work_dirs/occ_logits_val.npz

For other occupancy models: ensure the detector supports export_occ_logits (set model.export_occ_logits = True)
and test_step returns data_sample with 'occ_logits', 'voxel_semantics', 'mask_camera'.
"""

import argparse
import os
import numpy as np
import torch
from tqdm import tqdm

from mmengine.config import Config, DictAction
from mmengine.registry import DefaultScope
from mmengine.runner import Runner

try:
    from mmdet3d.utils import replace_ceph_backend
except ImportError:
    replace_ceph_backend = None
import mmdet3d  # noqa: F401 - register mmdet3d modules


def parse_args():
    parser = argparse.ArgumentParser(description='Export occ logits and GT for temperature scaling')
    parser.add_argument('config', help='Test config file path')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--output', type=str, default='occ_logits_val.npz',
                       help='Output .npz path (default: occ_logits_val.npz)')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Max samples to export (default: all)')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction, help='Override config options')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    DefaultScope.get_instance('mmdet3d', scope_name='mmdet3d')
    cfg = Config.fromfile(args.config)

    if hasattr(cfg, 'custom_imports') and cfg.custom_imports:
        import importlib
        for name in cfg.custom_imports.get('imports', []):
            importlib.import_module(name)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    if replace_ceph_backend is not None and getattr(args, 'ceph', False):
        cfg = replace_ceph_backend(cfg)

    if cfg.get('work_dir', None) is None:
        cfg.work_dir = os.path.join('./work_dirs', os.path.splitext(os.path.basename(args.config))[0])
    cfg.load_from = args.checkpoint

    # DistributedSampler는 단일 프로세스에서 init_process_group 없이 동작하지 않으므로
    # DefaultSampler(shuffle=False)로 교체한다.
    _DIST_SAMPLER_TYPES = {'DistributedSampler', 'DistributedGroupSampler',
                           'InfiniteSampler'}

    def _replace_sampler(dl_cfg):
        if not isinstance(dl_cfg, dict):
            return
        sampler = dl_cfg.get('sampler', {})
        if isinstance(sampler, dict) and sampler.get('type') in _DIST_SAMPLER_TYPES:
            dl_cfg['sampler'] = dict(type='DefaultSampler', shuffle=False)

    for key in ('test_dataloader', 'val_dataloader'):
        if hasattr(cfg, key):
            _replace_sampler(getattr(cfg, key))

    runner = Runner.from_cfg(cfg)
    from mmengine.runner import load_checkpoint
    load_checkpoint(runner.model, args.checkpoint, map_location='cpu', strict=False)

    # Enable export of occ logits (set on actual model if DDP-wrapped).
    # BEVFormerOcc uses getattr(self, 'export_occ_logits', False) in simple_test_pts; attribute is not in __init__.
    model = getattr(runner.model, 'module', runner.model)
    # breakpoint()
    model.export_occ_logits = True

    runner.model.eval()
    test_loop = runner.build_test_loop(runner._test_loop)
    dataloader = test_loop.dataloader

    all_logits, all_gt, all_mask = [], [], []
    n_export = 0
    max_samples = args.max_samples or float('inf')

    for batch in tqdm(dataloader, desc='Export logits'):
        if n_export >= max_samples:
            break
        with torch.no_grad():
            out = runner.model.test_step(batch)
        if not isinstance(out, list):
            out = [out]
        # breakpoint()
        for sample in out:
            if n_export >= max_samples:
                break
            if 'occ_logits' not in sample:
                continue
            logits = np.asarray(sample['occ_logits'])
            gt = sample.get('voxel_semantics')
            mask = sample.get('mask_camera')
            if gt is None:
                continue
            gt = np.asarray(gt)
            if mask is None:
                mask = np.ones(gt.shape, dtype=bool)
            else:
                mask = np.asarray(mask).astype(bool)
            # breakpoint()
            # Flatten: (H,W,Z,C) -> (N,C), (H,W,Z) -> (N,)
            logits_flat = logits.reshape(-1, logits.shape[-1])
            gt_flat = gt.ravel()
            mask_flat = mask.ravel()
            n = min(logits_flat.shape[0], gt_flat.shape[0], mask_flat.shape[0])
            logits_flat = logits_flat[:n]
            gt_flat = gt_flat[:n]
            mask_flat = mask_flat[:n]
            all_logits.append(logits_flat[mask_flat])
            all_gt.append(gt_flat[mask_flat])
            all_mask.append(np.ones(mask_flat.sum(), dtype=bool))
            n_export += 1

    if not all_logits:
        raise RuntimeError('No samples with occ_logits found. Is export_occ_logits supported and enabled?')

    logits_all = np.concatenate(all_logits, axis=0)
    gt_all = np.concatenate(all_gt, axis=0)
    mask_all = np.concatenate(all_mask, axis=0)
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    np.savez_compressed(args.output, logits=logits_all, gt=gt_all, mask=mask_all)
    print(f'Saved {logits_all.shape[0]} voxels to {args.output} (num_classes={logits_all.shape[1]})')


if __name__ == '__main__':
    main()
