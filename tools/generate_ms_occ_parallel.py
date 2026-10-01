#!/usr/bin/env python3
"""Parallel driver for tools/generate_ms_occ.py's per-sample downsampling.

generate_ms_occ.py's downsample_label()/downsample_mask() are pure-Python
triple-nested loops (~0.55s/sample single-threaded -> ~5.2h for the full
28130+6019 OpenOcc train+val set). This script reuses those exact same
functions unmodified (no behavior change, just parallelism across samples
with multiprocessing.Pool), and is resumable: a sample is skipped if its
three output files already exist.

Usage:
  python tools/generate_ms_occ_parallel.py --dataset openocc \
      --pkl_path data/nuscenes/stcocc-nuscenes_infos_train.pkl \
      --workers 28
"""
import argparse
import os
import sys
import time
import traceback

import numpy as np
import torch
from mmengine import fileio
from mmengine.utils import track_iter_progress
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_ms_occ import downsample_label, downsample_mask  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from projects.STCOcc.stcocc.utils.gt_resolver import resolve_occ_gt_dir  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description='Generate multi-scale occ (parallel driver)')
    parser.add_argument('--dataset', type=str, required=True, help='occ3d or openocc')
    parser.add_argument('--pkl_path', type=str, required=True, help='path to the pkl file')
    parser.add_argument('--workers', type=int, default=28)
    parser.add_argument('--limit', type=int, default=None, help='only process first N samples (debug)')
    return parser.parse_args()


def _process_one(args):
    occ_path, empty_cls_idx = args
    label_path = os.path.join(occ_path, 'labels.npz')
    save_path_1_2 = os.path.join(occ_path, 'labels_1_2.npz')
    save_path_1_4 = os.path.join(occ_path, 'labels_1_4.npz')
    save_path_1_8 = os.path.join(occ_path, 'labels_1_8.npz')

    if all(os.path.exists(p) for p in (save_path_1_2, save_path_1_4, save_path_1_8)):
        return ('skipped', occ_path, None)

    try:
        label_data = np.load(label_path)
        labels = torch.from_numpy(label_data['semantics'])
        camera_mask = label_data['mask_camera'] if 'mask_camera' in label_data else None
        lidar_mask = label_data['mask_lidar'] if 'mask_lidar' in label_data else None

        for downscale, save_path in ((2, save_path_1_2), (4, save_path_1_4), (8, save_path_1_8)):
            labels_ds = downsample_label(labels, downscale=downscale, empty_cls_idx=empty_cls_idx)
            # NOTE: upstream (Ref/STCOcc_ori/tools/generate_ms_occ.py) writes
            # flow=<downsampled label> as a placeholder, not a real downsampled flow
            # vector field. This is faithfully reproduced here (not fixed) because:
            # (1) it is upstream's own behavior, not a local regression, and
            # (2) it is never consumed downstream for occupancy-only profiles --
            # LoadOccGTFromFileOpenOcc(load_flow=False) never reads this key.
            # See research_openocc/implementation_changes.md.
            save_data = {'semantics': labels_ds, 'flow': labels_ds}
            if camera_mask is not None:
                save_data['mask_camera'] = downsample_mask(camera_mask, downscale=downscale)
            if lidar_mask is not None:
                save_data['mask_lidar'] = downsample_mask(lidar_mask, downscale=downscale)
            # Write to a temp file then rename, so a killed/crashed worker never
            # leaves a half-written labels_1_X.npz behind that a later resumed
            # run would mistake for "done".
            # np.savez_compressed silently appends '.npz' if the filename doesn't
            # already end with it, so the temp name must keep that suffix too.
            tmp_path = save_path[:-len('.npz')] + '.tmp-{}.npz'.format(os.getpid())
            np.savez_compressed(tmp_path, **save_data)
            os.replace(tmp_path, save_path)
        return ('ok', occ_path, None)
    except Exception as e:  # noqa: BLE001
        return ('error', occ_path, '{}: {}'.format(e, traceback.format_exc(limit=2)))


def main():
    args = parse_args()
    pkl = fileio.load(args.pkl_path)
    infos = pkl['infos']
    if args.limit:
        infos = infos[:args.limit]

    empty_cls_idx = 16 if args.dataset == 'openocc' else 17
    tasks = []
    for info in infos:
        occ_path = resolve_occ_gt_dir(info['occ_path'], args.dataset)
        tasks.append((occ_path, empty_cls_idx))

    print('[generate_ms_occ_parallel] dataset={} pkl={} samples={} workers={}'.format(
        args.dataset, args.pkl_path, len(tasks), args.workers))

    t0 = time.time()
    n_ok = n_skip = n_err = 0
    errors = []
    with Pool(processes=args.workers) as pool:
        for status, occ_path, err in track_iter_progress(
                (pool.imap_unordered(_process_one, tasks, chunksize=8), len(tasks))):
            if status == 'ok':
                n_ok += 1
            elif status == 'skipped':
                n_skip += 1
            else:
                n_err += 1
                errors.append((occ_path, err))

    dt = time.time() - t0
    print('[generate_ms_occ_parallel] done in {:.1f}s: ok={} skipped={} error={}'.format(
        dt, n_ok, n_skip, n_err))
    if errors:
        print('[generate_ms_occ_parallel] first errors:')
        for occ_path, err in errors[:10]:
            print(' -', occ_path, ':', err)
    return n_err


if __name__ == '__main__':
    n_err = main()
    sys.exit(1 if n_err else 0)
