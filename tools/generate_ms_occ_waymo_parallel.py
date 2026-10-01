#!/usr/bin/env python3
"""Parallel multi-scale GT generator for Occ3D-Waymo (voxel04, 200x200x16).

Reuses the exact same downsample_label/downsample_mask functions from
tools/generate_ms_occ.py (no algorithm change, see that file's docstring and
research_openocc/implementation_changes.md for why: majority-vote per block
with a 95%-empty threshold). Waymo-specific differences from the OpenOcc
generator (tools/generate_ms_occ_parallel.py):
  - directory layout: <occ_root>/<split>/<scene:03d>/<frame:03d>_04.npz
    (not scene_token/sample_token/labels.npz)
  - free sentinel is 23 (raw, unremapped -- kept as-is in the generated
    multi-scale files too, consistent with the full-resolution source; the
    23->15 remap happens at load time, same as CVT-Occ's reference loader
    and our own STCOccLoadOccGTFromFileWaymo)
  - three mask fields (origin_voxel_state, final_voxel_state, infov) are all
    downsampled the same way the OpenOcc/Occ3D generator already downsamples
    mask_camera/mask_lidar (majority-vote binary mask)

Usage:
  python tools/generate_ms_occ_waymo_parallel.py --split training --workers 28
  python tools/generate_ms_occ_waymo_parallel.py --split validation --workers 28
"""
import argparse
import os
import sys
import time

import numpy as np
from mmengine.utils import track_iter_progress
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_ms_occ import downsample_label, downsample_mask  # noqa: E402

FREE_LABEL_RAW = 23


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--occ-root', default='data/waymo/occ3d/voxel04')
    p.add_argument('--split', required=True, choices=['training', 'validation'])
    p.add_argument('--workers', type=int, default=28)
    p.add_argument('--limit', type=int, default=None)
    return p.parse_args()


def _process_one(args):
    base = args  # path without '_04.npz' suffix, e.g. .../549/000
    label_path = base + '_04.npz'
    save_1_2 = base + '_04_1_2.npz'
    save_1_4 = base + '_04_1_4.npz'
    save_1_8 = base + '_04_1_8.npz'

    if all(os.path.exists(p) for p in (save_1_2, save_1_4, save_1_8)):
        return ('skipped', base, None)

    try:
        d = np.load(label_path)
        import torch
        labels = torch.from_numpy(d['voxel_label'])
        infov = d['infov'].astype(np.uint8)
        # origin_voxel_state/final_voxel_state (lidar/camera masks) are NOT
        # downsampled here -- the baseline-none profile only needs voxel_label
        # (remapped) + infov (FOV validity) at every scale (see
        # LoadOccGTFromFileWaymo). Skipping these two extra downsample_mask
        # calls per scale roughly halves generation time; add them back here
        # if a future oracle/diagnostic profile needs lidar/camera mask at
        # multi-scale resolution.

        for downscale, save_path in ((2, save_1_2), (4, save_1_4), (8, save_1_8)):
            labels_ds = downsample_label(labels, downscale=downscale, empty_cls_idx=FREE_LABEL_RAW)
            infov_ds = downsample_mask(infov, downscale=downscale)
            save_data = {
                'voxel_label': labels_ds,
                'infov': infov_ds.astype(bool),
            }
            tmp_path = save_path[:-len('.npz')] + '.tmp-{}.npz'.format(os.getpid())
            np.savez_compressed(tmp_path, **save_data)
            os.replace(tmp_path, save_path)
        return ('ok', base, None)
    except Exception as e:  # noqa: BLE001
        import traceback
        return ('error', base, '{}: {}'.format(e, traceback.format_exc(limit=2)))


def main():
    args = parse_args()
    split_dir = os.path.join(args.occ_root, args.split)
    scenes = sorted(d for d in os.listdir(split_dir) if d.isdigit())

    tasks = []
    for scene in scenes:
        scene_dir = os.path.join(split_dir, scene)
        frames = sorted(f[:-len('_04.npz')] for f in os.listdir(scene_dir)
                         if f.endswith('_04.npz') and not f.endswith(('_1_2.npz', '_1_4.npz', '_1_8.npz')))
        for fr in frames:
            tasks.append(os.path.join(scene_dir, fr))
    if args.limit:
        tasks = tasks[:args.limit]

    print('[generate_ms_occ_waymo_parallel] split={} scenes={} samples={} workers={}'.format(
        args.split, len(scenes), len(tasks), args.workers))

    t0 = time.time()
    n_ok = n_skip = n_err = 0
    errors = []
    with Pool(processes=args.workers) as pool:
        for status, base, err in track_iter_progress(
                (pool.imap_unordered(_process_one, tasks, chunksize=8), len(tasks))):
            if status == 'ok':
                n_ok += 1
            elif status == 'skipped':
                n_skip += 1
            else:
                n_err += 1
                errors.append((base, err))

    dt = time.time() - t0
    print('[generate_ms_occ_waymo_parallel] done in {:.1f}s: ok={} skipped={} error={}'.format(
        dt, n_ok, n_skip, n_err))
    if errors:
        print('[generate_ms_occ_waymo_parallel] first errors:')
        for base, err in errors[:10]:
            print(' -', base, ':', err)
    return n_err


if __name__ == '__main__':
    n_err = main()
    sys.exit(1 if n_err else 0)
