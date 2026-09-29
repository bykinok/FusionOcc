"""E2 regression test: ray_metrics_occ3d.main() with lidar_origin_time_offset_list=None
(all existing callers) must produce byte-identical miou/mave/occ_score to when it's provided
(research_v2 E2 opt-in) -- the new argument may only ADD a printed origin-decomposition table,
never change the numeric result. Needs 1 GPU (DVR renderer is CUDA-only) but NOT a trained
checkpoint -- uses synthetic occupancy grids and fabricated origins.

Run: CUDA_VISIBLE_DEVICES=<idle_gpu> python research_v2/tests/test_rayiou_decomposition_regression.py
"""
import io
import contextlib
import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from projects.STCOcc.stcocc.datasets.ray_metrics_occ3d import main as ray_based_miou_occ3d

N_CLS = 18
FREE = N_CLS - 1


def make_synthetic_grid(seed):
    rng = np.random.RandomState(seed)
    grid = rng.randint(0, N_CLS, size=(200, 200, 16)).astype(np.uint8)
    # bias towards free so DVR rendering has a mix of hits/misses
    free_mask = rng.rand(200, 200, 16) < 0.6
    grid[free_mask] = FREE
    return grid


def make_synthetic_origins(rng, T=4):
    origins = rng.uniform(-30, 30, size=(T, 3)).astype(np.float32)
    origins[:, 2] = rng.uniform(-1, 2, size=T).astype(np.float32)
    origins[0] = [0.0, 0.0, 0.0]  # first origin = reference frame itself
    time_offsets = np.zeros(T, dtype=np.float32)
    time_offsets[0] = 0.0
    if T > 1:
        time_offsets[1] = -2.5   # past
    if T > 2:
        time_offsets[2] = 3.0    # future
    if T > 3:
        time_offsets[3] = -0.5   # past
    return torch.from_numpy(origins), torch.from_numpy(time_offsets)


def run_once(decompose):
    rng = np.random.RandomState(0)
    n_samples = 2
    sem_gt_list = [make_synthetic_grid(seed=i) for i in range(n_samples)]
    # imperfect predictions: flip ~10% of voxels so TP/FP/FN are all non-trivial
    sem_pred_list = []
    for gt in sem_gt_list:
        pred = gt.copy()
        flip = rng.rand(*pred.shape) < 0.1
        pred[flip] = rng.randint(0, N_CLS, size=int(flip.sum())).astype(np.uint8)
        sem_pred_list.append(pred)
    flow_gt_list = [np.zeros((200, 200, 16, 2), dtype=np.float16) for _ in range(n_samples)]
    flow_pred_list = [np.zeros((200, 200, 16, 2), dtype=np.float16) for _ in range(n_samples)]

    lidar_origin_list = []
    origin_time_offset_list = []
    for i in range(n_samples):
        origins, time_offsets = make_synthetic_origins(np.random.RandomState(100 + i))
        lidar_origin_list.append(origins.unsqueeze(0))
        origin_time_offset_list.append(time_offsets)

    kwargs = {}
    if decompose:
        kwargs['lidar_origin_time_offset_list'] = origin_time_offset_list

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        miou, mave, occ_score = ray_based_miou_occ3d(
            sem_pred_list, sem_gt_list, flow_pred_list, flow_gt_list, lidar_origin_list,
            logger=None, **kwargs)
    return miou, mave, occ_score, buf.getvalue()


def main():
    miou_a, mave_a, occ_a, stdout_a = run_once(decompose=False)
    miou_b, mave_b, occ_b, stdout_b = run_once(decompose=True)

    print(f'decompose=False: miou={miou_a:.6f} mave={mave_a} occ_score={occ_a:.6f}')
    print(f'decompose=True : miou={miou_b:.6f} mave={mave_b} occ_score={occ_b:.6f}')

    def nan_safe_eq(a, b):
        if np.isnan(a) and np.isnan(b):
            return True
        return a == b

    ok = True
    if not nan_safe_eq(miou_a, miou_b):
        print(f'FAIL: miou differs {miou_a} vs {miou_b}')
        ok = False
    if not nan_safe_eq(mave_a, mave_b):
        print(f'FAIL: mave differs {mave_a} vs {mave_b}')
        ok = False
    if not nan_safe_eq(occ_a, occ_b):
        print(f'FAIL: occ_score differs {occ_a} vs {occ_b}')
        ok = False

    if 'RayIoU by Origin Time Offset' in stdout_a:
        print('FAIL: decompose=False printed the origin table (should not)')
        ok = False
    if 'RayIoU by Origin Time Offset' not in stdout_b:
        print('FAIL: decompose=True did NOT print the origin table (should)')
        ok = False

    print('PASS' if ok else 'FAIL')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
