"""E1: free-class bias (delta) grid sweep on exported raw logits, CPU-only.

Design per research_v2/reports/E1_execution_plan.md section 4. Takes an npz produced by
tools/export_occ_logits.py --keep-invisible (keys: logits [N,C], gt [N], mask [N]=mask_camera,
NOT a filter here -- see that flag's docstring) and, for each delta in --delta-grid, shifts the
free-class logit by delta, re-decodes via argmax (equivalent to softmax+argmax since softmax is
monotonic -- matches STCOcc's actual native decode at stcocc.py:751), and recomputes:
  - overall_miou       : standard 17-class (excludes free) mIoU over ALL voxels
  - invisible_miou     : same, restricted to voxels where mask_camera == False
  - invisible_free_iou : binary free-vs-occupied IoU, restricted to invisible voxels

This is diagnostic-only (mIoU-style, CPU). It does NOT recompute RayIoU -- RayIoU needs the
CUDA-only DVR renderer on a full 3D volume per sample, not a flat voxel list (see E1 plan sec 3).

Usage:
  python research_v2/tools/bias_frontier_sweep.py \
    --logits-file /NAS/work_dirs/research_v2_e1_logits/nomask_calibration.npz \
    --delta-grid -2 -1.5 -1 -0.5 0 0.5 1 1.5 2 \
    --free-class 17 \
    --out research_v2/diagnostics/E1_bias/nomask_bias_sweep.csv
"""
import argparse
import csv
import os
import numpy as np

NUM_SEM_CLASSES = 18  # 0-16 semantic, 17 = free


def compute_miou(pred, gt, num_classes, free_class, voxel_mask=None):
    if voxel_mask is not None:
        pred = pred[voxel_mask]
        gt = gt[voxel_mask]
    ious = []
    for c in range(num_classes):
        if c == free_class:
            continue
        pred_c = (pred == c)
        gt_c = (gt == c)
        tp = np.logical_and(pred_c, gt_c).sum()
        union = pred_c.sum() + gt_c.sum() - tp
        if union > 0:
            ious.append(tp / union)
    return float(np.mean(ious)) if ious else float('nan')


def compute_binary_free_iou(pred, gt, free_class, voxel_mask=None):
    if voxel_mask is not None:
        pred = pred[voxel_mask]
        gt = gt[voxel_mask]
    pred_free = (pred == free_class)
    gt_free = (gt == free_class)
    tp = np.logical_and(pred_free, gt_free).sum()
    union = pred_free.sum() + gt_free.sum() - tp
    return float(tp / union) if union > 0 else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--logits-file', required=True)
    ap.add_argument('--delta-grid', type=float, nargs='+',
                     default=[-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2])
    ap.add_argument('--free-class', type=int, default=17)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    data = np.load(args.logits_file)
    logits = data['logits']  # [N, C] float32
    gt = data['gt']          # [N] uint8
    mask_camera = data['mask'].astype(bool)  # [N] True=visible (see export_occ_logits --keep-invisible)
    invisible = ~mask_camera

    print(f'Loaded {logits.shape[0]:,} voxels, {logits.shape[1]} classes, '
          f'{invisible.sum():,} invisible ({100*invisible.mean():.1f}%)')

    # 메모리 효율화: delta마다 전체 (N,C) 배열을 복사하지 않고 free 채널 컬럼만 덮어쓴다.
    # 769M x 18 float32 배열의 풀카피는 델타 1개당 ~55GB를 추가로 필요로 해서, 동시에 돌아가는
    # 다른 GPU export(각 ~50GB 상주)와 겹치면 시스템 OOM/스와핑을 유발한다(실측으로 확인됨).
    orig_free_col = logits[:, args.free_class].copy()
    rows = []
    for delta in args.delta_grid:
        logits[:, args.free_class] = orig_free_col + delta
        pred = logits.argmax(axis=1).astype(np.uint8)

        row = {
            'delta': delta,
            'overall_miou': compute_miou(pred, gt, NUM_SEM_CLASSES, args.free_class),
            'invisible_miou': compute_miou(pred, gt, NUM_SEM_CLASSES, args.free_class, voxel_mask=invisible),
            'invisible_free_iou': compute_binary_free_iou(pred, gt, args.free_class, voxel_mask=invisible),
            'pred_free_frac_invisible': float(pred[invisible].__eq__(args.free_class).mean()) if invisible.any() else float('nan'),
            'gt_free_frac_invisible': float((gt[invisible] == args.free_class).mean()) if invisible.any() else float('nan'),
        }
        rows.append(row)
        print(f"delta={delta:+.2f}  overall_miou={row['overall_miou']:.4f}  "
              f"invisible_miou={row['invisible_miou']:.4f}  "
              f"invisible_free_iou={row['invisible_free_iou']:.4f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Saved: {args.out}')


if __name__ == '__main__':
    main()
