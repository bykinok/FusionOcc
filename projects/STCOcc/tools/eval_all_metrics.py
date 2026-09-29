#!/usr/bin/env python3
"""One-shot evaluation pipeline for a single STCOcc experiment/checkpoint.

Computes mIoU, RayIoU@1/2/4/mean, AUROC/FPR95 (msp + entropy), ECE and NLL with
ONE command, and appends a single row to a shared summary CSV.

Why three subprocess stages instead of one process
---------------------------------------------------
mIoU+AUROC+ECE+NLL and RayIoU come from two structurally different evaluator
passes in this codebase (different eval_metric / use_image_mask / config), so a
literal single-process run isn't available without changing the shared
OccupancyMetric class used by every other STCOcc config in this repo -- out of
scope here. Instead this script *orchestrates* the existing tools so a single
command produces every number:

  1. tools/test.py <miou_config> <checkpoint>
       --cfg-options model.compute_uncertainty=True
       --save-predictions <out>/predictions.pkl --save-predictions-only
     Model-side `compute_uncertainty=True` is required for the model to return
     uncertainty_msp/uncertainty_entropy/softmax_probs at all (see
     stcocc/detectors/stcocc.py forward_test) -- without it AUROC/ECE/NLL are
     silently empty (no error, just missing keys). `--save-predictions-only`
     writes predictions incrementally to disk instead of accumulating them in
     RAM; the naive alternative (running compute_uncertainty_metrics=True
     in-process, letting OccupancyMetric hold all 6019 samples' softmax_probs
     in memory -- ~46 MB/sample -- at once) OOM-killed (exit 137) when run
     alongside a second concurrent eval job on this machine.

  2. tools/compute_metrics_from_file.py --predictions <out>/predictions_rank0.pkl
       --config <miou_config> --passes 3 --verbose
     Streams the saved predictions from disk in bounded chunks (3 passes over
     the file -- miou / ece_nll / auroc_fpr95 -- to further cap peak RAM) and
     prints the mIoU + AUROC/FPR95 + ECE/NLL summary block this script parses.

  3. tools/test.py <rayiou_config> <checkpoint>
     Separate eval_metric='rayiou' pass (no compute_uncertainty needed).

Stages run strictly sequentially (never concurrently) to keep peak RAM bounded
even when two experiments' pipelines are run in parallel on the two GPUs.

Usage
-----
  python projects/STCOcc/tools/eval_all_metrics.py \\
    --name stage1_nomask \\
    --miou-config projects/STCOcc/configs/stcocc_r50_704x256_16f_occ3d_e12_stage1_nomask.py \\
    --rayiou-config projects/STCOcc/configs/stcocc_r50_704x256_16f_occ3d_e12_stage1_nomask_rayiou.py \\
    --checkpoint /NAS/work_dirs/stcocc_r50_704x256_16f_occ3d_e12_stage1_nomask/iter_21096.pth \\
    --out-dir /NAS/work_dirs/stcocc_e12_eval/stage1_nomask \\
    --gpu 0 \\
    --csv /NAS/work_dirs/stcocc_e12_eval/summary.csv

Run from the repository root (same convention as tools/test.py).
Resumable: pass --skip-inference to reuse an existing predictions_rank0.pkl in
--out-dir, or --skip-rayiou to only do stages 1-2.
"""
import argparse
import csv
import os
import re
import subprocess
import sys
import time


def run(cmd, log_path, env):
    print(f"\n$ {' '.join(cmd)}\n  (log: {log_path})", flush=True)
    t0 = time.time()
    with open(log_path, 'w') as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    dt = time.time() - t0
    print(f"  -> exit={proc.returncode}  ({dt/60:.1f} min)", flush=True)
    return proc.returncode


def read(path):
    with open(path, 'r', errors='replace') as f:
        return f.read()


def parse_miou_uncertainty(text):
    out = {}
    m = re.search(r'^\s*mIoU:\s+([\d.]+)%', text, re.M)
    if m:
        out['mIoU'] = float(m.group(1))
    m = re.search(r'uncertainty_msp\s*\|\s*([\-\d.]+)\s*\|\s*([\-\d.]+)', text)
    if m:
        out['AUROC_msp'] = float(m.group(1))
        out['FPR95_msp'] = float(m.group(2))
    m = re.search(r'uncertainty_entropy\s*\|\s*([\-\d.]+)\s*\|\s*([\-\d.]+)', text)
    if m:
        out['AUROC_entropy'] = float(m.group(1))
        out['FPR95_entropy'] = float(m.group(2))
    m = re.search(r'ECE %:\s+([\-\d.]+)', text)
    if m:
        out['ECE'] = float(m.group(1))
    m = re.search(r'NLL:\s+([\-\d.]+)', text)
    if m:
        out['NLL'] = float(m.group(1))
    return out


def parse_rayiou(text):
    out = {}
    m = re.search(r'\|\s*MEAN\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|', text)
    if m:
        out['RayIoU_1'] = float(m.group(1))
        out['RayIoU_2'] = float(m.group(2))
        out['RayIoU_4'] = float(m.group(3))
    m = re.search(r'^MIOU:\s*([\d.]+)', text, re.M)
    if m:
        out['RayIoU_mean'] = float(m.group(1))
    m = re.search(r'^Occ score:\s*([\d.]+)', text, re.M)
    if m:
        out['Occ_score'] = float(m.group(1))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--name', required=True, help='Experiment name (CSV row label), e.g. stage2_invfree_l050')
    ap.add_argument('--miou-config', required=True)
    ap.add_argument('--rayiou-config', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--out-dir', required=True, help='Where per-stage logs and predictions.pkl are written')
    ap.add_argument('--csv', required=True, help='Summary CSV to append one row to (created with header if missing)')
    ap.add_argument('--gpu', type=int, default=0, help='CUDA_VISIBLE_DEVICES for all 3 stages')
    ap.add_argument('--chunk-size', type=int, default=200, help='compute_metrics_from_file.py chunk size')
    ap.add_argument('--skip-inference', action='store_true', help='Reuse existing <out-dir>/predictions_rank0.pkl')
    ap.add_argument('--skip-rayiou', action='store_true', help='Only run stages 1-2 (mIoU/AUROC/ECE/NLL)')
    ap.add_argument('--lambda-value', default='', help='Optional lambda_inv_free value to record as a CSV column')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    env = dict(os.environ)
    env['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    preds_base = os.path.join(args.out_dir, 'predictions')
    preds_rank0 = preds_base + '_rank0.pkl'

    row = {'name': args.name, 'lambda_inv_free': args.lambda_value}

    # ---- Stage 1: inference, streamed to disk ----
    if args.skip_inference and os.path.exists(preds_rank0):
        print(f"[skip] reusing existing predictions: {preds_rank0}")
    else:
        rc = run([
            sys.executable, 'tools/test.py', args.miou_config, args.checkpoint,
            '--cfg-options', 'model.compute_uncertainty=True',
            '--save-predictions', preds_base,
            '--save-predictions-only',
            '--work-dir', os.path.join(args.out_dir, 'infer_work_dir'),
        ], os.path.join(args.out_dir, '01_infer.log'), env)
        if rc != 0:
            print(f"FAILED at inference stage (exit {rc}). See {args.out_dir}/01_infer.log")
            sys.exit(rc)
        if not os.path.exists(preds_rank0):
            print(f"FAILED: expected predictions file not found: {preds_rank0}")
            sys.exit(1)

    # ---- Stage 2: streamed metric computation from the saved predictions ----
    metrics_log = os.path.join(args.out_dir, '02_metrics.log')
    rc = run([
        sys.executable, 'tools/compute_metrics_from_file.py',
        '--predictions', preds_rank0,
        '--config', args.miou_config,
        '--chunk-size', str(args.chunk_size),
        '--passes', '3',
        '--verbose',
    ], metrics_log, env)
    if rc != 0:
        print(f"FAILED at metrics-from-file stage (exit {rc}). See {metrics_log}")
        sys.exit(rc)
    row.update(parse_miou_uncertainty(read(metrics_log)))

    # ---- Stage 3: RayIoU (separate evaluator pass) ----
    if not args.skip_rayiou:
        rayiou_log = os.path.join(args.out_dir, '03_rayiou.log')
        rc = run([
            sys.executable, 'tools/test.py', args.rayiou_config, args.checkpoint,
            '--work-dir', os.path.join(args.out_dir, 'rayiou_work_dir'),
        ], rayiou_log, env)
        if rc != 0:
            print(f"FAILED at RayIoU stage (exit {rc}). See {rayiou_log}")
            sys.exit(rc)
        row.update(parse_rayiou(read(rayiou_log)))

    # ---- Write / append CSV row ----
    fieldnames = ['name', 'lambda_inv_free', 'mIoU', 'RayIoU_1', 'RayIoU_2', 'RayIoU_4',
                  'RayIoU_mean', 'Occ_score', 'AUROC_msp', 'FPR95_msp',
                  'AUROC_entropy', 'FPR95_entropy', 'ECE', 'NLL']
    write_header = not os.path.exists(args.csv)
    os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
    with open(args.csv, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, '') for k in fieldnames})

    print("\n" + "=" * 60)
    print(f"DONE: {args.name}")
    for k in fieldnames:
        if k in row:
            print(f"  {k:15s}: {row[k]}")
    print(f"Appended to {args.csv}")
    print("=" * 60)


if __name__ == '__main__':
    main()
