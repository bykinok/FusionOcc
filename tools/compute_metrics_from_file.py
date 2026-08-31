#!/usr/bin/env python3
"""
Compute occupancy metrics from a saved predictions file (no model run).
Use after: python tools/test.py CONFIG CHECKPOINT --save-predictions OUT.pkl

File format: one or more .pkl files, each with repeated pickle.dump(batch_list, f)
per batch (batch_list = list of pred_dicts with occ_results, index, uncertainty_msp, etc.).

Usage (single GPU):
  python tools/compute_metrics_from_file.py \\
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

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(description='Compute occupancy metrics from saved predictions file')
    parser.add_argument('--predictions', required=True, nargs='+',
                        help='Path(s) to .pkl file(s), e.g. work_dirs/predictions_rank0.pkl')
    parser.add_argument('--config', required=True, help='Config file (for ann_file, data_root, etc.)')
    parser.add_argument('--chunk-size', type=int, default=400,
                        help='Process this many predictions per chunk (default: 400)')
    parser.add_argument('--metric-group', choices=['miou', 'ece_nll', 'auroc_fpr95'], default=None,
                        help='Compute only this metric group (reduces memory). Default: all.')
    parser.add_argument('--passes', type=int, choices=[1, 3], default=1,
                        help='If 3: run file 3 times (miou, ece_nll, auroc_fpr95) and merge (saves memory for large pkl). Ignored if --metric-group is set.')
    parser.add_argument('--ann-file', default=None,
                        help='Override ann_file (default: from config test_evaluator)')
    parser.add_argument('--data-root', default=None,
                        help='Override data_root (default: from config)')
    parser.add_argument('--cfg-options', nargs='+', default=None,
                        help='Override config options, e.g. test_evaluator.ann_file=other.pkl')
    parser.add_argument('--verbose', action='store_true',
                        help='Print all metric keys (radius/height/class breakdown). Default: summary only.')
    parser.add_argument('--sort-by-timestamp', dest='sort_by_timestamp', action='store_true', default=True,
                        help='Sort ann_file data_infos by timestamp before matching pred index (default: True). '
                             'Only correct if the model\'s dataset ALSO iterates in timestamp order at prediction '
                             'time (true for SparseOcc/NuSceneOcc). Use --no-sort-by-timestamp for datasets '
                             '(e.g. GaussianFormer) whose ann_file is already stored in prediction-index order — '
                             're-sorting by timestamp there scrambles the pred<->GT correspondence entirely.')
    parser.add_argument('--no-sort-by-timestamp', dest='sort_by_timestamp', action='store_false',
                        help='Keep ann_file data_infos in their original (unsorted) order. See --sort-by-timestamp.')
    args = parser.parse_args()

    from mmengine.config import Config
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        # Simple key=value pairs
        opts = {}
        for x in args.cfg_options:
            if '=' in x:
                k, v = x.split('=', 1)
                opts[k] = v
        if opts:
            cfg.merge_from_dict(opts)

    # Resolve ann_file and data_root from config if not provided
    ann_file = args.ann_file
    data_root = args.data_root
    occ_gt_root = None  # GT 폴더 (token→scene 매핑에 사용, occ_path 보강용)
    if ann_file is None and hasattr(cfg, 'test_evaluator') and isinstance(cfg.test_evaluator, dict):
        ann_file = cfg.test_evaluator.get('ann_file')
    if ann_file is None and hasattr(cfg, 'val_evaluator') and isinstance(cfg.val_evaluator, dict):
        ann_file = cfg.val_evaluator.get('ann_file')
    if data_root is None and hasattr(cfg, 'test_evaluator') and isinstance(cfg.test_evaluator, dict):
        data_root = cfg.test_evaluator.get('data_root', '')
    if data_root is None and hasattr(cfg, 'val_evaluator') and isinstance(cfg.val_evaluator, dict):
        data_root = cfg.val_evaluator.get('data_root', '')
    # occ_gt_root: val_evaluator 또는 test_evaluator 또는 dataset config에서 추출
    for ev_key in ('val_evaluator', 'test_evaluator'):
        ev = getattr(cfg, ev_key, None)
        if isinstance(ev, dict) and ev.get('occ_gt_root'):
            occ_gt_root = ev['occ_gt_root']
            break
    if occ_gt_root is None:
        for dl_key in ('val_dataloader', 'test_dataloader'):
            dl = getattr(cfg, dl_key, None)
            if isinstance(dl, dict) and isinstance(dl.get('dataset'), dict):
                occ_gt_root = dl['dataset'].get('occ_gt_root')
                if occ_gt_root:
                    break
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
        raise SystemExit("Could not determine ann_file. Pass --ann-file or use a config with test_evaluator.ann_file.")
    # If ann_file is filename only, prepend data_root
    if data_root and not os.path.isabs(ann_file) and os.path.basename(ann_file) == ann_file:
        ann_file = os.path.join(data_root, ann_file)

    # Load only stcocc.evaluation (not full stcocc package) to avoid registry conflict:
    # stcocc/__init__.py imports view_transformation etc. and registers BEVFormerEncoder,
    # which conflicts with BEVFormer project already loaded by config.
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    stcocc_stcocc = os.path.join(root, 'projects', 'STCOcc', 'stcocc')
    if not os.path.isdir(stcocc_stcocc):
        raise SystemExit(f"STCOcc stcocc dir not found at {stcocc_stcocc}")
    if stcocc_stcocc not in sys.path:
        sys.path.insert(0, stcocc_stcocc)

    # Config load may have registered BEVFormer's OccupancyMetric; pop it so STCOcc's can register
    from mmengine.registry import METRICS as ENGINE_METRICS
    from mmdet3d.registry import METRICS as DET3D_METRICS
    ENGINE_METRICS._module_dict.pop('OccupancyMetric', None)
    DET3D_METRICS._module_dict.pop('OccupancyMetric', None)

    from evaluation.occupancy_metric import OccupancyMetric

    num_classes = 18
    if hasattr(cfg, 'test_evaluator') and isinstance(cfg.test_evaluator, dict):
        num_classes = cfg.test_evaluator.get('num_classes', 18)

    metric = OccupancyMetric(
        num_classes=num_classes,
        use_lidar_mask=False,
        use_image_mask=True,
        ann_file=ann_file,
        data_root=data_root or '',
        dataset_name='occ3d',
        eval_metric='miou',
        sort_by_timestamp=args.sort_by_timestamp,
        point_cloud_range=[-40.0, -40.0, -1.0, 40.0, 40.0, 5.4],
        compute_uncertainty_metrics=True,  # 파일 기반 스크립트에서는 uncertainty 메트릭 계산
    )

    # data_infos에 occ_path가 없을 경우 occ_gt_root를 이용해 보강
    # (SparseOcc_eccv 등 일부 프로젝트는 ann_file에 occ_path 필드가 없고,
    #  occ_gt_root + token→scene 매핑으로 런타임에 구축해야 함)
    if occ_gt_root and hasattr(metric, 'data_infos') and metric.data_infos:
        import glob
        need_enrich = any(
            not info.get('occ3d_gt_path') and not info.get('occ_path') and not info.get('occ_gt_path')
            for info in metric.data_infos[:10]
        )
        if need_enrich:
            token2scene = {}
            for npz_path in glob.glob(os.path.join(occ_gt_root, '*/*/*.npz')):
                parts = npz_path.replace('\\', '/').split('/')
                if len(parts) >= 3:
                    token2scene[parts[-2]] = parts[-3]
            enriched = 0
            for info in metric.data_infos:
                if not isinstance(info, dict):
                    continue
                if info.get('occ3d_gt_path') or info.get('occ_path') or info.get('occ_gt_path'):
                    continue
                token = info.get('token', info.get('sample_idx', ''))
                scene = token2scene.get(str(token), '')
                if scene:
                    info['occ_path'] = os.path.join(occ_gt_root, scene, str(token))
                    enriched += 1
            print(f"[occ_path 보강] {enriched}/{len(metric.data_infos)} data_infos에 occ_path 추가 "
                  f"(occ_gt_root={occ_gt_root})")
            if enriched == 0:
                print("  Warning: occ_path를 추가하지 못했습니다. "
                      "occ_gt_root 경로와 ann_file의 token 필드를 확인하세요.")

    if not hasattr(metric, 'compute_metrics_from_file'):
        raise SystemExit("OccupancyMetric does not have compute_metrics_from_file. Update STCOcc metric.")

    if args.metric_group:
        metric_groups = [args.metric_group]
        print(f"Reading predictions from {args.predictions}, chunk_size={args.chunk_size}, metric_group={args.metric_group}")
        results = metric.compute_metrics_from_file(args.predictions, chunk_size=args.chunk_size, metric_groups=metric_groups)
    elif args.passes == 3:
        print(f"Running 3 passes (miou, ece_nll, auroc_fpr95) to reduce memory. Reading from {args.predictions}, chunk_size={args.chunk_size}")
        results = {}
        for group in ['miou', 'ece_nll', 'auroc_fpr95']:
            print(f"\n--- Pass: {group} ---")
            out = metric.compute_metrics_from_file(args.predictions, chunk_size=args.chunk_size, metric_groups=[group])
            for k, v in out.items():
                if k == 'count' and v == 0 and results.get('count', 0) != 0:
                    continue  # keep count from miou pass
                results[k] = v
    else:
        print(f"Reading predictions from {args.predictions}, chunk_size={args.chunk_size}")
        results = metric.compute_metrics_from_file(args.predictions, chunk_size=args.chunk_size)

    # 요약만 출력 (mIoU 평가처럼 정리). 상세는 위 metric 내부에서 이미 출력됨.
    print("=" * 60)
    print("=== Metrics Summary ===")
    print("=" * 60)
    print(f"  mIoU:    {results.get('mIoU', 0):.2f}%")
    print(f"  count:   {results.get('count', 0)}")
    if 'AUROC_uncertainty_msp' in results or 'AUROC_uncertainty_entropy' in results:
        print("\n--- Uncertainty (global) ---")
        print(f"  {'Measure':>22s} | {'AUROC %':>8s} | {'FPR95 %':>8s}")
        print("  " + "-" * 42)
        if 'AUROC_uncertainty_msp' in results:
            print(f"  {'uncertainty_msp':>22s} | {results['AUROC_uncertainty_msp']:>7.2f} | {results.get('FPR95_uncertainty_msp', 0):>7.2f}")
        if 'AUROC_uncertainty_entropy' in results:
            print(f"  {'uncertainty_entropy':>22s} | {results['AUROC_uncertainty_entropy']:>7.2f} | {results.get('FPR95_uncertainty_entropy', 0):>7.2f}")
    if 'ECE' in results or 'NLL' in results:
        print("\n--- Calibration ---")
        if 'ECE' in results:
            print(f"  ECE %:   {results['ECE']}")
        if 'NLL' in results:
            print(f"  NLL:     {results['NLL']}")
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
