# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pickle
import os
import sys
from typing import List, Dict, Any, Optional, Sequence, Tuple
from mmengine.evaluator import BaseMetric
from mmdet3d.registry import METRICS

# For AUROC / FPR95 (uncertainty metrics)
try:
    from sklearn.metrics import roc_curve, auc
except ImportError:
    roc_curve = None
    auc = None

# For ECE / NLL (calibration)
try:
    from .occupancy_metric_utils import (
        compute_ece,
        compute_nll,
        nll_neglog_sum_count,
        ece_bin_stats_update,
        ece_from_bin_stats,
    )
except ImportError:
    # Loaded as standalone (e.g. by OccupancyMetricHybrid via importlib); load utils from same dir
    import importlib.util
    _utils_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'occupancy_metric_utils.py')
    if not os.path.isfile(_utils_path):
        raise ImportError(f"occupancy_metric_utils not found at {_utils_path}")
    _spec = importlib.util.spec_from_file_location('occupancy_metric_utils', _utils_path)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    compute_ece = _mod.compute_ece
    compute_nll = _mod.compute_nll
    nll_neglog_sum_count = _mod.nll_neglog_sum_count
    ece_bin_stats_update = _mod.ece_bin_stats_update
    ece_from_bin_stats = _mod.ece_from_bin_stats

# Import original metric functions
try:
    from mmdet3d.datasets.occ_metrics import Metric_mIoU, Metric_FScore
except ImportError:
    # Fallback if original metrics not available
    class Metric_mIoU:
        def __init__(self, **kwargs):
            pass
        def add_batch(self, *args, **kwargs):
            pass
        def count_miou(self):
            return ([], 0.0, 0, {}, "")
    
    class Metric_FScore:
        def __init__(self, **kwargs):
            pass
        def add_batch(self, *args, **kwargs):
            pass
        def count_fscore(self):
            return ([], 0.0, 0, {}, "")


def compute_auroc_fpr95(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float]:
    """Compute AUROC and FPR95 from binary labels and scores.

    Reusable for any model that provides uncertainty scores.
    Positive class (y_true=1) should be the "uncertain" / OOD / incorrect case;
    higher y_score should indicate more uncertainty.

    Args:
        y_true: Binary labels (0 or 1).
        y_score: Scores (e.g. uncertainty); higher = more likely positive.

    Returns:
        (auroc, fpr95): AUROC in [0,1], FPR95 in [0,1] (FPR when TPR >= 0.95).
    """
    if roc_curve is None or auc is None:
        return float('nan'), float('nan')
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return float('nan'), float('nan')
    fpr, tpr, thresholds = roc_curve(y_true=y_true, y_score=y_score)
    roc_auc = auc(fpr, tpr)
    fpr95 = 0.0
    for tpr_val, fpr_val in zip(tpr, fpr):
        if tpr_val >= 0.95:
            fpr95 = fpr_val
            break
    return float(roc_auc), float(fpr95)


# Radius/height bins (match mmdet3d/datasets/occ_metrics.py)
RADIUS_BINS = [0, 20, 25, 30, 35, 40, 45, 50]
HEIGHT_BINS_RELATIVE = [0, 2, 4, 6]  # display labels: 0-2m, 2-4m, 4-6m


def _get_radius_height_grids(shape, point_cloud_range):
    """Build radius and height (z) arrays per voxel. shape=(H,W,Z), point_cloud_range=(x_min,y_min,z_min,x_max,y_max,z_max)."""
    H, W, Z = int(shape[0]), int(shape[1]), int(shape[2])
    x_min, y_min, z_min = point_cloud_range[0], point_cloud_range[1], point_cloud_range[2]
    x_max, y_max, z_max = point_cloud_range[3], point_cloud_range[4], point_cloud_range[5]
    voxel_x = (x_max - x_min) / max(H, 1)
    voxel_y = (y_max - y_min) / max(W, 1)
    voxel_z = (z_max - z_min) / max(Z, 1)
    x_centers = np.arange(H, dtype=np.float64) * voxel_x + x_min + voxel_x / 2
    y_centers = np.arange(W, dtype=np.float64) * voxel_y + y_min + voxel_y / 2
    z_centers = np.arange(Z, dtype=np.float64) * voxel_z + z_min + voxel_z / 2
    x_grid = np.broadcast_to(x_centers[:, None, None], (H, W, Z)).copy()
    y_grid = np.broadcast_to(y_centers[None, :, None], (H, W, Z)).copy()
    z_grid = np.broadcast_to(z_centers[None, None, :], (H, W, Z)).copy()
    radius = np.sqrt(x_grid ** 2 + y_grid ** 2)
    return radius, z_grid


def _get_height_bins_actual(point_cloud_range):
    """Actual z bounds for height bins (same as occ_metrics)."""
    z_min = point_cloud_range[2]
    return [z_min + h for h in HEIGHT_BINS_RELATIVE]


def _print_auroc_fpr95_summary(metrics: Dict[str, float], class_names: Sequence, num_classes: int) -> None:
    """Print AUROC/FPR95 summary in the same style as distance/height mIoU in occ_metrics.count_miou."""
    has_global = 'AUROC_uncertainty_msp' in metrics or 'AUROC_uncertainty_entropy' in metrics
    has_per_class_msp = any(k.startswith('uncertainty_msp_AUROC_') for k in metrics)
    has_per_class_ent = any(k.startswith('uncertainty_entropy_AUROC_') for k in metrics)
    if not (has_global or has_per_class_msp or has_per_class_ent):
        return

    print('\n===> AUROC / FPR95 Summary (uncertainty vs correct/incorrect)')
    print('     (higher AUROC = uncertainty better separates wrong from correct; lower FPR95 = better)')

    if has_global:
        print('\n===> Global (all voxels):')
        print(f'{"Measure":>22s} | {"AUROC %":>8s} | {"FPR95 %":>8s}')
        print('-' * 44)
        if 'AUROC_uncertainty_msp' in metrics:
            print(f'{"uncertainty_msp":>22s} | {metrics["AUROC_uncertainty_msp"]:>7.2f} | {metrics.get("FPR95_uncertainty_msp", 0):>7.2f}')
        if 'AUROC_uncertainty_entropy' in metrics:
            print(f'{"uncertainty_entropy":>22s} | {metrics["AUROC_uncertainty_entropy"]:>7.2f} | {metrics.get("FPR95_uncertainty_entropy", 0):>7.2f}')

    if has_per_class_msp:
        print('\n===> Per-class AUROC/FPR95 (uncertainty_msp):')
        print(f'{"Class":>20s} | {"AUROC %":>8s} | {"FPR95 %":>8s}')
        print('-' * 42)
        for c in range(num_classes):
            name = class_names[c] if c < len(class_names) else f"class_{c}"
            key_auroc = f'uncertainty_msp_AUROC_{name}'
            key_fpr95 = f'uncertainty_msp_FPR95_{name}'
            if key_auroc in metrics:
                print(f'{name:>20s} | {metrics[key_auroc]:>7.2f} | {metrics.get(key_fpr95, 0):>7.2f}')
        if 'mAUROC_uncertainty_msp' in metrics:
            print('-' * 42)
            print(f'{"mean (mAUROC/mFPR95)":>20s} | {metrics["mAUROC_uncertainty_msp"]:>7.2f} | {metrics.get("mFPR95_uncertainty_msp", 0):>7.2f}')

    if has_per_class_ent:
        print('\n===> Per-class AUROC/FPR95 (uncertainty_entropy):')
        print(f'{"Class":>20s} | {"AUROC %":>8s} | {"FPR95 %":>8s}')
        print('-' * 42)
        for c in range(num_classes):
            name = class_names[c] if c < len(class_names) else f"class_{c}"
            key_auroc = f'uncertainty_entropy_AUROC_{name}'
            key_fpr95 = f'uncertainty_entropy_FPR95_{name}'
            if key_auroc in metrics:
                print(f'{name:>20s} | {metrics[key_auroc]:>7.2f} | {metrics.get(key_fpr95, 0):>7.2f}')
        if 'mAUROC_uncertainty_entropy' in metrics:
            print('-' * 42)
            print(f'{"mean (mAUROC/mFPR95)":>20s} | {metrics["mAUROC_uncertainty_entropy"]:>7.2f} | {metrics.get("mFPR95_uncertainty_entropy", 0):>7.2f}')
    print('')


def _print_ece_nll_summary(metrics: Dict[str, float], class_names: Sequence, num_classes: int) -> None:
    """Print ECE and NLL summary (overall + per-class)."""
    has_ece = 'ECE' in metrics
    has_nll = 'NLL' in metrics
    has_per_class_ece = any(k.startswith('ECE_') and k != 'ECE' for k in metrics)
    has_per_class_nll = any(k.startswith('NLL_') and k != 'NLL' for k in metrics)
    if not (has_ece or has_nll or has_per_class_ece or has_per_class_nll):
        return

    print('\n===> ECE / NLL Summary (calibration & likelihood)')
    print('     (ECE: lower = better calibrated; NLL: lower = better)')

    if has_ece or has_nll:
        print('\n===> Overall:')
        print(f'{"Metric":>12s} | {"Value":>10s}')
        print('-' * 26)
        if has_ece:
            print(f'{"ECE %":>12s} | {metrics["ECE"]:>9.2f}')
        if has_nll:
            print(f'{"NLL":>12s} | {metrics["NLL"]:>10.4f}')

    if has_per_class_ece:
        print('\n===> Per-class ECE (%):')
        print(f'{"Class":>20s} | {"ECE %":>8s}')
        print('-' * 32)
        for c in range(num_classes):
            name = class_names[c] if c < len(class_names) else f"class_{c}"
            key = f'ECE_{name}'
            if key in metrics:
                print(f'{name:>20s} | {metrics[key]:>7.2f}')
        if 'mECE' in metrics:
            print('-' * 32)
            print(f'{"mean (mECE)":>20s} | {metrics["mECE"]:>7.2f}')

    if has_per_class_nll:
        print('\n===> Per-class NLL:')
        print(f'{"Class":>20s} | {"NLL":>10s}')
        print('-' * 34)
        for c in range(num_classes):
            name = class_names[c] if c < len(class_names) else f"class_{c}"
            key = f'NLL_{name}'
            if key in metrics:
                print(f'{name:>20s} | {metrics[key]:>10.4f}')
        if 'mNLL' in metrics:
            print('-' * 34)
            print(f'{"mean (mNLL)":>20s} | {metrics["mNLL"]:>10.4f}')
    print('')


def _print_radius_height_uncertainty_summary(
    metrics: Dict[str, float],
    class_names: Optional[Sequence[str]] = None,
    num_classes: int = 0,
) -> None:
    """Print AUROC/FPR95/ECE/NLL by radius range and height; per-bin overall + per-class rows."""
    # Collect bin keys from any radius/height metric (AUROC_msp, ECE, or NLL) so ece_nll-only mode still prints.
    def _bin_sort_key(x: str) -> float:
        try:
            return float(x.split('-')[0])
        except (ValueError, IndexError):
            return 0.0

    radius_keys = set()
    for k in metrics:
        if not k.startswith('radius_'):
            continue
        parts = k.split('_', 2)
        if len(parts) != 3:
            continue
        bin_part = parts[1]
        if '-' in bin_part and bin_part.endswith('m'):
            try:
                float(bin_part.split('-')[0])
                float(bin_part.split('-')[1].replace('m', ''))
            except (ValueError, IndexError):
                continue
            radius_keys.add(bin_part)
    radius_keys = sorted(radius_keys, key=_bin_sort_key)

    height_keys = set()
    for k in metrics:
        if not k.startswith('height_'):
            continue
        parts = k.split('_', 2)
        if len(parts) != 3:
            continue
        bin_part = parts[1]
        if '-' in bin_part and bin_part.endswith('m'):
            try:
                float(bin_part.split('-')[0])
                float(bin_part.split('-')[1].replace('m', ''))
            except (ValueError, IndexError):
                continue
            height_keys.add(bin_part)
    height_keys = sorted(height_keys, key=_bin_sort_key)

    if not radius_keys and not height_keys:
        return
    class_names = class_names or []
    num_classes = num_classes or len(class_names)

    def _row(prefix: str, label: str, rkey: str, m: Dict[str, float], is_radius: bool) -> None:
        pre = 'radius_' if is_radius else 'height_'
        if label:
            auroc_m = m.get(f'{pre}{rkey}_{label}_AUROC_msp', np.nan)
            fpr95_m = m.get(f'{pre}{rkey}_{label}_FPR95_msp', np.nan)
            auroc_e = m.get(f'{pre}{rkey}_{label}_AUROC_entropy', np.nan)
            fpr95_e = m.get(f'{pre}{rkey}_{label}_FPR95_entropy', np.nan)
            ece = m.get(f'{pre}{rkey}_{label}_ECE', np.nan)
            nll = m.get(f'{pre}{rkey}_{label}_NLL', np.nan)
        else:
            auroc_m = m.get(f'{pre}{rkey}_AUROC_msp', np.nan)
            fpr95_m = m.get(f'{pre}{rkey}_FPR95_msp', np.nan)
            auroc_e = m.get(f'{pre}{rkey}_AUROC_entropy', np.nan)
            fpr95_e = m.get(f'{pre}{rkey}_FPR95_entropy', np.nan)
            ece = m.get(f'{pre}{rkey}_ECE', np.nan)
            nll = m.get(f'{pre}{rkey}_NLL', np.nan)
        auroc_m_s = f'{auroc_m:.2f}' if not np.isnan(auroc_m) else '  -'
        fpr95_m_s = f'{fpr95_m:.2f}' if not np.isnan(fpr95_m) else '  -'
        auroc_e_s = f'{auroc_e:.2f}' if not np.isnan(auroc_e) else '  -'
        fpr95_e_s = f'{fpr95_e:.2f}' if not np.isnan(fpr95_e) else '  -'
        ece_s = f'{ece:.2f}' if not np.isnan(ece) else '  -'
        nll_s = f'{nll:.4f}' if not np.isnan(nll) else '    -'
        print(f'{prefix:>12s} | {auroc_m_s:>8s} | {fpr95_m_s:>8s} | {auroc_e_s:>8s} | {fpr95_e_s:>8s} | {ece_s:>5s} | {nll_s:>8s}')

    print('\n===> Radius-based AUROC/FPR95/ECE/NLL Summary:')
    print('     (same radius bins as mIoU: 0-20m, 20-25m, ...)')
    if radius_keys:
        print(f'{"Range":>12s} | {"AUROC_msp":>9s} | {"FPR95_msp":>9s} | {"AUROC_ent":>9s} | {"FPR95_ent":>9s} | {"ECE %":>6s} | {"NLL":>8s}')
        print('-' * 80)
        for r in radius_keys:
            _row(r, '', r, metrics, is_radius=True)
            for c in range(num_classes):
                cname = class_names[c] if c < len(class_names) else f"class_{c}"
                has_any = (metrics.get(f'radius_{r}_{cname}_AUROC_msp') is not None
                           or metrics.get(f'radius_{r}_{cname}_ECE') is not None
                           or metrics.get(f'radius_{r}_{cname}_NLL') is not None)
                if has_any:
                    _row(f'  {cname}', cname, r, metrics, is_radius=True)
    print('\n===> Height-based AUROC/FPR95/ECE/NLL Summary:')
    print('     (same height bins as mIoU: 0-2m, 2-4m, 4-6m)')
    if height_keys:
        print(f'{"Height":>12s} | {"AUROC_msp":>9s} | {"FPR95_msp":>9s} | {"AUROC_ent":>9s} | {"FPR95_ent":>9s} | {"ECE %":>6s} | {"NLL":>8s}')
        print('-' * 80)
        for h in height_keys:
            _row(h, '', h, metrics, is_radius=False)
            for c in range(num_classes):
                cname = class_names[c] if c < len(class_names) else f"class_{c}"
                has_any = (metrics.get(f'height_{h}_{cname}_AUROC_msp') is not None
                           or metrics.get(f'height_{h}_{cname}_ECE') is not None
                           or metrics.get(f'height_{h}_{cname}_NLL') is not None)
                if has_any:
                    _row(f'  {cname}', cname, h, metrics, is_radius=False)
    print('')


@METRICS.register_module()
class OccupancyMetric(BaseMetric):
    """Occupancy prediction evaluation metric.
    
    This metric evaluates occupancy prediction using mIoU and F-Score metrics.
    
    Args:
        num_classes (int): Number of occupancy classes. Default: 17.
        use_lidar_mask (bool): Whether to use LiDAR mask. Default: False.
        use_image_mask (bool): Whether to use image mask. Default: False.
        sort_by_timestamp (bool): Whether to sort data_infos by timestamp.
            Set to True if model sorts dataset by timestamp during inference.
            Set to False if model uses original dataset order. Default: True.
        **kwargs: Additional arguments for BaseMetric.
    """
    
    def __init__(self,
                 num_classes: int = 17,
                 use_lidar_mask: bool = False,
                 use_image_mask: bool = False,
                 ann_file: Optional[str] = None,
                 data_root: Optional[str] = None,
                 dataset_name: str = 'occ3d',
                 eval_metric: str = 'miou',
                 sort_by_timestamp: bool = True,
                 point_cloud_range: Optional[List[float]] = None,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 compute_uncertainty_metrics: bool = False,
                 **kwargs):
        super().__init__(collect_device=collect_device, prefix=prefix)
        
        self.num_classes = num_classes
        self.use_lidar_mask = use_lidar_mask
        self.use_image_mask = use_image_mask
        self.ann_file = ann_file
        self.data_root = data_root
        self.dataset_name = dataset_name
        self.eval_metric = eval_metric
        self.sort_by_timestamp = sort_by_timestamp
        # When False, only mIoU is computed (e.g. during test.py); use compute_metrics_from_file.py for uncertainty
        self.compute_uncertainty_metrics = bool(compute_uncertainty_metrics)
        # For radius/height breakdown (same as occ_metrics: [-40,-40,-1,40,40,5.4])
        self.point_cloud_range = point_cloud_range or [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
        
        # Load data_infos if ann_file is provided
        self.data_infos = None
        if ann_file:
            try:
                import mmengine
                data = mmengine.load(ann_file)
                if isinstance(data, dict) and 'data_list' in data:
                    self.data_infos = data['data_list']
                elif isinstance(data, dict) and 'infos' in data:
                    self.data_infos = data['infos']
                elif isinstance(data, list):
                    self.data_infos = data
                
                # breakpoint()
                # Sort by timestamp if requested (depends on model's dataset ordering)
                if self.data_infos and len(self.data_infos) > 0:
                    if self.sort_by_timestamp and 'timestamp' in self.data_infos[0]:
                        self.data_infos = list(sorted(self.data_infos, key=lambda e: e['timestamp']))
                        print(f"Loaded and sorted {len(self.data_infos)} data_infos by timestamp from {ann_file}")
                    elif 'timestamp' in self.data_infos[0]:
                        print(f"Loaded {len(self.data_infos)} data_infos (not sorted, original order) from {ann_file}")
                    else:
                        print(f"Loaded {len(self.data_infos)} data_infos (no timestamp available) from {ann_file}")
                else:
                    print(f"Loaded {len(self.data_infos)} data_infos from {ann_file}")
            except Exception as e:
                print(f"Warning: Failed to load data_infos from {ann_file}: {e}")
        
        # Initialize metrics
        self.miou_metric = Metric_mIoU(
            num_classes=num_classes,
            use_lidar_mask=use_lidar_mask,
            use_image_mask=use_image_mask,
        )
        
        # Store predictions for later evaluation
        self.predictions = []
        
        # Optional F-Score metric (commented out for now to reduce complexity)
        # self.fscore_metric = Metric_FScore()
    
    def reset(self):
        """Reset metric state for new epoch."""
        self.miou_metric.reset()
        # Clear predictions
        self.predictions = []
    
    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.
        
        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        # data_samples is a list of dicts with keys like 'occ_results', 'flow_results', 'index'
        for data_sample in data_samples:
            # Extract prediction - STCOcc returns 'occ_results' key
            if 'occ_results' not in data_sample or 'index' not in data_sample:
                continue
            
            # Store predictions for later evaluation in compute_metrics
            pred_dict = {
                'occ_results': data_sample['occ_results'],
                'index': data_sample['index']
            }
            # Store per-voxel uncertainty for AUROC/FPR95 (used by BEVFormer and other models)
            if 'uncertainty_msp' in data_sample:
                u = data_sample['uncertainty_msp']
                pred_dict['uncertainty_msp'] = [u] if not isinstance(u, (list, tuple)) else u
            if 'uncertainty_entropy' in data_sample:
                u = data_sample['uncertainty_entropy']
                pred_dict['uncertainty_entropy'] = [u] if not isinstance(u, (list, tuple)) else u
            if 'softmax_probs' in data_sample:
                p = data_sample['softmax_probs']
                pred_dict['softmax_probs'] = [p] if not isinstance(p, (list, tuple)) else p

            # Also store flow if available (needed for rayiou)
            if 'flow_results' in data_sample:
                pred_dict['flow_results'] = data_sample['flow_results']
            
            # CRITICAL: Store in self.results (not self.predictions) so mmengine collects it
            # mmengine automatically gathers self.results from all ranks to rank 0
            self.results.append(pred_dict)
    
    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.
        
        Args:
            results (list): The processed results of each batch (gathered from all ranks).
            
        Returns:
            Dict[str, float]: The computed metrics.
        """
        if not self.data_infos:
            print("Warning: data_infos not loaded. Cannot compute metrics.")
            return {
                'mIoU': 0.0,
                'count': 0
            }
        
        # Use gathered results instead of self.predictions
        # In DDP, mmengine automatically collects self.results from all ranks to rank 0
        # and passes it as the 'results' parameter
        self.predictions = results
        
        try:
            if self.eval_metric == 'rayiou':
                result = self._compute_rayiou()
            else:
                result = self._compute_miou()
            return result
                
        except Exception as e:
            # Return default metrics if computation fails
            print(f"Warning: Failed to compute metrics: {e}")
            import traceback
            traceback.print_exc()
            return {
                'mIoU': 0.0,
                'count': 0
            }
        finally:
            # Reset metric state for next epoch
            self.miou_metric.reset()
            self.predictions = []
    
    def compute_metrics_from_file(
        self,
        file_paths: List[str],
        chunk_size: int = 400,
        metric_groups: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Compute metrics by reading predictions from file(s) in chunks.

        Optionally run only a subset of metrics to reduce peak memory (run 3 passes
        and merge results): metric_groups=['miou'], ['ece_nll'], ['auroc_fpr95'].

        File format: repeated pickle.dump(batch_list, f) per batch.
        """
        if not self.data_infos:
            print("Warning: data_infos not loaded. Cannot compute metrics.")
            return {'mIoU': 0.0, 'count': 0}
        groups = metric_groups if metric_groups is not None else ['miou', 'ece_nll', 'auroc_fpr95']
        self._f_metric_groups = set(groups)
        print('\nStarting Evaluation (from file)...')
        if len(self._f_metric_groups) < 3:
            print(f'  Metric groups: {sorted(self._f_metric_groups)} (single-pass mode)')
        from tqdm import tqdm
        self.miou_metric.reset()
        self._f_processed_set = set()
        self._f_height_bins_actual = _get_height_bins_actual(self.point_cloud_range)
        rk = [f'{RADIUS_BINS[i]}-{RADIUS_BINS[i+1]}m' for i in range(len(RADIUS_BINS) - 1)]
        hk = [f'{HEIGHT_BINS_RELATIVE[i]}-{HEIGHT_BINS_RELATIVE[i+1]}m' for i in range(len(HEIGHT_BINS_RELATIVE) - 1)]

        def _mk_list(bin_keys):
            return {k: [] for k in bin_keys}
        def _mkcls_list(bin_keys):
            return {k: {c: [] for c in range(self.num_classes)} for k in bin_keys}
        def _mk_bin_stats(bin_keys):
            return {k: [(0.0, 0.0, 0) for _ in range(10)] for k in bin_keys}
        def _mk_sum_count(bin_keys):
            return {k: 0.0 for k in bin_keys}
        def _mk_count(bin_keys):
            return {k: 0 for k in bin_keys}
        def _mkcls_bin_stats(bin_keys):
            return {k: {c: [(0.0, 0.0, 0) for _ in range(10)] for c in range(self.num_classes)} for k in bin_keys}
        def _mkcls_sum_count(bin_keys):
            return {k: {c: 0.0 for c in range(self.num_classes)} for k in bin_keys}
        def _mkcls_count(bin_keys):
            return {k: {c: 0 for c in range(self.num_classes)} for k in bin_keys}

        # AUROC/FPR95: list-based (exact, same as non-chunked)
        # Only created when selected — other groups' accumulators are not created (saves memory).
        if 'auroc_fpr95' in self._f_metric_groups:
            self._f_pairs_msp = []
            self._f_pairs_entropy = []
            self._f_class_pairs_msp = {c: [] for c in range(self.num_classes)}
            self._f_class_pairs_entropy = {c: [] for c in range(self.num_classes)}
            self._f_r_msp = _mk_list(rk)
            self._f_r_ent = _mk_list(rk)
            self._f_r_cls_msp = _mkcls_list(rk)
            self._f_r_cls_ent = _mkcls_list(rk)
            self._f_h_msp = _mk_list(hk)
            self._f_h_ent = _mk_list(hk)
            self._f_h_cls_msp = _mkcls_list(hk)
            self._f_h_cls_ent = _mkcls_list(hk)
        # ECE/NLL: streaming (fixed memory)
        # Only created when selected — other groups' accumulators are not created (saves memory).
        if 'ece_nll' in self._f_metric_groups:
            self._f_ece_bin_stats = [(0.0, 0.0, 0) for _ in range(10)]
            self._f_nll_neglog_sum = 0.0
            self._f_nll_count = 0
            self._f_class_ece_bin_stats = {c: [(0.0, 0.0, 0) for _ in range(10)] for c in range(self.num_classes)}
            self._f_class_nll_sum = {c: 0.0 for c in range(self.num_classes)}
            self._f_class_nll_count = {c: 0 for c in range(self.num_classes)}
            self._f_r_ece_bin_stats = _mk_bin_stats(rk)
            self._f_r_nll_sum = _mk_sum_count(rk)
            self._f_r_nll_count = _mk_count(rk)
            self._f_r_cls_ece_bin_stats = _mkcls_bin_stats(rk)
            self._f_r_cls_nll_sum = _mkcls_sum_count(rk)
            self._f_r_cls_nll_count = _mkcls_count(rk)
            self._f_h_ece_bin_stats = _mk_bin_stats(hk)
            self._f_h_nll_sum = _mk_sum_count(hk)
            self._f_h_nll_count = _mk_count(hk)
            self._f_h_cls_ece_bin_stats = _mkcls_bin_stats(hk)
            self._f_h_cls_nll_sum = _mkcls_sum_count(hk)
            self._f_h_cls_nll_count = _mkcls_count(hk)

        buffer = []
        for path in file_paths:
            if not os.path.isfile(path):
                continue
            with open(path, 'rb') as f:
                while True:
                    try:
                        batch = pickle.load(f)
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
    
    def _process_one_chunk(self, chunk: List[dict]) -> None:
        """Process one chunk of pred_dicts and update file-mode accumulators."""
        from tqdm import tqdm
        pred_sems, data_index = [], []
        u_msp, u_ent, sp = [], [], []
        for pred_dict in chunk:
            occ_results = pred_dict.get('occ_results')
            if occ_results is None or 'index' not in pred_dict or pred_dict['index'] is None:
                continue
            data_id = pred_dict['index']
            if not isinstance(data_id, (list, tuple, np.ndarray)):
                data_id = [data_id]
            for i, id in enumerate(data_id):
                if id in self._f_processed_set:
                    continue
                self._f_processed_set.add(id)
                if i >= len(occ_results):
                    continue
                data_index.append(id)
                pred_sems.append(occ_results[i])
                need_auroc = 'auroc_fpr95' in self._f_metric_groups
                need_ece_nll = 'ece_nll' in self._f_metric_groups
                if need_auroc or need_ece_nll:
                    u_msp.append(np.asarray(pred_dict['uncertainty_msp'][i]).astype(np.float64) if pred_dict.get('uncertainty_msp') and i < len(pred_dict['uncertainty_msp']) else None)
                    u_ent.append(np.asarray(pred_dict['uncertainty_entropy'][i]).astype(np.float64) if pred_dict.get('uncertainty_entropy') and i < len(pred_dict['uncertainty_entropy']) else None)
                else:
                    u_msp.append(None)
                    u_ent.append(None)
                if need_ece_nll:
                    sp.append(np.asarray(pred_dict['softmax_probs'][i]).astype(np.float64) if pred_dict.get('softmax_probs') and i < len(pred_dict['softmax_probs']) else None)
                else:
                    sp.append(None)

        if not data_index:
            return
        # Run GT loop for this chunk (same as in _compute_miou chunk branch)
        for index in tqdm(data_index, leave=False, desc='Chunk'):
            if index >= len(self.data_infos):
                break
            info = self.data_infos[index]
            occ_path = info.get('occ3d_gt_path') or info.get('occ_path') or info.get('occ_gt_path')
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
                mask_camera = occ_gt['mask_camera'].astype(bool) if (self.dataset_name == 'occ3d' or self.use_image_mask) else None
                if 'miou' in self._f_metric_groups:
                    self.miou_metric.add_batch(pr_semantics, gt_semantics, None, mask_camera)
                pr_flat = np.asarray(pr_semantics).reshape(-1)
                gt_flat = np.asarray(gt_semantics).reshape(-1)
                mask_flat = mask_camera.reshape(-1) if mask_camera is not None else np.ones(pr_flat.shape, dtype=bool)
                valid = mask_flat
                n_valid = valid.sum()
                if n_valid > 0:
                    gt_valid = gt_flat[valid].astype(np.int64)
                    pr_valid = pr_flat[valid].astype(np.int64)
                    y_incorrect = (pr_valid != gt_valid).astype(np.int64)
                    if 'auroc_fpr95' in self._f_metric_groups:
                        if idx < len(u_msp) and u_msp[idx] is not None:
                            u_msp_flat = np.asarray(u_msp[idx]).reshape(-1)[valid].astype(np.float64)
                            self._f_pairs_msp.append((y_incorrect, u_msp_flat))
                            for c in range(self.num_classes):
                                mask_c = (gt_valid == c)
                                if mask_c.sum() > 0:
                                    self._f_class_pairs_msp[c].append(((pr_valid[mask_c] != c).astype(np.int64), u_msp_flat[mask_c]))
                        if idx < len(u_ent) and u_ent[idx] is not None:
                            u_ent_flat = np.asarray(u_ent[idx]).reshape(-1)[valid].astype(np.float64)
                            self._f_pairs_entropy.append((y_incorrect, u_ent_flat))
                            for c in range(self.num_classes):
                                mask_c = (gt_valid == c)
                                if mask_c.sum() > 0:
                                    self._f_class_pairs_entropy[c].append(((pr_valid[mask_c] != c).astype(np.int64), u_ent_flat[mask_c]))
                    if 'ece_nll' in self._f_metric_groups:
                        if idx < len(u_msp) and u_msp[idx] is not None:
                            conf = (1.0 - np.asarray(u_msp[idx]).reshape(-1)[valid]).astype(np.float64)
                            acc = (pr_valid == gt_valid).astype(np.float64)
                            ece_bin_stats_update(self._f_ece_bin_stats, conf, acc, n_bins=10)
                            for c in range(self.num_classes):
                                mask_c = (gt_valid == c)
                                if mask_c.sum() > 0:
                                    ece_bin_stats_update(self._f_class_ece_bin_stats[c], conf[mask_c], acc[mask_c], n_bins=10)
                        if idx < len(sp) and sp[idx] is not None:
                            probs_flat = np.asarray(sp[idx]).reshape(-1, self.num_classes)[valid]
                            neglog_sum, cnt = nll_neglog_sum_count(probs_flat, gt_valid)
                            self._f_nll_neglog_sum += neglog_sum
                            self._f_nll_count += cnt
                            for c in range(self.num_classes):
                                mask_c = (gt_valid == c)
                                if mask_c.sum() > 0:
                                    ns_c, nc_c = nll_neglog_sum_count(probs_flat[mask_c], gt_valid[mask_c])
                                    self._f_class_nll_sum[c] += ns_c
                                    self._f_class_nll_count[c] += nc_c
                    try:
                        shp = np.asarray(pr_semantics).shape
                        if len(shp) >= 3:
                            radius_grid, z_grid = _get_radius_height_grids(shp, self.point_cloud_range)
                            radius_flat = radius_grid.reshape(-1)[valid]
                            height_flat = z_grid.reshape(-1)[valid]
                            for ri in range(len(RADIUS_BINS) - 1):
                                r_min, r_max = RADIUS_BINS[ri], RADIUS_BINS[ri + 1]
                                r_key = f'{r_min}-{r_max}m'
                                if ri == len(RADIUS_BINS) - 2:
                                    in_r = (radius_flat >= r_min)
                                else:
                                    in_r = (radius_flat >= r_min) & (radius_flat < r_max)
                                if in_r.sum() == 0:
                                    continue
                                gt_in_r = gt_valid[in_r]
                                if 'auroc_fpr95' in self._f_metric_groups:
                                    if idx < len(u_msp) and u_msp[idx] is not None:
                                        u_msp_flat = np.asarray(u_msp[idx]).reshape(-1)[valid].astype(np.float64)
                                        self._f_r_msp[r_key].append((y_incorrect[in_r], u_msp_flat[in_r]))
                                        for c in range(self.num_classes):
                                            mask_c = (gt_in_r == c)
                                            if mask_c.sum() > 0:
                                                self._f_r_cls_msp[r_key][c].append((y_incorrect[in_r][mask_c], u_msp_flat[in_r][mask_c]))
                                    if idx < len(u_ent) and u_ent[idx] is not None:
                                        u_ent_flat = np.asarray(u_ent[idx]).reshape(-1)[valid].astype(np.float64)
                                        self._f_r_ent[r_key].append((y_incorrect[in_r], u_ent_flat[in_r]))
                                        for c in range(self.num_classes):
                                            mask_c = (gt_in_r == c)
                                            if mask_c.sum() > 0:
                                                self._f_r_cls_ent[r_key][c].append((y_incorrect[in_r][mask_c], u_ent_flat[in_r][mask_c]))
                                if 'ece_nll' in self._f_metric_groups:
                                    if idx < len(u_msp) and u_msp[idx] is not None:
                                        conf = (1.0 - np.asarray(u_msp[idx]).reshape(-1)[valid]).astype(np.float64)
                                        acc = (pr_valid == gt_valid).astype(np.float64)
                                        ece_bin_stats_update(self._f_r_ece_bin_stats[r_key], conf[in_r], acc[in_r], n_bins=10)
                                        for c in range(self.num_classes):
                                            mask_c = (gt_in_r == c)
                                            if mask_c.sum() > 0:
                                                ece_bin_stats_update(self._f_r_cls_ece_bin_stats[r_key][c], conf[in_r][mask_c], acc[in_r][mask_c], n_bins=10)
                                    if idx < len(sp) and sp[idx] is not None:
                                        probs_flat = np.asarray(sp[idx]).reshape(-1, self.num_classes)[valid]
                                        ns, nc = nll_neglog_sum_count(probs_flat[in_r], gt_valid[in_r])
                                        self._f_r_nll_sum[r_key] += ns
                                        self._f_r_nll_count[r_key] += nc
                                        for c in range(self.num_classes):
                                            mask_c = (gt_in_r == c)
                                            if mask_c.sum() > 0:
                                                nsc, ncc = nll_neglog_sum_count(probs_flat[in_r][mask_c], gt_valid[in_r][mask_c])
                                                self._f_r_cls_nll_sum[r_key][c] += nsc
                                                self._f_r_cls_nll_count[r_key][c] += ncc
                            for hi in range(len(self._f_height_bins_actual) - 1):
                                h_min, h_max = self._f_height_bins_actual[hi], self._f_height_bins_actual[hi + 1]
                                h_key = f'{HEIGHT_BINS_RELATIVE[hi]}-{HEIGHT_BINS_RELATIVE[hi+1]}m'
                                if hi == len(self._f_height_bins_actual) - 2:
                                    in_h = (height_flat >= h_min)
                                else:
                                    in_h = (height_flat >= h_min) & (height_flat < h_max)
                                if in_h.sum() == 0:
                                    continue
                                gt_in_h = gt_valid[in_h]
                                if 'auroc_fpr95' in self._f_metric_groups:
                                    if idx < len(u_msp) and u_msp[idx] is not None:
                                        u_msp_flat = np.asarray(u_msp[idx]).reshape(-1)[valid].astype(np.float64)
                                        self._f_h_msp[h_key].append((y_incorrect[in_h], u_msp_flat[in_h]))
                                        for c in range(self.num_classes):
                                            mask_c = (gt_in_h == c)
                                            if mask_c.sum() > 0:
                                                self._f_h_cls_msp[h_key][c].append((y_incorrect[in_h][mask_c], u_msp_flat[in_h][mask_c]))
                                    if idx < len(u_ent) and u_ent[idx] is not None:
                                        u_ent_flat = np.asarray(u_ent[idx]).reshape(-1)[valid].astype(np.float64)
                                        self._f_h_ent[h_key].append((y_incorrect[in_h], u_ent_flat[in_h]))
                                        for c in range(self.num_classes):
                                            mask_c = (gt_in_h == c)
                                            if mask_c.sum() > 0:
                                                self._f_h_cls_ent[h_key][c].append((y_incorrect[in_h][mask_c], u_ent_flat[in_h][mask_c]))
                                if 'ece_nll' in self._f_metric_groups:
                                    if idx < len(u_msp) and u_msp[idx] is not None:
                                        conf = (1.0 - np.asarray(u_msp[idx]).reshape(-1)[valid]).astype(np.float64)
                                        acc = (pr_valid == gt_valid).astype(np.float64)
                                        ece_bin_stats_update(self._f_h_ece_bin_stats[h_key], conf[in_h], acc[in_h], n_bins=10)
                                        for c in range(self.num_classes):
                                            mask_c = (gt_in_h == c)
                                            if mask_c.sum() > 0:
                                                ece_bin_stats_update(self._f_h_cls_ece_bin_stats[h_key][c], conf[in_h][mask_c], acc[in_h][mask_c], n_bins=10)
                                    if idx < len(sp) and sp[idx] is not None:
                                        probs_flat = np.asarray(sp[idx]).reshape(-1, self.num_classes)[valid]
                                        ns, nc = nll_neglog_sum_count(probs_flat[in_h], gt_valid[in_h])
                                        self._f_h_nll_sum[h_key] += ns
                                        self._f_h_nll_count[h_key] += nc
                                        for c in range(self.num_classes):
                                            mask_c = (gt_in_h == c)
                                            if mask_c.sum() > 0:
                                                nsc, ncc = nll_neglog_sum_count(probs_flat[in_h][mask_c], gt_valid[in_h][mask_c])
                                                self._f_h_cls_nll_sum[h_key][c] += nsc
                                                self._f_h_cls_nll_count[h_key][c] += ncc
                    except Exception:
                        pass
            except Exception as e:
                print(f"Warning: Failed to load GT for index {index}: {e}")

    def _finalize_file_metrics(self) -> Dict[str, float]:
        """Compute final metrics dict from file-mode accumulators."""
        class_names, miou_array, cnt = self.miou_metric.count_miou()
        metrics = {}
        if 'miou' in self._f_metric_groups:
            mean_iou = np.nanmean(miou_array[:self.num_classes - 1]) * 100
            metrics['mIoU'] = mean_iou
            metrics['count'] = cnt
            for i, (class_name, iou) in enumerate(zip(class_names, miou_array)):
                if i < len(class_names):
                    metrics[f'IoU_{class_name}'] = round(iou * 100, 2)
        else:
            metrics['count'] = cnt

        has_auroc = 'auroc_fpr95' in self._f_metric_groups and getattr(self, '_f_pairs_msp', None) is not None and (self._f_pairs_msp or self._f_pairs_entropy)
        has_ece_nll = 'ece_nll' in self._f_metric_groups and (getattr(self, '_f_nll_count', 0) > 0 or any(b[2] > 0 for b in getattr(self, '_f_ece_bin_stats', [])))
        if not has_auroc and not has_ece_nll:
            return metrics

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

            def _add_per_class_auroc(class_pairs_dict, prefix, class_names):
                auroc_list, fpr95_list = [], []
                for c in range(self.num_classes):
                    if not class_pairs_dict[c]:
                        continue
                    y_all = np.concatenate([p[0] for p in class_pairs_dict[c]], axis=0)
                    s_all = np.concatenate([p[1] for p in class_pairs_dict[c]], axis=0)
                    auroc_c, fpr95_c = compute_auroc_fpr95(y_all, s_all)
                    name = class_names[c] if c < len(class_names) else f"class_{c}"
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

            if any(self._f_class_pairs_msp[c] for c in range(self.num_classes)):
                _add_per_class_auroc(self._f_class_pairs_msp, 'uncertainty_msp', class_names)
            if any(self._f_class_pairs_entropy[c] for c in range(self.num_classes)):
                _add_per_class_auroc(self._f_class_pairs_entropy, 'uncertainty_entropy', class_names)
            _print_auroc_fpr95_summary(metrics, class_names, self.num_classes)

        if 'ece_nll' in self._f_metric_groups:
            if any(b[2] > 0 for b in self._f_ece_bin_stats):
                ece_val = ece_from_bin_stats(self._f_ece_bin_stats, n_bins=10)
                metrics['ECE'] = round(ece_val * 100, 2)
            if self._f_nll_count > 0:
                metrics['NLL'] = round(self._f_nll_neglog_sum / self._f_nll_count, 4)

            if any(any(b[2] > 0 for b in self._f_class_ece_bin_stats[c]) for c in range(self.num_classes)):
                ece_list = []
                for c in range(self.num_classes):
                    if not any(b[2] > 0 for b in self._f_class_ece_bin_stats[c]):
                        continue
                    ece_c = ece_from_bin_stats(self._f_class_ece_bin_stats[c], n_bins=10)
                    name = class_names[c] if c < len(class_names) else f"class_{c}"
                    metrics[f'ECE_{name}'] = round(ece_c * 100, 2)
                    if not np.isnan(ece_c):
                        ece_list.append(ece_c * 100)
                if ece_list:
                    metrics['mECE'] = round(float(np.mean(ece_list)), 2)
            if any(self._f_class_nll_count[c] > 0 for c in range(self.num_classes)):
                nll_list = []
                for c in range(self.num_classes):
                    if self._f_class_nll_count[c] <= 0:
                        continue
                    nll_c = self._f_class_nll_sum[c] / self._f_class_nll_count[c]
                    name = class_names[c] if c < len(class_names) else f"class_{c}"
                    metrics[f'NLL_{name}'] = round(nll_c, 4)
                    if not np.isnan(nll_c):
                        nll_list.append(nll_c)
                if nll_list:
                    metrics['mNLL'] = round(float(np.mean(nll_list)), 4)
            _print_ece_nll_summary(metrics, class_names, self.num_classes)

            for r_key in sorted(self._f_r_ece_bin_stats.keys(), key=lambda x: float(x.split('-')[0])):
                if any(b[2] > 0 for b in self._f_r_ece_bin_stats[r_key]):
                    metrics[f'radius_{r_key}_ECE'] = round(ece_from_bin_stats(self._f_r_ece_bin_stats[r_key], n_bins=10) * 100, 2)
                if self._f_r_nll_count[r_key] > 0:
                    metrics[f'radius_{r_key}_NLL'] = round(self._f_r_nll_sum[r_key] / self._f_r_nll_count[r_key], 4)
                for c in range(self.num_classes):
                    cname = class_names[c] if c < len(class_names) else f"class_{c}"
                    if any(b[2] > 0 for b in self._f_r_cls_ece_bin_stats[r_key][c]):
                        metrics[f'radius_{r_key}_{cname}_ECE'] = round(ece_from_bin_stats(self._f_r_cls_ece_bin_stats[r_key][c], n_bins=10) * 100, 2)
                    if self._f_r_cls_nll_count[r_key][c] > 0:
                        metrics[f'radius_{r_key}_{cname}_NLL'] = round(self._f_r_cls_nll_sum[r_key][c] / self._f_r_cls_nll_count[r_key][c], 4)

            for h_key in sorted(self._f_h_ece_bin_stats.keys(), key=lambda x: float(x.split('-')[0])):
                if any(b[2] > 0 for b in self._f_h_ece_bin_stats[h_key]):
                    metrics[f'height_{h_key}_ECE'] = round(ece_from_bin_stats(self._f_h_ece_bin_stats[h_key], n_bins=10) * 100, 2)
                if self._f_h_nll_count[h_key] > 0:
                    metrics[f'height_{h_key}_NLL'] = round(self._f_h_nll_sum[h_key] / self._f_h_nll_count[h_key], 4)
                for c in range(self.num_classes):
                    cname = class_names[c] if c < len(class_names) else f"class_{c}"
                    if any(b[2] > 0 for b in self._f_h_cls_ece_bin_stats[h_key][c]):
                        metrics[f'height_{h_key}_{cname}_ECE'] = round(ece_from_bin_stats(self._f_h_cls_ece_bin_stats[h_key][c], n_bins=10) * 100, 2)
                    if self._f_h_cls_nll_count[h_key][c] > 0:
                        metrics[f'height_{h_key}_{cname}_NLL'] = round(self._f_h_cls_nll_sum[h_key][c] / self._f_h_cls_nll_count[h_key][c], 4)

        if 'auroc_fpr95' in self._f_metric_groups:
            for r_key in sorted(self._f_r_msp.keys(), key=lambda x: float(x.split('-')[0])):
                if self._f_r_msp[r_key]:
                    y_all = np.concatenate([p[0] for p in self._f_r_msp[r_key]], axis=0)
                    s_all = np.concatenate([p[1] for p in self._f_r_msp[r_key]], axis=0)
                    auroc, fpr95 = compute_auroc_fpr95(y_all, s_all)
                    metrics[f'radius_{r_key}_AUROC_msp'] = round(auroc * 100, 2)
                    metrics[f'radius_{r_key}_FPR95_msp'] = round(fpr95 * 100, 2)
                if self._f_r_ent[r_key]:
                    y_all = np.concatenate([p[0] for p in self._f_r_ent[r_key]], axis=0)
                    s_all = np.concatenate([p[1] for p in self._f_r_ent[r_key]], axis=0)
                    auroc, fpr95 = compute_auroc_fpr95(y_all, s_all)
                    metrics[f'radius_{r_key}_AUROC_entropy'] = round(auroc * 100, 2)
                    metrics[f'radius_{r_key}_FPR95_entropy'] = round(fpr95 * 100, 2)
                for c in range(self.num_classes):
                    cname = class_names[c] if c < len(class_names) else f"class_{c}"
                    if self._f_r_cls_msp[r_key][c]:
                        y_all = np.concatenate([p[0] for p in self._f_r_cls_msp[r_key][c]], axis=0)
                        s_all = np.concatenate([p[1] for p in self._f_r_cls_msp[r_key][c]], axis=0)
                        auroc_c, fpr95_c = compute_auroc_fpr95(y_all, s_all)
                        metrics[f'radius_{r_key}_{cname}_AUROC_msp'] = round(auroc_c * 100, 2)
                        metrics[f'radius_{r_key}_{cname}_FPR95_msp'] = round(fpr95_c * 100, 2)
                    if self._f_r_cls_ent[r_key][c]:
                        y_all = np.concatenate([p[0] for p in self._f_r_cls_ent[r_key][c]], axis=0)
                        s_all = np.concatenate([p[1] for p in self._f_r_cls_ent[r_key][c]], axis=0)
                        auroc_c, fpr95_c = compute_auroc_fpr95(y_all, s_all)
                        metrics[f'radius_{r_key}_{cname}_AUROC_entropy'] = round(auroc_c * 100, 2)
                        metrics[f'radius_{r_key}_{cname}_FPR95_entropy'] = round(fpr95_c * 100, 2)

            for h_key in sorted(self._f_h_msp.keys(), key=lambda x: float(x.split('-')[0])):
                if self._f_h_msp[h_key]:
                    y_all = np.concatenate([p[0] for p in self._f_h_msp[h_key]], axis=0)
                    s_all = np.concatenate([p[1] for p in self._f_h_msp[h_key]], axis=0)
                    auroc, fpr95 = compute_auroc_fpr95(y_all, s_all)
                    metrics[f'height_{h_key}_AUROC_msp'] = round(auroc * 100, 2)
                    metrics[f'height_{h_key}_FPR95_msp'] = round(fpr95 * 100, 2)
                if self._f_h_ent[h_key]:
                    y_all = np.concatenate([p[0] for p in self._f_h_ent[h_key]], axis=0)
                    s_all = np.concatenate([p[1] for p in self._f_h_ent[h_key]], axis=0)
                    auroc, fpr95 = compute_auroc_fpr95(y_all, s_all)
                    metrics[f'height_{h_key}_AUROC_entropy'] = round(auroc * 100, 2)
                    metrics[f'height_{h_key}_FPR95_entropy'] = round(fpr95 * 100, 2)
                for c in range(self.num_classes):
                    cname = class_names[c] if c < len(class_names) else f"class_{c}"
                    if self._f_h_cls_msp[h_key][c]:
                        y_all = np.concatenate([p[0] for p in self._f_h_cls_msp[h_key][c]], axis=0)
                        s_all = np.concatenate([p[1] for p in self._f_h_cls_msp[h_key][c]], axis=0)
                        auroc_c, fpr95_c = compute_auroc_fpr95(y_all, s_all)
                        metrics[f'height_{h_key}_{cname}_AUROC_msp'] = round(auroc_c * 100, 2)
                        metrics[f'height_{h_key}_{cname}_FPR95_msp'] = round(fpr95_c * 100, 2)
                    if self._f_h_cls_ent[h_key][c]:
                        y_all = np.concatenate([p[0] for p in self._f_h_cls_ent[h_key][c]], axis=0)
                        s_all = np.concatenate([p[1] for p in self._f_h_cls_ent[h_key][c]], axis=0)
                        auroc_c, fpr95_c = compute_auroc_fpr95(y_all, s_all)
                        metrics[f'height_{h_key}_{cname}_AUROC_entropy'] = round(auroc_c * 100, 2)
                        metrics[f'height_{h_key}_{cname}_FPR95_entropy'] = round(fpr95_c * 100, 2)

        if 'ece_nll' in self._f_metric_groups or 'auroc_fpr95' in self._f_metric_groups:
            _print_radius_height_uncertainty_summary(metrics, class_names, self.num_classes)

        return metrics
    
    def _compute_miou(self) -> Dict[str, float]:
        """Compute standard mIoU metric.
        
        CRITICAL: Match original STCOcc evaluation logic exactly:
        1. Use actual indices from result['index'] (not sequential)
        2. Remove duplicates with processed_set
        3. Match predictions with GT using actual indices
        """
        from tqdm import tqdm
        
        print(f'\nStarting Evaluation...')
        
        processed_set = set()
        
        # Accumulators for AUROC/FPR95/ECE/NLL and radius/height (created once, filled by chunk or full)
        pairs_msp = []
        pairs_entropy = []
        class_pairs_msp = {c: [] for c in range(self.num_classes)}
        class_pairs_entropy = {c: [] for c in range(self.num_classes)}
        ece_conf_acc = []
        nll_probs_gt = []
        class_ece = {c: [] for c in range(self.num_classes)}
        class_nll = {c: [] for c in range(self.num_classes)}
        def _make_bin_dict(bin_keys):
            return {k: [] for k in bin_keys}
        def _make_bin_class_dict(bin_keys):
            return {k: {c: [] for c in range(self.num_classes)} for k in bin_keys}
        radius_keys_list = [f'{RADIUS_BINS[i]}-{RADIUS_BINS[i+1]}m' for i in range(len(RADIUS_BINS) - 1)]
        height_keys_list = [f'{HEIGHT_BINS_RELATIVE[i]}-{HEIGHT_BINS_RELATIVE[i+1]}m' for i in range(len(HEIGHT_BINS_RELATIVE) - 1)]
        radius_bin_pairs_msp = _make_bin_dict(radius_keys_list)
        radius_bin_pairs_entropy = _make_bin_dict(radius_keys_list)
        radius_bin_ece = _make_bin_dict(radius_keys_list)
        radius_bin_nll = _make_bin_dict(radius_keys_list)
        radius_bin_class_msp = _make_bin_class_dict(radius_keys_list)
        radius_bin_class_entropy = _make_bin_class_dict(radius_keys_list)
        radius_bin_class_ece = _make_bin_class_dict(radius_keys_list)
        radius_bin_class_nll = _make_bin_class_dict(radius_keys_list)
        height_bin_pairs_msp = _make_bin_dict(height_keys_list)
        height_bin_pairs_entropy = _make_bin_dict(height_keys_list)
        height_bin_ece = _make_bin_dict(height_keys_list)
        height_bin_nll = _make_bin_dict(height_keys_list)
        height_bin_class_msp = _make_bin_class_dict(height_keys_list)
        height_bin_class_entropy = _make_bin_class_dict(height_keys_list)
        height_bin_class_ece = _make_bin_class_dict(height_keys_list)
        height_bin_class_nll = _make_bin_class_dict(height_keys_list)
        height_bins_actual = _get_height_bins_actual(self.point_cloud_range)

        def _extract_predictions(pred_dicts, _processed_set, _pred_sems, _data_index, _u_msp, _u_ent, _sp):
            for pred_dict in pred_dicts:
                occ_results = pred_dict.get('occ_results')
                if occ_results is None or 'index' not in pred_dict or pred_dict['index'] is None:
                    if 'index' not in pred_dict or pred_dict['index'] is None:
                        continue
                data_id = pred_dict['index']
                if not isinstance(data_id, (list, tuple, np.ndarray)):
                    data_id = [data_id]
                for i, id in enumerate(data_id):
                    if id in _processed_set:
                        continue
                    _processed_set.add(id)
                    if i >= len(occ_results):
                        continue
                    _data_index.append(id)
                    _pred_sems.append(occ_results[i])
                    u_msp = pred_dict.get('uncertainty_msp')
                    u_ent = pred_dict.get('uncertainty_entropy')
                    if u_msp is not None and i < len(u_msp):
                        _u_msp.append(np.asarray(u_msp[i]).astype(np.float64))
                    else:
                        _u_msp.append(None)
                    if u_ent is not None and i < len(u_ent):
                        _u_ent.append(np.asarray(u_ent[i]).astype(np.float64))
                    else:
                        _u_ent.append(None)
                    sp = pred_dict.get('softmax_probs')
                    if sp is not None and i < len(sp):
                        _sp.append(np.asarray(sp[i]).astype(np.float64))
                    else:
                        _sp.append(None)

        pred_sems = []
        data_index = []
        uncertainty_msp_sems = []
        uncertainty_entropy_sems = []
        softmax_probs_sems = []

        _extract_predictions(self.predictions, processed_set, pred_sems, data_index,
                             uncertainty_msp_sems, uncertainty_entropy_sems, softmax_probs_sems)

        for index in tqdm(data_index):
            if index >= len(self.data_infos):
                print(f"Warning: Index {index} >= dataset size {len(self.data_infos)}. Skipping.")
                break
            
            info = self.data_infos[index]

            # Priority: occ3d_gt_path (SurroundOcc format) > occ_path (STCOcc format)
            if 'occ3d_gt_path' in info:
                occ_path = info['occ3d_gt_path']
            else:
                occ_path = info['occ_path']
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
                if self.dataset_name == 'occ3d' or self.use_image_mask:
                    mask_camera = occ_gt['mask_camera'].astype(bool)
                else:
                    mask_camera = None
                self.miou_metric.add_batch(pr_semantics, gt_semantics, None, mask_camera)
                pr_flat = np.asarray(pr_semantics).reshape(-1)
                gt_flat = np.asarray(gt_semantics).reshape(-1)
                mask_flat = mask_camera.reshape(-1) if mask_camera is not None else np.ones(pr_flat.shape, dtype=bool)
                valid = mask_flat
                n_valid = valid.sum()
                if n_valid > 0 and self.compute_uncertainty_metrics:
                    gt_valid = gt_flat[valid].astype(np.int64)
                    pr_valid = pr_flat[valid].astype(np.int64)
                    y_incorrect = (pr_valid != gt_valid).astype(np.int64)
                    if idx < len(uncertainty_msp_sems) and uncertainty_msp_sems[idx] is not None:
                        u_msp = np.asarray(uncertainty_msp_sems[idx]).reshape(-1)[valid].astype(np.float64)
                        pairs_msp.append((y_incorrect, u_msp))
                        for c in range(self.num_classes):
                            mask_c = (gt_valid == c)
                            if mask_c.sum() > 0:
                                y_c = (pr_valid[mask_c] != c).astype(np.int64)
                                class_pairs_msp[c].append((y_c, u_msp[mask_c]))
                    if idx < len(uncertainty_entropy_sems) and uncertainty_entropy_sems[idx] is not None:
                        u_ent = np.asarray(uncertainty_entropy_sems[idx]).reshape(-1)[valid].astype(np.float64)
                        pairs_entropy.append((y_incorrect, u_ent))
                        for c in range(self.num_classes):
                            mask_c = (gt_valid == c)
                            if mask_c.sum() > 0:
                                y_c = (pr_valid[mask_c] != c).astype(np.int64)
                                class_pairs_entropy[c].append((y_c, u_ent[mask_c]))
                    if idx < len(uncertainty_msp_sems) and uncertainty_msp_sems[idx] is not None:
                        conf = (1.0 - np.asarray(uncertainty_msp_sems[idx]).reshape(-1)[valid]).astype(np.float64)
                        acc = (pr_valid == gt_valid).astype(np.float64)
                        ece_conf_acc.append((conf, acc))
                        for c in range(self.num_classes):
                            mask_c = (gt_valid == c)
                            if mask_c.sum() > 0:
                                class_ece[c].append((conf[mask_c], acc[mask_c]))
                    if idx < len(softmax_probs_sems) and softmax_probs_sems[idx] is not None:
                        probs_flat = np.asarray(softmax_probs_sems[idx]).reshape(-1, self.num_classes)[valid]
                        nll_probs_gt.append((probs_flat, gt_valid))
                        for c in range(self.num_classes):
                            mask_c = (gt_valid == c)
                            if mask_c.sum() > 0:
                                class_nll[c].append((probs_flat[mask_c], gt_valid[mask_c]))
                    try:
                        shp = np.asarray(pr_semantics).shape
                        if len(shp) >= 3:
                            radius_grid, z_grid = _get_radius_height_grids(shp, self.point_cloud_range)
                            radius_flat = radius_grid.reshape(-1)[valid]
                            height_flat = z_grid.reshape(-1)[valid]
                            for ri in range(len(RADIUS_BINS) - 1):
                                r_min, r_max = RADIUS_BINS[ri], RADIUS_BINS[ri + 1]
                                r_key = f'{r_min}-{r_max}m'
                                if ri == len(RADIUS_BINS) - 2:
                                    in_r = (radius_flat >= r_min)
                                else:
                                    in_r = (radius_flat >= r_min) & (radius_flat < r_max)
                                if in_r.sum() == 0:
                                    continue
                                gt_in_r = gt_valid[in_r]
                                if idx < len(uncertainty_msp_sems) and uncertainty_msp_sems[idx] is not None:
                                    radius_bin_pairs_msp[r_key].append((y_incorrect[in_r], u_msp[in_r]))
                                    for c in range(self.num_classes):
                                        mask_c = (gt_in_r == c)
                                        if mask_c.sum() > 0:
                                            radius_bin_class_msp[r_key][c].append((y_incorrect[in_r][mask_c], u_msp[in_r][mask_c]))
                                if idx < len(uncertainty_entropy_sems) and uncertainty_entropy_sems[idx] is not None:
                                    radius_bin_pairs_entropy[r_key].append((y_incorrect[in_r], u_ent[in_r]))
                                    for c in range(self.num_classes):
                                        mask_c = (gt_in_r == c)
                                        if mask_c.sum() > 0:
                                            radius_bin_class_entropy[r_key][c].append((y_incorrect[in_r][mask_c], u_ent[in_r][mask_c]))
                                if idx < len(uncertainty_msp_sems) and uncertainty_msp_sems[idx] is not None:
                                    radius_bin_ece[r_key].append((conf[in_r], acc[in_r]))
                                    for c in range(self.num_classes):
                                        mask_c = (gt_in_r == c)
                                        if mask_c.sum() > 0:
                                            radius_bin_class_ece[r_key][c].append((conf[in_r][mask_c], acc[in_r][mask_c]))
                                if idx < len(softmax_probs_sems) and softmax_probs_sems[idx] is not None:
                                    radius_bin_nll[r_key].append((probs_flat[in_r], gt_valid[in_r]))
                                    for c in range(self.num_classes):
                                        mask_c = (gt_in_r == c)
                                        if mask_c.sum() > 0:
                                            radius_bin_class_nll[r_key][c].append((probs_flat[in_r][mask_c], gt_valid[in_r][mask_c]))
                            for hi in range(len(height_bins_actual) - 1):
                                h_min, h_max = height_bins_actual[hi], height_bins_actual[hi + 1]
                                h_key = f'{HEIGHT_BINS_RELATIVE[hi]}-{HEIGHT_BINS_RELATIVE[hi+1]}m'
                                if hi == len(height_bins_actual) - 2:
                                    in_h = (height_flat >= h_min)
                                else:
                                    in_h = (height_flat >= h_min) & (height_flat < h_max)
                                if in_h.sum() == 0:
                                    continue
                                gt_in_h = gt_valid[in_h]
                                if idx < len(uncertainty_msp_sems) and uncertainty_msp_sems[idx] is not None:
                                    height_bin_pairs_msp[h_key].append((y_incorrect[in_h], u_msp[in_h]))
                                    for c in range(self.num_classes):
                                        mask_c = (gt_in_h == c)
                                        if mask_c.sum() > 0:
                                            height_bin_class_msp[h_key][c].append((y_incorrect[in_h][mask_c], u_msp[in_h][mask_c]))
                                if idx < len(uncertainty_entropy_sems) and uncertainty_entropy_sems[idx] is not None:
                                    height_bin_pairs_entropy[h_key].append((y_incorrect[in_h], u_ent[in_h]))
                                    for c in range(self.num_classes):
                                        mask_c = (gt_in_h == c)
                                        if mask_c.sum() > 0:
                                            height_bin_class_entropy[h_key][c].append((y_incorrect[in_h][mask_c], u_ent[in_h][mask_c]))
                                if idx < len(uncertainty_msp_sems) and uncertainty_msp_sems[idx] is not None:
                                    height_bin_ece[h_key].append((conf[in_h], acc[in_h]))
                                    for c in range(self.num_classes):
                                        mask_c = (gt_in_h == c)
                                        if mask_c.sum() > 0:
                                            height_bin_class_ece[h_key][c].append((conf[in_h][mask_c], acc[in_h][mask_c]))
                                if idx < len(softmax_probs_sems) and softmax_probs_sems[idx] is not None:
                                    height_bin_nll[h_key].append((probs_flat[in_h], gt_valid[in_h]))
                                    for c in range(self.num_classes):
                                        mask_c = (gt_in_h == c)
                                        if mask_c.sum() > 0:
                                            height_bin_class_nll[h_key][c].append((probs_flat[in_h][mask_c], gt_valid[in_h][mask_c]))
                    except Exception:
                        pass
            except Exception as e:
                print(f"Warning: Failed to load GT for index {index}: {e}")
                continue

        # Compute final metrics: mIoU always, then uncertainty metrics when available
        class_names, miou_array, cnt = self.miou_metric.count_miou()
        mean_iou = np.nanmean(miou_array[:self.num_classes - 1]) * 100
        metrics = {
            'mIoU': mean_iou,
            'count': cnt
        }
        for i, (class_name, iou) in enumerate(zip(class_names, miou_array)):
            if i < len(class_names):
                metrics[f'IoU_{class_name}'] = round(iou * 100, 2)

        if not self.compute_uncertainty_metrics:
            return metrics

        # Add uncertainty metrics when we have data
        if pairs_msp or pairs_entropy or ece_conf_acc or nll_probs_gt:
            if pairs_msp:
                y_all = np.concatenate([p[0] for p in pairs_msp], axis=0)
                s_all = np.concatenate([p[1] for p in pairs_msp], axis=0)
                auroc_msp, fpr95_msp = compute_auroc_fpr95(y_all, s_all)
                metrics['AUROC_uncertainty_msp'] = round(auroc_msp * 100, 2)
                metrics['FPR95_uncertainty_msp'] = round(fpr95_msp * 100, 2)
            if pairs_entropy:
                y_all = np.concatenate([p[0] for p in pairs_entropy], axis=0)
                s_all = np.concatenate([p[1] for p in pairs_entropy], axis=0)
                auroc_ent, fpr95_ent = compute_auroc_fpr95(y_all, s_all)
                metrics['AUROC_uncertainty_entropy'] = round(auroc_ent * 100, 2)
                metrics['FPR95_uncertainty_entropy'] = round(fpr95_ent * 100, 2)

            # Per-class AUROC and FPR95 (only for classes with both correct and incorrect voxels)
            def _add_per_class_metrics(class_pairs_dict, prefix, class_names):
                auroc_list, fpr95_list = [], []
                for c in range(self.num_classes):
                    if not class_pairs_dict[c]:
                        continue
                    y_all = np.concatenate([p[0] for p in class_pairs_dict[c]], axis=0)
                    s_all = np.concatenate([p[1] for p in class_pairs_dict[c]], axis=0)
                    auroc_c, fpr95_c = compute_auroc_fpr95(y_all, s_all)
                    name = class_names[c] if c < len(class_names) else f"class_{c}"
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

            if class_pairs_msp and any(class_pairs_msp[c] for c in range(self.num_classes)):
                _add_per_class_metrics(class_pairs_msp, 'uncertainty_msp', class_names)
            if class_pairs_entropy and any(class_pairs_entropy[c] for c in range(self.num_classes)):
                _add_per_class_metrics(class_pairs_entropy, 'uncertainty_entropy', class_names)

            _print_auroc_fpr95_summary(metrics, class_names, self.num_classes)

            if ece_conf_acc:
                conf_all = np.concatenate([p[0] for p in ece_conf_acc], axis=0)
                acc_all = np.concatenate([p[1] for p in ece_conf_acc], axis=0)
                ece_val = compute_ece(conf_all, acc_all, n_bins=10)
                metrics['ECE'] = round(ece_val * 100, 2)
                ece_list = []
                for c in range(self.num_classes):
                    if not class_ece[c]:
                        continue
                    conf_c = np.concatenate([p[0] for p in class_ece[c]], axis=0)
                    acc_c = np.concatenate([p[1] for p in class_ece[c]], axis=0)
                    ece_c = compute_ece(conf_c, acc_c, n_bins=10)
                    name = class_names[c] if c < len(class_names) else f"class_{c}"
                    metrics[f'ECE_{name}'] = round(ece_c * 100, 2)
                    if not np.isnan(ece_c):
                        ece_list.append(ece_c * 100)
                if ece_list:
                    metrics['mECE'] = round(float(np.mean(ece_list)), 2)
            if nll_probs_gt:
                probs_all = np.concatenate([p[0] for p in nll_probs_gt], axis=0)
                gt_all = np.concatenate([p[1] for p in nll_probs_gt], axis=0)
                nll_val = compute_nll(probs_all, gt_all)
                metrics['NLL'] = round(nll_val, 4)
                nll_list = []
                for c in range(self.num_classes):
                    if not class_nll[c]:
                        continue
                    probs_c = np.concatenate([p[0] for p in class_nll[c]], axis=0)
                    gt_c = np.concatenate([p[1] for p in class_nll[c]], axis=0)
                    nll_c = compute_nll(probs_c, gt_c)
                    name = class_names[c] if c < len(class_names) else f"class_{c}"
                    metrics[f'NLL_{name}'] = round(nll_c, 4)
                    if not np.isnan(nll_c):
                        nll_list.append(nll_c)
                if nll_list:
                    metrics['mNLL'] = round(float(np.mean(nll_list)), 4)

            _print_ece_nll_summary(metrics, class_names, self.num_classes)

            for r_key in sorted(radius_bin_pairs_msp.keys(), key=lambda x: float(x.split('-')[0])):
                if radius_bin_pairs_msp[r_key]:
                    y_all = np.concatenate([p[0] for p in radius_bin_pairs_msp[r_key]], axis=0)
                    s_all = np.concatenate([p[1] for p in radius_bin_pairs_msp[r_key]], axis=0)
                    auroc, fpr95 = compute_auroc_fpr95(y_all, s_all)
                    metrics[f'radius_{r_key}_AUROC_msp'] = round(auroc * 100, 2)
                    metrics[f'radius_{r_key}_FPR95_msp'] = round(fpr95 * 100, 2)
                if radius_bin_pairs_entropy[r_key]:
                    y_all = np.concatenate([p[0] for p in radius_bin_pairs_entropy[r_key]], axis=0)
                    s_all = np.concatenate([p[1] for p in radius_bin_pairs_entropy[r_key]], axis=0)
                    auroc, fpr95 = compute_auroc_fpr95(y_all, s_all)
                    metrics[f'radius_{r_key}_AUROC_entropy'] = round(auroc * 100, 2)
                    metrics[f'radius_{r_key}_FPR95_entropy'] = round(fpr95 * 100, 2)
                if radius_bin_ece[r_key]:
                    c_all = np.concatenate([p[0] for p in radius_bin_ece[r_key]], axis=0)
                    a_all = np.concatenate([p[1] for p in radius_bin_ece[r_key]], axis=0)
                    metrics[f'radius_{r_key}_ECE'] = round(compute_ece(c_all, a_all, n_bins=10) * 100, 2)
                if radius_bin_nll[r_key]:
                    p_all = np.concatenate([x[0] for x in radius_bin_nll[r_key]], axis=0)
                    g_all = np.concatenate([x[1] for x in radius_bin_nll[r_key]], axis=0)
                    metrics[f'radius_{r_key}_NLL'] = round(compute_nll(p_all, g_all), 4)
                for c in range(self.num_classes):
                    cname = class_names[c] if c < len(class_names) else f"class_{c}"
                    if radius_bin_class_msp[r_key][c]:
                        y_all = np.concatenate([p[0] for p in radius_bin_class_msp[r_key][c]], axis=0)
                        s_all = np.concatenate([p[1] for p in radius_bin_class_msp[r_key][c]], axis=0)
                        auroc_c, fpr95_c = compute_auroc_fpr95(y_all, s_all)
                        metrics[f'radius_{r_key}_{cname}_AUROC_msp'] = round(auroc_c * 100, 2)
                        metrics[f'radius_{r_key}_{cname}_FPR95_msp'] = round(fpr95_c * 100, 2)
                    if radius_bin_class_entropy[r_key][c]:
                        y_all = np.concatenate([p[0] for p in radius_bin_class_entropy[r_key][c]], axis=0)
                        s_all = np.concatenate([p[1] for p in radius_bin_class_entropy[r_key][c]], axis=0)
                        auroc_c, fpr95_c = compute_auroc_fpr95(y_all, s_all)
                        metrics[f'radius_{r_key}_{cname}_AUROC_entropy'] = round(auroc_c * 100, 2)
                        metrics[f'radius_{r_key}_{cname}_FPR95_entropy'] = round(fpr95_c * 100, 2)
                    if radius_bin_class_ece[r_key][c]:
                        c_all = np.concatenate([p[0] for p in radius_bin_class_ece[r_key][c]], axis=0)
                        a_all = np.concatenate([p[1] for p in radius_bin_class_ece[r_key][c]], axis=0)
                        metrics[f'radius_{r_key}_{cname}_ECE'] = round(compute_ece(c_all, a_all, n_bins=10) * 100, 2)
                    if radius_bin_class_nll[r_key][c]:
                        p_all = np.concatenate([x[0] for x in radius_bin_class_nll[r_key][c]], axis=0)
                        g_all = np.concatenate([x[1] for x in radius_bin_class_nll[r_key][c]], axis=0)
                        metrics[f'radius_{r_key}_{cname}_NLL'] = round(compute_nll(p_all, g_all), 4)

            for h_key in sorted(height_bin_pairs_msp.keys(), key=lambda x: float(x.split('-')[0])):
                if height_bin_pairs_msp[h_key]:
                    y_all = np.concatenate([p[0] for p in height_bin_pairs_msp[h_key]], axis=0)
                    s_all = np.concatenate([p[1] for p in height_bin_pairs_msp[h_key]], axis=0)
                    auroc, fpr95 = compute_auroc_fpr95(y_all, s_all)
                    metrics[f'height_{h_key}_AUROC_msp'] = round(auroc * 100, 2)
                    metrics[f'height_{h_key}_FPR95_msp'] = round(fpr95 * 100, 2)
                if height_bin_pairs_entropy[h_key]:
                    y_all = np.concatenate([p[0] for p in height_bin_pairs_entropy[h_key]], axis=0)
                    s_all = np.concatenate([p[1] for p in height_bin_pairs_entropy[h_key]], axis=0)
                    auroc, fpr95 = compute_auroc_fpr95(y_all, s_all)
                    metrics[f'height_{h_key}_AUROC_entropy'] = round(auroc * 100, 2)
                    metrics[f'height_{h_key}_FPR95_entropy'] = round(fpr95 * 100, 2)
                if height_bin_ece[h_key]:
                    c_all = np.concatenate([p[0] for p in height_bin_ece[h_key]], axis=0)
                    a_all = np.concatenate([p[1] for p in height_bin_ece[h_key]], axis=0)
                    metrics[f'height_{h_key}_ECE'] = round(compute_ece(c_all, a_all, n_bins=10) * 100, 2)
                if height_bin_nll[h_key]:
                    p_all = np.concatenate([x[0] for x in height_bin_nll[h_key]], axis=0)
                    g_all = np.concatenate([x[1] for x in height_bin_nll[h_key]], axis=0)
                    metrics[f'height_{h_key}_NLL'] = round(compute_nll(p_all, g_all), 4)
                for c in range(self.num_classes):
                    cname = class_names[c] if c < len(class_names) else f"class_{c}"
                    if height_bin_class_msp[h_key][c]:
                        y_all = np.concatenate([p[0] for p in height_bin_class_msp[h_key][c]], axis=0)
                        s_all = np.concatenate([p[1] for p in height_bin_class_msp[h_key][c]], axis=0)
                        auroc_c, fpr95_c = compute_auroc_fpr95(y_all, s_all)
                        metrics[f'height_{h_key}_{cname}_AUROC_msp'] = round(auroc_c * 100, 2)
                        metrics[f'height_{h_key}_{cname}_FPR95_msp'] = round(fpr95_c * 100, 2)
                    if height_bin_class_entropy[h_key][c]:
                        y_all = np.concatenate([p[0] for p in height_bin_class_entropy[h_key][c]], axis=0)
                        s_all = np.concatenate([p[1] for p in height_bin_class_entropy[h_key][c]], axis=0)
                        auroc_c, fpr95_c = compute_auroc_fpr95(y_all, s_all)
                        metrics[f'height_{h_key}_{cname}_AUROC_entropy'] = round(auroc_c * 100, 2)
                        metrics[f'height_{h_key}_{cname}_FPR95_entropy'] = round(fpr95_c * 100, 2)
                    if height_bin_class_ece[h_key][c]:
                        c_all = np.concatenate([p[0] for p in height_bin_class_ece[h_key][c]], axis=0)
                        a_all = np.concatenate([p[1] for p in height_bin_class_ece[h_key][c]], axis=0)
                        metrics[f'height_{h_key}_{cname}_ECE'] = round(compute_ece(c_all, a_all, n_bins=10) * 100, 2)
                    if height_bin_class_nll[h_key][c]:
                        p_all = np.concatenate([x[0] for x in height_bin_class_nll[h_key][c]], axis=0)
                        g_all = np.concatenate([x[1] for x in height_bin_class_nll[h_key][c]], axis=0)
                        metrics[f'height_{h_key}_{cname}_NLL'] = round(compute_nll(p_all, g_all), 4)

            _print_radius_height_uncertainty_summary(metrics, class_names, self.num_classes)

        return metrics
    
    def _compute_rayiou(self) -> Dict[str, float]:
        """Compute ray-based IoU metric."""
        try:
            from ..datasets.nuscenes_ego_pose_loader import nuScenesDataset
            from ..datasets.ray_metrics_occ3d import main as ray_based_miou_occ3d
            from ..datasets.ray_metrics_openocc import main as ray_based_miou_openocc
        except ImportError:
            import importlib.util as _ilu
            _eval_dir = os.path.dirname(os.path.abspath(__file__))

            def _load_mod(mod_name, filename):
                path = os.path.normpath(os.path.join(_eval_dir, '..', 'datasets', filename))
                spec = _ilu.spec_from_file_location(mod_name, path)
                mod = _ilu.module_from_spec(spec)
                spec.loader.exec_module(mod)
                return mod

            _pose_mod = _load_mod('nuscenes_ego_pose_loader', 'nuscenes_ego_pose_loader.py')
            nuScenesDataset = _pose_mod.nuScenesDataset
            _occ3d_mod = _load_mod('ray_metrics_occ3d', 'ray_metrics_occ3d.py')
            ray_based_miou_occ3d = _occ3d_mod.main
            _openocc_mod = _load_mod('ray_metrics_openocc', 'ray_metrics_openocc.py')
            ray_based_miou_openocc = _openocc_mod.main
        
        pred_sems, gt_sems = [], []
        pred_flows, gt_flows = [], []
        lidar_origins = []
        data_index = []
        
        print('\nStarting Evaluation...')
        
        # CRITICAL: Match original model's logic - use result['index'] instead of sequential index
        # Original: data_id = result['index'], then for i, id in enumerate(data_id): data_index.append(id)
        processed_set = set()
        for pred_dict in self.predictions:
            # Get index list from pred_dict (same as original: result['index'])
            if 'index' not in pred_dict:
                # Fallback: use sequential index if 'index' not available
                data_id = [len(data_index)]
            else:
                data_id = pred_dict['index']
                # Ensure data_id is a list
                if not isinstance(data_id, list):
                    data_id = [data_id]
            
            occ_results = pred_dict['occ_results']
            for i, id in enumerate(data_id):
                # CRITICAL: Skip duplicates using processed_set (same as original)
                if id in processed_set:
                    continue
                processed_set.add(id)
                
                # Ensure index is within bounds
                if i >= len(occ_results):
                    continue
                
                pred_sem = occ_results[i]
                
                data_index.append(id)  # CRITICAL: Use actual id from result['index'], not sequential index
                pred_sems.append(pred_sem)
                
                # Get flow if available
                if 'flow_results' in pred_dict and i < len(pred_dict['flow_results']):
                    pred_flows.append(pred_dict['flow_results'][i])
                else:
                    pred_flows.append(np.zeros(pred_sem.shape + (2,)))
        
        # Load NuScenes dataset for lidar origins
        from nuscenes.nuscenes import NuScenes
        nusc = NuScenes('v1.0-trainval', 'data/nuscenes/')
        nusdata = nuScenesDataset(nusc, 'val')
        
        # CRITICAL: Load ground truth in the same order as data_index (same as original)
        for index in data_index:
            if index >= len(self.data_infos):
                break
            info = self.data_infos[index]

            # Priority: occ3d_gt_path (SurroundOcc/BEVFormer format) > occ_path (STCOcc format)
            if 'occ3d_gt_path' in info:
                occ_path = info['occ3d_gt_path']
            else:
                occ_path = info['occ_path']
                if self.dataset_name == 'openocc':
                    occ_path = occ_path.replace('gts', 'openocc_v2')
            
            # Only append 'labels.npz' if not already present (BEVFormer includes it)
            if not occ_path.endswith('labels.npz'):
                occ_path = os.path.join(occ_path, 'labels.npz')
            
            # Prepend data_root if provided and path is relative; avoid duplicating data_root
            if self.data_root and not os.path.isabs(occ_path):
                root = os.path.normpath(self.data_root.rstrip(os.sep))
                path_norm = os.path.normpath(occ_path)
                if not path_norm.startswith(root + os.sep) and path_norm != root:
                    occ_path = os.path.join(self.data_root, occ_path)
            
            occ_gt = np.load(occ_path, allow_pickle=True)
            
            gt_semantics = occ_gt['semantics'].astype(np.uint8)
            if self.dataset_name == 'occ3d':
                gt_flow = np.zeros((200, 200, 16, 2), dtype=np.float16)
            elif self.dataset_name == 'openocc':
                gt_flow = occ_gt['flow'].astype(np.float16)
            
            gt_sems.append(gt_semantics)
            gt_flows.append(gt_flow)
            
            # Get lidar origin
            ref_sample_token, output_origin_tensor = nusdata.__getitem__(index)
            lidar_origins.append(output_origin_tensor.unsqueeze(0))
        
        # Compute ray-based IoU
        if self.dataset_name == 'openocc':
            miou, mave, occ_score = ray_based_miou_openocc(pred_sems, gt_sems, pred_flows, gt_flows, lidar_origins, logger=None)
        elif self.dataset_name == 'occ3d':
            miou, mave, occ_score = ray_based_miou_occ3d(pred_sems, gt_sems, pred_flows, gt_flows, lidar_origins, logger=None)
        
        return {
            'mIoU': miou,
            'mAVE': mave,
            'occ_score': occ_score,
            'count': len(data_index)
        }
