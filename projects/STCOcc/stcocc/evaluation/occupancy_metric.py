# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from typing import List, Dict, Any, Optional, Sequence
from mmengine.evaluator import BaseMetric
from mmdet3d.registry import METRICS

# Import original metric functions
import sys
import os
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


@METRICS.register_module()
class OccupancyMetric(BaseMetric):
    """Occupancy prediction evaluation metric.
    
    This metric evaluates occupancy prediction using mIoU and F-Score metrics.
    
    Args:
        num_classes (int): Number of occupancy classes. Default: 17.
        use_lidar_mask (bool): Whether to use LiDAR mask. Default: False.
        use_image_mask (bool): Whether to use image mask. Default: False.
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
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 **kwargs):
        super().__init__(collect_device=collect_device, prefix=prefix)
        
        self.num_classes = num_classes
        self.use_lidar_mask = use_lidar_mask
        self.use_image_mask = use_image_mask
        self.ann_file = ann_file
        self.data_root = data_root
        self.dataset_name = dataset_name
        self.eval_metric = eval_metric
        
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
                # CRITICAL: Sort by timestamp to match dataset's data_infos order
                # Dataset sorts data_infos in load_data_list(), so metric must use same order
                if self.data_infos and len(self.data_infos) > 0:
                    if 'timestamp' in self.data_infos[0]:
                        self.data_infos = list(sorted(self.data_infos, key=lambda e: e['timestamp']))
                        print(f"Loaded and sorted {len(self.data_infos)} data_infos from {ann_file}")
                    else:
                        print(f"Loaded {len(self.data_infos)} data_infos from {ann_file} (no timestamp, not sorted)")
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
            
            # Also store flow if available (needed for rayiou)
            if 'flow_results' in data_sample:
                pred_dict['flow_results'] = data_sample['flow_results']
            
            self.predictions.append(pred_dict)
        
        # Store processed results (will be computed in compute_metrics)
        self.results.append({})
    
    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.
        
        Args:
            results (list): The processed results of each batch.
            
        Returns:
            Dict[str, float]: The computed metrics.
        """
        if not self.data_infos:
            print("Warning: data_infos not loaded. Cannot compute metrics.")
            return {
                'mIoU': 0.0,
                'count': 0
            }
        
        try:
            if self.eval_metric == 'rayiou':
                return self._compute_rayiou()
            else:
                return self._compute_miou()
                
        except Exception as e:
            # Return default metrics if computation fails
            print(f"Warning: Failed to compute metrics: {e}")
            import traceback
            traceback.print_exc()
            return {
                'mIoU': 0.0,
                'count': 0
            }
    
    def _compute_miou(self) -> Dict[str, float]:
        """Compute standard mIoU metric."""
        pred_sems, gt_sems = [], []
        data_index = []
        
        print(f'\nStarting Evaluation...')
        from tqdm import tqdm
        
        # Collect predictions in order (ignore reported indices as they may be incorrect)
        sample_idx = 0
        for pred_dict in self.predictions:
            occ_results = pred_dict['occ_results']
            for i in range(len(occ_results)):
                pred_sem = occ_results[i]
                data_index.append(sample_idx)
                pred_sems.append(pred_sem)
                sample_idx += 1
        
        # Load ground truth and evaluate
        for index in tqdm(data_index):
            if index >= len(self.data_infos):
                break
            info = self.data_infos[index]
            
            occ_path = info['occ_path']
            if self.dataset_name == 'openocc':
                occ_path = occ_path.replace('gts', 'openocc_v2')
            occ_path = os.path.join(occ_path, 'labels.npz')
            
            try:
                occ_gt = np.load(occ_path, allow_pickle=True)
                gt_semantics = occ_gt['semantics']
                pr_semantics = pred_sems[data_index.index(index)]
                
                if self.dataset_name == 'occ3d' or self.use_image_mask:
                    mask_camera = occ_gt['mask_camera'].astype(bool)
                else:
                    mask_camera = None
                
                self.miou_metric.add_batch(pr_semantics, gt_semantics, None, mask_camera)
            except Exception as e:
                print(f"Warning: Failed to load GT for index {index}: {e}")
                continue
        
        # Compute final metrics
        class_names, miou_array, cnt = self.miou_metric.count_miou()
        
        # Calculate mean IoU
        mean_iou = np.nanmean(miou_array[:self.num_classes]) * 100
        
        metrics = {
            'mIoU': mean_iou,
            'count': cnt
        }
        
        # Add per-class IoU
        for i, (class_name, iou) in enumerate(zip(class_names, miou_array)):
            if i < len(class_names):
                metrics[f'IoU_{class_name}'] = round(iou * 100, 2)
        
        return metrics
    
    def _compute_rayiou(self) -> Dict[str, float]:
        """Compute ray-based IoU metric."""
        from ..datasets.nuscenes_ego_pose_loader import nuScenesDataset
        from ..datasets.ray_metrics_occ3d import main as ray_based_miou_occ3d
        from ..datasets.ray_metrics_openocc import main as ray_based_miou_openocc
        
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

            breakpoint()
            
            occ_path = info['occ_path']
            if self.dataset_name == 'openocc':
                occ_path = occ_path.replace('gts', 'openocc_v2')
            occ_path = os.path.join(occ_path, 'labels.npz')
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
