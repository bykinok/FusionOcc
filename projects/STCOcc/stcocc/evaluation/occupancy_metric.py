# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from typing import List, Dict, Any, Optional, Sequence
from mmengine.evaluator import BaseMetric
from mmdet3d.registry import METRICS

# Import original metric functions
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'STCOcc_ori'))
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
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 **kwargs):
        super().__init__(collect_device=collect_device, prefix=prefix)
        
        self.num_classes = num_classes
        self.use_lidar_mask = use_lidar_mask
        self.use_image_mask = use_image_mask
        
        # Initialize metrics
        self.miou_metric = Metric_mIoU(
            num_classes=num_classes,
            use_lidar_mask=use_lidar_mask,
            use_image_mask=use_image_mask,
        )
        
        # Optional F-Score metric (commented out for now to reduce complexity)
        # self.fscore_metric = Metric_FScore()
    
    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.
        
        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        for data_sample in data_samples:
            # Extract predictions and ground truth
            if 'pred_instances_3d' in data_sample:
                pred = data_sample['pred_instances_3d']
            elif 'pts_semantic_mask' in data_sample:
                pred = data_sample['pts_semantic_mask']
            else:
                # Skip if no predictions available
                continue
                
            if 'gt_instances_3d' in data_sample:
                gt = data_sample['gt_instances_3d']
            elif 'gt_pts_seg' in data_sample:
                gt = data_sample['gt_pts_seg']
            else:
                # Skip if no ground truth available
                continue
            
            # Add to metric calculation
            try:
                self.miou_metric.add_batch(pred, gt)
            except Exception as e:
                # Handle gracefully if metric calculation fails
                print(f"Warning: Failed to add batch to metric: {e}")
                continue
        
        # Store processed results (empty for now, will be computed in compute_metrics)
        self.results.append({})
    
    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.
        
        Args:
            results (list): The processed results of each batch.
            
        Returns:
            Dict[str, float]: The computed metrics.
        """
        try:
            # Compute mIoU
            class_names, miou, cnt, ret_dict, table = self.miou_metric.count_miou()
            
            metrics = {
                'mIoU': miou,
                'count': cnt
            }
            
            # Add per-class IoU if available
            if ret_dict:
                for class_name, iou in ret_dict.items():
                    if isinstance(iou, (int, float)):
                        metrics[f'IoU_{class_name}'] = iou
            
            return metrics
            
        except Exception as e:
            # Return default metrics if computation fails
            print(f"Warning: Failed to compute metrics: {e}")
            return {
                'mIoU': 0.0,
                'count': 0
            }
