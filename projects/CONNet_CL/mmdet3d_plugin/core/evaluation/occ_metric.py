# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from mmengine.evaluator import BaseMetric
from mmdet3d.registry import METRICS

from ...utils.metric_util import SSCMetrics
from ...utils.formating import cm_to_ious, format_SC_results, format_SSC_results


@METRICS.register_module()
class OccMetric(BaseMetric):
    """Occupancy evaluation metric using SSC metrics.
    
    This metric evaluates both scene completion and semantic segmentation
    of 3D occupancy grids.
    
    Args:
        class_names (list): List of class names.
        empty_idx (int): Index for empty voxels. Default: 0.
        ignore_idx (int): Index for ignored voxels. Default: 255.
        collect_device (str): Device for collecting results. Default: 'cpu'.
        prefix (str): Prefix for metric names. Default: None.
    """
    
    def __init__(self, 
                 class_names=None,
                 empty_idx=0,
                 ignore_idx=255,
                 collect_device='cpu',
                 prefix=None,
                 **kwargs):
        super().__init__(collect_device=collect_device, prefix=prefix)
        
        if class_names is None:
            # Default nuScenes occupancy classes
            self.class_names = [
                'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
                'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
                'driveable_surface', 'other_flat', 'sidewalk',
                'terrain', 'manmade', 'vegetation'
            ]
        else:
            self.class_names = class_names
            
        self.empty_idx = empty_idx
        self.ignore_idx = ignore_idx
        self.ssc_metrics = SSCMetrics(
            class_names=self.class_names,
            ignore_idx=ignore_idx,
            empty_idx=empty_idx
        )
        
    def process(self, data_batch, data_samples):
        """Process one batch of data samples.
        
        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (list): A batch of outputs from the model.
        """
        for data_sample in data_samples:
            # Extract predictions and ground truth
            pred_occ = data_sample.get('pred_occ', None)
            gt_occ = data_sample.get('gt_occ', None)
            
            if pred_occ is None or gt_occ is None:
                continue
                
            # Convert to numpy if needed
            if hasattr(pred_occ, 'cpu'):
                pred_occ = pred_occ.cpu().numpy()
            if hasattr(gt_occ, 'cpu'):
                gt_occ = gt_occ.cpu().numpy()
                
            # Ensure predictions are class indices
            if pred_occ.ndim > 3:  # If logits, take argmax
                pred_occ = np.argmax(pred_occ, axis=0)
                
            # Ensure same shape
            assert pred_occ.shape == gt_occ.shape, \
                f"Pred shape {pred_occ.shape} != GT shape {gt_occ.shape}"
            
            # Add batch dimension if needed
            if pred_occ.ndim == 3:
                pred_occ = pred_occ[None, ...]
                gt_occ = gt_occ[None, ...]
                
            # Add to metrics computation
            self.ssc_metrics.add_batch(pred_occ, gt_occ)
            
    def compute_metrics(self, results):
        """Compute the metrics from processed results.
        
        Args:
            results (list): The processed results of each batch.
            
        Returns:
            dict: The computed metrics.
        """
        # Get statistics from SSC metrics
        stats = self.ssc_metrics.get_stats()
        
        # Format results similar to formating.py
        precision = stats['precision']
        recall = stats['recall'] 
        iou = stats['iou']
        iou_ssc = stats['iou_ssc']
        iou_ssc_mean = stats['iou_ssc_mean']
        
        # Create result dictionary
        result_dict = {
            'SC_Precision': precision * 100,
            'SC_Recall': recall * 100, 
            'SC_IoU': iou * 100,
            'SSC_mean': iou_ssc_mean * 100,
        }
        
        # Add per-class IoU
        for i, class_name in enumerate(self.class_names):
            if i < len(iou_ssc):
                result_dict[f'SSC_{class_name}'] = iou_ssc[i] * 100
                
        return result_dict
        
    def __del__(self):
        """Reset metrics when object is destroyed."""
        if hasattr(self, 'ssc_metrics'):
            self.ssc_metrics.reset()
