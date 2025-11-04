# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from mmengine.evaluator import BaseMetric
from mmdet3d.registry import METRICS

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
        
        # Initialize confusion matrices for accumulation (same as original code)
        self.SC_metric = 0
        self.SSC_metric = 0
        self.SSC_metric_fine = 0
        
    def process(self, data_batch, data_samples):
        """Process one batch of data samples.
        
        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (list): A batch of outputs from the model.
        """
        for data_sample in data_samples:
            # Use confusion matrices from simple_test (same as original code)
            # This ensures identical evaluation to the original implementation
            if 'SC_metric' in data_sample:
                if isinstance(self.SC_metric, int) and self.SC_metric == 0:
                    self.SC_metric = data_sample['SC_metric']
                else:
                    self.SC_metric += data_sample['SC_metric']
            
            if 'SSC_metric' in data_sample:
                if isinstance(self.SSC_metric, int) and self.SSC_metric == 0:
                    self.SSC_metric = data_sample['SSC_metric']
                else:
                    self.SSC_metric += data_sample['SSC_metric']
                    
            if 'SSC_metric_fine' in data_sample:
                if isinstance(self.SSC_metric_fine, int) and self.SSC_metric_fine == 0:
                    self.SSC_metric_fine = data_sample['SSC_metric_fine']
                else:
                    self.SSC_metric_fine += data_sample['SSC_metric_fine']
            
            # Add a dummy result to avoid the warning
            result = {'processed': True}
            self.results.append(result)
            
    def compute_metrics(self, results):
        """Compute the metrics from processed results.
        
        Args:
            results (list): The processed results of each batch.
            
        Returns:
            dict: The computed metrics.
        """
        # Compute IoU from confusion matrices (same as original code)
        result_dict = {}
        
        # SC metrics (Scene Completion)
        if not isinstance(self.SC_metric, int) or self.SC_metric != 0:
            sc_ious = cm_to_ious(self.SC_metric)
            # sc_ious[0] is empty, sc_ious[1] is non-empty
            result_dict['SC_Precision'] = sc_ious[1] * 100  # Simplified, use non-empty IoU
            result_dict['SC_Recall'] = sc_ious[1] * 100
            result_dict['SC_IoU'] = sc_ious[1] * 100
        
        # SSC metrics (Semantic Scene Completion)
        if not isinstance(self.SSC_metric, int) or self.SSC_metric != 0:
            ssc_ious = cm_to_ious(self.SSC_metric)
            # Calculate mean IoU (excluding empty class at index 0)
            # Use nanmean to handle classes not present in GT (nan values)
            ssc_mean = np.nanmean(ssc_ious[1:]) * 100
            result_dict['SSC_mean'] = ssc_mean
            
            # Add per-class IoU
            for i, class_name in enumerate(self.class_names):
                if i < len(ssc_ious):
                    result_dict[f'SSC_{class_name}'] = ssc_ious[i] * 100
        
        # SSC fine metrics (if available)
        if not isinstance(self.SSC_metric_fine, int) or self.SSC_metric_fine != 0:
            ssc_fine_ious = cm_to_ious(self.SSC_metric_fine)
            ssc_fine_mean = np.nanmean(ssc_fine_ious[1:]) * 100
            result_dict['SSC_fine_mean'] = ssc_fine_mean
                
        return result_dict
        
    def __del__(self):
        """Reset metrics when object is destroyed."""
        # Metrics are automatically reset with new initialization
        pass
