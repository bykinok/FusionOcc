"""OccupancyMetricHybrid for SurroundOcc using STCOcc metric for occ3d GT format."""
import numpy as np
import torch
from mmengine.registry import METRICS as ENGINE_METRICS
from mmdet3d.registry import METRICS as DET3D_METRICS
from mmengine.evaluator import BaseMetric
from typing import Optional, Dict, Sequence, List


@ENGINE_METRICS.register_module()
@DET3D_METRICS.register_module()
class OccupancyMetricHybrid(BaseMetric):
    """Occupancy Metric for SurroundOcc using STCOcc's metric for occ3d format.
    
    This metric uses STCOcc's OccupancyMetric for evaluating 3D occupancy predictions
    with occ3d ground truth format (18 classes).
    """
    
    def __init__(self, 
                 dataset_name: Optional[str] = None,
                 num_classes: int = 17,
                 use_lidar_mask: bool = False,
                 use_image_mask: bool = False,
                 ann_file: Optional[str] = None,
                 data_root: Optional[str] = None,
                 class_names: Optional[List[str]] = None,
                 collect_device='cpu', 
                 prefix=None):
        super().__init__(collect_device=collect_device, prefix=prefix)
        
        self.dataset_name = dataset_name
        self.num_classes = num_classes
        self.class_names = class_names or []
        
        # Import STCOcc metric directly without triggering registry conflicts
        import importlib.util
        import sys
        from pathlib import Path
        from mmengine.registry import METRICS as ENGINE_METRICS
        from mmdet3d.registry import METRICS as DET3D_METRICS
        
        stcocc_metric_path = Path(__file__).resolve().parents[3] / 'STCOcc' / 'stcocc' / 'evaluation' / 'occupancy_metric.py'
        
        if not stcocc_metric_path.exists():
            raise ImportError(f"STCOcc metric file not found at {stcocc_metric_path}")
        
        # Temporarily store existing OccupancyMetric registrations from both registries
        existing_engine_metric = ENGINE_METRICS._module_dict.pop('OccupancyMetric', None)
        existing_det3d_metric = DET3D_METRICS._module_dict.pop('OccupancyMetric', None)
        
        # Load STCOcc metric module
        spec = importlib.util.spec_from_file_location("stcocc_metric_module", stcocc_metric_path)
        stcocc_metric_module = importlib.util.module_from_spec(spec)
        
        # Execute module (this will register STCOcc's OccupancyMetric without conflict)
        spec.loader.exec_module(stcocc_metric_module)
        
        # Get STCOcc's OccupancyMetric class
        STCOccMetric = stcocc_metric_module.OccupancyMetric
        
        # Restore original SurroundOcc's OccupancyMetric registrations
        # Remove STCOcc's registration first
        ENGINE_METRICS._module_dict.pop('OccupancyMetric', None)
        DET3D_METRICS._module_dict.pop('OccupancyMetric', None)
        
        # Restore original registrations
        if existing_engine_metric is not None:
            ENGINE_METRICS._module_dict['OccupancyMetric'] = existing_engine_metric
        if existing_det3d_metric is not None:
            DET3D_METRICS._module_dict['OccupancyMetric'] = existing_det3d_metric
        
        # Initialize STCOcc metric with 18 classes for occ3d
        self.stcocc_metric = STCOccMetric(
            num_classes=18,  # occ3d uses 18 classes
            use_lidar_mask=use_lidar_mask,
            use_image_mask=use_image_mask,
            dataset_name=dataset_name,
            ann_file=ann_file,
            data_root=data_root,
            eval_metric='miou',
            collect_device=collect_device,
            prefix=prefix
        )
    
    def reset(self):
        """Reset evaluation metrics."""
        self.stcocc_metric.reset()
    
    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples.
        
        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        # STCOcc expects predictions in occ_results format
        # Model's predict method already sets occ_results with 18 classes (occ3d format)
        # Just ensure index is set for each sample
        for data_sample in data_samples:
            # Handle both dict and object types
            is_dict = isinstance(data_sample, dict)
            
            # Ensure index is set
            has_index = 'index' in data_sample if is_dict else hasattr(data_sample, 'index')
            
            if not has_index:
                # Try to get sample_idx from metainfo
                if is_dict:
                    metainfo = data_sample.get('metainfo', {})
                    sample_idx = metainfo.get('sample_idx', 0) if isinstance(metainfo, dict) else 0
                    data_sample['index'] = sample_idx
                else:
                    if hasattr(data_sample, 'metainfo') and 'sample_idx' in data_sample.metainfo:
                        data_sample.index = data_sample.metainfo['sample_idx']
                    else:
                        data_sample.index = 0
        
        self.stcocc_metric.process(data_batch, data_samples)
    
    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.
        
        Args:
            results (list): The processed results of each batch.
            
        Returns:
            Dict[str, float]: The computed metrics.
        """
        return self.stcocc_metric.compute_metrics(results)

