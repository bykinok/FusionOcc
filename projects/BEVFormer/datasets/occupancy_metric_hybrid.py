"""OccupancyMetricHybrid for BEVFormer using STCOcc metric for occ3d GT format."""
import numpy as np
import torch
from mmengine.registry import METRICS as ENGINE_METRICS
from mmdet3d.registry import METRICS as DET3D_METRICS
from mmengine.evaluator import BaseMetric
from typing import Optional, Dict, Sequence, List


@ENGINE_METRICS.register_module()
@DET3D_METRICS.register_module()
class OccupancyMetricHybrid(BaseMetric):
    """Occupancy Metric for BEVFormer using STCOcc's metric for occ3d format.
    
    This metric uses STCOcc's OccupancyMetric for evaluating 3D occupancy predictions
    with occ3d ground truth format (18 classes).
    
    Args:
        dataset_name (str, optional): Dataset name ('occ3d' for occ3d format). Defaults to None.
        num_classes (int): Number of classes. Defaults to 18.
        use_lidar_mask (bool): Whether to use lidar mask. Defaults to False.
        use_image_mask (bool): Whether to use image mask. Defaults to False.
        ann_file (str, optional): Path to annotation file. Defaults to None.
        data_root (str, optional): Root directory of dataset. Defaults to None.
        class_names (List[str], optional): List of class names. Defaults to None.
        eval_metric (str): Evaluation metric type. Defaults to 'miou'.
        sort_by_timestamp (bool): Whether to sort results by timestamp. Defaults to True.
        collect_device (str): Device for collecting results. Defaults to 'cpu'.
        prefix (str, optional): Prefix for metric names. Defaults to None.
    """
    
    def __init__(self, 
                 dataset_name: Optional[str] = None,
                 num_classes: int = 18,
                 use_lidar_mask: bool = False,
                 use_image_mask: bool = False,
                 ann_file: Optional[str] = None,
                 data_root: Optional[str] = None,
                 class_names: Optional[List[str]] = None,
                 eval_metric: str = 'miou',
                 sort_by_timestamp: bool = True,  # BEVFormer dataset sorts by timestamp (line 100 in nuscenes_occ.py)
                 collect_device='cpu', 
                 prefix=None):
        super().__init__(collect_device=collect_device, prefix=prefix)
        
        self.dataset_name = dataset_name
        self.num_classes = num_classes
        self.class_names = class_names or []
        self.eval_metric = eval_metric
        self.sort_by_timestamp = sort_by_timestamp
        
        # Import STCOcc metric directly without triggering registry conflicts
        import importlib.util
        import sys
        from pathlib import Path
        
        # Get the path to STCOcc metric file
        # This file is in: projects/BEVFormer/datasets/occupancy_metric_hybrid.py
        # STCOcc is in: projects/STCOcc/stcocc/evaluation/occupancy_metric.py
        stcocc_metric_path = Path(__file__).resolve().parents[2] / 'STCOcc' / 'stcocc' / 'evaluation' / 'occupancy_metric.py'
        
        if not stcocc_metric_path.exists():
            raise ImportError(
                f"STCOcc metric file not found at {stcocc_metric_path}\n"
                f"Please ensure STCOcc project is available in the projects directory."
            )
        
        # Temporarily store existing OccupancyMetric registrations from both registries
        existing_engine_metric = ENGINE_METRICS._module_dict.pop('OccupancyMetric', None)
        existing_det3d_metric = DET3D_METRICS._module_dict.pop('OccupancyMetric', None)
        
        # Load STCOcc metric module
        spec = importlib.util.spec_from_file_location("stcocc_metric_module", stcocc_metric_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to load spec from {stcocc_metric_path}")
        
        stcocc_metric_module = importlib.util.module_from_spec(spec)
        
        # Execute module (this will register STCOcc's OccupancyMetric)
        spec.loader.exec_module(stcocc_metric_module)
        
        # Get STCOcc's OccupancyMetric class
        STCOccMetric = stcocc_metric_module.OccupancyMetric
        
        # Restore original registrations
        # Remove STCOcc's registration first
        ENGINE_METRICS._module_dict.pop('OccupancyMetric', None)
        DET3D_METRICS._module_dict.pop('OccupancyMetric', None)
        
        # Restore original registrations if they existed
        if existing_engine_metric is not None:
            ENGINE_METRICS._module_dict['OccupancyMetric'] = existing_engine_metric
        if existing_det3d_metric is not None:
            DET3D_METRICS._module_dict['OccupancyMetric'] = existing_det3d_metric
        
        # Initialize STCOcc metric with 18 classes for occ3d
        self.stcocc_metric = STCOccMetric(
            num_classes=num_classes,  # occ3d uses 18 classes
            use_lidar_mask=use_lidar_mask,
            use_image_mask=use_image_mask,
            dataset_name=dataset_name,
            ann_file=ann_file,
            data_root=data_root,
            eval_metric=eval_metric,
            sort_by_timestamp=self.sort_by_timestamp,  # Use config value (default: True for BEVFormer)
            collect_device=collect_device,
            prefix=prefix
        )
        
        # CRITICAL: Convert BEVFormer's 'occ_gt_path' to STCOcc's expected 'occ3d_gt_path'
        # BEVFormer annotation files use 'occ_gt_path' but STCOcc expects 'occ3d_gt_path' or 'occ_path'
        if hasattr(self.stcocc_metric, 'data_infos') and self.stcocc_metric.data_infos:
            for info in self.stcocc_metric.data_infos:
                if 'occ_gt_path' in info and 'occ3d_gt_path' not in info:
                    # Convert occ_gt_path to occ3d_gt_path for STCOcc compatibility
                    info['occ3d_gt_path'] = info['occ_gt_path']
        
        print(f"[OccupancyMetricHybrid] Initialized with STCOcc metric for {dataset_name} "
              f"with eval_metric={eval_metric}, sort_by_timestamp={self.sort_by_timestamp}")
    
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
        # Model's predict method should set occ_results with 18 classes (occ3d format)
        # Ensure index is set for each sample
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
        
        # Delegate to STCOcc metric (so it can use the same data internally if needed)
        self.stcocc_metric.process(data_batch, data_samples)
        
        # CRITICAL: MMEngine gathers this metric's self.results, not stcocc_metric.results.
        # So we must append to self.results here; otherwise compute_metrics receives [].
        for data_sample in data_samples:
            occ = data_sample.get('occ_results') if isinstance(data_sample, dict) else getattr(data_sample, 'occ_results', None)
            idx = data_sample.get('index') if isinstance(data_sample, dict) else getattr(data_sample, 'index', None)
            if occ is not None and idx is not None:
                pred_dict = {'occ_results': occ, 'index': idx}
                if isinstance(data_sample, dict) and 'flow_results' in data_sample:
                    pred_dict['flow_results'] = data_sample['flow_results']
                self.results.append(pred_dict)
    
    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.
        
        Args:
            results (list): The processed results of each batch.
            
        Returns:
            Dict[str, float]: The computed metrics.
        """
        return self.stcocc_metric.compute_metrics(results)
