"""OccupancyMetricHybrid for LiCROcc using STCOcc metric for occ3d GT format."""
import numpy as np
import torch
from mmengine.registry import METRICS as ENGINE_METRICS
from mmdet3d.registry import METRICS as DET3D_METRICS
from mmengine.evaluator import BaseMetric
from typing import Optional, Dict, Sequence, List


@ENGINE_METRICS.register_module()
@DET3D_METRICS.register_module()
class OccupancyMetricHybrid(BaseMetric):
    """Occupancy Metric for LiCROcc using STCOcc's metric for occ3d format.
    
    This metric uses STCOcc's OccupancyMetric for evaluating 3D occupancy predictions
    with occ3d ground truth format (18 classes). It delegates evaluation to STCOcc's
    implementation to ensure consistency with the occ3d benchmark.
    
    Args:
        dataset_name (str): Dataset name ('occ3d' for occ3d dataset).
        num_classes (int): Number of classes (18 for occ3d). Default: 18.
        use_lidar_mask (bool): Whether to use LiDAR mask. Default: False.
        use_image_mask (bool): Whether to use image mask (camera visibility). Default: True for occ3d.
        ann_file (str): Path to annotation file.
        data_root (str): Root directory of the dataset.
        class_names (List[str]): List of class names.
        eval_metric (str): Evaluation metric type ('miou' or 'rayiou'). Default: 'miou'.
        sort_by_timestamp (bool): Whether to sort data by timestamp. Default: False.
        collect_device (str): Device for collecting results. Default: 'cpu'.
        prefix (str): Prefix for metric names.
    """
    
    def __init__(self, 
                 dataset_name: Optional[str] = 'occ3d',
                 num_classes: int = 18,
                 use_lidar_mask: bool = False,
                 use_image_mask: bool = True,
                 ann_file: Optional[str] = None,
                 data_root: Optional[str] = None,
                 class_names: Optional[List[str]] = None,
                 eval_metric: str = 'miou',
                 sort_by_timestamp: bool = False,
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
        from mmengine.registry import METRICS as ENGINE_METRICS
        from mmdet3d.registry import METRICS as DET3D_METRICS
        
        # Find STCOcc metric path (relative to this file)
        current_file = Path(__file__).resolve()
        project_root = current_file.parents[4]  # Go up to projects/
        stcocc_metric_path = project_root / 'STCOcc' / 'stcocc' / 'evaluation' / 'occupancy_metric.py'
        
        if not stcocc_metric_path.exists():
            raise ImportError(
                f"STCOcc metric file not found at: {stcocc_metric_path}\n\n"
                f"Required directory structure:\n"
                f"  projects/\n"
                f"  ├── LiCROcc/  (current project)\n"
                f"  └── STCOcc/\n"
                f"      └── stcocc/\n"
                f"          └── evaluation/\n"
                f"              └── occupancy_metric.py  <-- Missing!\n\n"
                f"Current file: {current_file}\n"
                f"Project root: {project_root}\n\n"
                f"Please install STCOcc:\n"
                f"  1. Clone STCOcc repository in projects/ directory\n"
                f"  2. Ensure the structure matches the required layout above"
            )
        
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
        
        # Restore original registrations
        # Remove STCOcc's registration first
        ENGINE_METRICS._module_dict.pop('OccupancyMetric', None)
        DET3D_METRICS._module_dict.pop('OccupancyMetric', None)
        
        # Restore original registrations if they existed
        if existing_engine_metric is not None:
            ENGINE_METRICS._module_dict['OccupancyMetric'] = existing_engine_metric
        if existing_det3d_metric is not None:
            DET3D_METRICS._module_dict['OccupancyMetric'] = existing_det3d_metric
        
        # Initialize STCOcc metric
        self.stcocc_metric = STCOccMetric(
            num_classes=num_classes,
            use_lidar_mask=use_lidar_mask,
            use_image_mask=use_image_mask,
            dataset_name=dataset_name,
            ann_file=ann_file,
            data_root=data_root,
            eval_metric=eval_metric,
            sort_by_timestamp=sort_by_timestamp,
            collect_device=collect_device,
            prefix=prefix
        )
        
        print(f"[OccupancyMetricHybrid] Initialized with STCOcc metric for {dataset_name} "
              f"({num_classes} classes, eval_metric={eval_metric})")
    
    def reset(self):
        """Reset evaluation metrics."""
        self.stcocc_metric.reset()
    
    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples.
        
        Converts LiCROcc output format to STCOcc format and delegates to STCOcc metric.
        
        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        # STCOcc expects predictions in dict format with 'occ_results' and 'index' keys
        # Convert data_samples to STCOcc-compatible format
        converted_samples = []
        
        for data_sample in data_samples:
            # Handle both dict and object types
            is_dict = isinstance(data_sample, dict)
            
            if is_dict:
                # Already dict, validate required keys
                if 'occ_results' not in data_sample:
                    raise KeyError(
                        f"'occ_results' not found in data_sample. "
                        f"Available keys: {list(data_sample.keys())}. "
                        f"For occ3d evaluation, model forward() must return data_samples with 'occ_results' key. "
                        f"Check that dataset_name='occ3d' is set in model config."
                    )
                
                if 'index' not in data_sample:
                    raise KeyError(
                        f"'index' not found in data_sample. "
                        f"Available keys: {list(data_sample.keys())}. "
                        f"For occ3d evaluation, model forward() must return data_samples with 'index' key. "
                        f"Check that dataset provides 'sample_idx' in meta_dict."
                    )
                
                converted_samples.append(data_sample)
            else:
                # Convert object to dict for STCOcc compatibility
                sample_dict = {}
                
                # Extract occ_results (STCOcc expects numpy array)
                if not hasattr(data_sample, 'occ_results'):
                    raise AttributeError(
                        f"data_sample object does not have 'occ_results' attribute. "
                        f"For occ3d evaluation, model forward() must return data_samples with 'occ_results'. "
                        f"Check that dataset_name='occ3d' is set in model config."
                    )
                
                occ_results = data_sample.occ_results
                if isinstance(occ_results, torch.Tensor):
                    occ_results = occ_results.cpu().numpy()
                sample_dict['occ_results'] = occ_results
                
                # Extract index
                if hasattr(data_sample, 'index'):
                    sample_dict['index'] = data_sample.index
                elif hasattr(data_sample, 'metainfo') and 'sample_idx' in data_sample.metainfo:
                    sample_dict['index'] = data_sample.metainfo['sample_idx']
                else:
                    raise AttributeError(
                        f"data_sample object does not have 'index' attribute or 'metainfo.sample_idx'. "
                        f"For occ3d evaluation, model forward() must return data_samples with 'index'. "
                        f"Check that dataset provides 'sample_idx' in meta_dict."
                    )
                
                # Extract flow_results if available (for rayiou)
                if hasattr(data_sample, 'flow_results'):
                    sample_dict['flow_results'] = data_sample.flow_results
                
                converted_samples.append(sample_dict)
        
        # Pass converted dict samples to STCOcc metric
        self.stcocc_metric.process(data_batch, converted_samples)
    
    def evaluate(self, size: int):
        """Override evaluate to properly delegate to STCOcc metric.
        
        For STCOcc mode:
        1. process() delegates to self.stcocc_metric.process()
        2. But mmengine only collects OccupancyMetricHybrid's self.results, not self.stcocc_metric.results
        3. So we need to manually call stcocc_metric.evaluate() which handles its own collection
        
        Args:
            size (int): Length of the dataset.
            
        Returns:
            dict: Computed metrics.
        """
        # Directly call the internal metric's evaluate()
        # This allows it to handle its own result collection and computation
        return self.stcocc_metric.evaluate(size)
    
    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.
        
        Args:
            results (list): The processed results of each batch.
            
        Returns:
            Dict[str, float]: The computed metrics.
        """
        return self.stcocc_metric.compute_metrics(results)
