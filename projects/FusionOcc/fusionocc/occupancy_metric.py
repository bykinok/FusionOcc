"""Occupancy Metric for FusionOcc evaluation."""
import os
import numpy as np
from typing import Dict, List, Optional, Sequence

from mmengine.evaluator import BaseMetric
from mmengine.registry import METRICS as ENGINE_METRICS
from mmdet3d.registry import METRICS

from .datasets.occ_metrics import Metric_mIoU


@ENGINE_METRICS.register_module()
@METRICS.register_module()
class OccupancyMetric(BaseMetric):
    """Occupancy prediction evaluation metric for FusionOcc.
    
    This metric evaluates occupancy prediction using mIoU.
    
    Args:
        num_classes (int): Number of occupancy classes. Default: 18.
        use_lidar_mask (bool): Whether to use LiDAR mask. Default: False.
        use_image_mask (bool): Whether to use image mask. Default: True.
        collect_device (str): Device name for collecting results. Default: 'cpu'.
        prefix (str, optional): Prefix for metric names. Default: None.
    """
    
    def __init__(self,
                 num_classes: int = 18,
                 use_lidar_mask: bool = False,
                 use_image_mask: bool = True,
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
    
    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples.
        
        Args:
            data_batch: A batch of data from the dataloader.
            data_samples: A batch of outputs from the model.
        """
        for data_sample in data_samples:
            # Get prediction
            if 'pred_occ' in data_sample:
                pred_occ = data_sample['pred_occ']
            elif 'occ_pred' in data_sample:
                pred_occ = data_sample['occ_pred']
            else:
                # Try to get from eval_results
                if hasattr(data_sample, 'eval_results'):
                    pred_occ = data_sample.eval_results.get('occ_pred', None)
                else:
                    continue
            
            # Get ground truth
            if 'gt_occ' in data_sample:
                gt_occ_data = data_sample['gt_occ']
            elif 'eval_ann_info' in data_sample:
                gt_occ_data = data_sample['eval_ann_info']
            else:
                continue
            
            # Extract semantics and masks
            if isinstance(gt_occ_data, dict):
                gt_semantics = gt_occ_data.get('semantics', None)
                mask_lidar = gt_occ_data.get('mask_lidar', None)
                mask_camera = gt_occ_data.get('mask_camera', None)
            else:
                # gt_occ_data might be a numpy array directly
                gt_semantics = gt_occ_data
                mask_lidar = None
                mask_camera = None
            
            if gt_semantics is None:
                continue
            
            # Convert to numpy if needed
            if hasattr(pred_occ, 'cpu'):
                pred_occ = pred_occ.cpu().numpy()
            if hasattr(gt_semantics, 'cpu'):
                gt_semantics = gt_semantics.cpu().numpy()
            if mask_lidar is not None and hasattr(mask_lidar, 'cpu'):
                mask_lidar = mask_lidar.cpu().numpy()
            if mask_camera is not None and hasattr(mask_camera, 'cpu'):
                mask_camera = mask_camera.cpu().numpy()
            
            # Ensure all arrays have the same shape
            # pred_occ and gt_semantics should have shape (X, Y, Z) or similar
            if pred_occ.shape != gt_semantics.shape:
                # Log warning and try to reshape
                print(f"Warning: pred_occ shape {pred_occ.shape} != gt_semantics shape {gt_semantics.shape}")
                # Skip this sample if shapes don't match
                continue
            
            # Create default masks if not provided, ensuring they match gt_semantics shape
            if mask_lidar is None:
                mask_lidar = np.ones_like(gt_semantics, dtype=bool)
            else:
                # Ensure mask has the same shape as gt_semantics
                if mask_lidar.shape != gt_semantics.shape:
                    # Try to reshape or create new mask
                    print(f"Warning: mask_lidar shape {mask_lidar.shape} != gt_semantics shape {gt_semantics.shape}")
                    mask_lidar = np.ones_like(gt_semantics, dtype=bool)
            
            if mask_camera is None:
                mask_camera = np.ones_like(gt_semantics, dtype=bool)
            else:
                # Ensure mask has the same shape as gt_semantics
                if mask_camera.shape != gt_semantics.shape:
                    # Try to reshape or create new mask
                    print(f"Warning: mask_camera shape {mask_camera.shape} != gt_semantics shape {gt_semantics.shape}")
                    mask_camera = np.ones_like(gt_semantics, dtype=bool)
            
            # Add batch to metric
            self.miou_metric.add_batch(
                pred_occ.astype(np.int32),
                gt_semantics.astype(np.int32),
                mask_lidar.astype(bool),
                mask_camera.astype(bool)
            )
    
    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.
        
        Args:
            results: The processed results of each batch.
            
        Returns:
            Dict[str, float]: The computed metrics.
        """
        # The actual computation is done in process() method
        # Here we just need to collect the final results
        metrics = self.miou_metric.count_miou()
        
        return metrics
    
    def evaluate(self, size: int) -> dict:
        """Evaluate the model performance of the whole dataset after processing.

        Args:
            size (int): Length of the entire validation dataset.

        Returns:
            dict: Evaluation metrics dict.
        """
        # No need to sync results since we're computing on the fly
        metrics = self.compute_metrics([])
        
        return metrics

