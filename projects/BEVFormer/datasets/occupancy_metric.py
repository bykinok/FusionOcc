# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from typing import Dict, List, Optional, Sequence
import torch

from mmengine.evaluator import BaseMetric
from mmengine.registry import METRICS as ENGINE_METRICS
from mmdet3d.registry import METRICS

from .occ_metrics import Metric_mIoU


@ENGINE_METRICS.register_module()
@METRICS.register_module()
class OccupancyMetric(BaseMetric):
    """Occupancy prediction evaluation metric for BEVFormer.
    
    This metric evaluates occupancy prediction using mIoU.
    
    Args:
        num_classes (int): Number of occupancy classes. Default: 18.
        use_lidar_mask (bool): Whether to use LiDAR mask. Default: False.
        use_image_mask (bool): Whether to use image mask. Default: False.
        collect_device (str): Device name for collecting results. Default: 'cpu'.
        prefix (str, optional): Prefix for metric names. Default: None.
    """
    
    def __init__(self,
                 num_classes: int = 18,
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
    
    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.
        
        Args:
            data_batch: A batch of data from the dataloader.
            data_samples: A batch of outputs from the model.
        """
        for data_sample in data_samples:
            # Extract predictions and ground truth from data_sample
            # data_sample should be a dict containing model predictions
            
            if isinstance(data_sample, dict):
                # Get predicted occupancy (prioritize 'pred_occ')
                if 'pred_occ' in data_sample:
                    pred_occ = data_sample['pred_occ']
                elif 'occ_pred' in data_sample:
                    pred_occ = data_sample['occ_pred']
                elif 'occ_preds' in data_sample:
                    pred_occ = data_sample['occ_preds']
                else:
                    # Skip this sample if no prediction is available
                    continue
                
                # Get ground truth (prioritize 'voxel_semantics')
                if 'voxel_semantics' in data_sample:
                    gt_occ = data_sample['voxel_semantics']
                elif 'gt_occ' in data_sample:
                    # gt_occ might be a dict with 'semantics' key (FusionOcc style)
                    gt_occ_data = data_sample['gt_occ']
                    if isinstance(gt_occ_data, dict) and 'semantics' in gt_occ_data:
                        gt_occ = gt_occ_data['semantics']
                    else:
                        gt_occ = gt_occ_data
                else:
                    # Skip this sample if no ground truth is available
                    continue
                
                # Get masks
                mask_lidar = data_sample.get('mask_lidar', None)
                mask_camera = data_sample.get('mask_camera', None)
                
                # Also check in gt_occ dict if available
                if mask_lidar is None and 'gt_occ' in data_sample and isinstance(data_sample['gt_occ'], dict):
                    mask_lidar = data_sample['gt_occ'].get('mask_lidar', None)
                if mask_camera is None and 'gt_occ' in data_sample and isinstance(data_sample['gt_occ'], dict):
                    mask_camera = data_sample['gt_occ'].get('mask_camera', None)
                
            else:
                # Handle case where data_sample might be a custom object
                pred_occ = data_sample.pred_occ if hasattr(data_sample, 'pred_occ') else data_sample.occ_preds
                gt_occ = data_sample.gt_occ if hasattr(data_sample, 'gt_occ') else data_sample.voxel_semantics
                mask_lidar = data_sample.mask_lidar if hasattr(data_sample, 'mask_lidar') else None
                mask_camera = data_sample.mask_camera if hasattr(data_sample, 'mask_camera') else None
            
            # Convert to numpy if needed
            if isinstance(pred_occ, torch.Tensor):
                pred_occ = pred_occ.detach().cpu().numpy()
            if isinstance(gt_occ, torch.Tensor):
                gt_occ = gt_occ.detach().cpu().numpy()
            if mask_lidar is not None and isinstance(mask_lidar, torch.Tensor):
                mask_lidar = mask_lidar.detach().cpu().numpy()
            if mask_camera is not None and isinstance(mask_camera, torch.Tensor):
                mask_camera = mask_camera.detach().cpu().numpy()
            
            # Handle default masks
            if mask_lidar is None:
                mask_lidar = np.ones_like(gt_occ, dtype=bool)
            if mask_camera is None:
                mask_camera = np.ones_like(gt_occ, dtype=bool)
            
            # CRITICAL: Convert masks to bool dtype for proper boolean indexing
            # uint8 masks cause fancy indexing which explodes memory
            if mask_lidar is not None:
                mask_lidar = mask_lidar.astype(bool)
            if mask_camera is not None:
                mask_camera = mask_camera.astype(bool)
            
            # Add to metric accumulator
            self.miou_metric.add_batch(pred_occ, gt_occ, mask_lidar, mask_camera)
    
    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.
        
        Args:
            results: The processed results of each batch.
            
        Returns:
            Dict[str, float]: The computed metrics.
        """
        # Get per-class IoU
        mIoU = self.miou_metric.per_class_iu(self.miou_metric.hist)
        
        # Calculate mean IoU (excluding the 'free' class which is the last one)
        mIoU_mean = np.nanmean(mIoU[:self.num_classes-1]) * 100
        
        # Build result dictionary
        result_dict = {
            'mIoU': mIoU_mean
        }
        
        # Add per-class IoU
        for i, class_name in enumerate(self.miou_metric.class_names[:self.num_classes]):
            if i < len(mIoU):
                result_dict[f'IoU_{class_name}'] = mIoU[i] * 100
        
        return result_dict

