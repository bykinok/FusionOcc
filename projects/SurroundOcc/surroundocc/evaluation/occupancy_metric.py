import numpy as np
import torch
from mmengine.evaluator import BaseMetric
from mmengine.registry import METRICS as ENGINE_METRICS
from mmdet3d.registry import METRICS as DET3D_METRICS
from typing import Dict, List, Optional, Sequence


@ENGINE_METRICS.register_module()
@DET3D_METRICS.register_module()
class OccupancyMetric(BaseMetric):
    """Occupancy prediction evaluation metric.
    
    Args:
        use_semantic (bool): Whether to evaluate semantic occupancy. Default: True.
        num_classes (int): Number of classes for semantic evaluation. Default: 17.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    """
    
    def __init__(self,
                 use_semantic: bool = True,
                 num_classes: int = 17,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.use_semantic = use_semantic
        self.num_classes = num_classes
        
    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.
        
        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        for data_sample in data_samples:
            pred_occ = data_sample.get('pred_occ', None)
            gt_occ = data_sample.get('gt_occ', None)
            
            if pred_occ is None or gt_occ is None:
                # Skip if no prediction or ground truth
                continue
                
            # Convert to numpy if needed
            if isinstance(pred_occ, torch.Tensor):
                pred_occ = pred_occ.cpu().numpy()
            if isinstance(gt_occ, torch.Tensor):
                gt_occ = gt_occ.cpu().numpy()
            
            # Calculate metrics
            result = self._calculate_metrics(pred_occ, gt_occ)
            self.results.append(result)
    
    def _calculate_metrics(self, pred_occ: np.ndarray, gt_occ: np.ndarray) -> Dict:
        """Calculate occupancy metrics.
        
        Args:
            pred_occ (np.ndarray): Predicted occupancy.
            gt_occ (np.ndarray): Ground truth occupancy.
            
        Returns:
            Dict: Dictionary containing calculated metrics.
        """
        result = {}
        
        if self.use_semantic:
            # Semantic occupancy evaluation
            # Assume pred_occ and gt_occ are class indices
            if pred_occ.shape != gt_occ.shape:
                # If shapes don't match, resize or handle accordingly
                pred_occ = self._resize_prediction(pred_occ, gt_occ.shape)
            
            # Calculate per-class IoU
            ious = []
            for class_id in range(self.num_classes):
                pred_mask = (pred_occ == class_id)
                gt_mask = (gt_occ == class_id)
                
                intersection = np.logical_and(pred_mask, gt_mask).sum()
                union = np.logical_or(pred_mask, gt_mask).sum()
                
                if union > 0:
                    iou = intersection / union
                else:
                    iou = float('nan')  # Undefined IoU for classes not present
                
                ious.append(iou)
            
            # Calculate mean IoU (excluding NaN values)
            valid_ious = [iou for iou in ious if not np.isnan(iou)]
            mean_iou = np.mean(valid_ious) if valid_ious else 0.0
            
            result['semantic_ious'] = ious
            result['semantic_miou'] = mean_iou
            
            # Overall accuracy
            accuracy = (pred_occ == gt_occ).sum() / gt_occ.size
            result['semantic_accuracy'] = accuracy
            
        else:
            # Binary occupancy evaluation
            # Convert to binary (occupied vs empty)
            pred_binary = (pred_occ > 0).astype(np.uint8)
            gt_binary = (gt_occ > 0).astype(np.uint8)
            
            # Calculate IoU for occupied voxels
            intersection = np.logical_and(pred_binary, gt_binary).sum()
            union = np.logical_or(pred_binary, gt_binary).sum()
            
            if union > 0:
                iou = intersection / union
            else:
                iou = 0.0
            
            result['binary_iou'] = iou
            
            # Calculate accuracy
            accuracy = (pred_binary == gt_binary).sum() / gt_binary.size
            result['binary_accuracy'] = accuracy
            
            # Calculate precision and recall
            true_positives = intersection
            false_positives = pred_binary.sum() - true_positives
            false_negatives = gt_binary.sum() - true_positives
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
            
            result['binary_precision'] = precision
            result['binary_recall'] = recall
        
        return result
    
    def _resize_prediction(self, pred_occ: np.ndarray, target_shape: tuple) -> np.ndarray:
        """Resize prediction to match ground truth shape."""
        # Simple nearest neighbor resizing
        # In practice, you might want to use more sophisticated interpolation
        from scipy.ndimage import zoom
        
        # Handle different dimensionalities
        if pred_occ.shape == target_shape:
            return pred_occ.astype(pred_occ.dtype)
        
        # If shapes have same number of dimensions, compute zoom factors directly
        if len(pred_occ.shape) == len(target_shape):
            zoom_factors = [target_shape[i] / pred_occ.shape[i] for i in range(len(target_shape))]
        else:
            # If different number of dimensions, only resize spatial dimensions
            # Assume last 3 dimensions are spatial (H, W, D)
            spatial_dims = min(3, len(pred_occ.shape), len(target_shape))
            zoom_factors = [1.0] * len(pred_occ.shape)  # Initialize with 1.0
            
            # Set zoom factors for spatial dimensions
            for i in range(spatial_dims):
                pred_dim_idx = len(pred_occ.shape) - spatial_dims + i
                target_dim_idx = len(target_shape) - spatial_dims + i
                zoom_factors[pred_dim_idx] = target_shape[target_dim_idx] / pred_occ.shape[pred_dim_idx]
        
        resized_pred = zoom(pred_occ, zoom_factors, order=0)  # Nearest neighbor
        
        return resized_pred.astype(pred_occ.dtype)
    
    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.
        
        Args:
            results (list): The processed results of each batch.
            
        Returns:
            Dict[str, float]: The computed metrics.
        """
        if not results:
            return {}
        
        metrics = {}
        
        if self.use_semantic:
            # Aggregate semantic metrics
            all_ious = []
            all_accuracy = []
            
            for result in results:
                if 'semantic_ious' in result:
                    all_ious.append(result['semantic_ious'])
                if 'semantic_accuracy' in result:
                    all_accuracy.append(result['semantic_accuracy'])
            
            if all_ious:
                # Calculate mean IoU across all samples
                all_ious = np.array(all_ious)
                mean_ious_per_class = np.nanmean(all_ious, axis=0)
                overall_miou = np.nanmean(mean_ious_per_class)
                
                metrics['semantic_mIoU'] = overall_miou
                
                # Per-class IoU
                for i, iou in enumerate(mean_ious_per_class):
                    if not np.isnan(iou):
                        metrics[f'semantic_IoU_class_{i}'] = iou
            
            if all_accuracy:
                metrics['semantic_accuracy'] = np.mean(all_accuracy)
                
        else:
            # Aggregate binary metrics
            binary_metrics = ['binary_iou', 'binary_accuracy', 'binary_precision', 'binary_recall']
            
            for metric_name in binary_metrics:
                values = [result[metric_name] for result in results if metric_name in result]
                if values:
                    metrics[metric_name] = np.mean(values)
        
        return metrics