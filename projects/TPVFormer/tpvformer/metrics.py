import numpy as np
import torch
from mmengine.registry import METRICS
from mmengine.evaluator import BaseMetric


@METRICS.register_module()
class OccupancyMetric(BaseMetric):
    """Occupancy prediction evaluation metric.
    
    Compatible with TPVFormer original eval.py evaluation style.
    Evaluates both point-wise and voxel-wise predictions.
    """
    
    def __init__(self, class_indices=None, ignore_label=0, collect_device='cpu', prefix=None):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.class_indices = class_indices or list(range(18))  # Default: 18 classes
        self.ignore_label = ignore_label
        self.num_classes = 18  # Fixed for NuScenes occupancy
        
        # Initialize confusion matrices for both pts and vox evaluation
        self.reset()
    
    def reset(self):
        """Reset evaluation metrics."""
        self.confusion_matrix_pts = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        self.confusion_matrix_vox = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
    
    def process(self, data_batch, data_samples):
        """Process one batch of data samples.
        
        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (list): A batch of outputs from the model.
        """
        for idx, data_sample in enumerate(data_samples):
            # Add result to satisfy BaseMetric requirements (must be done for each sample)
            self.results.append({})
            
            # Get voxel and point predictions (if available)
            # Handle both dict and object forms
            if isinstance(data_sample, dict):
                pred_logits_vox = data_sample.get('pred_logits_vox', None)
                pred_logits_pts = data_sample.get('pred_logits_pts', None)
            else:
                pred_logits_vox = getattr(data_sample, 'pred_logits_vox', None)
                pred_logits_pts = getattr(data_sample, 'pred_logits_pts', None)
            
            # Debug: First sample only
            if len(self.results) == 1:
                print(f"  - has pred_logits_vox: {pred_logits_vox is not None}")
                print(f"  - has pred_logits_pts: {pred_logits_pts is not None}")
                if pred_logits_vox is not None:
                    print(f"  - pred_logits_vox shape: {pred_logits_vox.shape}")
                    # Check predicted classes
                    pred_classes = torch.argmax(pred_logits_vox, dim=0)
                    print(f"  - pred_classes unique: {torch.unique(pred_classes).tolist()}")
                if pred_logits_pts is not None:
                    print(f"  - pred_logits_pts shape: {pred_logits_pts.shape}")
            
            # Get ground truth point labels and voxel coordinates
            gt_pts_labels = None
            voxel_coords = None
            
            # Handle both dict and object forms of data_sample
            if isinstance(data_sample, dict):
                # Dict form (after serialization in distributed training)
                if 'gt_pts_seg' in data_sample:
                    gt_pts_seg = data_sample['gt_pts_seg']
                    
                    # Get from dict or object
                    if isinstance(gt_pts_seg, dict):
                        gt_pts_labels = gt_pts_seg.get('pts_semantic_mask')
                        voxel_coords = gt_pts_seg.get('voxel_coords')
                    elif hasattr(gt_pts_seg, 'pts_semantic_mask'):
                        gt_pts_labels = gt_pts_seg.pts_semantic_mask
                        voxel_coords = getattr(gt_pts_seg, 'voxel_coords', None)
            else:
                # Object form (Det3DDataSample)
                if hasattr(data_sample, 'gt_pts_seg'):
                    gt_pts_seg = data_sample.gt_pts_seg
                    
                    # Get point-level GT labels
                    if hasattr(gt_pts_seg, 'pts_semantic_mask'):
                        gt_pts_labels = gt_pts_seg.pts_semantic_mask
                    elif isinstance(gt_pts_seg, dict) and 'pts_semantic_mask' in gt_pts_seg:
                        gt_pts_labels = gt_pts_seg['pts_semantic_mask']
                    
                    # Get voxel coordinates (needed for vox evaluation)
                    if hasattr(gt_pts_seg, 'voxel_coords'):
                        voxel_coords = gt_pts_seg.voxel_coords
                    elif isinstance(gt_pts_seg, dict) and 'voxel_coords' in gt_pts_seg:
                        voxel_coords = gt_pts_seg['voxel_coords']
            
            # Debug: First sample only
            if len(self.results) == 1:
                print(f"  - has gt_pts_labels: {gt_pts_labels is not None}")
                print(f"  - has voxel_coords: {voxel_coords is not None}")
                if gt_pts_labels is not None:
                    gt_shape = gt_pts_labels.shape if hasattr(gt_pts_labels, 'shape') else 'N/A'
                    print(f"  - gt_pts_labels shape: {gt_shape}")
                    if hasattr(gt_pts_labels, 'cpu'):
                        gt_unique = torch.unique(gt_pts_labels).tolist()
                    else:
                        gt_unique = np.unique(gt_pts_labels).tolist()
                    print(f"  - gt_pts_labels unique: {gt_unique}")
                if voxel_coords is not None:
                    print(f"  - voxel_coords shape: {voxel_coords.shape if hasattr(voxel_coords, 'shape') else 'N/A'}")
            
            # Skip if we don't have minimum required data
            if gt_pts_labels is None:
                continue
            
            # Convert to numpy
            if hasattr(gt_pts_labels, 'cpu'):
                gt_pts_labels = gt_pts_labels.cpu().numpy()
            
            # Point-wise evaluation
            if pred_logits_pts is not None:
                if hasattr(pred_logits_pts, 'cpu'):
                    pred_logits_pts = pred_logits_pts.cpu()
                pred_pts = torch.argmax(pred_logits_pts.squeeze(-1).squeeze(-1), dim=0).numpy()
                gt_pts = gt_pts_labels.squeeze() if gt_pts_labels.ndim > 1 else gt_pts_labels
                
                self._update_confusion_matrix(pred_pts, gt_pts, self.confusion_matrix_pts)
            
            # Voxel-wise evaluation (sample voxel predictions at point locations)
            if pred_logits_vox is not None and voxel_coords is not None:
                if hasattr(pred_logits_vox, 'cpu'):
                    pred_logits_vox = pred_logits_vox.cpu()
                if hasattr(voxel_coords, 'cpu'):
                    voxel_coords_np = voxel_coords.cpu().numpy()
                elif isinstance(voxel_coords, torch.Tensor):
                    voxel_coords_np = voxel_coords.numpy()
                else:
                    voxel_coords_np = voxel_coords
                
                # Get voxel predictions at point locations (following original eval.py line 167)
                pred_vox_full = torch.argmax(pred_logits_vox, dim=0)  # [W, H, Z]
                
                # Voxel coordinates should be [N, 3] with order [w, h, z]
                voxel_coords_int = voxel_coords_np.astype(np.int64)
                
                # Sample voxel predictions at point locations
                pred_vox_at_pts = pred_vox_full[voxel_coords_int[:, 0], voxel_coords_int[:, 1], voxel_coords_int[:, 2]].numpy()
                gt_pts = gt_pts_labels.squeeze() if gt_pts_labels.ndim > 1 else gt_pts_labels
                
                self._update_confusion_matrix(pred_vox_at_pts, gt_pts, self.confusion_matrix_vox)
    
    def _update_confusion_matrix(self, pred, gt, confusion_matrix):
        """Update confusion matrix with predictions and ground truth.
        
        Args:
            pred (np.ndarray): Predicted labels (1D array).
            gt (np.ndarray): Ground truth labels (1D array).
            confusion_matrix (np.ndarray): Confusion matrix to update.
        """
        # Ensure 1D arrays
        pred = pred.flatten()
        gt = gt.flatten()
        
        # Filter out ignore labels
        mask = (gt != self.ignore_label)
        pred = pred[mask]
        gt = gt[mask]
        
        # Update confusion matrix (following original MeanIoU._after_step)
        for i in range(len(pred)):
            if gt[i] < self.num_classes and pred[i] < self.num_classes:
                confusion_matrix[gt[i], pred[i]] += 1
    
    def compute_metrics(self, results):
        """Compute metrics compatible with original eval.py style.
        
        Args:
            results (list): List of processed results (not used, we use accumulated confusion matrices).
            
        Returns:
            dict: Dictionary of computed metrics.
        """
        metrics = {}
        
        # Debug: Print confusion matrix sums
        pts_sum = np.sum(self.confusion_matrix_pts)
        vox_sum = np.sum(self.confusion_matrix_vox)
        
        # Compute metrics for point-wise evaluation
        if pts_sum > 0:
            metrics_pts = self._compute_metrics_from_confusion_matrix(
                self.confusion_matrix_pts, suffix='_pts')
            metrics.update(metrics_pts)
        
        # Compute metrics for voxel-wise evaluation
        if vox_sum > 0:
            metrics_vox = self._compute_metrics_from_confusion_matrix(
                self.confusion_matrix_vox, suffix='_vox')
            metrics.update(metrics_vox)
        
        return metrics
    
    def _compute_metrics_from_confusion_matrix(self, confusion_matrix, suffix=''):
        """Compute IoU metrics from confusion matrix (following original MeanIoU._after_epoch).
        
        Args:
            confusion_matrix (np.ndarray): Confusion matrix [num_classes, num_classes].
            suffix (str): Suffix to add to metric names.
            
        Returns:
            dict: Computed metrics.
        """
        # Class names (excluding ignore_label)
        class_names = [
            'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
            'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
            'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
            'vegetation'
        ]
        
        # Compute IoU for each class (exclude ignore_label)
        iou_per_class = []
        # Use class_indices from config (exclude ignore_label)
        valid_classes = [c for c in self.class_indices if c != self.ignore_label]
        
        for i in valid_classes:
            # True positives
            tp = confusion_matrix[i, i]
            # False positives
            fp = confusion_matrix[:, i].sum() - tp
            # False negatives
            fn = confusion_matrix[i, :].sum() - tp
            
            # Total GT occurrences of class i
            total_seen = tp + fn
            
            # IoU (following original MeanIoU calculation exactly)
            if total_seen == 0:
                # Original logic: if class not in GT, treat as 100% (perfect)
                iou = 1.0
            elif tp + fp + fn > 0:
                iou = tp / (tp + fp + fn)
            else:
                iou = 0.0
            iou_per_class.append(iou * 100)  # Convert to percentage
        
        # Compute mean IoU
        miou = np.mean(iou_per_class)
        
        # Prepare metrics dictionary
        metrics = {f'mIoU{suffix}': miou}
        
        # Add per-class IoU
        for class_name, iou_value in zip(class_names, iou_per_class):
            metrics[f'{class_name}{suffix}'] = iou_value
        
        return metrics
