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
        # breakpoint()

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
                
                # ★ data_sample 최상위 레벨에서도 확인 (config의 meta_keys가 top-level로 추가되는 경우)
                if gt_pts_labels is None and 'pts_semantic_mask' in data_sample:
                    gt_pts_labels = data_sample.get('pts_semantic_mask')
                if voxel_coords is None and 'voxel_coords' in data_sample:
                    voxel_coords = data_sample.get('voxel_coords')
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
                
                # ★ data_sample 최상위 속성에서도 확인 (object form)
                if gt_pts_labels is None and hasattr(data_sample, 'pts_semantic_mask'):
                    gt_pts_labels = data_sample.pts_semantic_mask
                if voxel_coords is None and hasattr(data_sample, 'voxel_coords'):
                    voxel_coords = data_sample.voxel_coords
            
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
        # breakpoint()
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

            # Print per-class IoU for points (원본과 동일)
            print("Validation per class iou pts:", flush=True)
            class_names = [
                'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
                'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
                'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
                'vegetation'
            ]
            for class_name in class_names:
                iou_value = metrics_pts.get(f'{class_name}_pts', 0.0)
                print(f"{class_name} : {iou_value:.2f}%", flush=True)
        
        # Compute metrics for voxel-wise evaluation
        if vox_sum > 0:
            metrics_vox = self._compute_metrics_from_confusion_matrix(
                self.confusion_matrix_vox, suffix='_vox')
            metrics.update(metrics_vox)

            # Print per-class IoU for voxels (원본과 동일)
            print("Validation per class iou vox:", flush=True)
            class_names = [
                'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
                'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
                'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
                'vegetation'
            ]
            for class_name in class_names:
                iou_value = metrics_vox.get(f'{class_name}_vox', 0.0)
                print(f"{class_name} : {iou_value:.2f}%", flush=True)
            
        # Print mIoU summary (원본과 동일)
        if pts_sum > 0:
            print(f"Current val miou pts is {metrics.get('mIoU_pts', 0.0):.3f}", flush=True)
        if vox_sum > 0:
            print(f"Current val miou vox is {metrics.get('mIoU_vox', 0.0):.3f}", flush=True)
        
        print("="*80, flush=True)

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
        
        # Compute IoU for each class (exclude ignore_label = 0)
        iou_per_class = []
        valid_classes = self.class_indices  # 원본과 동일하게 class_indices 사용
        
        for i in valid_classes:
            # total_seen = sum(targets == c) = GT에서 클래스 c의 개수
            total_seen = confusion_matrix[i, :].sum()
            # total_correct = sum((targets == c) & (outputs == c)) = TP
            total_correct = confusion_matrix[i, i]
            # total_positive = sum(outputs == c) = 예측에서 클래스 c의 개수
            total_positive = confusion_matrix[:, i].sum()
            
            # 원본과 동일: seen=0이면 IoU=1 (100%)
            if total_seen == 0:
                iou = 1.0
            else:
                # IoU = TP / (seen + positive - correct)
                # 원본 라인 48-50과 동일
                iou = total_correct / (total_seen + total_positive - total_correct)
            
            iou_per_class.append(iou * 100)  # Convert to percentage
        
        # Compute mean IoU
        miou = np.mean(iou_per_class)
        
        # Prepare metrics dictionary
        metrics = {f'mIoU{suffix}': miou}
        
        # Add per-class IoU
        for class_name, iou_value in zip(class_names, iou_per_class):
            metrics[f'{class_name}{suffix}'] = iou_value
        
        return metrics
