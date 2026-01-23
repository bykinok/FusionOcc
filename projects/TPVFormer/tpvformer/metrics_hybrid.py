"""Hybrid OccupancyMetric supporting both occ3d and traditional GT formats."""
import numpy as np
import torch
from mmengine.registry import METRICS
from mmengine.evaluator import BaseMetric
from typing import Optional, Dict, Sequence


@METRICS.register_module()
class OccupancyMetricHybrid(BaseMetric):
    """Hybrid Occupancy Metric supporting multiple GT formats.
    
    Supports:
    - Traditional GT (point-wise, voxel-wise): dataset_name=None
    - occ3d format: dataset_name='occ3d'
    - openocc format: dataset_name='openocc' (future)
    
    This metric automatically delegates to the appropriate metric based on
    dataset_name configuration.
    """
    
    def __init__(self, 
                 dataset_name: Optional[str] = None,
                 class_indices=None, 
                 ignore_label=0,
                 num_classes: int = 18,
                 use_lidar_mask: bool = False,
                 use_image_mask: bool = False,
                 ann_file: Optional[str] = None,
                 data_root: Optional[str] = None,
                 eval_metric: str = 'miou',
                 sort_by_timestamp: bool = False,
                 collect_device='cpu', 
                 prefix=None):
        super().__init__(collect_device=collect_device, prefix=prefix)
        
        self.dataset_name = dataset_name
        self.use_occ3d = (dataset_name == 'occ3d')
        self.use_openocc = (dataset_name == 'openocc')
        
        if self.use_occ3d or self.use_openocc:
            # Use STCOcc's metric for occ3d/openocc formats
            try:
                from projects.STCOcc.stcocc.evaluation.occupancy_metric import OccupancyMetric as STCOccMetric
                self.stcocc_metric = STCOccMetric(
                    num_classes=num_classes,
                    use_lidar_mask=use_lidar_mask,
                    use_image_mask=use_image_mask if self.use_occ3d else False,
                    dataset_name=dataset_name,
                    ann_file=ann_file,
                    data_root=data_root,
                    eval_metric=eval_metric,
                    sort_by_timestamp=sort_by_timestamp,
                    collect_device=collect_device,
                    prefix=prefix
                )
                self.mode = 'stcocc'
            except ImportError:
                print("Warning: STCOcc metric not available, falling back to traditional metric")
                self.mode = 'traditional'
                self._init_traditional_metric(class_indices, ignore_label)
        else:
            # Use traditional TPVFormer metric
            self.mode = 'traditional'
            self._init_traditional_metric(class_indices, ignore_label)
    
    def _init_traditional_metric(self, class_indices, ignore_label):
        """Initialize traditional metric components."""
        self.class_indices = class_indices or list(range(18))
        self.ignore_label = ignore_label
        self.num_classes = 18
        self.reset()
    
    def reset(self):
        """Reset evaluation metrics."""
        if self.mode == 'traditional':
            self.confusion_matrix_pts = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
            self.confusion_matrix_vox = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        elif self.mode == 'stcocc':
            self.stcocc_metric.reset()
    
    def process(self, data_batch, data_samples):
        """Process one batch of data samples.
        
        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence): A batch of outputs from the model.
        """
        if self.mode == 'stcocc':
            # Delegate to STCOcc metric
            self.stcocc_metric.process(data_batch, data_samples)
        else:
            # Traditional metric processing
            self._process_traditional(data_batch, data_samples)
    
    def _process_traditional(self, data_batch, data_samples):
        """Process traditional GT format."""
        for idx, data_sample in enumerate(data_samples):
            self.results.append({})
            
            # Get predictions
            if isinstance(data_sample, dict):
                pred_logits_vox = data_sample.get('pred_logits_vox', None)
                pred_logits_pts = data_sample.get('pred_logits_pts', None)
            else:
                pred_logits_vox = getattr(data_sample, 'pred_logits_vox', None)
                pred_logits_pts = getattr(data_sample, 'pred_logits_pts', None)
            
            # Get ground truth
            gt_pts_labels = None
            voxel_coords = None
            
            if isinstance(data_sample, dict):
                if 'gt_pts_seg' in data_sample:
                    gt_pts_seg = data_sample['gt_pts_seg']
                    if isinstance(gt_pts_seg, dict):
                        gt_pts_labels = gt_pts_seg.get('pts_semantic_mask')
                        voxel_coords = gt_pts_seg.get('voxel_coords')
                    elif hasattr(gt_pts_seg, 'pts_semantic_mask'):
                        gt_pts_labels = gt_pts_seg.pts_semantic_mask
                        voxel_coords = getattr(gt_pts_seg, 'voxel_coords', None)
                
                if gt_pts_labels is None and 'pts_semantic_mask' in data_sample:
                    gt_pts_labels = data_sample.get('pts_semantic_mask')
                if voxel_coords is None and 'voxel_coords' in data_sample:
                    voxel_coords = data_sample.get('voxel_coords')
            else:
                if hasattr(data_sample, 'gt_pts_seg'):
                    gt_pts_seg = data_sample.gt_pts_seg
                    if hasattr(gt_pts_seg, 'pts_semantic_mask'):
                        gt_pts_labels = gt_pts_seg.pts_semantic_mask
                    if hasattr(gt_pts_seg, 'voxel_coords'):
                        voxel_coords = gt_pts_seg.voxel_coords
                
                if gt_pts_labels is None and hasattr(data_sample, 'pts_semantic_mask'):
                    gt_pts_labels = data_sample.pts_semantic_mask
                if voxel_coords is None and hasattr(data_sample, 'voxel_coords'):
                    voxel_coords = data_sample.voxel_coords
            
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
            
            # Voxel-wise evaluation
            if pred_logits_vox is not None and voxel_coords is not None:
                if hasattr(pred_logits_vox, 'cpu'):
                    pred_logits_vox = pred_logits_vox.cpu()
                if hasattr(voxel_coords, 'cpu'):
                    voxel_coords_np = voxel_coords.cpu().numpy()
                elif isinstance(voxel_coords, torch.Tensor):
                    voxel_coords_np = voxel_coords.numpy()
                else:
                    voxel_coords_np = voxel_coords
                
                pred_vox_full = torch.argmax(pred_logits_vox, dim=0)
                voxel_coords_int = voxel_coords_np.astype(np.int64)
                pred_vox_at_pts = pred_vox_full[
                    voxel_coords_int[:, 0], 
                    voxel_coords_int[:, 1], 
                    voxel_coords_int[:, 2]
                ].numpy()
                gt_pts = gt_pts_labels.squeeze() if gt_pts_labels.ndim > 1 else gt_pts_labels
                self._update_confusion_matrix(pred_vox_at_pts, gt_pts, self.confusion_matrix_vox)
    
    def _update_confusion_matrix(self, pred, gt, confusion_matrix):
        """Update confusion matrix."""
        pred = pred.flatten()
        gt = gt.flatten()
        mask = (gt != self.ignore_label)
        pred = pred[mask]
        gt = gt[mask]
        for i in range(len(pred)):
            if gt[i] < self.num_classes and pred[i] < self.num_classes:
                confusion_matrix[gt[i], pred[i]] += 1
    
    def evaluate(self, size: int):
        """Override evaluate to handle both traditional and stcocc modes.
        
        For traditional mode:
        1. process() directly accumulates confusion matrices on each rank
        2. We need to aggregate confusion matrices from all ranks before computing metrics
        3. all_reduce must be called on ALL ranks (not just rank 0)
        
        For stcocc mode:
        1. process() delegates to self.stcocc_metric.process()
        2. But mmengine only collects OccupancyMetricHybrid's self.results, not self.stcocc_metric.results
        3. So we need to manually call stcocc_metric.evaluate() which handles its own collection
        
        Args:
            size (int): Length of the dataset.
            
        Returns:
            dict: Computed metrics.
        """
        import torch.distributed as dist
        
        if self.mode == 'stcocc':
            # For STCOcc mode, directly call the internal metric's evaluate()
            # This allows it to handle its own result collection and computation
            return self.stcocc_metric.evaluate(size)
        else:
            # Traditional mode: aggregate confusion matrices
            if dist.is_available() and dist.is_initialized():
                backend = dist.get_backend()
                device = torch.device('cuda' if backend == 'nccl' else 'cpu')
                
                # Convert to tensor for all_reduce
                cm_pts_tensor = torch.from_numpy(self.confusion_matrix_pts.astype(np.int64)).to(device)
                cm_vox_tensor = torch.from_numpy(self.confusion_matrix_vox.astype(np.int64)).to(device)
                
                # All-reduce: sum confusion matrices from all processes
                dist.all_reduce(cm_pts_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(cm_vox_tensor, op=dist.ReduceOp.SUM)
                
                # Convert back to numpy
                self.confusion_matrix_pts = cm_pts_tensor.cpu().numpy()
                self.confusion_matrix_vox = cm_vox_tensor.cpu().numpy()
                
                # Clear tensors
                del cm_pts_tensor, cm_vox_tensor
                if backend == 'nccl':
                    torch.cuda.empty_cache()
            
            # Call parent's evaluate() which will call compute_metrics() on rank 0
            return super().evaluate(size)
    
    def compute_metrics(self, results):
        """Compute metrics from processed results."""
        if self.mode == 'stcocc':
            return self.stcocc_metric.compute_metrics(results)
        else:
            return self._compute_traditional_metrics(results)
    
    def _compute_traditional_metrics(self, results):
        """Compute traditional metrics."""
        metrics = {}
        
        pts_sum = np.sum(self.confusion_matrix_pts)
        vox_sum = np.sum(self.confusion_matrix_vox)
        
        if pts_sum > 0:
            metrics_pts = self._compute_metrics_from_confusion_matrix(
                self.confusion_matrix_pts, suffix='_pts')
            metrics.update(metrics_pts)
        
        if vox_sum > 0:
            metrics_vox = self._compute_metrics_from_confusion_matrix(
                self.confusion_matrix_vox, suffix='_vox')
            metrics.update(metrics_vox)
        
        return metrics
    
    def _compute_metrics_from_confusion_matrix(self, confusion_matrix, suffix=''):
        """Compute IoU metrics from confusion matrix."""
        class_names = [
            'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
            'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
            'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
            'vegetation'
        ]
        
        iou_per_class = []
        valid_classes = self.class_indices
        
        for i in valid_classes:
            total_seen = confusion_matrix[i, :].sum()
            total_correct = confusion_matrix[i, i]
            total_positive = confusion_matrix[:, i].sum()
            
            if total_seen == 0:
                iou = 1.0
            else:
                iou = total_correct / (total_seen + total_positive - total_correct)
            
            iou_per_class.append(iou * 100)
        
        miou = np.mean(iou_per_class)
        metrics = {f'mIoU{suffix}': miou}
        
        for class_name, iou_value in zip(class_names, iou_per_class):
            metrics[f'{class_name}{suffix}'] = iou_value
        
        return metrics

