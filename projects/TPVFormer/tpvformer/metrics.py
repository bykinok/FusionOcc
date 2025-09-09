import numpy as np
from mmengine.registry import METRICS

from mmdet3d.evaluation.metrics import SegMetric


@METRICS.register_module()
class OccupancyMetric(SegMetric):
    """Occupancy prediction evaluation metric.
    
    Compatible with tpv04_occupancy.py evaluation style.
    Computes mIoU, precision, recall, and class-wise IoU metrics.
    """
    
    def __init__(self, class_indices=None, ignore_label=0, **kwargs):
        super().__init__(**kwargs)
        self.metric_names = ['mIoU', 'IoU', 'Precision', 'Recall', 'class_IoU']
        self.class_indices = class_indices or list(range(18))  # Default: 18 classes
        self.ignore_label = ignore_label
        self.num_classes = len(self.class_indices)
    
    def process(self, data_batch, data_samples):
        """Process one batch of data samples.
        
        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (list): A batch of outputs from the model.
        """
        for data_sample in data_samples:
            # Get predictions and ground truth
            pred = data_sample.get('pred_occ_sem_seg', None)  # Occupancy prediction
            gt = data_sample.get('gt_pts_seg', {}).get('voxel_semantic_mask', None)  # Ground truth voxel labels
            
            # Convert to numpy if needed
            if hasattr(pred, 'cpu'):
                pred = pred.cpu().numpy()
            if hasattr(gt, 'cpu'):
                gt = gt.cpu().numpy()
            
            # Process the data
            self._process_single(pred, gt)
    
    def _process_single(self, pred, gt):
        """Process single sample.
        
        Args:
            pred (np.ndarray): Predicted occupancy grid.
            gt (np.ndarray): Ground truth occupancy grid.
        """
        # Ensure both have the same shape
        if pred.shape != gt.shape:
            # Resize prediction to match ground truth if needed
            pred = self._resize_prediction(pred, gt.shape)
        
        # Flatten for processing
        pred_flat = pred.flatten()
        gt_flat = gt.flatten()
        
        # Compute confusion matrix
        num_classes = max(pred_flat.max(), gt_flat.max()) + 1
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
        
        for i in range(len(pred_flat)):
            confusion_matrix[gt_flat[i], pred_flat[i]] += 1
        
        # Store confusion matrix for final computation
        if not hasattr(self, 'confusion_matrix'):
            self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
        
        self.confusion_matrix += confusion_matrix
    
    def _resize_prediction(self, pred, target_shape):
        """Resize prediction to target shape.
        
        Args:
            pred (np.ndarray): Prediction array.
            target_shape (tuple): Target shape.
            
        Returns:
            np.ndarray: Resized prediction.
        """
        # Simple nearest neighbor interpolation for now
        # In practice, you might want more sophisticated resizing
        from scipy.ndimage import zoom
        
        zoom_factors = [target_shape[i] / pred.shape[i] for i in range(len(pred.shape))]
        resized = zoom(pred, zoom_factors, order=0)
        
        return resized
    
    def compute_metrics(self, results):
        """Compute metrics compatible with tpv04_occupancy.py style.
        
        Args:
            results (list): List of processed results.
            
        Returns:
            dict: Dictionary of computed metrics.
        """
        if not hasattr(self, 'confusion_matrix'):
            return {}
        
        # Compute metrics from confusion matrix (tpv04 style)
        confusion_matrix = self.confusion_matrix
        
        # Compute IoU for each class (exclude ignore_label if exists)
        iou_per_class = []
        valid_classes = [i for i in range(confusion_matrix.shape[0]) if i != self.ignore_label]
        
        for i in valid_classes:
            # True positives
            tp = confusion_matrix[i, i]
            # False positives
            fp = confusion_matrix[:, i].sum() - tp
            # False negatives
            fn = confusion_matrix[i, :].sum() - tp
            
            # IoU (same as tpv04 MeanIoU calculation)
            if tp + fp + fn > 0:
                iou = tp / (tp + fp + fn)
            else:
                iou = 0.0
            iou_per_class.append(iou)
        
        # Compute tpv04-style metrics
        mIoU = np.mean(iou_per_class)  # mIoU SSC
        
        # Overall precision and recall (non-empty classes only)
        valid_confusion = confusion_matrix[1:, 1:]  # Exclude background/empty
        total_tp = np.sum(valid_confusion.diagonal())
        total_pred = np.sum(valid_confusion) 
        total_gt = np.sum(confusion_matrix[1:, :])
        
        precision = total_tp / total_pred if total_pred > 0 else 0.0
        recall = total_tp / total_gt if total_gt > 0 else 0.0
        
        # Overall IoU (completion IoU)
        overall_iou = total_tp / (total_pred + total_gt - total_tp) if (total_pred + total_gt - total_tp) > 0 else 0.0
        
        # Prepare results (tpv04 compatible format)
        metrics = {
            'mIoU': mIoU * 100,  # Convert to percentage like tpv04
            'IoU': overall_iou * 100,
            'Precision': precision * 100,
            'Recall': recall * 100,
            'class_IoU': [iou * 100 for iou in iou_per_class]
        }
        
        return metrics
    
    def get_class_names(self):
        """Get class names for the dataset.
        
        Returns:
            list: List of class names.
        """
        # Default class names for occupancy prediction
        return [
            'noise', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
            'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
            'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
            'vegetation', 'occupied'
        ]
