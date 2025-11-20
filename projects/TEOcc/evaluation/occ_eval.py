"""Occupancy evaluation metric for mmengine."""

from mmengine.evaluator import BaseMetric
from mmdet3d.registry import METRICS
import numpy as np
import os
import pickle
from .occ_metrics import Metric_mIoU, Metric_FScore


@METRICS.register_module()
class OccupancyMetric(BaseMetric):
    """Occupancy prediction evaluation metric.
    
    Args:
        num_classes (int): Number of classes. Default: 18.
        use_lidar_mask (bool): Whether to use lidar mask. Default: False.
        use_image_mask (bool): Whether to use image mask. Default: True.
        collect_device (str): Device name for collecting results. Default: 'cpu'.
        prefix (str): Prefix for metric names. Default: None.
        ann_file (str): Annotation file path for loading data_infos. Default: None.
    """
    
    def __init__(self,
                 num_classes=18,
                 use_lidar_mask=False,
                 use_image_mask=True,
                 collect_device='cpu',
                 prefix=None,
                 ann_file=None,
                 **kwargs):
        super().__init__(collect_device=collect_device, prefix=prefix)
        
        self.num_classes = num_classes
        self.use_lidar_mask = use_lidar_mask
        self.use_image_mask = use_image_mask
        self.ann_file = ann_file
        
        # Load data_infos from ann_file to access occ_path
        self.data_infos = None
        if ann_file and os.path.exists(ann_file):
            print(f"Loading data_infos from {ann_file}")
            with open(ann_file, 'rb') as f:
                data = pickle.load(f)
                # Handle different pkl formats
                if isinstance(data, dict) and 'data_list' in data:
                    self.data_infos = data['data_list']
                elif isinstance(data, dict) and 'infos' in data:
                    self.data_infos = data['infos']
                elif isinstance(data, list):
                    self.data_infos = data
                else:
                    print(f"Warning: Unknown data format in {ann_file}")
            
            # Dataset과 동일하게 timestamp 기준으로 정렬
            if self.data_infos:
                self.data_infos = list(sorted(self.data_infos, key=lambda e: e.get('timestamp', 0)))
            
            print(f"Loaded {len(self.data_infos) if self.data_infos else 0} data_infos")
        
        # Initialize metric calculator
        self.metric_calculator = Metric_mIoU(
            num_classes=num_classes,
            use_lidar_mask=use_lidar_mask,
            use_image_mask=use_image_mask
        )
        
        # Store sample index for matching predictions with ground truth
        self.sample_idx = 0
    
    def process(self, data_batch, data_samples):
        """Process one batch of data samples.
        
        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        # data_samples는 simple_test의 반환값 [occ_res]가 배치로 묶인 형태
        # 원본 코드처럼 각 occ_pred를 직접 처리
        for pred in data_samples:
            # pred는 [occ_res] 형태의 리스트이거나 occ_res 자체일 수 있음
            if isinstance(pred, list) and len(pred) > 0:
                # 리스트인 경우 첫 번째 요소 사용 (simple_test가 [occ_res] 반환)
                occ_pred = pred[0] if isinstance(pred[0], np.ndarray) else pred
            elif isinstance(pred, np.ndarray):
                occ_pred = pred
            else:
                continue
            
            # 원본 코드와 동일하게 numpy array인지 확인
            if not isinstance(occ_pred, np.ndarray):
                continue
            
            # Store prediction with current sample index
            result = {
                'semantics_pred': occ_pred,
                'sample_idx': self.sample_idx
            }
            
            self.sample_idx += 1
            self.results.append(result)
    
    def compute_metrics(self, results):
        """Compute the metrics from processed results.
        
        Args:
            results (list): The processed results of each batch.
        
        Returns:
            dict: The computed metrics including per-class IoU and mIoU.
        """
        if not self.data_infos:
            print("Warning: No data_infos loaded. Cannot evaluate without ground truth.")
            return {'num_samples': len(results), 'mIoU': 0.0}
        
        print('\nStarting Evaluation...')
        print(f"Evaluating {len(results)} predictions against ground truth...")
        
        # breakpoint()

        # Process all results
        loaded_count = 0
        for idx, result in enumerate(results):
            semantics_pred = result['semantics_pred']
            sample_idx = result.get('sample_idx', idx)
            
            # Load ground truth from data_infos like original code
            semantics_gt = None
            mask_lidar = None
            mask_camera = None
            
            try:
                # Get data info for this sample
                if sample_idx < len(self.data_infos):
                    info = self.data_infos[sample_idx]
                    occ_path = info.get('occ_path', None)
                    
                    # 원본 코드와 동일하게 경로 처리
                    if occ_path:
                        # 원본 코드와 정확히 동일: os.path.join(info['occ_path'], 'labels.npz')
                        labels_path = os.path.join(occ_path, 'labels.npz')
                        if os.path.exists(labels_path):
                            occ_gt = np.load(labels_path)
                            semantics_gt = occ_gt['semantics']
                            mask_lidar = occ_gt['mask_lidar'].astype(bool)
                            mask_camera = occ_gt['mask_camera'].astype(bool)
                            loaded_count += 1
                        else:
                            if idx < 3:
                                print(f"Warning: labels.npz not found at {labels_path}")
                    else:
                        if idx < 3:
                            print(f"Warning: occ_path not found for sample {sample_idx}")
            except Exception as e:
                if idx < 3:  # Only print first few errors
                    print(f"Warning: Failed to load ground truth for sample {sample_idx}: {e}")
            
            # If ground truth is available, compute metrics
            if semantics_gt is not None:
                # Add to metric calculator - following original code
                self.metric_calculator.add_batch(
                    semantics_pred,
                    semantics_gt,
                    mask_lidar,
                    mask_camera
                )
        
        print(f"Successfully loaded ground truth for {loaded_count}/{len(results)} samples")
        
        # Compute final metrics
        if self.metric_calculator.cnt > 0:
            # This will print per-class IoU like original code
            class_names, mIoU, cnt = self.metric_calculator.count_miou()
            
            # Build metrics dict
            metrics = {}
            for i, class_name in enumerate(class_names):
                if i < len(mIoU):
                    metrics[f'IoU_{class_name}'] = float(mIoU[i] * 100)
            
            # Compute mean IoU (excluding 'free' class which is index 17)
            metrics['mIoU'] = float(np.nanmean(mIoU[:self.num_classes-1]) * 100)
            metrics['num_samples'] = cnt
            
            return metrics
        else:
            print("Warning: No ground truth was loaded. Please check occ_path in data_infos.")
            return {'num_samples': len(results), 'mIoU': 0.0}

