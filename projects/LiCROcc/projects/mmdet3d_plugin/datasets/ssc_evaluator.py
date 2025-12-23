"""SSC Evaluator for MMEngine 2.x."""

import os.path as osp
import pickle
from typing import List, Optional, Sequence

from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger
from mmdet3d.registry import METRICS


@METRICS.register_module()
class SSCEvaluator(BaseMetric):
    """Evaluator for Semantic Scene Completion (SSC) task.
    
    This evaluator calculates:
    - Scene Completion (SC) metrics: Precision, Recall, IoU
    - Semantic Scene Completion (SSC) metrics: mIoU, per-class IoU
    
    Args:
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (Optional[str]): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Default: None.
    """
    
    default_prefix: Optional[str] = 'SSC'
    
    def __init__(self,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 save_results: bool = True,
                 out_file_path: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.save_results = save_results
        self.out_file_path = out_file_path
    
    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.
        
        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.
        
        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        import torch
        from ..ssc_rs.utils.ssc_metric import SSCMetrics
        
        for data_sample in data_samples:
            result = dict()
            
            # Check if this is already processed results
            if 'ssc_results' in data_sample:
                result['ssc_results'] = data_sample['ssc_results']
            elif 'ssc_scores' in data_sample:
                result['ssc_scores'] = data_sample['ssc_scores']
            # Otherwise, compute from output_voxels and target_voxels
            elif 'output_voxels' in data_sample and 'target_voxels' in data_sample:
                output_voxels = data_sample['output_voxels']
                target_voxels = data_sample['target_voxels']
                
                # Convert to predictions (argmax over class dimension)
                if output_voxels.dim() == 5:  # (B, C, X, Y, Z)
                    pred = torch.argmax(output_voxels, dim=1)
                else:
                    pred = output_voxels
                
                # Compute SSC metrics for this sample
                ssc_metric = SSCMetrics(n_classes=17)
                if pred.is_cuda:
                    ssc_metric = ssc_metric.cuda()
                
                # compute_single returns tuple: (tp, fp, fn, tp_sum, fp_sum, fn_sum)
                ssc_results_i = ssc_metric.compute_single(
                    y_pred=pred,
                    y_true=target_voxels
                )
                result['ssc_results'] = ssc_results_i
            
            self.results.append(result)
    
    def compute_metrics(self, results: List[dict]) -> dict:
        """Compute the metrics from processed results.
        
        Args:
            results (list): The processed results of each batch.
        
        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        
        # Save results to file if requested
        if self.save_results and self.out_file_path:
            logger.info(f'Saving results to {self.out_file_path}')
            with open(self.out_file_path, 'wb') as f:
                pickle.dump(results, f)
        
        # Compute metrics
        eval_results = {}
        
        # Check if we have ssc_scores (already computed) or ssc_results (need to compute)
        if results and 'ssc_scores' in results[0]:
            # Scores are already computed, just extract them
            ssc_scores = results[0]['ssc_scores']
            
            class_ssc_iou = ssc_scores['iou_ssc'].tolist()
            res_dic = {
                "SC_Precision": ssc_scores['precision'].item(),
                "SC_Recall": ssc_scores['recall'].item(),
                "SC_IoU": ssc_scores['iou'],
                "SSC_mIoU": ssc_scores['iou_ssc_mean'],
            }
            
            # Add per-class IoU (use default class names)
            class_names = [
                'empty', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
                'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
                'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
                'vegetation'
            ]
            
            for name, iou in zip(class_names, class_ssc_iou):
                res_dic[f"SSC_{name}_IoU"] = iou
                
        elif results and 'ssc_results' in results[0]:
            # Need to aggregate results from all samples
            import numpy as np
            
            ssc_results = [r['ssc_results'] for r in results]
            
            completion_tp = sum([x[0] for x in ssc_results])
            completion_fp = sum([x[1] for x in ssc_results])
            completion_fn = sum([x[2] for x in ssc_results])
            
            tps = sum([x[3] for x in ssc_results])
            fps = sum([x[4] for x in ssc_results])
            fns = sum([x[5] for x in ssc_results])
            
            precision = completion_tp / (completion_tp + completion_fp)
            recall = completion_tp / (completion_tp + completion_fn)
            iou = completion_tp / (completion_tp + completion_fp + completion_fn)
            iou_ssc = tps / (tps + fps + fns + 1e-5)
            
            class_ssc_iou = iou_ssc.tolist() if hasattr(iou_ssc, 'tolist') else iou_ssc
            
            res_dic = {
                "SC_Precision": float(precision),
                "SC_Recall": float(recall),
                "SC_IoU": float(iou),
                "SSC_mIoU": float(iou_ssc[1:].mean()),
            }
            
            # Add per-class IoU
            class_names = [
                'empty', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
                'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
                'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
                'vegetation'
            ]
            
            for name, iou_val in zip(class_names, class_ssc_iou):
                res_dic[f"SSC_{name}_IoU"] = float(iou_val)
        else:
            logger.warning('No SSC results or scores found in data samples!')
            return eval_results
        
        # Convert to percentage and add to eval_results with prefix
        for key, val in res_dic.items():
            eval_results[key] = round(val * 100, 2)
        
        # Add combined metric
        eval_results['combined_IoU'] = eval_results['SC_IoU'] + eval_results['SSC_mIoU']
        
        # Log results
        logger.info('SSC Evaluation Results:')
        logger.info(f"SC_Precision: {eval_results['SC_Precision']:.2f}")
        logger.info(f"SC_Recall: {eval_results['SC_Recall']:.2f}")
        logger.info(f"SC_IoU: {eval_results['SC_IoU']:.2f}")
        logger.info(f"SSC_mIoU: {eval_results['SSC_mIoU']:.2f}")
        logger.info(f"Combined_IoU: {eval_results['combined_IoU']:.2f}")
        
        return eval_results

