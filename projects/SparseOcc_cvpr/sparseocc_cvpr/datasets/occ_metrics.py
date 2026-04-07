"""NuScenesOccMetric: mmengine BaseMetric 인터페이스로 NuScenes occupancy 평가를 수행합니다.

기존 mmdet3d의 dataset.evaluate() 방식을 mmengine Evaluator 패턴으로 래핑합니다.
"""

import numpy as np
import torch
import torch.nn.functional as F

from mmengine.evaluator import BaseMetric

try:
    from mmdet3d.registry import METRICS
except ImportError:
    from mmengine.registry import METRICS


@METRICS.register_module()
class NuScenesOccMetric(BaseMetric):
    """NuScenes 3D Occupancy 평가 메트릭.

    SSC (Semantic Scene Completion) 기준으로 mIoU를 계산합니다.
    """

    CLASS_NAMES = [
        'empty', 'barrier', 'bicycle', 'bus', 'car',
        'construction_vehicle', 'motorcycle', 'pedestrian',
        'traffic_cone', 'trailer', 'truck', 'driveable_surface',
        'other_flat', 'sidewalk', 'terrain', 'manmade', 'vegetation',
    ]

    def __init__(self, class_names=None, collect_device='cpu', prefix=None, **kwargs):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.class_names = class_names or self.CLASS_NAMES
        self.n_classes = len(self.class_names)

    def process(self, data_batch, data_samples):
        """배치 예측 결과를 받아 per-sample SSC 결과를 축적합니다.

        Args:
            data_batch: 원본 입력 배치 (미사용).
            data_samples: val_step 반환 값 (forward_test 출력 dict 또는 리스트).
        """
        if data_samples is None:
            return

        # forward_test가 dict 또는 list를 반환할 수 있음
        if isinstance(data_samples, dict):
            samples_list = [data_samples]
        elif isinstance(data_samples, (list, tuple)):
            samples_list = data_samples
        else:
            return

        for sample in samples_list:
            if not isinstance(sample, dict):
                continue

            output_voxels = sample.get('output_voxels', None)
            target_voxels = sample.get('target_voxels', None)
            output_voxels_refine = sample.get('output_voxel_refine', None)
            evaluation_semantic = sample.get('evaluation_semantic', None)

            result = {}

            if evaluation_semantic is not None:
                # lidar-seg 평가
                result['evaluation_semantic'] = (
                    evaluation_semantic.cpu().numpy()
                    if isinstance(evaluation_semantic, torch.Tensor)
                    else np.array(evaluation_semantic)
                )
            elif output_voxels is not None and target_voxels is not None:
                # SSC 평가
                ssc = self._compute_ssc_single(output_voxels, target_voxels)
                result['ssc_result'] = ssc

                if output_voxels_refine is not None:
                    ssc_refine = self._compute_ssc_single(
                        output_voxels_refine, target_voxels)
                    result['ssc_result_refine'] = ssc_refine

            self.results.append(result)

    def _compute_ssc_single(self, output_voxels, target_voxels):
        """단일 샘플에 대한 SSC TP/FP/FN 계산.

        Args:
            output_voxels: [C, H, W, D] 또는 [1, C, H, W, D] 로짓 텐서.
            target_voxels: [H, W, D] 또는 [1, H, W, D] GT 텐서.

        Returns:
            tuple: (completion_tp, completion_fp, completion_fn, tps, fps, fns)
        """
        # 텐서로 변환
        if not isinstance(output_voxels, torch.Tensor):
            output_voxels = torch.tensor(output_voxels)
        if not isinstance(target_voxels, torch.Tensor):
            target_voxels = torch.tensor(target_voxels)

        # batch 차원 제거
        if output_voxels.dim() == 5:
            output_voxels = output_voxels[0]   # [C, H, W, D]
        if target_voxels.dim() == 4:
            target_voxels = target_voxels[0]   # [H, W, D]

        # 예측 클래스: argmax over channel dim
        if output_voxels.dim() == 4 and output_voxels.shape[0] > 1:
            y_pred = torch.argmax(output_voxels, dim=0)  # [H, W, D]
        else:
            y_pred = output_voxels.squeeze(0)

        y_pred = y_pred.long().cpu()
        y_true = target_voxels.long().cpu()

        # 유효 마스크 (255는 ignore)
        mask = y_true != 255
        y_pred_m = y_pred[mask]
        y_true_m = y_true[mask]

        # Completion: non-empty vs empty
        pred_nonempty = (y_pred_m > 0)
        true_nonempty = (y_true_m > 0)

        completion_tp = int((pred_nonempty & true_nonempty).sum().item())
        completion_fp = int((pred_nonempty & ~true_nonempty).sum().item())
        completion_fn = int((~pred_nonempty & true_nonempty).sum().item())

        # Semantic completion: per-class TP/FP/FN
        n_cls = self.n_classes
        tps = np.zeros(n_cls, dtype=np.float64)
        fps = np.zeros(n_cls, dtype=np.float64)
        fns = np.zeros(n_cls, dtype=np.float64)

        for cls_idx in range(n_cls):
            tp = int(((y_true_m == cls_idx) & (y_pred_m == cls_idx)).sum().item())
            fp = int(((y_true_m != cls_idx) & (y_pred_m == cls_idx)).sum().item())
            fn = int(((y_true_m == cls_idx) & (y_pred_m != cls_idx)).sum().item())
            tps[cls_idx] = tp
            fps[cls_idx] = fp
            fns[cls_idx] = fn

        return (completion_tp, completion_fp, completion_fn, tps, fps, fns)

    def compute_metrics(self, results):
        """축적된 결과에서 최종 메트릭을 계산합니다."""
        if not results:
            return {}

        # lidar-seg 평가
        if results and 'evaluation_semantic' in results[0]:
            return self._compute_lidarseg_metrics(results)

        # SSC 평가
        return self._compute_ssc_metrics(results)

    def _compute_lidarseg_metrics(self, results):
        """LiDAR segmentation 메트릭 계산."""
        try:
            from ..utils import cm_to_ious, format_results
        except ImportError:
            return {}

        combined = sum([r['evaluation_semantic'] for r in results])
        ious = cm_to_ious(combined)
        _, res_dic = format_results(ious, return_dic=True)
        return {f'nuScenes_lidarseg_{k}': v for k, v in res_dic.items()}

    def _compute_ssc_metrics(self, results):
        """SSC mIoU 메트릭 계산."""
        ssc_results = [r['ssc_result'] for r in results if 'ssc_result' in r]
        if not ssc_results:
            return {}

        completion_tp = sum(x[0] for x in ssc_results)
        completion_fp = sum(x[1] for x in ssc_results)
        completion_fn = sum(x[2] for x in ssc_results)

        tps = sum(x[3] for x in ssc_results)
        fps = sum(x[4] for x in ssc_results)
        fns = sum(x[5] for x in ssc_results)

        denom_c = completion_tp + completion_fp + completion_fn
        precision = completion_tp / (completion_tp + completion_fp + 1e-7)
        recall = completion_tp / (completion_tp + completion_fn + 1e-7)
        iou = completion_tp / (denom_c + 1e-7)

        iou_ssc = tps / (tps + fps + fns + 1e-5)
        miou = float(iou_ssc[1:17].mean())

        eval_results = {
            'nuScenes_SC_Precision': round(float(precision) * 100, 2),
            'nuScenes_SC_Recall': round(float(recall) * 100, 2),
            'nuScenes_SC_IoU': round(float(iou) * 100, 2),
            'nuScenes_SSC_mIoU': round(miou * 100, 2),
        }

        for name, val in zip(self.class_names, iou_ssc.tolist()):
            eval_results[f'nuScenes_SSC_{name}_IoU'] = round(val * 100, 2)

        eval_results['nuScenes_combined_IoU'] = (
            eval_results['nuScenes_SC_IoU'] + eval_results['nuScenes_SSC_mIoU'])

        # refine 결과가 있으면 추가
        ssc_results_refine = [
            r['ssc_result_refine'] for r in results if 'ssc_result_refine' in r
        ]
        if ssc_results_refine:
            completion_tp = sum(x[0] for x in ssc_results_refine)
            completion_fp = sum(x[1] for x in ssc_results_refine)
            completion_fn = sum(x[2] for x in ssc_results_refine)
            tps = sum(x[3] for x in ssc_results_refine)
            fps = sum(x[4] for x in ssc_results_refine)
            fns = sum(x[5] for x in ssc_results_refine)

            denom_c = completion_tp + completion_fp + completion_fn
            iou = completion_tp / (denom_c + 1e-7)
            iou_ssc = tps / (tps + fps + fns + 1e-5)
            miou_refine = float(iou_ssc[1:17].mean())

            eval_results['nuScenes_SC_refine_IoU'] = round(float(iou) * 100, 2)
            eval_results['nuScenes_SSC_refine_mIoU'] = round(miou_refine * 100, 2)
            for name, val in zip(self.class_names, iou_ssc.tolist()):
                eval_results[f'nuScenes_SSC_refine_{name}_IoU'] = round(val * 100, 2)

        return eval_results
