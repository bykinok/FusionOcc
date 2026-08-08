from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from mmengine.evaluator import BaseMetric
from mmdet3d.registry import METRICS

NUSC_CLASS_INDICES = list(range(1, 17))
NUSC_CLASS_NAMES = [
    'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
    'vegetation',
]


@METRICS.register_module()
class GaussianFormerOccupancyMetric(BaseMetric):
    """mMengine ``BaseMetric`` wrapper reproducing the per-class mIoU counting
    used by GaussianFormer_ori's ``misc/metric_util.py::MeanIoU`` /ored the way
    ``train.py``/``eval.py`` call it (see their ``miou_metric._after_step``
    loop keyed on ``final_occ``/``sampled_label``/``occ_mask``).

    ``MeanIoU`` itself (ported unchanged in ``evaluation/mean_iou.py``)
    accumulates into persistent CUDA buffers and calls ``dist.all_reduce``
    directly, which assumes GaussianFormer_ori's own single bespoke process
    group -- that does not compose safely with mmengine's ``BaseMetric``,
    whose ``process()``/``compute_metrics()`` split already gathers
    per-sample results across ranks itself (via ``collect_results``) before
    ``compute_metrics`` runs once on the main process. So this class
    reproduces the exact same per-class seen/correct/positive counting and
    IoU formulas, just split to fit that contract instead of calling
    ``MeanIoU`` directly.
    """

    def __init__(
        self,
        class_indices: List[int] = NUSC_CLASS_INDICES,
        empty_label: int = 17,
        class_names: List[str] = NUSC_CLASS_NAMES,
        collect_device: str = 'cpu',
        prefix: Optional[str] = None,
    ):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.class_indices = class_indices
        self.empty_label = empty_label
        self.class_names = class_names
        self.num_classes = len(class_indices)

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        for sample in data_samples:
            outputs = sample['final_occ']
            targets = sample['sampled_label']
            mask = sample.get('occ_mask', None)
            if mask is not None:
                mask = mask.flatten()
                outputs = outputs[mask]
                targets = targets[mask]

            # rows 0..num_classes-1: per-class [seen, correct, positive];
            # last row: binary occupied-vs-empty [seen, correct, positive].
            counts = torch.zeros(self.num_classes + 1, 3)
            for i, c in enumerate(self.class_indices):
                counts[i, 0] = torch.sum(targets == c)
                counts[i, 1] = torch.sum((targets == c) & (outputs == c))
                counts[i, 2] = torch.sum(outputs == c)
            counts[-1, 0] = torch.sum(targets != self.empty_label)
            counts[-1, 1] = torch.sum((targets != self.empty_label) & (outputs != self.empty_label))
            counts[-1, 2] = torch.sum(outputs != self.empty_label)

            self.results.append(counts.cpu().numpy())

    def compute_metrics(self, results: list) -> Dict[str, float]:
        counts = np.sum(np.stack(results, axis=0), axis=0)  # (num_classes + 1, 3)
        seen, correct, positive = counts[:, 0], counts[:, 1], counts[:, 2]

        metrics = {}
        ious = []
        for i, name in enumerate(self.class_names):
            if seen[i] == 0:
                iou = 1.0
            else:
                iou = correct[i] / (seen[i] + positive[i] - correct[i])
            ious.append(iou)
            metrics[f'IoU.{name}'] = float(iou * 100)

        miou = float(np.mean(ious) * 100)
        denom = seen[-1] + positive[-1] - correct[-1]
        occ_iou = float(correct[-1] / denom * 100) if denom > 0 else 100.0

        metrics['mIoU'] = miou
        metrics['IoU'] = occ_iou
        return metrics
