from typing import Dict, Optional, Sequence

import numpy as np
import torch
from mmengine.evaluator import BaseMetric
from mmdet3d.registry import METRICS
from mmdet3d.datasets.occ_metrics import Metric_mIoU


def _to_numpy(x, dtype):
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy().astype(dtype)
    return np.asarray(x, dtype=dtype)


@METRICS.register_module()
class GaussianFormerOcc3DMetric(BaseMetric):
    """Occ3D-nuScenes mIoU evaluation for GaussianFormer, delegating the
    actual counting/reporting to ``mmdet3d.datasets.occ_metrics.Metric_mIoU``
    -- the same canonical class ``projects/STCOcc``/``projects/FusionOcc``
    use directly, and that ``projects/SurroundOcc``'s ``OccupancyMetricHybrid``
    (eval_metric='miou') uses transitively through STCOcc's
    ``evaluation/occupancy_metric.py``. Using it here (instead of the
    hand-rolled per-class counting in ``occupancy_metric.py::
    GaussianFormerOccupancyMetric``, which stays as the evaluator for
    ``configs/gaussianformer/nuscenes_gs25600.py``'s SurroundOcc-format GT,
    since ``Metric_mIoU`` hardcodes Occ3D's own `pc_range`/grid and would
    mislabel SurroundOcc GT's different grid in its radius/height breakdown)
    means the printed evaluation output -- per-class IoU/TP/FP/FN, plus the
    radius-bin and height-bin mIoU breakdown tables -- is in the exact same
    format other projects' Occ3D eval logs are.

    One behavioural difference from the previous hand-rolled metric:
    ``Metric_mIoU`` uses Occ3D's full 18-class numbering (0='others',
    1-16=semantic classes, 17='free') and averages mIoU over classes 0-16
    (**including** 'others') -- the standard Occ3D-nuScenes benchmark
    convention used by CONet/SurroundOcc/STCOcc, whereas the previous
    16-class list (matching GaussianFormer_ori's own SurroundOcc-GT
    evaluation, which has no 'others' class) excluded it entirely.
    """

    def __init__(
        self,
        use_lidar_mask: bool = False,
        use_image_mask: bool = True,
        num_classes: int = 18,
        ann_file: Optional[str] = None,
        data_root: Optional[str] = None,
        collect_device: str = 'cpu',
        prefix: Optional[str] = None,
    ):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.use_lidar_mask = use_lidar_mask
        self.use_image_mask = use_image_mask
        self.num_classes = num_classes
        # Not used by this class itself (process()/compute_metrics() get
        # occ_results/voxel_semantics/mask_camera directly, no GT reload
        # needed) -- stored purely so tools/compute_metrics_from_file.py's
        # `cfg.val_evaluator.get('ann_file'/'data_root')` config-introspection
        # can find the flat companion pkl it needs (see
        # projects/GaussianFormer/tools/build_flat_ann_file.py), the same
        # way it reads them off SurroundOcc/CONet's `OccupancyMetricHybrid`.
        self.ann_file = ann_file
        self.data_root = data_root

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        for sample in data_samples:
            pred = _to_numpy(sample['occ_results'], np.uint8)
            gt = _to_numpy(sample['voxel_semantics'], np.uint8)
            mask_camera = sample.get('mask_camera')
            mask_lidar = sample.get('mask_lidar')
            mask_camera = (_to_numpy(mask_camera, bool) if mask_camera is not None
                           else np.ones_like(gt, dtype=bool))
            mask_lidar = (_to_numpy(mask_lidar, bool) if mask_lidar is not None
                          else np.ones_like(gt, dtype=bool))
            self.results.append((pred, gt, mask_lidar, mask_camera))

    def compute_metrics(self, results: list) -> Dict[str, float]:
        metric = Metric_mIoU(
            num_classes=self.num_classes,
            use_lidar_mask=self.use_lidar_mask,
            use_image_mask=self.use_image_mask)
        for pred, gt, mask_lidar, mask_camera in results:
            metric.add_batch(pred, gt, mask_lidar, mask_camera)

        # count_miou() prints the per-class IoU/TP/FP/FN table plus the
        # radius-bin and height-bin mIoU breakdown tables (same format as
        # STCOcc/FusionOcc/SurroundOcc's own Occ3D eval logs); it returns
        # only the per-class array + count, matching those projects' own
        # metrics-dict granularity too (the breakdown tables are log/console
        # output there as well, not part of any structured return value).
        class_names, iou_per_class, cnt = metric.count_miou()

        metrics = {'count': cnt}
        for name, iou in zip(class_names, iou_per_class):
            metrics[f'IoU_{name}'] = float(iou * 100)
        metrics['mIoU'] = float(np.nanmean(iou_per_class[:self.num_classes - 1]) * 100)
        return metrics
