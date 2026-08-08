import os
import importlib.util
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from mmengine.evaluator import BaseMetric
from mmdet3d.registry import METRICS


def _load_ray_metrics_occ3d():
    """Dynamically loads STCOcc's ray_metrics_occ3d.py by file path (same
    trick projects/SurroundOcc/surroundocc/evaluation/occupancy_metric_hybrid.py
    uses for occupancy_metric.py) rather than importing `projects.STCOcc...`,
    to avoid depending on that project's own package __init__ side effects.
    Requires CUDA_HOME to point at a CUDA toolkit matching the installed
    torch build (see projects/GaussianFormer/README.md) -- the module JIT
    compiles a CUDA ray-casting extension (`libs/dvr`) on first import.
    """
    path = (Path(__file__).resolve().parents[3] / 'STCOcc' / 'stcocc'
            / 'datasets' / 'ray_metrics_occ3d.py')
    if not path.exists():
        raise ImportError(f'ray_metrics_occ3d.py not found at {path}')
    spec = importlib.util.spec_from_file_location('gaussianformer_ray_metrics_occ3d', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@METRICS.register_module()
class GaussianFormerRayIoUMetric(BaseMetric):
    """RayIoU evaluation for GaussianFormer, reusing the shared DVR-based
    ray-casting kernel from projects/STCOcc/stcocc/datasets/ray_metrics_occ3d.py
    (the same one projects/SurroundOcc's OccupancyMetricHybrid delegates to
    for its own `_rayiou` configs).

    Unlike OccupancyMetricHybrid, this does NOT cross-reference predictions
    against a separate flat `ann_file` by integer index, nor does it use
    STCOcc's `nuscenes_ego_pose_loader.nuScenesDataset` (which enumerates
    val samples in nuscenes-devkit's own scene order) -- GaussianFormer's
    own dataset iterates `self.keyframes` sorted by
    `scene_token + zero-padded frame index` (dataset/dataset.py), a
    DIFFERENT order, so index-based cross-referencing against a
    nuscenes-devkit-ordered list would silently misalign predictions with
    the wrong ego/lidar pose. Instead, each sample's own lidar-in-ego-frame
    origin is computed directly in the pipeline
    (LoadOccupancyOcc3D -> `results['lidar_origin']`, from the same
    `ego2lidar` extrinsic `dataset.py::get_data_info` already computes) and
    carried through `BEVSegmentor.predict()` -- no external ordering
    dependency.

    The lidar-in-ego origin computed this way was verified against
    `ray_metrics_occ3d.py`'s own example comment (`[0.9858, 0.0, 1.8402]`,
    the standard nuScenes LIDAR_TOP-to-ego offset): GaussianFormer's
    `LoadOccupancyOcc3D`-computed `lidar_origin` for a real val sample
    matched it to 4 decimal places, confirming the coordinate-frame
    assumption. What was *not* verified is the resulting RayIoU number
    against a published reference value (that requires a full trained
    checkpoint + val-set pass), so treat absolute scores with appropriate
    caution until cross-checked against a trained model.
    """

    def __init__(self, collect_device: str = 'cpu', prefix: Optional[str] = None):
        super().__init__(collect_device=collect_device, prefix=prefix)

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        for sample in data_samples:
            pred = sample['occ_results']
            gt = sample['voxel_semantics']
            lidar_origin = sample.get('lidar_origin')
            pred = pred.byte().cpu().numpy() if isinstance(pred, torch.Tensor) else np.asarray(pred, dtype=np.uint8)
            gt = gt.byte().cpu().numpy() if isinstance(gt, torch.Tensor) else np.asarray(gt, dtype=np.uint8)
            if lidar_origin is None:
                lidar_origin = np.zeros(3, dtype=np.float32)
            elif isinstance(lidar_origin, torch.Tensor):
                lidar_origin = lidar_origin.cpu().numpy()
            self.results.append((pred, gt, np.asarray(lidar_origin, dtype=np.float32)))

    def compute_metrics(self, results: list) -> Dict[str, float]:
        ray_metrics_occ3d = _load_ray_metrics_occ3d()

        pred_sems = [r[0] for r in results]
        gt_sems = [r[1] for r in results]
        flow_zeros = [np.zeros((200, 200, 16, 2), dtype=np.float16) for _ in results]
        lidar_origins = [torch.from_numpy(r[2]).view(1, 1, 3) for r in results]

        miou, mave, occ_score = ray_metrics_occ3d.main(
            pred_sems, gt_sems, flow_zeros, flow_zeros, lidar_origins, logger=None)

        return {'RayIoU': miou, 'mAVE': mave, 'occ_score': occ_score, 'count': len(results)}
