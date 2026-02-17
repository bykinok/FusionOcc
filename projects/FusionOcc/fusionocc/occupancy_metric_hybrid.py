"""OccupancyMetricHybrid for FusionOcc using STCOcc metric for occ3d GT format."""
from mmengine.registry import METRICS as ENGINE_METRICS
from mmdet3d.registry import METRICS as DET3D_METRICS
from mmengine.evaluator import BaseMetric
from typing import Optional, Dict, Sequence, List


@ENGINE_METRICS.register_module()
@DET3D_METRICS.register_module()
class OccupancyMetricHybrid(BaseMetric):
    """Occupancy Metric for FusionOcc using STCOcc's metric for occ3d format.

    Same pattern as SurroundOcc, CONet, BEVFormer: temporarily unregister
    local OccupancyMetric, load STCOcc module, restore registry, then use
    STCOcc's OccupancyMetric instance for process/compute_metrics.
    """

    def __init__(self,
                 dataset_name: Optional[str] = None,
                 num_classes: int = 18,
                 use_lidar_mask: bool = False,
                 use_image_mask: bool = False,
                 ann_file: Optional[str] = None,
                 data_root: Optional[str] = None,
                 class_names: Optional[List[str]] = None,
                 eval_metric: str = 'miou',
                 sort_by_timestamp: bool = True,
                 collect_device: str = 'cpu',
                 prefix=None,
                 **kwargs):
        # kwargs: absorb config keys like metric='bbox', backend_args=None from FusionOcc config
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.dataset_name = dataset_name
        self.num_classes = num_classes
        self.class_names = class_names or []
        self.eval_metric = eval_metric
        self.sort_by_timestamp = sort_by_timestamp

        import importlib.util
        from pathlib import Path

        # projects/FusionOcc/fusionocc/occupancy_metric_hybrid.py -> projects/
        stcocc_metric_path = Path(__file__).resolve().parents[2] / 'STCOcc' / 'stcocc' / 'evaluation' / 'occupancy_metric.py'
        if not stcocc_metric_path.exists():
            raise ImportError(
                f"STCOcc metric file not found at {stcocc_metric_path}\n"
                "Please ensure STCOcc project is available in the projects directory."
            )

        # Temporarily remove FusionOcc's OccupancyMetric to avoid duplicate registration
        existing_engine_metric = ENGINE_METRICS._module_dict.pop('OccupancyMetric', None)
        existing_det3d_metric = DET3D_METRICS._module_dict.pop('OccupancyMetric', None)

        spec = importlib.util.spec_from_file_location("stcocc_metric_module", stcocc_metric_path)
        if spec is None or spec.loader is None:
            if existing_engine_metric is not None:
                ENGINE_METRICS._module_dict['OccupancyMetric'] = existing_engine_metric
            if existing_det3d_metric is not None:
                DET3D_METRICS._module_dict['OccupancyMetric'] = existing_det3d_metric
            raise ImportError(f"Failed to load spec from {stcocc_metric_path}")

        stcocc_metric_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(stcocc_metric_module)
        STCOccMetric = stcocc_metric_module.OccupancyMetric

        # Remove STCOcc's registration and restore FusionOcc's
        ENGINE_METRICS._module_dict.pop('OccupancyMetric', None)
        DET3D_METRICS._module_dict.pop('OccupancyMetric', None)
        if existing_engine_metric is not None:
            ENGINE_METRICS._module_dict['OccupancyMetric'] = existing_engine_metric
        if existing_det3d_metric is not None:
            DET3D_METRICS._module_dict['OccupancyMetric'] = existing_det3d_metric

        self.stcocc_metric = STCOccMetric(
            num_classes=num_classes,
            use_lidar_mask=use_lidar_mask,
            use_image_mask=use_image_mask,
            dataset_name=dataset_name,
            ann_file=ann_file,
            data_root=data_root,
            eval_metric=eval_metric,
            sort_by_timestamp=self.sort_by_timestamp,
            collect_device=collect_device,
            prefix=prefix,
        )

        # FusionOcc pkl may use 'occ_gt_path'; STCOcc expects 'occ3d_gt_path' (STCOcc avoids data_root duplication in occupancy_metric.py)
        if hasattr(self.stcocc_metric, 'data_infos') and self.stcocc_metric.data_infos:
            for info in self.stcocc_metric.data_infos:
                if isinstance(info, dict) and 'occ_gt_path' in info and 'occ3d_gt_path' not in info:
                    info['occ3d_gt_path'] = info['occ_gt_path']

        print(f"[OccupancyMetricHybrid] Initialized with STCOcc metric for {dataset_name} "
              f"with eval_metric={eval_metric}, sort_by_timestamp={self.sort_by_timestamp}")

    def reset(self):
        self.stcocc_metric.reset()

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        # -------------------------------------------------------------------------
        # Where "index" is set in the codebase (for GT matching in STCOcc metric):
        # 1. Dataset: fusionocc_dataset.py get_data_info() -> input_dict['index'] = index
        # 2. Pipeline: loading.py FormatDataSamples meta_keys includes 'index' -> data['img_metas']['index']
        # 3. Batch: dataloader returns data; collate may make img_metas['index'] a list, e.g. [0]
        # 4. Model: fusion_occ.py test_step() reads data['img_metas'] and sets data_sample['index']
        # 5. Here: we override each data_sample['index'] so STCOcc gets correct 0,1,2,...
        # -------------------------------------------------------------------------
        n = len(data_samples)
        batch_indices = []
        if isinstance(data_batch, dict):
            img_metas = data_batch.get('img_metas', None)
            if img_metas is not None and n > 0:
                if isinstance(img_metas, list) and len(img_metas) >= n:
                    for i in range(n):
                        m = img_metas[i]
                        idx = m.get('index', m.get('sample_idx', i))
                        if isinstance(idx, (list, tuple)):
                            idx = int(idx[0]) if len(idx) else i
                        else:
                            try:
                                idx = int(idx)
                            except (TypeError, ValueError):
                                idx = i
                        batch_indices.append(idx)
                elif isinstance(img_metas, dict):
                    idx = img_metas.get('index', img_metas.get('sample_idx', 0))
                    if isinstance(idx, (list, tuple)):
                        idx = int(idx[0]) if len(idx) else 0
                    else:
                        try:
                            idx = int(idx)
                        except (TypeError, ValueError):
                            idx = 0
                    batch_indices = [idx + i for i in range(n)]
        # When we have indices from img_metas: append to OUR self.results only (correct 0,1,2).
        # MMEngine collects self.results -> compute_metrics(results) -> STCOcc's self.predictions = results.
        if batch_indices:
            for i, data_sample in enumerate(data_samples):
                if i >= len(batch_indices):
                    break
                if isinstance(data_sample, dict) and 'occ_results' in data_sample:
                    pred_dict = {
                        'occ_results': data_sample['occ_results'],
                        'index': [batch_indices[i]],
                    }
                    if 'flow_results' in data_sample:
                        pred_dict['flow_results'] = data_sample['flow_results']
                    self.results.append(pred_dict)
        else:
            self.stcocc_metric.process(data_batch, data_samples)
            self.results.extend(list(self.stcocc_metric.results))

    def compute_metrics(self, results: list) -> Dict[str, float]:
        return self.stcocc_metric.compute_metrics(results)
