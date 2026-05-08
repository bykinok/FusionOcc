"""OccupancyMetricHybrid for SparseOcc_eccv using STCOcc metric for occ3d GT format."""
import glob
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from mmengine.registry import METRICS as ENGINE_METRICS
from mmdet3d.registry import METRICS as DET3D_METRICS
from mmengine.evaluator import BaseMetric


@ENGINE_METRICS.register_module()
@DET3D_METRICS.register_module()
class OccupancyMetricHybrid(BaseMetric):
    """Occupancy Metric for SparseOcc_eccv using STCOcc's metric for occ3d format.

    BEVFormer / FusionOcc 와 동일한 패턴:
    - STCOcc OccupancyMetric을 동적으로 로드하여 인스턴스화
    - process()에서 Det3DDataSample.pred_occ로부터 occ_results / index를 추출해
      self.results에 추가
    - compute_metrics()를 STCOcc 인스턴스에 위임

    SparseOcc_eccv 고유 처리:
    - data_sample이 Det3DDataSample 객체 (plain dict가 아님)
    - pred_occ = {'occ_results': [dense_np_array], 'index': [int]}
    - pkl에 occ_path가 없으므로 occ_gt_root + token→scene 매핑으로 직접 구축
    """

    def __init__(self,
                 occ_gt_root: str = 'data/nuscenes/gts',
                 dataset_name: Optional[str] = 'occ3d',
                 num_classes: int = 18,
                 use_lidar_mask: bool = False,
                 use_image_mask: bool = True,
                 ann_file: Optional[str] = None,
                 data_root: Optional[str] = None,
                 eval_metric: str = 'miou',
                 sort_by_timestamp: bool = True,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 **kwargs):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.occ_gt_root = occ_gt_root
        self.dataset_name = dataset_name
        self.num_classes = num_classes
        self.eval_metric = eval_metric
        self.sort_by_timestamp = sort_by_timestamp

        # ------------------------------------------------------------------
        # STCOcc OccupancyMetric 동적 로드
        # 이 파일: projects/SparseOcc_eccv/sparseocc_eccv/datasets/
        # STCOcc:  projects/STCOcc/stcocc/evaluation/occupancy_metric.py
        # ------------------------------------------------------------------
        import importlib.util

        stcocc_metric_path = (
            Path(__file__).resolve().parents[3]
            / 'STCOcc' / 'stcocc' / 'evaluation' / 'occupancy_metric.py'
        )
        if not stcocc_metric_path.exists():
            raise ImportError(
                f"STCOcc metric file not found at {stcocc_metric_path}. "
                "Please ensure STCOcc project is available in the projects directory."
            )

        # 기존 OccupancyMetric 등록 임시 제거 (중복 등록 방지)
        existing_engine = ENGINE_METRICS._module_dict.pop('OccupancyMetric', None)
        existing_det3d = DET3D_METRICS._module_dict.pop('OccupancyMetric', None)

        spec = importlib.util.spec_from_file_location('stcocc_metric_module', stcocc_metric_path)
        if spec is None or spec.loader is None:
            if existing_engine is not None:
                ENGINE_METRICS._module_dict['OccupancyMetric'] = existing_engine
            if existing_det3d is not None:
                DET3D_METRICS._module_dict['OccupancyMetric'] = existing_det3d
            raise ImportError(f"Failed to load spec from {stcocc_metric_path}")

        stcocc_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(stcocc_mod)
        STCOccMetric = stcocc_mod.OccupancyMetric

        # STCOcc 자기 등록 제거 후 원래 등록 복원
        ENGINE_METRICS._module_dict.pop('OccupancyMetric', None)
        DET3D_METRICS._module_dict.pop('OccupancyMetric', None)
        if existing_engine is not None:
            ENGINE_METRICS._module_dict['OccupancyMetric'] = existing_engine
        if existing_det3d is not None:
            DET3D_METRICS._module_dict['OccupancyMetric'] = existing_det3d

        # ------------------------------------------------------------------
        # STCOcc 메트릭 인스턴스 생성
        # ------------------------------------------------------------------
        self.stcocc_metric = STCOccMetric(
            num_classes=num_classes,
            use_lidar_mask=use_lidar_mask,
            use_image_mask=use_image_mask,
            dataset_name=dataset_name,
            ann_file=ann_file,
            data_root=data_root,
            eval_metric=eval_metric,
            sort_by_timestamp=sort_by_timestamp,
            collect_device=collect_device,
            prefix=prefix,
        )

        # ------------------------------------------------------------------
        # data_infos에 occ_path 추가
        # SparseOcc_eccv pkl에는 occ_path 필드가 없으므로 직접 구축한다.
        # ------------------------------------------------------------------
        if hasattr(self.stcocc_metric, 'data_infos') and self.stcocc_metric.data_infos:
            self._enrich_data_infos_with_occ_path()

        print(f"[OccupancyMetricHybrid] Initialized STCOcc metric "
              f"(dataset={dataset_name}, eval_metric={eval_metric}, "
              f"sort_by_timestamp={sort_by_timestamp})")

    # ------------------------------------------------------------------
    # occ_path 보강
    # ------------------------------------------------------------------
    def _enrich_data_infos_with_occ_path(self) -> None:
        """glob으로 token→scene 매핑을 구축하고 data_infos에 occ_path를 추가한다."""
        token2scene: Dict[str, str] = {}
        for npz_path in glob.glob(os.path.join(self.occ_gt_root, '*/*/*.npz')):
            parts = npz_path.replace('\\', '/').split('/')
            if len(parts) >= 3:
                token2scene[parts[-2]] = parts[-3]

        enriched = 0
        for info in self.stcocc_metric.data_infos:
            if not isinstance(info, dict):
                continue
            if 'occ_path' in info or 'occ3d_gt_path' in info:
                continue
            token = info.get('token', info.get('sample_idx', ''))
            scene = token2scene.get(str(token), '')
            if scene:
                info['occ_path'] = os.path.join(self.occ_gt_root, scene, str(token))
                enriched += 1

        print(f"[OccupancyMetricHybrid] Enriched {enriched}/"
              f"{len(self.stcocc_metric.data_infos)} data_infos with occ_path")

    # ------------------------------------------------------------------
    # BaseMetric 인터페이스
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """평가 상태 초기화."""
        self.stcocc_metric.reset()
        self.results.clear()

    def process(self, data_batch: dict, data_samples: Sequence) -> None:
        """배치 결과를 처리하여 self.results에 추가한다.

        SparseOcc_eccv의 predict()는 각 Det3DDataSample에
        occ_results / index 를 직접 field로 저장한다.
        """
        # breakpoint()
        for data_sample in data_samples:
            # Det3DDataSample(BaseDataElement), dict 모두 지원
            # BaseDataElement.get(key, default)는 getattr(self, key, default)로 위임
            if isinstance(data_sample, dict):
                occ_results = data_sample.get('occ_results')
                index = data_sample.get('index')
            else:
                # BaseDataElement: .get() 또는 getattr 모두 동작
                occ_results = data_sample.get('occ_results', None)
                index = data_sample.get('index', None)

            if occ_results is None or index is None:
                print(f"[OccupancyMetricHybrid] WARNING: sample missing occ_results or index "
                      f"(type={type(data_sample).__name__})")
                continue

            # 리스트로 정규화
            if not isinstance(occ_results, (list, tuple)):
                occ_results = [occ_results]
            if not isinstance(index, (list, tuple)):
                index = [index]

            self.results.append({
                'occ_results': list(occ_results),
                'index': list(index),
            })

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """MMEngine이 gather한 results로 최종 메트릭을 계산한다."""
        return self.stcocc_metric.compute_metrics(results)
