"""Save predictions to file during test to avoid OOM; metrics are computed later from file."""
import os
import pickle
from typing import Dict, Optional, Sequence

try:
    from mmengine.registry import METRICS as ENGINE_METRICS
    from mmdet3d.registry import METRICS as DET3D_METRICS
    from mmengine.evaluator import BaseMetric
except ImportError:
    BaseMetric = object
    ENGINE_METRICS = None
    DET3D_METRICS = None


def _get_rank():
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass
    return 0


if ENGINE_METRICS is not None and DET3D_METRICS is not None:

    @ENGINE_METRICS.register_module()
    @DET3D_METRICS.register_module()
    class SavePredictionsEvaluator(BaseMetric):
        """Evaluator that writes each batch's predictions to a file instead of keeping in memory.
        Use with --save-predictions PATH; then run tools/compute_metrics_from_file.py to compute metrics.
        """

        def __init__(self,
                     save_path: str,
                     collect_device: str = 'cpu',
                     prefix: Optional[str] = None,
                     **kwargs):
            super().__init__(collect_device=collect_device, prefix=prefix)
            self.save_path = save_path
            self._file = None
            self._rank = _get_rank()

        def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
            # Convert to same pred_dict format as OccupancyMetricHybrid (for compatibility)
            pred_dicts = []
            for data_sample in data_samples:
                occ = data_sample.get('occ_results') if isinstance(data_sample, dict) else getattr(data_sample, 'occ_results', None)
                idx = data_sample.get('index') if isinstance(data_sample, dict) else getattr(data_sample, 'index', None)
                if occ is None or idx is None:
                    continue
                pred_dict = {'occ_results': occ, 'index': idx}

                def _get(key):
                    """Retrieve a field from both dict-type and object-type data_samples."""
                    if isinstance(data_sample, dict):
                        return data_sample.get(key)
                    return getattr(data_sample, key, None)

                flow = _get('flow_results')
                if flow is not None:
                    pred_dict['flow_results'] = flow
                for uk in ('uncertainty_msp', 'uncertainty_entropy'):
                    u = _get(uk)
                    if u is not None:
                        pred_dict[uk] = [u] if not isinstance(u, (list, tuple)) else u
                sp = _get('softmax_probs')
                if sp is not None:
                    pred_dict['softmax_probs'] = [sp] if not isinstance(sp, (list, tuple)) else sp

                pred_dicts.append(pred_dict)

            if not pred_dicts:
                return

            # Append to file (one pickle.dump per batch)
            base, ext = os.path.splitext(self.save_path)
            path_rank = f"{base}_rank{self._rank}{ext}" if ext else f"{base}_rank{self._rank}.pkl"
            if self._file is None:
                self._file = open(path_rank, 'wb')
            pickle.dump(pred_dicts, self._file, protocol=pickle.HIGHEST_PROTOCOL)
            self._file.flush()
            # Do NOT append to self.results so we don't hold all in memory

        def compute_metrics(self, results: list) -> Dict[str, float]:
            if self._file is not None:
                self._file.close()
                self._file = None
            return {}

        def __del__(self):
            f = getattr(self, '_file', None)
            if f is not None:
                try:
                    f.close()
                except Exception:
                    pass
                self._file = None
