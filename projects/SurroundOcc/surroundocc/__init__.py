# Register mmdet3d transforms in mmengine so pipeline can build LoadPointsFromFile (for DepthSV)
import mmdet3d.datasets.transforms  # noqa: F401 - ensure LoadPointsFromFile is registered
from mmdet3d.registry import TRANSFORMS as TRANSFORMS_MMDET3D
from mmengine.registry import TRANSFORMS as TRANSFORMS_MMENGINE

_mmdet3d_transforms_for_surroundocc = [
    'LoadPointsFromFile',
]
for _name in _mmdet3d_transforms_for_surroundocc:
    if _name in TRANSFORMS_MMDET3D and _name not in TRANSFORMS_MMENGINE:
        TRANSFORMS_MMENGINE.register_module(
            name=_name,
            module=TRANSFORMS_MMDET3D.get(_name),
        )

from .detectors import SurroundOcc
from .dense_heads import OccHead
from .modules import PerceptionTransformer, SpatialCrossAttention, MSDeformableAttention3D, OccEncoder, OccLayer
from .datasets import CustomNuScenesOccDataset
from .transforms import LoadOccupancy
from .evaluation import OccupancyMetric
from .evaluation.occupancy_metric_hybrid import OccupancyMetricHybrid
from .loss import multiscale_supervision, geo_scal_loss, sem_scal_loss

__all__ = [
    'SurroundOcc', 'OccHead', 'PerceptionTransformer', 'SpatialCrossAttention', 
    'MSDeformableAttention3D', 'OccEncoder', 'OccLayer', 'CustomNuScenesOccDataset',
    'LoadOccupancy', 'OccupancyMetric', 'OccupancyMetricHybrid', 
    'multiscale_supervision', 'geo_scal_loss', 'sem_scal_loss'
]
