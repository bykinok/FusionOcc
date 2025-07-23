# Copyright (c) OpenMMLab. All rights reserved.
from .fusion_occ import FusionOCC, FusionDepthSeg
from .lidar_encoder import CustomSparseEncoder

try:
    from .datasets.fusionocc_dataset import FusionOccDataset
    __all__ = ['FusionOCC', 'FusionDepthSeg', 'CustomSparseEncoder', 'FusionOccDataset']
except ImportError:
    __all__ = ['FusionOCC', 'FusionDepthSeg', 'CustomSparseEncoder'] 