# Copyright (c) OpenMMLab. All rights reserved.
from .fusion_occ import FusionOCC, FusionDepthSeg
from .lidar_encoder import CustomSparseEncoder

try:
    from .datasets.fusionocc_dataset import FusionOccDataset, NuScenesDatasetOccpancy
    dataset_classes = ['FusionOccDataset', 'NuScenesDatasetOccpancy']
except ImportError:
    dataset_classes = []

try:
    from .transforms import *
    transform_classes = ['PrepareImageSeg', 'LoadOccGTFromFile', 'LoadAnnotationsAll', 'FuseAdjacentSweeps', 'FormatDataSamples']
except ImportError:
    transform_classes = []

__all__ = ['FusionOCC', 'FusionDepthSeg', 'CustomSparseEncoder'] + dataset_classes + transform_classes 