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

try:
    from .occupancy_metric import OccupancyMetric
    metric_classes = ['OccupancyMetric']
except ImportError:
    metric_classes = []

try:
    from .backbones import CustomResNet3D, CustomResNet, SwinTransformer
    backbone_classes = ['CustomResNet3D', 'CustomResNet', 'SwinTransformer']
except ImportError:
    backbone_classes = []

try:
    from .necks import FPN_LSS, LSSFPN3D, CrossModalLSS, LSSViewTransformer, LSSViewTransformerBEVDepth, LSSViewTransformerBEVStereo
    neck_classes = ['FPN_LSS', 'LSSFPN3D', 'CrossModalLSS', 'LSSViewTransformer', 'LSSViewTransformerBEVDepth', 'LSSViewTransformerBEVStereo']
except ImportError:
    neck_classes = []

__all__ = ['FusionOCC', 'FusionDepthSeg', 'CustomSparseEncoder'] + dataset_classes + transform_classes + metric_classes + backbone_classes + neck_classes 