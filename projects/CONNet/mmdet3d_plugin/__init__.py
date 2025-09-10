# Copyright (c) OpenMMLab. All rights reserved.
# Import and register mmdet models for ResNet backbone support
import mmdet.models
from mmdet3d.registry import MODELS
from mmdet.models.backbones import ResNet

# Register ResNet from mmdet to mmdet3d
MODELS.register_module(module=ResNet, force=True)

# Skip evaluation hooks for now due to mmengine compatibility issues
# from .core.evaluation.eval_hooks import OccDistEvalHook, OccEvalHook
# from .core.evaluation.efficiency_hooks import OccEfficiencyHook

# Skip some pipelines for now due to compatibility issues
# from .core.visualizer import save_occ
# from .datasets.pipelines import (
#   PhotoMetricDistortionMultiViewImage, PadMultiViewImage, 
#   NormalizeMultiviewImage,  CustomCollect3D)

# Import core models and datasets
from .occupancy import *
from .datasets import *
# Import only the OccMetric directly to avoid eval_hooks import issues
from .core.evaluation.occ_metric import OccMetric
# Import pipelines for transform registration
from .datasets.pipelines import *