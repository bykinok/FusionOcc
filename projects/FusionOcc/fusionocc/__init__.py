# Copyright (c) OpenMMLab. All rights reserved.
import sys
import subprocess

# Check and remove spconv-cu113 if installed (incompatible with FusionOcc)
def check_and_remove_spconv_cu113():
    """Check if spconv-cu113 is installed and remove it if found.
    
    FusionOcc requires a different version of spconv that is incompatible
    with spconv-cu113. This function ensures spconv-cu113 is removed before
    FusionOcc modules are loaded.
    """
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'show', 'spconv-cu113'],
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode == 0:
            print("⚠️  spconv-cu113이 설치되어 있습니다. FusionOcc와 호환되지 않아 제거합니다...")
            uninstall_result = subprocess.run(
                [sys.executable, '-m', 'pip', 'uninstall', '-y', 'spconv-cu113'],
                capture_output=True,
                text=True,
                check=False
            )
            if uninstall_result.returncode == 0:
                print("✅ spconv-cu113이 성공적으로 제거되었습니다.")
                print("   FusionOcc는 다른 버전의 spconv를 사용합니다.")
            else:
                print(f"❌ spconv-cu113 제거 중 오류 발생: {uninstall_result.stderr}")
        # If not installed, no action needed (silent)
    except Exception as e:
        print(f"⚠️  spconv-cu113 확인 중 오류 발생: {e}")

# Remove spconv-cu113 before importing any FusionOcc modules
check_and_remove_spconv_cu113()

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
    from .occupancy_metric_hybrid import OccupancyMetricHybrid
    metric_classes = ['OccupancyMetric', 'OccupancyMetricHybrid']
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

try:
    from .hooks import SyncBNHook, SyncbnControlHook, EMAHookSafeForTest
    hook_classes = ['SyncBNHook', 'SyncbnControlHook', 'EMAHookSafeForTest']
except ImportError:
    hook_classes = []

__all__ = ['FusionOCC', 'FusionDepthSeg', 'CustomSparseEncoder'] + dataset_classes + transform_classes + metric_classes + backbone_classes + neck_classes + hook_classes 