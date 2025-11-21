# Copyright (c) OpenMMLab. All rights reserved.
"""
STCOcc Registry Helper

This module ensures all STCOcc components are properly registered with mmdet3d registries.
Import this module to automatically register all STCOcc models, transforms, and hooks.
"""

# CRITICAL: Import mmdet3d.models first to initialize the registry
# This ensures MODELS registry is available before registering custom modules
import mmdet3d.models  # noqa: F401

# Import all required modules at module level
try:
    # First, import core model components that must be registered
    # Import these BEFORE importing the full stcocc package to avoid
    # dependency issues with optional packages like prettytable
    from projects.STCOcc.stcocc.models.necks.fpn import CustomFPN
    from projects.STCOcc.stcocc.models.backbones.resnet import CustomResNet3D
    from projects.STCOcc.stcocc.detectors.stcocc import STCOcc
    
    # Import view transformation modules
    from projects.STCOcc.stcocc.view_transformation.forward_projection.bevdet_utils.lss_transformation import LSSVStereoForwardPorjection
    from projects.STCOcc.stcocc.view_transformation.forward_projection.bevdet_stereo_projection import BEVDetStereoForwardProjection
    from projects.STCOcc.stcocc.view_transformation.backward_projection.bevformer_utils.positional_encoding import CustormLearnedPositionalEncoding
    
    # Import heads and modules
    from projects.STCOcc.stcocc.heads.occ_head import OccHead
    from projects.STCOcc.stcocc.heads.occ_flow_head import OccFlowHead
    from projects.STCOcc.stcocc.modules.temporal_fusion import SparseFusion
    
    # Import evaluation modules
    from projects.STCOcc.stcocc.evaluation.occupancy_metric import OccupancyMetric
    
    # Import hooks
    from projects.STCOcc.stcocc.hooks.ema_hook import MEGVIIEMAHook
    
    # Import specific transforms (avoid * to prevent conflicts)
    import projects.STCOcc.stcocc.transforms.pipelines.loading
    import projects.STCOcc.stcocc.transforms.pipelines.formating
    import projects.STCOcc.stcocc.transforms.pipelines.compose
    import projects.STCOcc.stcocc.transforms.pipelines.test_time_aug
    
    # Try to import datasets (may fail if optional dependencies are missing)
    # This is done last since it may have optional dependencies
    try:
        from projects.STCOcc.stcocc.datasets.nuscenes_dataset_occ import NuScenesDatasetOccpancy
    except ImportError as dataset_import_error:
        print(f"⚠️ Warning: Could not import NuScenesDatasetOccpancy: {dataset_import_error}")
        print("   This is usually due to missing optional dependencies (e.g., prettytable)")
        print("   Model registration will continue, but dataset functionality may be limited.")
    
    # Verify registration of critical modules
    from mmdet3d.registry import MODELS
    if 'CustomFPN' not in MODELS._module_dict:
        raise RuntimeError("CustomFPN failed to register in mmdet3d::model registry")
    
    print("✅ STCOcc modules registered successfully")
    
except ImportError as e:
    import traceback
    print(f"⚠️ STCOcc module registration failed: {e}")
    traceback.print_exc()
    # If allow_failed_imports is False, re-raise the error
    raise
except Exception as e:
    import traceback
    print(f"⚠️ STCOcc module registration failed with unexpected error: {e}")
    traceback.print_exc()
    raise

def register_stcocc_modules():
    """Function to manually trigger registration if needed."""
    return True
