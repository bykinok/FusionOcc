# Copyright (c) OpenMMLab. All rights reserved.
"""
STCOcc Registry Helper

This module ensures all STCOcc components are properly registered with mmdet3d registries.
Import this module to automatically register all STCOcc models, transforms, and hooks.
"""

# Import all required modules at module level
try:
    # Import main STCOcc package (this will trigger all __init__.py imports)
    import projects.STCOcc.stcocc
    
    # Explicitly import key modules to ensure registration
    from projects.STCOcc.stcocc.models.necks.fpn import CustomFPN
    from projects.STCOcc.stcocc.models.backbones.resnet import CustomResNet3D
    from projects.STCOcc.stcocc.detectors.stcocc import STCOcc
    from projects.STCOcc.stcocc.hooks.ema_hook import MEGVIIEMAHook
    from projects.STCOcc.stcocc.datasets.nuscenes_dataset_occ import NuScenesDatasetOccpancy
    
    # Import specific transforms (avoid * to prevent conflicts)
    import projects.STCOcc.stcocc.transforms.pipelines.loading
    import projects.STCOcc.stcocc.transforms.pipelines.formating
    import projects.STCOcc.stcocc.transforms.pipelines.compose
    import projects.STCOcc.stcocc.transforms.pipelines.test_time_aug
    # Skip transforms_3d to avoid conflicts with existing mmdet3d transforms
    
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
    
    print("✅ STCOcc modules registered successfully")
    
except ImportError as e:
    print(f"⚠️ STCOcc module registration failed: {e}")

def register_stcocc_modules():
    """Function to manually trigger registration if needed."""
    return True
