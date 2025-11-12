# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.ops import (RoIAlign, SigmoidFocalLoss, get_compiler_version,
                      get_compiling_cuda_version, nms, roi_align,
                      sigmoid_focal_loss)
from mmcv.ops.assign_score_withk import assign_score_withk
from mmcv.ops.ball_query import ball_query
from mmcv.ops.furthest_point_sample import (furthest_point_sample,
                                            furthest_point_sample_with_dist)
from mmcv.ops.gather_points import gather_points
from mmcv.ops.group_points import GroupAll, QueryAndGroup, grouping_operation
from mmcv.ops.knn import knn
from mmcv.ops.points_in_boxes import (points_in_boxes_all, points_in_boxes_cpu,
                                      points_in_boxes_part)
from mmcv.ops.points_sampler import PointsSampler as Points_Sampler
from mmcv.ops.roiaware_pool3d import RoIAwarePool3d
from mmcv.ops.roipoint_pool3d import RoIPointPool3d
from mmcv.ops.scatter_points import DynamicScatter, dynamic_scatter
from mmcv.ops.three_interpolate import three_interpolate
from mmcv.ops.three_nn import three_nn
from mmcv.ops.voxelize import Voxelization, voxelization

from .bev_pool_v2.bev_pool import bev_pool_v2, TRTBEVPoolv2

try:
    from .dgcnn_modules import DGCNNFAModule, DGCNNFPModule, DGCNNGFModule
except ImportError as e:
    print(f"Warning: Could not import dgcnn_modules: {e}")
    DGCNNFAModule = DGCNNFPModule = DGCNNGFModule = None

try:
    from .norm import NaiveSyncBatchNorm1d, NaiveSyncBatchNorm2d
except ImportError as e:
    print(f"Warning: Could not import norm: {e}")
    NaiveSyncBatchNorm1d = NaiveSyncBatchNorm2d = None

try:
    from .paconv import PAConv, PAConvCUDA
except ImportError as e:
    print(f"Warning: Could not import paconv: {e}")
    PAConv = PAConvCUDA = None

try:
    from .pointnet_modules import (PAConvCUDASAModule, PAConvCUDASAModuleMSG,
                                   PAConvSAModule, PAConvSAModuleMSG,
                                   PointFPModule, PointSAModule, PointSAModuleMSG,
                                   build_sa_module)
except ImportError as e:
    print(f"Warning: Could not import pointnet_modules: {e}")
    PAConvCUDASAModule = PAConvCUDASAModuleMSG = None
    PAConvSAModule = PAConvSAModuleMSG = None
    PointFPModule = PointSAModule = PointSAModuleMSG = None
    build_sa_module = None

try:
    from .sparse_block import (SparseBasicBlock, SparseBottleneck,
                               make_sparse_convmodule)
except ImportError as e:
    print(f"Warning: Could not import sparse_block: {e}")
    SparseBasicBlock = SparseBottleneck = make_sparse_convmodule = None

__all__ = [
    'nms', 'soft_nms', 'RoIAlign', 'roi_align', 'get_compiler_version',
    'get_compiling_cuda_version', 'NaiveSyncBatchNorm1d',
    'NaiveSyncBatchNorm2d', 'batched_nms', 'Voxelization', 'voxelization',
    'dynamic_scatter', 'DynamicScatter', 'sigmoid_focal_loss',
    'SigmoidFocalLoss', 'SparseBasicBlock', 'SparseBottleneck',
    'RoIAwarePool3d', 'points_in_boxes_part', 'points_in_boxes_cpu',
    'make_sparse_convmodule', 'ball_query', 'knn', 'furthest_point_sample',
    'furthest_point_sample_with_dist', 'three_interpolate', 'three_nn',
    'gather_points', 'grouping_operation', 'GroupAll', 'QueryAndGroup',
    'PointSAModule', 'PointSAModuleMSG', 'PointFPModule', 'DGCNNFPModule',
    'DGCNNGFModule', 'DGCNNFAModule', 'points_in_boxes_all',
    'get_compiler_version', 'assign_score_withk', 'get_compiling_cuda_version',
    'Points_Sampler', 'build_sa_module', 'PAConv', 'PAConvCUDA',
    'PAConvSAModuleMSG', 'PAConvSAModule', 'PAConvCUDASAModule',
    'PAConvCUDASAModuleMSG', 'RoIPointPool3d', 'bev_pool_v2', 'TRTBEVPoolv2'
]
