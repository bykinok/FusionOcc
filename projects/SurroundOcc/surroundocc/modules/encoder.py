# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import copy
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmdet3d.registry import MODELS as DET3D_MODELS
from mmengine.registry import MODELS as ENGINE_MODELS
from mmcv.cnn import build_conv_layer, build_norm_layer


@DET3D_MODELS.register_module()
@ENGINE_MODELS.register_module()
class OccEncoder(BaseModule):
    """Occupancy Encoder for SurroundOcc.
    
    This encoder processes volume queries using spatial cross-attention to
    aggregate features from multi-view images.
    
    Args:
        num_layers (list): Number of layers at each level.
        pc_range (list): Point cloud range.
        return_intermediate (bool): Whether to return intermediate outputs.
        transformerlayers (dict): Config for transformer layers.
    """

    def __init__(self,
                 num_layers: list = [1, 3, 6],
                 pc_range: list = None,
                 return_intermediate: bool = False,
                 transformerlayers: dict = None,
                 init_cfg: dict = None,
                 **kwargs):
        super(OccEncoder, self).__init__(init_cfg=init_cfg)
        
        self.num_layers = num_layers if isinstance(num_layers, list) else [num_layers]
        self.return_intermediate = return_intermediate
        self.pc_range = pc_range
        
        # Build transformer layers
        self.layers = nn.ModuleList()
        if transformerlayers is not None:
            for i in range(len(self.num_layers)):
                layer_config = copy.deepcopy(transformerlayers)
                layer = ENGINE_MODELS.build(layer_config)
                self.layers.append(layer)

    @staticmethod
    def get_reference_points(H: int, W: int, Z: int, bs: int = 1, 
                           device: str = 'cuda', dtype: torch.dtype = torch.float):
        """Get reference points for 3D volume.
        
        Args:
            H: Height of volume.
            W: Width of volume.
            Z: Depth of volume.
            bs: Batch size.
            device: Device.
            dtype: Data type.
            
        Returns:
            torch.Tensor: Reference points with shape (bs, 1, H*W*Z, 3).
        """
        zs = torch.linspace(0.5, Z - 0.5, Z, dtype=dtype,
                          device=device).view(Z, 1, 1).expand(Z, H, W) / Z
        xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                          device=device).view(1, 1, W).expand(Z, H, W) / W
        ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                          device=device).view(1, H, 1).expand(Z, H, W) / H
        ref_3d = torch.stack((xs, ys, zs), -1)
        ref_3d = ref_3d.permute(3, 0, 1, 2).flatten(1).permute(1, 0)
        ref_3d = ref_3d[None, None].repeat(bs, 1, 1, 1)
        return ref_3d

    def point_sampling(self, reference_points: torch.Tensor, pc_range: list, img_metas: list):
        """Sample points in 3D space and project to camera coordinates.
        
        Args:
            reference_points: Reference points in normalized coordinates.
            pc_range: Point cloud range.
            img_metas: Image meta information.
            
        Returns:
            tuple: Reference points in camera coordinates and masks.
        """
        lidar2img = []
        for img_meta in img_metas:
            lidar2img.append(img_meta['lidar2img'])
        lidar2img = np.asarray(lidar2img)
        lidar2img = reference_points.new_tensor(lidar2img)  # (B, N, 4, 4)
        
        reference_points = reference_points.clone()

        # Convert normalized coordinates to world coordinates
        reference_points[..., 0:1] = reference_points[..., 0:1] * \
            (pc_range[3] - pc_range[0]) + pc_range[0]
        reference_points[..., 1:2] = reference_points[..., 1:2] * \
            (pc_range[4] - pc_range[1]) + pc_range[1]
        reference_points[..., 2:3] = reference_points[..., 2:3] * \
            (pc_range[5] - pc_range[2]) + pc_range[2]

        reference_points = torch.cat(
            (reference_points, torch.ones_like(reference_points[..., :1])), -1)

        reference_points = reference_points.permute(1, 0, 2, 3)
        D, B, num_query = reference_points.size()[:3]
        
        # lidar2img should be [batch_size, num_cam, 4, 4]
        if lidar2img.dim() == 4 and lidar2img.size(0) == 1 and B > 1:
            # This case should not happen but handle it
            lidar2img = lidar2img.squeeze(0)
        elif lidar2img.dim() == 4 and lidar2img.size(0) != B:
            # lidar2img might be [1, B, num_cam, 4, 4] format, squeeze first dim
            lidar2img = lidar2img.squeeze(0)
        
        # Now lidar2img should be [batch_size, num_cam, 4, 4]
        if lidar2img.dim() == 3:
            # Missing batch dimension, add it
            lidar2img = lidar2img.unsqueeze(0)
            
        batch_size = lidar2img.size(0)
        num_cam = lidar2img.size(1)
        
        # Ensure B matches batch_size
        assert B == batch_size, f"Mismatch: reference_points B={B} vs lidar2img batch_size={batch_size}"

        reference_points = reference_points.view(
            D, B, 1, num_query, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1)

        lidar2img = lidar2img.view(
            1, B, num_cam, 1, 4, 4).repeat(D, 1, 1, num_query, 1, 1)

        reference_points_cam = torch.matmul(lidar2img.to(torch.float32),
                                          reference_points.to(torch.float32)).squeeze(-1)
        eps = 1e-5

        bev_mask = (reference_points_cam[..., 2:3] > eps)
        reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps)

        reference_points_cam[..., 0] /= img_metas[0]['img_shape'][0][1]
        reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0][0]

        bev_mask = (bev_mask & (reference_points_cam[..., 1:2] > 0.0)
                   & (reference_points_cam[..., 1:2] < 1.0)
                   & (reference_points_cam[..., 0:1] < 1.0)
                   & (reference_points_cam[..., 0:1] > 0.0))
        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            bev_mask = torch.nan_to_num(bev_mask)

        reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)
        bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1)

        return reference_points_cam, bev_mask

    def forward(self,
                bev_query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                volume_h: int,
                volume_w: int,
                volume_z: int,
                spatial_shapes: torch.Tensor = None,
                level_start_index: torch.Tensor = None,
                img_metas: list = None,
                **kwargs) -> torch.Tensor:
        """Forward function.
        
        Args:
            bev_query: Volume queries.
            key: Key features.
            value: Value features.
            volume_h: Height of volume.
            volume_w: Width of volume.
            volume_z: Depth of volume.
            spatial_shapes: Spatial shapes of features.
            level_start_index: Start index for each level.
            img_metas: Image meta information.
            
        Returns:
            torch.Tensor: Encoded volume features.
        """
        output = bev_query
        intermediate = []

        bs, num_query, _ = bev_query.shape

        # Get reference points for 3D volume
        ref_3d = self.get_reference_points(
            volume_h, volume_w, volume_z, bs, 
            device=bev_query.device, dtype=bev_query.dtype)

        reference_points_cam, bev_mask = self.point_sampling(
            ref_3d, self.pc_range, img_metas)

        # Process through transformer layers
        for lid, layer in enumerate(self.layers):
            output = layer(
                bev_query,
                key,
                value,
                bev_pos=None,
                ref_2d=reference_points_cam,
                bev_h=volume_h,
                bev_w=volume_w,
                bev_z=volume_z,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                bev_mask=bev_mask,
                **kwargs)

            bev_query = output
            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


@DET3D_MODELS.register_module() 
@ENGINE_MODELS.register_module()
class OccLayer(BaseModule):
    """Occupancy Layer with spatial cross-attention and feed-forward network.
    
    Args:
        attn_cfgs (list): Attention configurations.
        feedforward_channels (int): Channels of feed-forward network.
        ffn_dropout (float): Dropout rate for FFN.
        embed_dims (int): Embedding dimensions.
        conv_num (int): Number of convolution layers.
        operation_order (tuple): Order of operations.
    """

    def __init__(self,
                 attn_cfgs: list = None,
                 feedforward_channels: int = 512,
                 ffn_dropout: float = 0.1,
                 embed_dims: int = 256,
                 conv_num: int = 2,
                 operation_order: tuple = ('cross_attn', 'norm', 'ffn', 'norm', 'conv'),
                 init_cfg: dict = None,
                 **kwargs):
        super(OccLayer, self).__init__(init_cfg=init_cfg)
        
        self.operation_order = operation_order
        self.embed_dims = embed_dims
        self.conv_num = conv_num
        
        # Build attention layers
        self.attentions = nn.ModuleList()
        if attn_cfgs is not None:
            for attn_cfg in attn_cfgs:
                self.attentions.append(ENGINE_MODELS.build(attn_cfg))
        
        # Build normalization layers
        self.norms = nn.ModuleList()
        for _ in range(operation_order.count('norm')):
            self.norms.append(nn.LayerNorm(embed_dims))
        
        # Build feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dims, feedforward_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(ffn_dropout),
            nn.Linear(feedforward_channels, embed_dims),
            nn.Dropout(ffn_dropout)
        )
        
        # Build convolution layers
        self.convs = nn.ModuleList()
        for _ in range(conv_num):
            conv = nn.Sequential(
                nn.Conv1d(embed_dims, embed_dims, 1),
                nn.ReLU(inplace=True)
            )
            self.convs.append(conv)

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor = None,
                value: torch.Tensor = None,
                bev_pos: torch.Tensor = None,
                ref_2d: torch.Tensor = None,
                bev_h: int = None,
                bev_w: int = None,
                bev_z: int = None,
                spatial_shapes: torch.Tensor = None,
                level_start_index: torch.Tensor = None,
                bev_mask: torch.Tensor = None,
                **kwargs) -> torch.Tensor:
        """Forward function."""
        norm_index = 0
        attn_index = 0
        conv_index = 0
        identity = query
        
        for op in self.operation_order:
            if op == 'cross_attn':
                if attn_index < len(self.attentions):
                    query = self.attentions[attn_index](
                        query, key, value,
                        reference_points=ref_2d,
                        spatial_shapes=spatial_shapes,
                        level_start_index=level_start_index,
                        **kwargs)
                    attn_index += 1
                    
            elif op == 'norm':
                if norm_index < len(self.norms):
                    query = self.norms[norm_index](query)
                    norm_index += 1
                    
            elif op == 'ffn':
                query = self.ffn(query) + query
                
            elif op == 'conv':
                if conv_index < len(self.convs):
                    bs, num_query, embed_dims = query.shape
                    query = query.permute(0, 2, 1)  # (bs, embed_dims, num_query)
                    query = self.convs[conv_index](query)
                    query = query.permute(0, 2, 1)  # (bs, num_query, embed_dims)
                    conv_index += 1

        return query


# Import required for version checking
try:
    from mmcv.utils import TORCH_VERSION, digit_version
except ImportError:
    import torch
    TORCH_VERSION = torch.__version__
    def digit_version(version_str):
        return tuple(map(int, version_str.split('.')[:2]))
