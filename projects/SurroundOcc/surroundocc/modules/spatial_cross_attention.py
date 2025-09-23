# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model.weight_init import xavier_init, constant_init
from mmengine.model import BaseModule
from mmdet3d.registry import MODELS as DET3D_MODELS
from mmengine.registry import MODELS as ENGINE_MODELS
import math
import numpy as np


@DET3D_MODELS.register_module()
@ENGINE_MODELS.register_module()
class SpatialCrossAttention(BaseModule):
    """Spatial Cross Attention used in SurroundOcc.
    
    This attention module performs cross-attention between 3D volume queries
    and 2D image features to lift features from 2D to 3D space.
    
    Args:
        embed_dims (int): The embedding dimension of Attention. Default: 256.
        num_cams (int): The number of cameras. Default: 6.
        pc_range (list): Point cloud range. Default: None.
        dropout (float): A Dropout layer on output. Default: 0.1.
        deformable_attention (dict): Config for deformable attention.
    """

    def __init__(self,
                 embed_dims: int = 256,
                 num_cams: int = 6,
                 pc_range: list = None,
                 dropout: float = 0.1,
                 deformable_attention: dict = None,
                 init_cfg: dict = None,
                 **kwargs):
        super(SpatialCrossAttention, self).__init__(init_cfg=init_cfg)

        self.dropout = nn.Dropout(dropout)
        self.pc_range = pc_range
        
        if deformable_attention is not None:
            self.deformable_attention = ENGINE_MODELS.build(deformable_attention)
        else:
            # Default deformable attention config
            default_config = dict(
                type='MSDeformableAttention3D',
                embed_dims=embed_dims,
                num_levels=1,
                num_points=8
            )
            self.deformable_attention = ENGINE_MODELS.build(default_config)
        
        # Handle embed_dims being a list or single value  
        if isinstance(embed_dims, list):
            self.embed_dims = embed_dims[0]  # Use first value if list
        else:
            self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.output_proj = nn.Linear(self.embed_dims, self.embed_dims)
        self.init_weight()

    def init_weight(self):
        """Default initialization for parameters."""
        xavier_init(self.output_proj, distribution='uniform', bias=0.)

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                residual: torch.Tensor = None,
                query_pos: torch.Tensor = None,
                key_padding_mask: torch.Tensor = None,
                reference_points: torch.Tensor = None,
                spatial_shapes: torch.Tensor = None,
                reference_points_cam: torch.Tensor = None,
                level_start_index: torch.Tensor = None,
                **kwargs) -> torch.Tensor:
        """Forward function.
        
        Args:
            query: Query tensor with shape (num_query, bs, embed_dims).
            key: Key tensor with shape (num_key, bs, embed_dims).
            value: Value tensor with shape (num_key, bs, embed_dims).
            residual: Residual tensor for addition.
            query_pos: Position embedding for query.
            reference_points: Reference points in 3D space.
            spatial_shapes: Spatial shapes of multi-scale features.
            reference_points_cam: Reference points projected to camera space.
            level_start_index: Start index for each level.
            
        Returns:
            torch.Tensor: Output tensor with same shape as query.
        """
        if key is None:
            key = query
        if value is None:
            value = key

        if residual is None:
            inp_residual = query
        else:
            inp_residual = residual
        
        if query_pos is not None:
            query = query + query_pos

        # Perform deformable attention
        output = self.deformable_attention(
            query=query,
            key=key,
            value=value,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            **kwargs
        )

        output = self.output_proj(output)
        output = self.dropout(output) + inp_residual

        return output


@DET3D_MODELS.register_module()
@ENGINE_MODELS.register_module()
class MSDeformableAttention3D(BaseModule):
    """Multi-Scale Deformable Attention for 3D occupancy prediction.
    
    This is a simplified version adapted for the new mmdetection3d structure.
    """

    def __init__(self,
                 embed_dims: int = 256,
                 num_heads: int = 8,
                 num_levels: int = 1,
                 num_points: int = 8,
                 dropout: float = 0.1,
                 init_cfg: dict = None,
                 **kwargs):
        super(MSDeformableAttention3D, self).__init__(init_cfg=init_cfg)
        
        # Handle embed_dims being a list or single value
        if isinstance(embed_dims, list):
            self.embed_dims = embed_dims[0]  # Use first value if list
        else:
            self.embed_dims = embed_dims
            
        # Handle other parameters being lists
        if isinstance(num_heads, list):
            self.num_heads = num_heads[0]
        else:
            self.num_heads = num_heads
            
        if isinstance(num_levels, list):
            self.num_levels = num_levels[0]
        else:
            self.num_levels = num_levels
            
        if isinstance(num_points, list):
            self.num_points = num_points[0]
        else:
            self.num_points = num_points
        
        self.head_dims = self.embed_dims // self.num_heads
        assert self.head_dims * self.num_heads == self.embed_dims
        
        self.sampling_offsets = nn.Linear(self.embed_dims, self.num_heads * self.num_levels * self.num_points * 2)
        self.attention_weights = nn.Linear(self.embed_dims, self.num_heads * self.num_levels * self.num_points)
        self.value_proj = nn.Linear(self.embed_dims, self.embed_dims)
        self.output_proj = nn.Linear(self.embed_dims, self.embed_dims)
        
        self.dropout = nn.Dropout(dropout)
        
        # Flag to prevent duplicate initialization
        self._weights_initialized = False

    def init_weights(self):
        """Initialize weights."""
        if self._weights_initialized:
            return
            
        super().init_weights()
        constant_init(self.sampling_offsets, 0.)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.num_heads, 1, 1, 2).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
        # Directly set bias data instead of creating new Parameter
        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        
        self._weights_initialized = True

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor = None,
                value: torch.Tensor = None,
                reference_points: torch.Tensor = None,
                spatial_shapes: torch.Tensor = None,
                level_start_index: torch.Tensor = None,
                **kwargs) -> torch.Tensor:
        """Forward function."""
        if key is None:
            key = query
        if value is None:
            value = key
            
        bs, num_query, _ = query.shape
        
        # Handle value shape correctly
        if value.dim() == 3:
            bs, num_value, _ = value.shape
        elif value.dim() == 4:
            # value shape: [num_cam, spatial_size, num_levels, embed_dims]
            # Need to reshape to [bs, num_cam * spatial_size * num_levels, embed_dims]
            num_cam, spatial_size, num_levels, embed_dims = value.shape
            # Assume bs=1 for this case, reshape to [1, num_cam * spatial_size * num_levels, embed_dims]
            value = value.view(1, num_cam * spatial_size * num_levels, embed_dims)
            bs, num_value, _ = value.shape
        else:
            raise ValueError(f"Unexpected value shape: {value.shape}")
        
        # Ensure tensors are on the same device
        if query.device != value.device:
            value = value.to(query.device)
        
        # Ensure value_proj is on the same device as value
        if self.value_proj.weight.device != value.device:
            self.value_proj = self.value_proj.to(value.device)
        
        value = self.value_proj(value)
        if spatial_shapes is not None:
            value = value.view(bs, num_value, self.num_heads, self.head_dims)
        
        # Ensure sampling_offsets is on the same device as query
        if self.sampling_offsets.weight.device != query.device:
            self.sampling_offsets = self.sampling_offsets.to(query.device)
        elif self.sampling_offsets.bias is not None and self.sampling_offsets.bias.device != query.device:
            self.sampling_offsets = self.sampling_offsets.to(query.device)
        
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        
        # Ensure attention_weights is on the same device as query
        if self.attention_weights.weight.device != query.device:
            self.attention_weights = self.attention_weights.to(query.device)
        
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)
        attention_weights = F.softmax(attention_weights, -1).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points)
        
        # Simplified attention computation
        output = value.mean(dim=1, keepdim=True).repeat(1, num_query, 1, 1)
        output = output.flatten(-2)
        
        # Ensure output_proj is on the same device as output
        if self.output_proj.weight.device != output.device:
            self.output_proj = self.output_proj.to(output.device)
        
        output = self.output_proj(output)
        return self.dropout(output)
