# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import numpy as np
import torch
import torch.nn as nn
from mmengine.model.weight_init import xavier_init
from mmengine.model import BaseModule
from mmdet3d.registry import MODELS as DET3D_MODELS
from mmengine.registry import MODELS as ENGINE_MODELS
from torch.nn.init import normal_

from .spatial_cross_attention import MSDeformableAttention3D


@DET3D_MODELS.register_module()
@ENGINE_MODELS.register_module()
class PerceptionTransformer(BaseModule):
    """Perception Transformer for SurroundOcc.
    
    This transformer implements spatial cross-attention to lift 2D image features
    to 3D volume space for occupancy prediction.
    
    Args:
        num_feature_levels (int): Number of feature maps from FPN. Default: 4.
        num_cams (int): Number of cameras. Default: 6.
        encoder (dict): Config of the transformer encoder.
        embed_dims (int): Embedding dimension. Default: 256.
        use_cams_embeds (bool): Whether to use camera embeddings. Default: True.
        rotate_center (list): Center for rotation. Default: [100, 100].
    """

    def __init__(self,
                 num_feature_levels: int = 4,
                 num_cams: int = 6,
                 encoder: dict = None,
                 embed_dims: int = 256,
                 use_cams_embeds: bool = True,
                 rotate_center: list = [100, 100],
                 init_cfg: dict = None,
                 **kwargs):
        super(PerceptionTransformer, self).__init__(init_cfg=init_cfg)

        if encoder is not None:
            self.encoder = ENGINE_MODELS.build(encoder)
        else:
            self.encoder = None

        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.use_cams_embeds = use_cams_embeds
        self.rotate_center = rotate_center

        self.init_layers()

    def init_layers(self):
        """Initialize layers of the transformer."""
        self.level_embeds = nn.Parameter(torch.Tensor(
            self.num_feature_levels, self.embed_dims))
        if self.use_cams_embeds:
            self.cams_embeds = nn.Parameter(
                torch.Tensor(self.num_cams, self.embed_dims))
        self.reference_points = nn.Linear(self.embed_dims, 3)

    def init_weights(self):
        """Initialize the transformer weights."""
        super().init_weights()
        
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformableAttention3D):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()
        normal_(self.level_embeds)
        if self.use_cams_embeds:
            normal_(self.cams_embeds)
        xavier_init(self.reference_points, distribution='uniform', bias=0.)

    def forward(self,
                mlvl_feats: list,
                volume_queries: torch.Tensor,
                volume_h: int,
                volume_w: int,
                volume_z: int,
                **kwargs) -> torch.Tensor:
        """Forward function.
        
        Args:
            mlvl_feats: Multi-level features from FPN.
            volume_queries: Volume queries for 3D space.
            volume_h: Height of volume.
            volume_w: Width of volume.
            volume_z: Depth of volume.
            
        Returns:
            torch.Tensor: Volume embeddings.
        """
        bs = mlvl_feats[0].size(0)
        
        # Check if volume_queries already has batch dimension
        if volume_queries.dim() == 2:
            # volume_queries shape: (num_query, embed_dims) -> (bs, num_query, embed_dims)
            volume_queries = volume_queries.unsqueeze(0).repeat(bs, 1, 1)
        elif volume_queries.dim() == 3 and volume_queries.size(0) == bs:
            # volume_queries already has correct shape: (bs, num_query, embed_dims)
            pass
        else:
            raise ValueError(f"Unexpected volume_queries shape: {volume_queries.shape}, expected batch size: {bs}")

        feat_flatten = []
        spatial_shapes = []
        
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2)  # num_cam, bs, hw, c

            if self.use_cams_embeds:
                feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
            feat = feat + self.level_embeds[None, None, lvl:lvl + 1, :].to(feat.dtype)
            
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = torch.cat(feat_flatten, 2)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        feat_flatten = feat_flatten.permute(0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims)

        if self.encoder is not None:
            volume_embed = self.encoder(
                volume_queries,
                feat_flatten,
                feat_flatten,
                volume_h=volume_h,
                volume_w=volume_w,
                volume_z=volume_z,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                **kwargs
            )
        else:
            volume_embed = volume_queries

        return volume_embed
