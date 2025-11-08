# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional

from mmdet3d.registry import MODELS as DET3D_MODELS
from mmengine.registry import MODELS as ENGINE_MODELS
from mmengine.model import BaseModule
from mmcv.cnn import build_conv_layer, build_norm_layer, build_upsample_layer
from mmengine.model.weight_init import constant_init

try:
    from projects.SurroundOcc.surroundocc.loss.loss_utils import multiscale_supervision, geo_scal_loss, sem_scal_loss
except ImportError:
    from ..loss.loss_utils import multiscale_supervision, geo_scal_loss, sem_scal_loss


@DET3D_MODELS.register_module()
@ENGINE_MODELS.register_module() 
class OccHead(BaseModule): 
    """Occupancy Head for SurroundOcc.
    
    Args:
        transformer_template (dict): Template config for transformer.
        num_classes (int): Number of occupancy classes.
        volume_h (list): Height of volume features at different levels.
        volume_w (list): Width of volume features at different levels.
        volume_z (list): Depth of volume features at different levels.
        upsample_strides (list): Upsample strides for deconv layers.
        out_indices (list): Output indices for multi-scale supervision.
        conv_input (list): Input channels for conv layers.
        conv_output (list): Output channels for conv layers.
        embed_dims (list): Embedding dimensions at different levels.
        img_channels (list): Image feature channels at different levels.
        use_semantic (bool): Whether to use semantic occupancy prediction.
    """
    
    def __init__(self,
                 transformer_template: Optional[dict] = None,
                 num_classes: int = 17,
                 volume_h: List[int] = [200],
                 volume_w: List[int] = [200],
                 volume_z: List[int] = [16],
                 upsample_strides: List[int] = [1, 2, 1, 2],
                 out_indices: List[int] = [0, 2, 4, 6],
                 conv_input: Optional[List[int]] = None,
                 conv_output: Optional[List[int]] = None,
                 embed_dims: Optional[List[int]] = None,
                 img_channels: Optional[List[int]] = None,
                 use_semantic: bool = True,
                 init_cfg: Optional[dict] = None,
                 **kwargs):
        super(OccHead, self).__init__(init_cfg=init_cfg)
        
        self.conv_input = conv_input or [512, 256, 128, 64, 64]
        self.conv_output = conv_output or [256, 128, 64, 64, 32]
        self.num_classes = num_classes
        self.volume_h = volume_h if isinstance(volume_h, list) else [volume_h]
        self.volume_w = volume_w if isinstance(volume_w, list) else [volume_w]
        self.volume_z = volume_z if isinstance(volume_z, list) else [volume_z]
        self.img_channels = img_channels or [512, 512, 512]
        self.use_semantic = use_semantic
        self.embed_dims = embed_dims or [128, 256, 512]
        self.fpn_level = len(self.embed_dims)
        self.upsample_strides = upsample_strides
        self.out_indices = out_indices
        self.transformer_template = transformer_template

        self._init_layers()

    def _init_layers(self):
        """Initialize layers - following original SurroundOcc implementation."""
        # Build transformers for different FPN levels
        self.transformer = nn.ModuleList()
        if self.transformer_template is not None:
            for i in range(self.fpn_level):
                transformer = copy.deepcopy(self.transformer_template)
                
                # Update transformer embed_dims (must access from template since it's a list)
                transformer['embed_dims'] = self.transformer_template['embed_dims'][i]
                
                # Update encoder parameters
                if 'encoder' in transformer:
                    encoder_config = transformer['encoder']
                    
                    # CRITICAL: Set num_layers for this FPN level (matches original line 90)
                    # This determines how many OccLayer instances are created
                    # num_layers = [1, 3, 6] for levels [0, 1, 2]
                    encoder_config['num_layers'] = self.transformer_template['encoder']['num_layers'][i]
                    
                    if 'transformerlayers' in encoder_config:
                        
                        layer_config = encoder_config['transformerlayers']
                        
                        # Update layer embed_dims
                        layer_config['embed_dims'] = self.transformer_template['encoder']['transformerlayers']['embed_dims'][i]
                        
                        # Update feedforward_channels (use from template, not embed_dims * 2)
                        if 'feedforward_channels' in layer_config:
                            layer_config['feedforward_channels'] = self.transformer_template['encoder']['transformerlayers']['feedforward_channels'][i]
                        
                        # Update attention configs
                        if 'attn_cfgs' in layer_config:
                            for attn_cfg in layer_config['attn_cfgs']:
                                # Update attention embed_dims
                                if 'embed_dims' in attn_cfg:
                                    attn_cfg['embed_dims'] = self.transformer_template['encoder']['transformerlayers']['attn_cfgs'][0]['embed_dims'][i]
                                
                                # Update deformable attention
                                if 'deformable_attention' in attn_cfg:
                                    deform_attn = attn_cfg['deformable_attention']
                                    template_deform = self.transformer_template['encoder']['transformerlayers']['attn_cfgs'][0]['deformable_attention']
                                    
                                    # Update embed_dims
                                    if 'embed_dims' in deform_attn:
                                        deform_attn['embed_dims'] = template_deform['embed_dims'][i]
                                    
                                    # Update num_points (critical!)
                                    if 'num_points' in deform_attn:
                                        deform_attn['num_points'] = template_deform['num_points'][i]
                
                transformer_i = ENGINE_MODELS.build(transformer)
                self.transformer.append(transformer_i)

        # Build deconv blocks
        self.deblocks = nn.ModuleList()
        norm_cfg = dict(type='GN', num_groups=16, requires_grad=True)
        upsample_cfg = dict(type='deconv3d', bias=False)
        conv_cfg = dict(type='Conv3d', bias=False)

        for i, out_channel in enumerate(self.conv_output):
            stride = self.upsample_strides[i]
            if stride > 1:
                upsample_layer = build_upsample_layer(
                    upsample_cfg,
                    in_channels=self.conv_input[i],
                    out_channels=out_channel,
                    kernel_size=self.upsample_strides[i],
                    stride=self.upsample_strides[i])
            else:
                upsample_layer = build_conv_layer(
                    conv_cfg,
                    in_channels=self.conv_input[i],
                    out_channels=out_channel,
                    kernel_size=3,
                    stride=1,
                    padding=1)

            deblock = nn.Sequential(
                upsample_layer,
                build_norm_layer(norm_cfg, out_channel)[1],
                nn.ReLU(inplace=True))

            self.deblocks.append(deblock)

        # Build occupancy prediction layers
        self.occ = nn.ModuleList()
        for i in self.out_indices:
            if self.use_semantic:
                occ = build_conv_layer(
                    conv_cfg,
                    in_channels=self.conv_output[i],
                    out_channels=self.num_classes,
                    kernel_size=1,
                    stride=1,
                    padding=0)
            else:
                occ = build_conv_layer(
                    conv_cfg,
                    in_channels=self.conv_output[i],
                    out_channels=1,
                    kernel_size=1,
                    stride=1,
                    padding=0)
            self.occ.append(occ)

        # Build volume embeddings
        self.volume_embedding = nn.ModuleList()
        for i in range(self.fpn_level):
            self.volume_embedding.append(nn.Embedding(
                self.volume_h[i] * self.volume_w[i] * self.volume_z[i], 
                self.embed_dims[i]))

        # Build transfer conv layers
        self.transfer_conv = nn.ModuleList()
        conv_cfg = dict(type='Conv2d', bias=True)
        for i in range(self.fpn_level):
            transfer_layer = build_conv_layer(
                conv_cfg,
                in_channels=self.img_channels[i],
                out_channels=self.embed_dims[i],
                kernel_size=1,
                stride=1)
            transfer_block = nn.Sequential(
                transfer_layer,
                nn.ReLU(inplace=True))
            self.transfer_conv.append(transfer_block)

    def init_weights(self):
        """Initialize weights of the head."""
        super().init_weights()
        
        # Initialize transformer weights
        for i in range(self.fpn_level):
            if hasattr(self.transformer[i], 'init_weights'):
                self.transformer[i].init_weights()
                
        # Initialize deformable conv weights
        for m in self.modules():
            if hasattr(m, 'conv_offset'):
                constant_init(m.conv_offset, 0)

    def forward(self, mlvl_feats: List[torch.Tensor], img_metas: List[dict]) -> dict:
        """Forward function.
        
        Args:
            mlvl_feats: Multi-level image features.
            img_metas: Image meta information.
            
        Returns:
            dict: Predictions containing volume embeddings and occupancy predictions.
        """
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype

        # Extract volume embeddings at different levels
        volume_embed = []
        for i in range(self.fpn_level):
            # IMPORTANT: Keep volume_queries as (num_query, embed_dims) without batch dimension
            # The transformer will handle batching internally
            # Ensure volume_queries is on the same device as input features
            volume_queries = self.volume_embedding[i].weight.to(dtype=dtype, device=mlvl_feats[i].device)
            
            volume_h = self.volume_h[i] if i < len(self.volume_h) else self.volume_h[-1]
            volume_w = self.volume_w[i] if i < len(self.volume_w) else self.volume_w[-1]
            volume_z = self.volume_z[i] if i < len(self.volume_z) else self.volume_z[-1]

            _, _, C, H, W = mlvl_feats[i].shape
            view_features = self.transfer_conv[i](
                mlvl_feats[i].reshape(bs*num_cam, C, H, W)
            ).reshape(bs, num_cam, -1, H, W)

            volume_embed_i = self.transformer[i](
                [view_features],
                volume_queries,
                volume_h=volume_h,
                volume_w=volume_w,
                volume_z=volume_z,
                img_metas=img_metas
            )
            volume_embed.append(volume_embed_i)

        # Reshape volume embeddings to 3D
        volume_embed_reshape = []
        for i in range(self.fpn_level):
            volume_h = self.volume_h[i] if i < len(self.volume_h) else self.volume_h[-1]
            volume_w = self.volume_w[i] if i < len(self.volume_w) else self.volume_w[-1]
            volume_z = self.volume_z[i] if i < len(self.volume_z) else self.volume_z[-1]

            volume_embed_reshape_i = volume_embed[i].reshape(
                bs, volume_z, volume_h, volume_w, -1
            ).permute(0, 4, 3, 2, 1)
            
            volume_embed_reshape.append(volume_embed_reshape_i)

        # Progressive upsampling with skip connections
        outputs = []
        result = volume_embed_reshape.pop()
        
        for i in range(len(self.deblocks)):
            result = self.deblocks[i](result)

            if i in self.out_indices:
                outputs.append(result)
            elif i < len(self.deblocks) - 2:  # Skip connection
                volume_embed_temp = volume_embed_reshape.pop()
                
                # Handle shape mismatch by interpolating to match result's shape
                if result.shape != volume_embed_temp.shape:
                    import torch.nn.functional as F
                    # Interpolate volume_embed_temp to match result's spatial dimensions
                    target_shape = result.shape[2:]  # Get spatial dimensions (H, W, D)
                    volume_embed_temp = F.interpolate(
                        volume_embed_temp, 
                        size=target_shape, 
                        mode='trilinear', 
                        align_corners=False
                    )
                
                result = result + volume_embed_temp

        # Generate occupancy predictions
        occ_preds = []
        for i in range(len(outputs)):
            occ_pred = self.occ[i](outputs[i])
            occ_preds.append(occ_pred)

        outs = {
            'volume_embed': volume_embed,
            'occ_preds': occ_preds,
        }

        return outs

    def loss(self, gt_occ: torch.Tensor, preds_dicts: dict, img_metas: List[dict]) -> dict:
        """Calculate loss.
        
        Args:
            gt_occ: Ground truth occupancy.
            preds_dicts: Prediction dictionaries.
            img_metas: Image meta information.
            
        Returns:
            dict: Loss dictionary.
        """
        loss_dict = {}
        
        if not self.use_semantic:
            # Binary occupancy loss
            for i in range(len(preds_dicts['occ_preds'])):
                pred = preds_dicts['occ_preds'][i][:, 0]
                
                # Multi-scale supervision using original implementation
                ratio = 2 ** (len(preds_dicts['occ_preds']) - 1 - i)
                gt = multiscale_supervision(gt_occ.clone(), ratio, preds_dicts['occ_preds'][i].shape)
                
                # Add geo_scal_loss to binary loss (following original implementation)
                loss_occ_i = (F.binary_cross_entropy_with_logits(pred, gt) + 
                             geo_scal_loss(pred, gt.long(), semantic=False))
                loss_weight = (0.5) ** (len(preds_dicts['occ_preds']) - 1 - i)
                loss_dict[f'loss_occ_{i}'] = loss_occ_i * loss_weight
        else:
            # Semantic occupancy loss
            criterion = nn.CrossEntropyLoss(ignore_index=255, reduction="mean")
            
            for i in range(len(preds_dicts['occ_preds'])):
                pred = preds_dicts['occ_preds'][i]
                ratio = 2 ** (len(preds_dicts['occ_preds']) - 1 - i)
                
                # Use original multiscale_supervision implementation
                gt = multiscale_supervision(gt_occ.clone(), ratio, preds_dicts['occ_preds'][i].shape)
                
                # CRITICAL FIX: Add sem_scal_loss and geo_scal_loss (matching original line 292)
                # This is the main difference between original and reimplementation
                loss_occ_i = (criterion(pred, gt.long()) + 
                             sem_scal_loss(pred, gt.long()) + 
                             geo_scal_loss(pred, gt.long()))
                
                loss_weight = (0.5) ** (len(preds_dicts['occ_preds']) - 1 - i)
                loss_dict[f'loss_occ_{i}'] = loss_occ_i * loss_weight

        return loss_dict
