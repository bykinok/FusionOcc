# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_norm_layer
from mmengine.model import BaseModule
from mmdet3d.registry import MODELS
from functools import partial
from timm.models.layers import DropPath, Mlp
import time


class BasicBlock3D(nn.Module):
    """Basic 3D ResNet block."""
    
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = ConvModule(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            conv_cfg=dict(type='Conv3d'),
            norm_cfg=dict(type='BN3d'),
            act_cfg=dict(type='ReLU', inplace=True)
        )
        self.conv2 = ConvModule(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            conv_cfg=dict(type='Conv3d'),
            norm_cfg=dict(type='BN3d'),
            act_cfg=None  # No activation after conv2
        )
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        # 원본 모델과 동일한 순서: downsample을 먼저 확인
        if self.downsample is not None:
            identity = self.downsample(x)
        else:
            identity = x
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + identity
        return self.relu(x)


@MODELS.register_module()
class CustomResNet3D(BaseModule):
    """Custom 3D ResNet for BEV encoding with BasicBlock structure."""
    
    def __init__(self, 
                 numC_input=32,
                 num_layer=[1, 2, 4],
                 num_channels=[32, 64, 128],
                 stride=[1, 2, 2],
                 backbone_output_ids=[0, 1, 2],
                 with_cp=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_layer = num_layer
        self.num_channels = num_channels
        self.stride = stride
        self.backbone_output_ids = backbone_output_ids
        
        # 원본 모델과 동일하게: nn.Sequential 사용
        layers = []
        curr_numC = numC_input
        
        for i in range(len(num_layer)):
            layer = [
                BasicBlock3D(
                    curr_numC,
                    num_channels[i],
                    stride=stride[i],
                    downsample=ConvModule(
                        curr_numC,
                        num_channels[i],
                        kernel_size=3,
                        stride=stride[i],
                        padding=1,
                        bias=False,
                        conv_cfg=dict(type='Conv3d'),
                        norm_cfg=dict(type='BN3d'),
                        act_cfg=None
                    )
                )
            ]
            curr_numC = num_channels[i]
            layer.extend([
                BasicBlock3D(curr_numC, curr_numC)
                for _ in range(num_layer[i] - 1)
            ])
            layers.append(nn.Sequential(*layer))
        
        self.layers = nn.Sequential(*layers)
        self.with_cp = with_cp
            
    def forward(self, x):
        """Forward pass."""
        feats = []
        x_tmp = x
        for lid, layer in enumerate(self.layers):
            if self.with_cp:
                from torch.utils.checkpoint import checkpoint
                x_tmp = checkpoint(layer, x_tmp)
            else:
                x_tmp = layer(x_tmp)
            if lid in self.backbone_output_ids:
                feats.append(x_tmp)
        return feats


@MODELS.register_module()
class LSSFPN3D(BaseModule):
    """LSS FPN 3D for feature pyramid network."""
    
    def __init__(self, in_channels, out_channels, with_cp=False, **kwargs):
        super().__init__(**kwargs)
        self.up1 = nn.Upsample(
            scale_factor=2, mode='trilinear', align_corners=True)
        self.up2 = nn.Upsample(
            scale_factor=4, mode='trilinear', align_corners=True)

        self.conv = ConvModule(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            conv_cfg=dict(type='Conv3d'),
            norm_cfg=dict(type='BN3d'),
            act_cfg=dict(type='ReLU', inplace=True))
        self.with_cp = with_cp

    def forward(self, feats):
        """Forward pass with multi-scale features."""
        x_8, x_16, x_32 = feats
        x_16 = self.up1(x_16)
        x_32 = self.up2(x_32)
        x = torch.cat([x_8, x_16, x_32], dim=1)
        if self.with_cp:
            x = checkpoint(self.conv, x)
        else:
            x = self.conv(x)
        return x


def get_paddings_indicator(actual_num, max_num, axis=0):
    """Create boolean mask by actually number of a padded tensor."""
    actual_num = torch.unsqueeze(actual_num, axis + 1)
    max_num_shape = [1] * len(actual_num.shape)
    max_num_shape[axis + 1] = -1
    max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(
        max_num_shape
    )
    paddings_indicator = actual_num.int() > max_num
    return paddings_indicator


class RFNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, norm_cfg=None, last_layer=False):
        super().__init__()
        self.name = "RFNLayer"
        self.last_vfe = last_layer
        self.units = out_channels

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)
        self.norm_cfg = norm_cfg

        self.linear = nn.Linear(in_channels, self.units, bias=False)
        self.norm = build_norm_layer(self.norm_cfg, self.units)[1]

    def forward(self, inputs):
        x = self.linear(inputs)
        torch.backends.cudnn.enabled = False
        x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        torch.backends.cudnn.enabled = True
        x = F.relu(x)

        if self.last_vfe:
            x_max = torch.max(x, dim=1, keepdim=True)[0]
            return x_max
        else:
            return x


class PointEmbed(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_c, out_c, 1),
            nn.BatchNorm1d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_c, out_c, 1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(out_c*2, out_c*2, 1),
            nn.BatchNorm1d(out_c*2),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_c*2, out_c, 1)
        )

    def forward(self, points):
        bs, n, c = points.shape
        feature = self.conv1(points.transpose(2, 1))  # bs c n
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # bs c 1
        
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)# bs c*2 n
        feature = self.conv2(feature) # bs c*2 n

        return feature.transpose(2, 1)


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query, feat):
        B, N, C = query.shape
        _, M, _ = feat.shape
        
        q = self.q(query).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(feat).reshape(B, M, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Extractor(nn.Module):
    def __init__(self, dim, num_heads=6, n_points=4, n_levels=1, deform_ratio=1.0,
                 with_cffn=True, cffn_ratio=0.25, drop=0., drop_path=0.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), with_cp=False):
        super().__init__()
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = CrossAttention(dim, num_heads, qkv_bias=False, attn_drop=drop, proj_drop=drop)
        self.with_cffn = with_cffn
        self.with_cp = with_cp
        if with_cffn:
            self.ffn = Mlp(in_features=dim, hidden_features=int(dim * cffn_ratio), act_layer=nn.GELU, drop=drop)
            self.ffn_norm = norm_layer(dim)
            self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    
    def forward(self, query, feat):
        def _inner_forward(query, feat):
            attn = self.attn(self.query_norm(query), self.feat_norm(feat))
            query = query + attn
            
            # if self.with_cffn:
                # query = query + self.drop_path(self.ffn(self.ffn_norm(query)))
            query = self.drop_path(self.ffn(self.ffn_norm(query)))
            return query
        
        if self.with_cp and query.requires_grad:
            import torch.utils.checkpoint as cp
            query = cp.checkpoint(_inner_forward, query, feat)
        else:
            query = _inner_forward(query, feat)
        
        return query


class Injector(nn.Module):
    def __init__(self, dim, num_heads=6, n_points=4, n_levels=1, deform_ratio=1.0,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), init_values=0., with_cp=False, drop=0.):
        super().__init__()
        self.with_cp = with_cp
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = CrossAttention(dim, num_heads, qkv_bias=False, attn_drop=drop, proj_drop=drop)
        self.gamma = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        
    def forward(self, query, feat):
        def _inner_forward(query, feat):
            attn = self.attn(self.query_norm(query), self.feat_norm(feat))
            # return query + self.gamma * attn
            return self.gamma * attn
        
        if self.with_cp and query.requires_grad:
            import torch.utils.checkpoint as cp
            query = cp.checkpoint(_inner_forward, query, feat)
        else:
            query = _inner_forward(query, feat)
        
        return query


class SelfAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=6, n_points=4, n_levels=1, deform_ratio=1.0,
                 with_cffn=True, cffn_ratio=0.25, drop=0., drop_path=0.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), with_cp=False):
        super().__init__()
        self.query_norm = norm_layer(dim)
        self.attn = SparseSelfAttention(dim, num_heads, dropout=drop)
        self.with_cffn = with_cffn
        self.with_cp = with_cp
        if with_cffn:
            self.ffn = Mlp(in_features=dim, hidden_features=int(dim * 2), act_layer=nn.GELU, drop=drop)
            self.ffn_norm = norm_layer(dim)
            self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    
    def forward(self, feat, points):
        def _inner_forward(feat, points):
            identity = feat
            feat = self.query_norm(feat)
            feat = self.attn(points, feat)
            feat = feat + identity
            
            feat = self.drop_path(self.ffn(self.ffn_norm(feat)))
            return feat
        
        query = _inner_forward(feat, points)
        
        return query


class SparseSelfAttention(nn.Module):
    def __init__(self, embed_dims=256, num_heads=8, dropout=0.1):
        super().__init__()
        from mmcv.cnn.bricks.transformer import MultiheadAttention
        
        self.attention = MultiheadAttention(embed_dims, num_heads, dropout, batch_first=True)
        self.gen_tau = nn.Linear(embed_dims, num_heads)

    @torch.no_grad()
    def init_weights(self):
        nn.init.zeros_(self.gen_tau.weight)
        nn.init.uniform_(self.gen_tau.bias, 0.0, 2.0)

    def inner_forward(self, query_bbox, query_feat, pre_attn_mask):
        dist = self.calc_bbox_dists(query_bbox)
        tau = self.gen_tau(query_feat)

        tau = tau.permute(0, 2, 1)
        attn_mask = dist[:, None, :, :] * tau[..., None]
        if pre_attn_mask is not None:
            attn_mask[:, :, pre_attn_mask] = float('-inf')
        attn_mask = attn_mask.flatten(0, 1)
        return self.attention(query_feat, attn_mask=attn_mask)

    def forward(self, query_bbox, query_feat, pre_attn_mask=None):
        return self.inner_forward(query_bbox, query_feat, pre_attn_mask)

    @torch.no_grad()
    def calc_bbox_dists(self, points):
        centers = points[..., :2]

        dist = []
        for b in range(centers.shape[0]):
            dist_b = torch.norm(centers[b].reshape(-1, 1, 2) - centers[b].reshape(1, -1, 2), dim=-1)
            dist.append(dist_b[None, ...])

        dist = torch.cat(dist, dim=0)
        dist = -dist

        return dist


@MODELS.register_module()
class RadarEncoder(nn.Module):
    def __init__(
        self,
        in_channels=4,
        feat_channels=(64,),
        with_distance=False,
        voxel_size=(0.2, 0.2, 4),
        point_cloud_range=(0, -40, -3, 70.4, 40, 1),
        norm_cfg=None,
        with_pos_embed=False,
        return_rcs=False,
        drop=0.0,
        permute_injection_extraction=False,
    ):
        super().__init__()
        self.return_rcs = return_rcs
        self.name = "RadarFeatureNetAdapterNoMask"
        assert len(feat_channels) > 0

        self.in_channels = in_channels
        in_channels = in_channels + 2
        self._with_distance = with_distance

        # Create PillarFeatureNet layers
        feat_channels = [in_channels] + list(feat_channels)
        rfn_layers = []
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            if i < len(feat_channels) - 2:
                last_layer = False
            else:
                last_layer = False
            rfn_layers.append(
                RFNLayer(
                    in_filters, out_filters, norm_cfg=norm_cfg, last_layer=last_layer
                )
            )
        self.rfn_layers = nn.ModuleList(rfn_layers)

        # num_heads = 8 or 6
        num_heads = 2

        if permute_injection_extraction:
            injector = []
            for i in range(1, len(feat_channels)):
                injector.append(
                    Extractor(feat_channels[i], num_heads=num_heads, cffn_ratio=1,drop=drop, drop_path=drop)
                )
            self.injector = nn.ModuleList(injector)

            extractor = []
            for i in range(1, len(feat_channels)):
                extractor.append(
                    Injector(feat_channels[i], num_heads=num_heads,drop=drop)
                )
            self.extractor = nn.ModuleList(extractor)
        else:
            injector = []
            for i in range(1, len(feat_channels)):
                injector.append(
                    Injector(feat_channels[i], num_heads=num_heads,drop=drop)
                )
            self.injector = nn.ModuleList(injector)

            extractor = []
            for i in range(1, len(feat_channels)):
                extractor.append(
                    Extractor(feat_channels[i], num_heads=num_heads, cffn_ratio=1,drop=drop, drop_path=drop)
                )
            self.extractor = nn.ModuleList(extractor)

        adapterblock = []
        for i in range(1, len(feat_channels)):
            adapterblock.append(
                SelfAttentionBlock(feat_channels[i], num_heads=num_heads, cffn_ratio=1,drop=drop, drop_path=drop)
            )
        self.adapterblock = nn.ModuleList(adapterblock)

        linear_module = []
        for i in range(1, len(feat_channels)-1):
            linear_module.append(
                nn.Linear(feat_channels[i], feat_channels[i+1])
            )
        self.linear_module = nn.ModuleList(linear_module)

        self.out_linear = nn.Linear(feat_channels[-1]*2, feat_channels[-1])

        # Need pillar (voxel) size and x/y offset in order to calculate pillar offset
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]
        self.pc_range = point_cloud_range

        if with_pos_embed:
            embed_dims = feat_channels[1]
            self.pos_embed = nn.Sequential(
                        nn.Linear(3, embed_dims), 
                        nn.LayerNorm(embed_dims),
                        nn.ReLU(inplace=True),
                        nn.Linear(embed_dims, embed_dims),
                        nn.LayerNorm(embed_dims),
                        nn.ReLU(inplace=True),
                    )
        self.with_pos_embed = with_pos_embed
        
        self.point_embed = PointEmbed(in_channels+2, feat_channels[1])
    
    def compress(self, x):
        x = x.max(dim=1)[0]
        x = x.unsqueeze(dim=0)
        return x

    def forward(self, features, num_voxels, coors):
        dtype = features.dtype
        
        f_center = torch.zeros_like(features[:, :, :2])
        f_center[:, :, 0] = features[:, :, 0] - (
            coors[:, 1].to(dtype).unsqueeze(1) * self.vx + self.x_offset
        )
        f_center[:, :, 1] = features[:, :, 1] - (
            coors[:, 2].to(dtype).unsqueeze(1) * self.vy + self.y_offset
        )

        # normalize x,y,z to [0, 1]
        features[:, :, 0:1] = (features[:, :, 0:1] - self.pc_range[0]) / (self.pc_range[3] - self.pc_range[0])
        features[:, :, 1:2] = (features[:, :, 1:2] - self.pc_range[1]) / (self.pc_range[4] - self.pc_range[1])
        features[:, :, 2:3] = (features[:, :, 2:3] - self.pc_range[2]) / (self.pc_range[5] - self.pc_range[2])

        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_voxels, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)
        
        features_mean = torch.zeros_like(features[:, :, :2])

        features_mean[:, :, 0] = features[:, :, 0] - ((features[:, :, 0] * mask.squeeze()).sum(dim=1) / mask.squeeze().sum(dim=1)).unsqueeze(1)
        features_mean[:, :, 1] = features[:, :, 1] - ((features[:, :, 1] * mask.squeeze()).sum(dim=1) / mask.squeeze().sum(dim=1)).unsqueeze(1)

        rcs_features = features.clone()
        c = torch.cat([features, features_mean, f_center], dim=-1)
        x = torch.cat([features, f_center], dim=-1)

        # The feature decorations were calculated without regard to whether pillar was empty. Need to ensure that
        # empty pillars remain set to zeros.
        x *= mask
        c *= mask

        c = self.point_embed(c)
        if self.with_pos_embed:
            c = c + self.pos_embed(features[:, :, 0:3])
        points_coors = features[:, :, 0:3].detach()
        
        # Forward pass through PFNLayers
        batch_size = coors[-1, 0] + 1
        if batch_size>1:
            bs_list = [0]
            bs_info = coors[:, 0]
            pre = bs_info[0]
            for i in range(1, len(bs_info)):
                if pre != bs_info[i]:
                    bs_list.append(i)
                    pre = bs_info[i]
            bs_list.append(len(bs_info))
            bs_list = [bs_list[i+1]-bs_list[i] for i in range(len(bs_list)-1)]
        elif batch_size == 1:
            bs_list = [len(coors[:, 0])]
        else:
            assert False

        points_coors_split = torch.split(points_coors, bs_list)

        i = 0
        
        for rfn in self.rfn_layers:
            x = rfn(x)
            x_split = torch.split(x, bs_list)
            c_split = torch.split(c, bs_list)
            
            x_out_list = []
            c_out_list = []
            for bs in range(len(x_split)):
                c_tmp = c_split[bs]
                x_tmp = x_split[bs]
                points_coors_tmp = points_coors_split[bs]
                
                c_tmp = c_tmp + self.extractor[i](self.compress(c_tmp), self.compress(x_tmp)).transpose(1, 0).expand_as(c_tmp)
                x_tmp = x_tmp + self.injector[i](self.compress(x_tmp), self.compress(c_tmp)).transpose(1, 0).expand_as(x_tmp)
                c_tmp = self.adapterblock[i](self.compress(c_tmp), self.compress(points_coors_tmp)).transpose(1, 0).expand_as(c_tmp)
                if i < len(self.rfn_layers)-1:
                    c_tmp = self.linear_module[i](c_tmp)
                
                c_out_list.append(c_tmp)
                x_out_list.append(x_tmp)
            
            x = torch.cat(x_out_list, dim=0)
            c = torch.cat(c_out_list, dim=0)
            i += 1
        
        c = self.out_linear(torch.cat([c, x], dim=-1))
        c = torch.max(c, dim=1, keepdim=True)[0]
        
        if not self.return_rcs:
            return c.squeeze()
        else:
            rcs = (rcs_features*mask).sum(dim=1)/mask.sum(dim=1)
            return c.squeeze(), rcs.squeeze()


@MODELS.register_module()
class CustomFPN(BaseModule):
    """Custom Feature Pyramid Network."""
    
    def __init__(self, 
                 in_channels=[1024, 2048],
                 out_channels=256,
                 num_outs=1,
                 start_level=0,
                 out_ids=[0],
                 **kwargs):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_outs = num_outs
        self.start_level = start_level
        self.out_ids = out_ids
        
        # Lateral convs (using ConvModule to match checkpoint structure)
        self.lateral_convs = nn.ModuleList()
        for i in range(len(in_channels)):
            self.lateral_convs.append(
                ConvModule(
                    in_channels[i],
                    out_channels,
                    1,
                    conv_cfg=None,
                    norm_cfg=None,
                    act_cfg=None,
                    inplace=False
                )
            )
        
        # FPN convs (using ConvModule to match checkpoint structure)
        self.fpn_convs = nn.ModuleList()
        for i in range(num_outs):
            self.fpn_convs.append(
                ConvModule(
                    out_channels,
                    out_channels,
                    3,
                    padding=1,
                    conv_cfg=None,
                    norm_cfg=None,
                    act_cfg=None,
                    inplace=False
                )
            )
    
    def forward(self, inputs):
        """Forward pass."""
        # image_encoder already selects the correct features (backbone_feats[-2:])
        # So inputs already contains the right features for lateral convs
        # inputs[0]: 1024 channels
        # inputs[1]: 2048 channels
        
        # Build laterals directly from inputs
        laterals = []
        for i, lateral_conv in enumerate(self.lateral_convs):
            laterals.append(lateral_conv(inputs[i]))
        
        # Build top-down path
        for i in range(len(laterals) - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=prev_shape, mode='nearest')
        
        # Build outputs
        outs = []
        for i in range(self.num_outs):
            if i < len(laterals):
                outs.append(self.fpn_convs[i](laterals[i]))
            
        # Return only specified outputs
        if self.out_ids:
            return [outs[i] for i in self.out_ids if i < len(outs)]
        return outs
