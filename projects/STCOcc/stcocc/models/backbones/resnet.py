# Copyright (c) Phigent Robotics. All rights reserved.
import torch.utils.checkpoint as checkpoint
from torch import nn

from mmcv.cnn.bricks.conv_module import ConvModule
from mmdet3d.registry import MODELS
from mmdet.models.backbones.resnet import BasicBlock, Bottleneck

# Import debug utilities
import sys
from pathlib import Path
debug_path = Path(__file__).parent.parent.parent.parent
if str(debug_path) not in sys.path:
    sys.path.insert(0, str(debug_path))
try:
    from debug_layer_comparison import log_layer
except ImportError:
    def log_layer(*args, **kwargs):
        pass

@MODELS.register_module()
class CustomResNet(nn.Module):

    def __init__(
            self,
            numC_input,
            num_layer=[2, 2, 2],
            num_channels=None,
            stride=[2, 2, 2],
            backbone_output_ids=None,
            norm_cfg=dict(type='BN'),
            with_cp=False,
            block_type='Basic',
    ):
        super(CustomResNet, self).__init__()
        # build backbone
        assert len(num_layer) == len(stride)
        num_channels = [numC_input*2**(i+1) for i in range(len(num_layer))] \
            if num_channels is None else num_channels
        self.backbone_output_ids = range(len(num_layer)) \
            if backbone_output_ids is None else backbone_output_ids
        layers = []
        if block_type == 'BottleNeck':
            curr_numC = numC_input
            for i in range(len(num_layer)):
                layer = [
                    Bottleneck(
                        curr_numC,
                        num_channels[i] // 4,
                        stride=stride[i],
                        downsample=nn.Conv2d(curr_numC, num_channels[i], 3,
                                             stride[i], 1),
                        norm_cfg=norm_cfg)
                ]
                curr_numC = num_channels[i]
                layer.extend([
                    Bottleneck(curr_numC, curr_numC // 4, norm_cfg=norm_cfg)
                    for _ in range(num_layer[i] - 1)
                ])
                layers.append(nn.Sequential(*layer))
        elif block_type == 'Basic':
            curr_numC = numC_input
            for i in range(len(num_layer)):
                layer = [
                    BasicBlock(
                        curr_numC,
                        num_channels[i],
                        stride=stride[i],
                        downsample=nn.Conv2d(curr_numC, num_channels[i], 3,
                                             stride[i], 1),
                        norm_cfg=norm_cfg)
                ]
                curr_numC = num_channels[i]
                layer.extend([
                    BasicBlock(curr_numC, curr_numC, norm_cfg=norm_cfg)
                    for _ in range(num_layer[i] - 1)
                ])
                layers.append(nn.Sequential(*layer))
        else:
            assert False
        self.layers = nn.Sequential(*layers)

        self.with_cp = with_cp

    def forward(self, x):
        feats = []
        x_tmp = x
        for lid, layer in enumerate(self.layers):
            if self.with_cp:
                x_tmp = checkpoint.checkpoint(layer, x_tmp)
            else:
                x_tmp = layer(x_tmp)
            if lid in self.backbone_output_ids:
                feats.append(x_tmp)
        
        # DEBUG: Log BEV encoder output
        log_layer("bev_encoder", outputs={"feats_" + str(i): feat for i, feat in enumerate(feats)})
        
        return feats


class BasicBlock3D(nn.Module):
    def __init__(self,
                 channels_in, channels_out, stride=1, downsample=None):
        super(BasicBlock3D, self).__init__()
        self.conv1 = ConvModule(
            channels_in,
            channels_out,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            conv_cfg=dict(type='Conv3d'),
            norm_cfg=dict(type='BN3d', ),
            act_cfg=dict(type='ReLU',inplace=True))
        self.conv2 = ConvModule(
            channels_out,
            channels_out,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            conv_cfg=dict(type='Conv3d'),
            norm_cfg=dict(type='BN3d', ),
            act_cfg=None)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: [bs, c, z, h, w]
        if self.downsample is not None:
            identity = self.downsample(x)
        else:
            identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + identity
        return self.relu(x)

@MODELS.register_module()
class CustomResNet3D(nn.Module):

    def __init__(
            self,
            numC_input,
            num_layer=[2, 2, 2],
            num_channels=None,
            stride=[2, 2, 2],
            backbone_output_ids=None,
            adjust_number_channel=None,
            with_cp=False,
    ):
        super(CustomResNet3D, self).__init__()
        # build backbone
        assert len(num_layer) == len(stride)
        num_channels = [numC_input*2**(i+1) for i in range(len(num_layer))] \
            if num_channels is None else num_channels
        self.backbone_output_ids = range(len(num_layer)) \
            if backbone_output_ids is None else backbone_output_ids
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
                        norm_cfg=dict(type='BN3d', ),
                        act_cfg=None))
            ]
            curr_numC = num_channels[i]
            layer.extend([
                BasicBlock3D(curr_numC, curr_numC)
                for _ in range(num_layer[i] - 1)
            ])
            layers.append(nn.Sequential(*layer))
        self.layers = nn.Sequential(*layers)
        self.adjust_number_channel = adjust_number_channel
        self.adjust_layer = nn.ModuleList()
        if self.adjust_number_channel is not None:
            if adjust_number_channel == num_channels[0]:
                adjust_number = len(num_channels) - 1
            else:
                adjust_number = len(num_channels)
            self.adjust_number = adjust_number
            for i in range(adjust_number):
                adjust_layer = ConvModule(
                        num_channels[i] if adjust_number==len(num_channels) else num_channels[i+1],
                        adjust_number_channel,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=False,
                        conv_cfg=dict(type='Conv3d'),
                        norm_cfg=dict(type='BN3d', ),
                        act_cfg=dict(type='ReLU',inplace=True)
                )
                self.adjust_layer.append(adjust_layer)

        self.with_cp = with_cp

    def forward(self, x):
        # DEBUG: Log BEV encoder input
        log_layer("bev_encoder", inputs={"x": x})
        
        feats = []
        x_tmp = x
        for lid, layer in enumerate(self.layers):
            # DEBUG: Check LAYER0 weights before forward
            if lid == 0 and not hasattr(self, '_layer0_weights_debug'):
                print(f"\n[BEV_ENCODER_LAYER0_WEIGHTS]:")
                # First block
                first_block = layer[0]
                if hasattr(first_block, 'conv1'):
                    # ConvModule uses .conv attribute
                    conv1 = first_block.conv1.conv if hasattr(first_block.conv1, 'conv') else first_block.conv1
                    print(f"  conv1.weight: mean={conv1.weight.mean().item():.6f}, std={conv1.weight.std().item():.6f}")
                if hasattr(first_block, 'bn1'):
                    # BN might be in .bn attribute
                    bn1 = first_block.bn1.bn if hasattr(first_block.bn1, 'bn') else first_block.bn1
                    print(f"  bn1.weight: mean={bn1.weight.mean().item():.6f}, std={bn1.weight.std().item():.6f}")
                    print(f"  bn1.running_mean: mean={bn1.running_mean.mean().item():.6f}, std={bn1.running_mean.std().item():.6f}")
                    print(f"  bn1.running_var: mean={bn1.running_var.mean().item():.6f}, min={bn1.running_var.min().item():.6f}")
                    print(f"  bn1.training={bn1.training}")
                self._layer0_weights_debug = True
            
            if self.with_cp:
                x_tmp = checkpoint.checkpoint(layer, x_tmp)
            else:
                x_tmp = layer(x_tmp)
            
            # DEBUG: Print output after each layer
            if not hasattr(self, f'_layer{lid}_output_debug'):
                print(f"\n[BEV_ENCODER_LAYER{lid}]:")
                print(f"  Shape: {x_tmp.shape}")
                print(f"  Mean: {x_tmp.mean().item():.6f}, Std: {x_tmp.std().item():.6f}")
                print(f"  Min: {x_tmp.min().item():.6f}, Max: {x_tmp.max().item():.6f}")
                setattr(self, f'_layer{lid}_output_debug', True)
            
            if lid in self.backbone_output_ids:
                feats.append(x_tmp)

        if self.adjust_number_channel is not None:
            for i in range(self.adjust_number):
                if self.adjust_number == len(feats):
                    feats[i] = self.adjust_layer[i](feats[i])
                else:
                    feats[i+1] = self.adjust_layer[i](feats[i+1])

        return feats