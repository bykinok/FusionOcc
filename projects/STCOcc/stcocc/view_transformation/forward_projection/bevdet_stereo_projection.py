# Copyright (c) Phigent Robotics. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet3d.registry import MODELS
from mmdet.models.backbones.resnet import ResNet

# from mmdet3d.models.detectors.bevdet import BEVDepth4D  # module not available
from mmengine.model import BaseModule

# Import debug utilities
import sys
from pathlib import Path
debug_path = Path(__file__).parent.parent.parent.parent.parent
if str(debug_path) not in sys.path:
    sys.path.insert(0, str(debug_path))


@MODELS.register_module()
class BEVDetStereoForwardProjection(BaseModule):
    def __init__(self,
                 return_intermediate=False,
                 stereo_feat_index=0,
                 cv_downsample=4,
                 adjust_channel=None,
                 img_context_idx=0,
                 # BEVDepth4D specific parameters that we need to handle
                 num_adj=1,
                 img_backbone=None,
                 img_neck=None,
                 img_view_transformer=None,
                 img_bev_encoder_backbone=None,
                 align_after_view_transfromation=False,
                 **kwargs):
        # Filter out kwargs that BaseModule doesn't accept
        base_kwargs = {k: v for k, v in kwargs.items() if k in ['init_cfg']}
        super(BEVDetStereoForwardProjection, self).__init__(**base_kwargs)
        
        # Initialize attributes that were expected from BEVDepth4D
        self.num_adj = num_adj
        self.num_frame = num_adj + 1
        self.with_prev = self.num_frame > 1
        
        # Build components using MODELS.build
        if img_backbone is not None:
            self.img_backbone = MODELS.build(img_backbone)
        if img_neck is not None:
            self.img_neck = MODELS.build(img_neck)
        if img_view_transformer is not None:
            self.img_view_transformer = MODELS.build(img_view_transformer)
        if img_bev_encoder_backbone is not None:
            self.img_bev_encoder_backbone = MODELS.build(img_bev_encoder_backbone)
            
        self.align_after_view_transfromation = align_after_view_transfromation
        self.extra_ref_frames = 1
        self.temporal_frame = self.num_frame
        self.num_frame += self.extra_ref_frames
        self.return_intermediate = return_intermediate
        self.stereo_feat_index = stereo_feat_index
        self.cv_downsample = cv_downsample
        self.img_context_idx = img_context_idx
        if adjust_channel is not None:
            self.adjust_channel_conv = nn.Conv2d(in_channels=self.img_view_transformer.out_channels, out_channels=adjust_channel, kernel_size=1, stride=1, padding=0)
        else:
            self.adjust_channel_conv = None
            
        # Initialize grid_mask as None (can be set later if needed)
        self.grid_mask = None
        
        # Initialize other required attributes
        self.with_img_neck = img_neck is not None
        
        # Initialize pre_process attribute (usually a preprocessing network)
        self.pre_process = None

    def prepare_inputs(self, inputs, stereo=False):

        # breakpoint() 
           
        # split the inputs into each frame
        B, N, C, H, W = inputs[0].shape
        N = N // self.num_frame
        imgs = inputs[0].view(B, N, self.num_frame, C, H, W)
        imgs = torch.split(imgs, 1, 2)
        imgs = [t.squeeze(2) for t in imgs]
        sensor2egos, ego2globals, intrins, post_augs, bda = inputs[1:]

        sensor2egos = sensor2egos.view(B, self.num_frame, N, 4, 4)
        ego2globals = ego2globals.view(B, self.num_frame, N, 4, 4)

        # calculate the transformation from sweep sensor to key ego
        keyego2global = ego2globals[:, 0, 0, ...].unsqueeze(1).unsqueeze(1)
        global2keyego = torch.inverse(keyego2global.double())
        sensor2keyegos = global2keyego @ ego2globals.double() @ sensor2egos.double()
        sensor2keyegos = sensor2keyegos.float()

        curr2adjsensor = None
        if stereo:
            sensor2egos_cv, ego2globals_cv = sensor2egos, ego2globals
            sensor2egos_curr = sensor2egos_cv[:, :self.temporal_frame, ...].double()
            ego2globals_curr = ego2globals_cv[:, :self.temporal_frame, ...].double()
            sensor2egos_adj = sensor2egos_cv[:, 1:self.temporal_frame + 1, ...].double()
            ego2globals_adj = ego2globals_cv[:, 1:self.temporal_frame + 1, ...].double()
            curr2adjsensor = torch.inverse(ego2globals_adj @ sensor2egos_adj) @ ego2globals_curr @ sensor2egos_curr
            curr2adjsensor = curr2adjsensor.float()
            curr2adjsensor = torch.split(curr2adjsensor, 1, 1)
            curr2adjsensor = [p.squeeze(1) for p in curr2adjsensor]
            curr2adjsensor.extend([None for _ in range(self.extra_ref_frames)])
            assert len(curr2adjsensor) == self.num_frame

        extra = [
            sensor2keyegos,
            ego2globals,
            intrins.view(B, self.num_frame, N, 4, 4),
            post_augs.view(B, self.num_frame, N, 4, 4)
        ]
        extra = [torch.split(t, 1, 1) for t in extra]
        extra = [[p.squeeze(1) for p in t] for t in extra]
        sensor2keyegos, ego2globals, intrins, post_augs = extra
        return imgs, sensor2keyegos, ego2globals, intrins, post_augs, bda, curr2adjsensor

    def with_specific_component(self, component_name):
        """Whether the model owns a specific component"""
        return getattr(self, component_name, None) is not None

    def bev_encoder(self, x):
        if self.with_specific_component('img_bev_encoder_backbone'):
            x = self.img_bev_encoder_backbone(x)
        if self.with_specific_component('img_bev_encoder_neck'):
            x = self.img_bev_encoder_neck(x)

        if self.return_intermediate:
            return x
        else:
            if type(x) in [list, tuple]:
                x = x[0]
            return x

    def extract_stereo_ref_feat(self, x):
        B, N, C, imH, imW = x.shape
        x = x.view(B * N, C, imH, imW)
        if isinstance(self.img_backbone, ResNet):
            if self.img_backbone.deep_stem:
                x = self.img_backbone.stem(x)
            else:
                x = self.img_backbone.conv1(x)
                x = self.img_backbone.norm1(x)
                x = self.img_backbone.relu(x)
            x = self.img_backbone.maxpool(x)
            for i, layer_name in enumerate(self.img_backbone.res_layers):
                res_layer = getattr(self.img_backbone, layer_name)
                x = res_layer(x)
                if i == self.stereo_feat_index:
                    return x
        else:
            x, hw_shape = self.img_backbone.patch_embed(x)
            if self.img_backbone.use_abs_pos_embed:
                absolute_pos_embed = F.interpolate(self.img_backbone.absolute_pos_embed, size=hw_shape, mode='bicubic')
                x = x + absolute_pos_embed.flatten(2).transpose(1, 2)
            x = self.img_backbone.drop_after_pos(x)

            for i, stage in enumerate(self.img_backbone.stages):
                x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
                out = out.view(-1,  *out_hw_shape,
                               self.img_backbone.num_features[i])
                out = out.permute(0, 3, 1, 2).contiguous()
                return out

    def image_encoder(self, img, stereo=False):
        imgs = img
        B, N, C, imH, imW = imgs.shape
        
        imgs = imgs.view(B * N, C, imH, imW)
        if self.grid_mask is not None:
            imgs = self.grid_mask(imgs)
        
        
        x = self.img_backbone(imgs)
        
        stereo_feat = None
        if stereo:
            stereo_feat = x[0]
            x = x[1:]
        if self.with_img_neck:
            x = self.img_neck(x)
            if type(x) is not list and type(x) is not tuple:
                x = [x]

        img_feats_reshape = []
        for feat in x:
            img_feats_reshape.append(feat.view(B, N, *feat.shape[1:]))
        return img_feats_reshape, stereo_feat

    def prepare_bev_feat(self,
                         img,
                         sensor2keyego,
                         ego2global,
                         intrin,
                         post_aug,
                         bda,
                         mlp_input,
                         feat_prev_iv,
                         k2s_sensor,
                         extra_ref_frame):
        if extra_ref_frame:
            stereo_feat = self.extract_stereo_ref_feat(img)
            return None, None, stereo_feat, None, None

        img_feats_reshape, stereo_feat = self.image_encoder(img, stereo=True)
        
        # CRITICAL FIX: post_aug stores translation in column 2 (index 2), not column 3!
        # This matches the original implementation in STCOcc_ori
        post_rot = torch.zeros_like(post_aug[:, :, :3, :3])
        post_rot[:, :, :2, :2] = post_aug[:, :, :2, :2]
        post_rot[:, :, 2, 2] = 1
        post_tran = torch.zeros_like(post_aug[:, :, :3, 0])
        post_tran[:, :, :2] = post_aug[:, :, :2, 2]  # Read from column 2, not 3!

        metas = dict(
            k2s_sensor=k2s_sensor,
            intrins=intrin,
            post_rots=post_rot,
            post_trans=post_tran,
            frustum=self.img_view_transformer.cv_frustum.to(img_feats_reshape[self.img_context_idx]),
            cv_downsample=self.cv_downsample,
            downsample=self.img_view_transformer.downsample,
            grid_config=self.img_view_transformer.grid_config,
            cv_feat_list=[feat_prev_iv, stereo_feat]
        )

        img_view_transformer_input = [img_feats_reshape[self.img_context_idx], sensor2keyego, ego2global, intrin, post_rot, post_tran, bda, mlp_input]
        bev_feat, depth, tran_feat = self.img_view_transformer(img_view_transformer_input, metas)
        
        if self.pre_process:
            bev_feat = self.pre_process_net(bev_feat)[0]

        return bev_feat, depth, stereo_feat, tran_feat, img_feats_reshape

    def extract_img_feat(self,
                         img,
                         img_metas,
                         pred_prev=False,
                         sequential=False,
                         **kwargs):
        if sequential:
            # Todo
            assert False

        # Prepare Input.
        imgs, sensor2keyegos, ego2globals, intrins, post_augs, bda, curr2adjsensor = self.prepare_inputs(img, stereo=True)

        # Extract features of images.
        bev_feat_list = []
        depth_key_frame = None
        tran_feat_key_frame = None
        ms_feat_key_frame = None
        feat_prev_iv = None
        # Use actual number of frames available
        actual_num_frames = len(imgs)
        for fid in range(actual_num_frames-1, -1, -1):
            # check if key_frame
            key_frame = fid == 0
            # get fid input
            img, sensor2keyego, ego2global, intrin, post_aug = imgs[fid], sensor2keyegos[fid], ego2globals[fid], intrins[fid], post_augs[fid]

            extra_ref_frame = fid == self.num_frame-self.extra_ref_frames

            if key_frame or self.with_prev:
                # get cam params mlp input
                mlp_input = self.img_view_transformer.get_mlp_input(sensor2keyegos[0], ego2globals[0], intrin, post_aug, bda)
                inputs_curr = (img,
                               sensor2keyego, ego2global, intrin, post_aug, bda,
                               mlp_input,
                               feat_prev_iv, curr2adjsensor[fid],
                               extra_ref_frame)
                # store the key frame features
                if key_frame:
                    bev_feat, depth, feat_curr_iv, tran_feat, img_feats_reshape = self.prepare_bev_feat(*inputs_curr)
                    depth_key_frame = depth
                    tran_feat_key_frame = tran_feat
                    ms_feat_key_frame = img_feats_reshape
                else:
                    with torch.no_grad():
                        bev_feat, depth, feat_curr_iv, tran_feat, img_feats_reshape = self.prepare_bev_feat(*inputs_curr)

                if not extra_ref_frame:
                    bev_feat_list.append(bev_feat)

                feat_prev_iv = feat_curr_iv

        # breakpoint()

        # bev_encoder to fuse multi-frame bev features
        bev_feat = torch.cat(bev_feat_list, dim=1)
        x = self.bev_encoder(bev_feat)
        
        cam_params_key_frame = [sensor2keyegos[0], ego2globals[0], intrins[0], post_augs[0], bda]

        # adjust channel
        if self.adjust_channel_conv is not None:
            tran_feat_key_frame = self.adjust_channel_conv(tran_feat_key_frame)

        return x, depth_key_frame, tran_feat_key_frame, ms_feat_key_frame, cam_params_key_frame

    def extract_feat(self, points, img, img_metas, **kwargs):
        """Extract features from images and points."""
        voxel_feats, depth, tran_feats, ms_feat_key_frame, cam_params = self.extract_img_feat(img, img_metas, **kwargs)
        return voxel_feats, depth, tran_feats, ms_feat_key_frame, cam_params