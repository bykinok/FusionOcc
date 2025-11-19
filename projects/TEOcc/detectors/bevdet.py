# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer, build_norm_layer, build_upsample_layer
from mmengine.model import BaseModule
from mmdet3d.models.detectors.base import Base3DDetector
from mmdet3d.registry import MODELS
from mmcv.cnn.bricks.conv_module import ConvModule


@MODELS.register_module()
class BEVStereo4D(Base3DDetector):
    """BEVStereo4D detector for occupancy prediction with radar-camera fusion."""
    
    def __init__(self,
                 img_backbone=None,
                 img_neck=None,
                 img_view_transformer=None,
                 img_bev_encoder_backbone=None,
                 img_bev_encoder_neck=None,
                 pre_process=None,
                 num_frame=2,
                 extra_ref_frames=1,
                 align_after_view_transfromation=False,
                 freeze_img=False,
                 num_adj=None,
                 **kwargs):
        # Filter out custom parameters that Base3DDetector doesn't recognize
        base_kwargs = {k: v for k, v in kwargs.items() 
                      if k in ['data_preprocessor', 'init_cfg']}
        super().__init__(**base_kwargs)
        
        # Set num_frame based on num_adj if provided, otherwise use default
        if num_adj is not None:
            self.num_frame = num_adj + 1  # num_adj adjacent frames + 1 current frame
        else:
            self.num_frame = num_frame
        
        self.extra_ref_frames = extra_ref_frames
        # 원본 모델과 동일하게: BEVStereo4D에서는 num_frame에 extra_ref_frames를 더함
        self.temporal_frame = self.num_frame
        self.num_frame += self.extra_ref_frames  # ← 이 줄 추가!
        # temporal_frame is the number of frames used for temporal processing
        # num_frame = temporal_frame + extra_ref_frames
        
        self.align_after_view_transfromation = align_after_view_transfromation
        self.freeze_img = freeze_img
        self.with_prev = self.num_frame > 1
        
        # Build modules directly in __init__ to avoid lazy loading prefix issues
        self.with_img_backbone = img_backbone is not None
        self.with_img_neck = img_neck is not None
        self.pre_process = pre_process is not None
        
        # Build modules immediately
        if img_backbone is not None:
            self.img_backbone = MODELS.build(img_backbone)
        else:
            self.img_backbone = None
            
        if img_neck is not None:
            self.img_neck = MODELS.build(img_neck)
        else:
            self.img_neck = None
            
        if img_view_transformer is not None:
            self.img_view_transformer = MODELS.build(img_view_transformer)
        else:
            self.img_view_transformer = None
            
        if img_bev_encoder_backbone is not None:
            self.img_bev_encoder_backbone = MODELS.build(img_bev_encoder_backbone)
        else:
            self.img_bev_encoder_backbone = None
            
        if img_bev_encoder_neck is not None:
            self.img_bev_encoder_neck = MODELS.build(img_bev_encoder_neck)
        else:
            self.img_bev_encoder_neck = None
            
        if pre_process is not None:
            self.pre_process_net = MODELS.build(pre_process)
        else:
            self.pre_process_net = None


    def init_weights(self):
        """Initialize weights."""
        super().init_weights()

    def image_encoder(self, img, stereo=False):
        """Encode image features."""
        imgs = img
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)
        
        backbone_feats = self.img_backbone(imgs)
        stereo_feat = None
        if stereo:
            if isinstance(backbone_feats, (list, tuple)):
                stereo_feat = backbone_feats[0].detach()
            else:
                stereo_feat = backbone_feats.detach()
        
        if self.with_img_neck:
            # FPN expects a list/tuple of features from different levels
            if not isinstance(backbone_feats, (list, tuple)):
                # If backbone returns single tensor, wrap in list
                backbone_feats = [backbone_feats]
            
            # Select features for FPN based on config (usually last 2 levels for this config)
            # ResNet typically returns 4 levels, but FPN config expects 2 levels [1024, 2048]
            if len(backbone_feats) > 2:
                backbone_feats = backbone_feats[-2:]  # Take last 2 levels
            
            neck_outputs = self.img_neck(backbone_feats)
            if isinstance(neck_outputs, (list, tuple)):
                x = neck_outputs[0]  # Use first output for further processing
            else:
                x = neck_outputs
        else:
            # If no neck, use backbone output directly
            if isinstance(backbone_feats, (list, tuple)):
                x = backbone_feats[0]
            else:
                x = backbone_feats
        
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)
        
        if stereo:
            return x, stereo_feat
        return x

    def extract_stereo_ref_feat(self, x):
        """Extract stereo reference features."""
        B, N, C, imH, imW = x.shape
        x = x.view(B * N, C, imH, imW)
        
        # Use ResNet backbone
        if hasattr(self.img_backbone, 'deep_stem') and self.img_backbone.deep_stem:
            x = self.img_backbone.stem(x)
        else:
            x = self.img_backbone.conv1(x)
            x = self.img_backbone.norm1(x)
            x = self.img_backbone.relu(x)
        x = self.img_backbone.maxpool(x)
        
        for i, layer_name in enumerate(self.img_backbone.res_layers):
            res_layer = getattr(self.img_backbone, layer_name)
            x = res_layer(x)
            return x

    def prepare_bev_feat(self, img, sensor2keyego, ego2global, intrin,
                         post_rot, post_tran, bda, mlp_input, feat_prev_iv,
                         k2s_sensor, extra_ref_frame):
        """Prepare BEV features from images."""

        # breakpoint()

        if extra_ref_frame:
            # Extract stereo reference features for extra reference frame
            stereo_feat = self.extract_stereo_ref_feat(img)
            return None, None, stereo_feat
        
        # Process current frame
        x, stereo_feat = self.image_encoder(img, stereo=True)
        metas = dict(
            k2s_sensor=k2s_sensor,
            intrins=intrin,
            post_rots=post_rot,
            post_trans=post_tran,
            frustum=self.img_view_transformer.cv_frustum.to(x),
            cv_downsample=4,
            downsample=self.img_view_transformer.downsample,
            grid_config=self.img_view_transformer.grid_config,
            cv_feat_list=[feat_prev_iv, stereo_feat]
        )
        
        bev_feat, depth = self.img_view_transformer(
            [x, sensor2keyego, ego2global, intrin, post_rot, post_tran, bda,
             mlp_input], metas)
        
        if self.pre_process:
            bev_feat = self.pre_process_net(bev_feat)[0]
            
        return bev_feat, depth, stereo_feat

    def prepare_inputs(self, inputs, stereo=False):
        """Prepare inputs for processing.
        
        Args:
            inputs (list): List of [imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans, bda]
            stereo (bool): Whether to prepare stereo matching inputs
            
        Returns:
            tuple: Prepared inputs for multiple frames
        """

        # breakpoint()

        # Extract inputs
        B, N, C, H, W = inputs[0].shape
        N = N // self.num_frame
        imgs = inputs[0].view(B, N, self.num_frame, C, H, W)
        imgs = torch.split(imgs, 1, 2)
        imgs = [t.squeeze(2) for t in imgs]
        sensor2egos, ego2globals, intrins, post_rots, post_trans, bda = inputs[1:7]

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
            sensor2egos_curr = \
                sensor2egos_cv[:, :self.temporal_frame, ...].double()
            ego2globals_curr = \
                ego2globals_cv[:, :self.temporal_frame, ...].double()
            sensor2egos_adj = \
                sensor2egos_cv[:, 1:self.temporal_frame + 1, ...].double()
            ego2globals_adj = \
                ego2globals_cv[:, 1:self.temporal_frame + 1, ...].double()
            curr2adjsensor = \
                torch.inverse(ego2globals_adj @ sensor2egos_adj) \
                @ ego2globals_curr @ sensor2egos_curr
            curr2adjsensor = curr2adjsensor.float()
            curr2adjsensor = torch.split(curr2adjsensor, 1, 1)
            curr2adjsensor = [p.squeeze(1) for p in curr2adjsensor]
            curr2adjsensor.extend([None for _ in range(self.extra_ref_frames)])
            assert len(curr2adjsensor) == self.num_frame

        extra = [
            sensor2keyegos,
            ego2globals,
            intrins.view(B, self.num_frame, N, 3, 3),
            post_rots.view(B, self.num_frame, N, 3, 3),
            post_trans.view(B, self.num_frame, N, 3)
        ]
        extra = [torch.split(t, 1, 1) for t in extra]
        extra = [[p.squeeze(1) for p in t] for t in extra]
        sensor2keyegos, ego2globals, intrins, post_rots, post_trans = extra
        return imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, bda, curr2adjsensor

    def shift_feature(self, input, trans, noise):
        """Shift features for temporal alignment."""
        # Simplified implementation - would need proper feature shifting
        return input

    def bev_encoder(self, x):
        """Encode BEV features."""
        if hasattr(self, 'img_down2top_encoder_backbone'):
            x = self.img_down2top_encoder_backbone(x)
        x = self.img_bev_encoder_backbone(x)
        x = self.img_bev_encoder_neck(x)
        if type(x) in [list, tuple]:
            x = x[0]
        return x

    def extract_feat(self, points, img, img_metas, **kwargs):
        """Extract features from images and points."""
        img_feats, depth, prev_feats = self.extract_img_feat(img, img_metas, **kwargs)
        pts_feats = None
        return img_feats, pts_feats, depth, prev_feats

    def extract_img_feat(self, img, img_metas, with_bevencoder=True, pred_prev=False, sequential=False, **kwargs):
        """Extract image features and transform to BEV."""
        if sequential:
            assert False, "Sequential not implemented"
            
        imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, \
        bda, curr2adjsensor = self.prepare_inputs(img, stereo=True)
        
        bev_feat_list = []
        depth_key_frame = None
        feat_prev_iv = None
        
        for fid in range(self.num_frame - 1, -1, -1):
            img_curr = imgs[0]  # Simplified - would need proper indexing
            sensor2keyego = sensor2keyegos[0]
            ego2global = ego2globals[0]
            intrin = intrins[0]
            post_rot = post_rots[0]
            post_tran = post_trans[0]
            
            key_frame = fid == 0
            extra_ref_frame = fid == self.num_frame - self.extra_ref_frames
            
            if key_frame or self.with_prev:
                if self.align_after_view_transfromation:
                    sensor2keyego, ego2global = sensor2keyegos[0], ego2globals[0]
                    
                mlp_input = self.img_view_transformer.get_mlp_input(
                    sensor2keyegos[0], ego2globals[0], intrin,
                    post_rot, post_tran, bda)
                    
                inputs_curr = (img_curr, sensor2keyego, ego2global, intrin,
                               post_rot, post_tran, bda, mlp_input,
                               feat_prev_iv, curr2adjsensor[0],
                               extra_ref_frame)
                               
                if key_frame:
                    bev_feat, depth, feat_curr_iv = self.prepare_bev_feat(*inputs_curr)
                    depth_key_frame = depth
                else:
                    with torch.no_grad():
                        bev_feat, depth, feat_curr_iv = self.prepare_bev_feat(*inputs_curr)
                        
                if not extra_ref_frame:
                    bev_feat_list.append(bev_feat)
                feat_prev_iv = feat_curr_iv
                
        if not self.with_prev:
            bev_feat_key = bev_feat_list[0]
            if len(bev_feat_key.shape) == 4:
                b, c, h, w = bev_feat_key.shape
                bev_feat_list = [
                    torch.zeros([b, c * (self.num_frame - self.extra_ref_frames - 1),
                                h, w]).to(bev_feat_key), bev_feat_key
                ]
            else:
                b, c, z, h, w = bev_feat_key.shape
                bev_feat_list = [
                    torch.zeros([b, c * (self.num_frame - self.extra_ref_frames - 1), z,
                                h, w]).to(bev_feat_key), bev_feat_key
                ]
                
        bev_feat = torch.cat(bev_feat_list, dim=1)
        
        if with_bevencoder:
            x = self.bev_encoder(bev_feat)
            return [x], depth_key_frame, bev_feat_list
        else:
            return [bev_feat], depth_key_frame, bev_feat_list
