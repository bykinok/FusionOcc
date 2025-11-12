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
from debug_layer_comparison import log_layer


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
        # split the inputs into each frame
        # Check if inputs is already in original format (imgs, sensor2egos, ego2globals, cam2imgs, post_augs)
        if isinstance(inputs, tuple) and len(inputs) >= 4:
            # Original format from PrepareImageInputs
            img_tensor = inputs[0]
        elif isinstance(inputs, (list, tuple)) and isinstance(inputs[0], (tuple, list)):
            # Handle case where inputs[0] might be a tuple
            # If inputs[0] is a tuple/list, get the first tensor
            img_tensor = inputs[0][0] if len(inputs[0]) > 0 else inputs[0]
        else:
            img_tensor = inputs[0] if isinstance(inputs, (list, tuple)) else inputs
        
        # Handle different input shapes
        # Original format: (B, N*num_frame, C, H, W) where N=num_cameras
        # mmengine format for single batch: (N*num_frame, C, H, W) where N=num_cameras
        
        if len(img_tensor.shape) == 5:
            # Original format: (B, N*num_frame, C, H, W)
            B, N_total, C, H, W = img_tensor.shape
            N = N_total // self.num_frame  # num_cameras
            imgs = img_tensor.view(B, N, self.num_frame, C, H, W)
            imgs = torch.split(imgs, 1, 2)
            imgs = [t.squeeze(2) for t in imgs]
        elif len(img_tensor.shape) == 4:
            # Handle 4D tensor case: (B*N, C, H, W)
            BN, C, H, W = img_tensor.shape
            # Assume B=1 for test mode
            B = 1
            N_total = BN  # This is actually num_cameras * num_frame
            N = N_total // self.num_frame  # Extract num_cameras
            
            # Reshape to (B, N, num_frame, C, H, W)
            imgs = img_tensor.view(B, N, self.num_frame, C, H, W)
            imgs = torch.split(imgs, 1, 2)
            imgs = [t.squeeze(2) for t in imgs]
        elif len(img_tensor.shape) == 3:
            # Handle 3D tensor: (N*num_frame, C, H, W) - mmengine single batch format
            N_total, C, H, W = img_tensor.shape
            B = 1  # Single batch
            N = N_total // self.num_frame  # Extract num_cameras
            
            # Reshape to (B, N, num_frame, C, H, W)
            imgs = img_tensor.view(B, N, self.num_frame, C, H, W)
            imgs = torch.split(imgs, 1, 2)
            imgs = [t.squeeze(2) for t in imgs]
        else:
            raise ValueError(f"Unsupported tensor shape: {img_tensor.shape}")
        # Handle case where inputs[1:] might contain tuples
        remaining_inputs = inputs[1:]
        sensor2egos, ego2globals, intrins, post_augs, bda = remaining_inputs
        
        # Convert tuples to tensors if necessary
        if isinstance(sensor2egos, tuple):
            sensor2egos = torch.stack([torch.tensor(x) if not torch.is_tensor(x) else x for x in sensor2egos])
        if isinstance(ego2globals, tuple):
            ego2globals = torch.stack([torch.tensor(x) if not torch.is_tensor(x) else x for x in ego2globals])
        if isinstance(intrins, tuple):
            intrins = torch.stack([torch.tensor(x) if not torch.is_tensor(x) else x for x in intrins])
        if isinstance(post_augs, tuple):
            post_augs = torch.stack([torch.tensor(x) if not torch.is_tensor(x) else x for x in post_augs])
        if isinstance(bda, tuple):
            bda = torch.stack([torch.tensor(x) if not torch.is_tensor(x) else x for x in bda])
        
        # CRITICAL FIX: Ensure bda has batch dimension (B, 4, 4)
        if torch.is_tensor(bda):
            if bda.dim() == 2:  # (4, 4) -> (1, 4, 4)
                bda = bda.unsqueeze(0)
            elif bda.dim() == 1:  # Malformed, create identity
                bda = torch.eye(4, device=bda.device).unsqueeze(0)

        # Reshape sensor2egos and ego2globals to (B, num_frame, N, 4, 4)
        # Expected input shape: (B*N*num_frame, 4, 4) or (N*num_frame, 4, 4) for single batch
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
            
            # Handle case where lengths don't match
            if len(curr2adjsensor) != self.num_frame:
                # Only print this warning once per session to reduce log noise
                if not hasattr(self, '_curr2adj_mismatch_warned'):
                    print(f"Warning: curr2adjsensor length mismatch ({len(curr2adjsensor)} vs {self.num_frame}), adjusting automatically.")
                    self._curr2adj_mismatch_warned = True
                if len(curr2adjsensor) < self.num_frame:
                    # Pad with None values
                    curr2adjsensor.extend([None for _ in range(self.num_frame - len(curr2adjsensor))])
                else:
                    # Truncate to match self.num_frame
                    curr2adjsensor = curr2adjsensor[:self.num_frame]
            
            assert len(curr2adjsensor) == self.num_frame

        # intrins and post_augs:
        # In original STCOcc, cam2imgs and post_augs are extended for adjacent frames in PrepareImageInputs
        # Shape after PrepareImageInputs: (B*N*num_frame, 3/4, 3/4) for current + adjacent frames
        # We need to view it as (B, num_frame, N, 3/4, 3/4)
        
        # Calculate total cameras (should be B * N * self.num_frame)
        total_cams = intrins.shape[0]
        expected_cams = B * N * self.num_frame
        
        # DEBUG: print once
        if not hasattr(self, '_intrins_reshape_debug'):
            print(f"DEBUG prepare_inputs: intrins/post_augs reshaping")
            print(f"  B={B}, N={N}, num_frame={self.num_frame}")
            print(f"  total_cams={total_cams}, expected_cams={expected_cams}")
            print(f"  intrins.shape={intrins.shape}, post_augs.shape={post_augs.shape}")
            self._intrins_reshape_debug = True
        
        if total_cams == expected_cams:
            # Extended format: reshape to (B, num_frame, N, H, W)
            intrins_reshaped = intrins.view(B, self.num_frame, N, intrins.shape[-2], intrins.shape[-1])
            post_augs_reshaped = post_augs.view(B, self.num_frame, N, post_augs.shape[-2], post_augs.shape[-1])
            # DEBUG: print once
            if not hasattr(self, '_intrins_extended_debug'):
                print(f"DEBUG prepare_inputs: Using extended format")
                print(f"  intrins_reshaped.shape={intrins_reshaped.shape}")
                self._intrins_extended_debug = True
        else:
            # Not extended format: only current frame, repeat for num_frame
            intrins_reshaped = intrins.view(B, N, intrins.shape[-2], intrins.shape[-1]).unsqueeze(1)  # (B, 1, N, 3/4, 3/4)
            intrins_reshaped = intrins_reshaped.repeat(1, self.num_frame, 1, 1, 1)  # (B, num_frame, N, 3/4, 3/4)
            
            post_augs_reshaped = post_augs.view(B, N, post_augs.shape[-2], post_augs.shape[-1]).unsqueeze(1)  # (B, 1, N, 3/4, 3/4)
            post_augs_reshaped = post_augs_reshaped.repeat(1, self.num_frame, 1, 1, 1)  # (B, num_frame, N, 3/4, 3/4)
            # DEBUG: print once
            if not hasattr(self, '_intrins_repeat_debug'):
                print(f"DEBUG prepare_inputs: Using repeat format")
                print(f"  intrins_reshaped.shape={intrins_reshaped.shape}")
                self._intrins_repeat_debug = True

        extra = [
            sensor2keyegos,
            ego2globals,
            intrins_reshaped,
            post_augs_reshaped
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
        
        # DEBUG: Log input images
        log_layer("image_backbone", 
                 inputs={"imgs": imgs},
                 extra={"B": B, "N": N, "C": C, "H": imH, "W": imW})
        
        imgs = imgs.view(B * N, C, imH, imW)
        if self.grid_mask is not None:
            imgs = self.grid_mask(imgs)
        
        # DEBUG: Check ResNet layer by layer
        if not hasattr(self, '_resnet_layer_debug'):
            print(f"\n[RESNET_DEBUG] Layer-by-layer analysis:")
            print(f"  Input: Mean={imgs.mean().item():.6f}, Std={imgs.std().item():.6f}")
            
            # Check conv1
            if hasattr(self.img_backbone, 'conv1'):
                conv1_out = self.img_backbone.conv1(imgs)
                print(f"  After conv1: Mean={conv1_out.mean().item():.6f}, Std={conv1_out.std().item():.6f}")
                
                # Check BN1
                if hasattr(self.img_backbone, 'bn1'):
                    bn1 = self.img_backbone.bn1
                    print(f"    BN1 training={bn1.training}")
                    print(f"    BN1 running_mean: mean={bn1.running_mean.mean().item():.6f}, std={bn1.running_mean.std().item():.6f}")
                    print(f"    BN1 running_var: mean={bn1.running_var.mean().item():.6f}, min={bn1.running_var.min().item():.6f}")
                    bn1_out = bn1(conv1_out)
                    print(f"  After bn1: Mean={bn1_out.mean().item():.6f}, Std={bn1_out.std().item():.6f}")
                    
                    # After relu
                    if hasattr(self.img_backbone, 'relu'):
                        relu_out = self.img_backbone.relu(bn1_out)
                        print(f"  After relu: Mean={relu_out.mean().item():.6f}, Std={relu_out.std().item():.6f}")
                        
                        # After maxpool
                        if hasattr(self.img_backbone, 'maxpool'):
                            maxpool_out = self.img_backbone.maxpool(relu_out)
                            print(f"  After maxpool: Mean={maxpool_out.mean().item():.6f}, Std={maxpool_out.std().item():.6f}")
            
            self._resnet_layer_debug = True
        
        x = self.img_backbone(imgs)
        
        # DEBUG: print backbone output stats
        if not hasattr(self, '_backbone_output_debug'):
            print(f"\n[BACKBONE_OUT] After img_backbone:")
            if isinstance(x, (list, tuple)):
                for i, feat in enumerate(x):
                    print(f"  [{i}] Shape: {feat.shape}, Mean: {feat.mean().item():.6f}, Std: {feat.std().item():.6f}")
            else:
                print(f"  Shape: {x.shape}, Mean: {x.mean().item():.6f}, Std: {x.std().item():.6f}")
            self._backbone_output_debug = True
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
        
        # DEBUG: Log backbone output
        if isinstance(img_feats_reshape, (list, tuple)):
            log_layer("image_backbone",
                     outputs={"feat_" + str(i): feat for i, feat in enumerate(img_feats_reshape)},
                     extra={"num_levels": len(img_feats_reshape)})

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
        
        # DEBUG: print depth and bev_feat stats
        if not hasattr(self, '_depth_bev_debug'):
            print(f"\n[DEPTH] After img_view_transformer:")
            print(f"  Shape: {depth.shape}, Mean: {depth.mean().item():.6f}, Std: {depth.std().item():.6f}")
            print(f"[BEV_FEAT] After img_view_transformer:")
            print(f"  Shape: {bev_feat.shape}, Mean: {bev_feat.mean().item():.6f}, Std: {bev_feat.std().item():.6f}")
            self._depth_bev_debug = True

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

        # DEBUG: print once with statistics
        if not hasattr(self, '_bev_feat_list_debug'):
            print(f"\n[BEV_FEAT_LIST] Processing:")
            print(f"  len(bev_feat_list)={len(bev_feat_list)}, temporal_frame={self.temporal_frame}")
            print(f"  with_prev={self.with_prev}")
            if len(bev_feat_list) > 0:
                for i, feat in enumerate(bev_feat_list):
                    print(f"  [{i}] Shape: {feat.shape}, Mean: {feat.mean().item():.6f}, Std: {feat.std().item():.6f}")
            self._bev_feat_list_debug = True
        
        # Handle case when we need to add dummy frames (when bev_feat_list has fewer frames than expected)
        # If we don't have enough frames (only key frame), add dummy frames for adjacent frames
        if len(bev_feat_list) == 1:
            bev_feat_key = bev_feat_list[0]
            if len(bev_feat_key.shape) == 4:
                b, c, h, w = bev_feat_key.shape
                # Create dummy features for (temporal_frame-1) adjacent frames
                bev_feat_list = \
                    [torch.zeros([b,
                                  c * (self.temporal_frame - 1),  # Adjacent frames
                                  h, w]).to(bev_feat_key), bev_feat_key]  # Key frame
                # DEBUG: print once
                if not hasattr(self, '_bev_dummy_4d_debug'):
                    print(f"DEBUG extract_img_feat: Added dummy 4D features")
                    print(f"  dummy shape: {bev_feat_list[0].shape}")
                    self._bev_dummy_4d_debug = True
            else:
                b, c, z, h, w = bev_feat_key.shape
                # Create dummy features for 3D case
                bev_feat_list = \
                    [torch.zeros([b,
                                  c * (self.temporal_frame - 1), z,  # Adjacent frames
                                  h, w]).to(bev_feat_key), bev_feat_key]  # Key frame
                # DEBUG: print once
                if not hasattr(self, '_bev_dummy_5d_debug'):
                    print(f"DEBUG extract_img_feat: Added dummy 5D features")
                    print(f"  dummy shape: {bev_feat_list[0].shape}")
                    self._bev_dummy_5d_debug = True
        
        # bev_encoder to fuse multi-frame bev features
        bev_feat = torch.cat(bev_feat_list, dim=1)
        # DEBUG: print once with statistics
        if not hasattr(self, '_bev_cat_debug'):
            print(f"\n[BEV_CAT] After concatenation:")
            print(f"  Shape: {bev_feat.shape}")
            print(f"  Mean: {bev_feat.mean().item():.6f}, Std: {bev_feat.std().item():.6f}")
            print(f"  Min: {bev_feat.min().item():.6f}, Max: {bev_feat.max().item():.6f}")
            print(f"  Non-zero ratio: {(bev_feat != 0).float().mean().item():.4f}")
            if len(bev_feat_list) > 0:
                for i, feat in enumerate(bev_feat_list):
                    print(f"  BEV[{i}] Mean: {feat.mean().item():.6f}, Std: {feat.std().item():.6f}")
            self._bev_cat_debug = True
        x = self.bev_encoder(bev_feat)
        
        # DEBUG: print bev_encoder output stats once
        if not hasattr(self, '_bev_encoder_out_debug'):
            print(f"\n[BEV_ENCODER_OUT]:")
            if isinstance(x, (list, tuple)):
                for i, feat in enumerate(x):
                    print(f"  [{i}] Shape: {feat.shape}, Mean: {feat.mean().item():.6f}, Std: {feat.std().item():.6f}")
            else:
                print(f"  Shape: {x.shape}, Mean: {x.mean().item():.6f}, Std: {x.std().item():.6f}")
            self._bev_encoder_out_debug = True

        cam_params_key_frame = [sensor2keyegos[0], ego2globals[0], intrins[0], post_augs[0], bda]

        # adjust channel
        if self.adjust_channel_conv is not None:
            tran_feat_key_frame = self.adjust_channel_conv(tran_feat_key_frame)

        return x, depth_key_frame, tran_feat_key_frame, ms_feat_key_frame, cam_params_key_frame

    def extract_feat(self, points, img, img_metas, **kwargs):
        """Extract features from images and points."""
        voxel_feats, depth, tran_feats, ms_feat_key_frame, cam_params = self.extract_img_feat(img, img_metas, **kwargs)
        return voxel_feats, depth, tran_feats, ms_feat_key_frame, cam_params