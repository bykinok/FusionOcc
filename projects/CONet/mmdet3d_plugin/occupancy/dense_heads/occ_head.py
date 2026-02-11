import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# Remove unused reduce_mean import - function not used in this file
import torch.distributed as dist
from mmdet3d.registry import MODELS as HEADS
from mmcv.cnn import build_conv_layer, build_norm_layer
from .lovasz_softmax import lovasz_softmax
from ...utils import coarse_to_fine_coordinates, project_points_on_img
from ...utils.nusc_param import nusc_class_frequencies, occ3d_class_frequencies, nusc_class_names
from ...utils.semkitti import geo_scal_loss, sem_scal_loss, CE_ssc_loss

@HEADS.register_module(force=True)
class OccHead(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channel,
        num_level=1,
        num_img_level=1,
        soft_weights=False,
        loss_weight_cfg=None,
        conv_cfg=dict(type='Conv3d', bias=False),
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        fine_topk=20000,
        point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        final_occ_size=[256, 256, 20],
        empty_idx=0,
        visible_loss=False,
        balance_cls_weight=True,
        cascade_ratio=1,
        sample_from_voxel=False,
        sample_from_img=False,
        train_cfg=None,
        test_cfg=None,
    ):
        super(OccHead, self).__init__()
        
        if type(in_channels) is not list:
            in_channels = [in_channels]
        
        self.in_channels = in_channels
        self.out_channel = out_channel
        self.num_level = num_level
        self.fine_topk = fine_topk
        
        self.point_cloud_range = torch.tensor(np.array(point_cloud_range)).float()
        self.final_occ_size = final_occ_size
        self.visible_loss = visible_loss
        self.cascade_ratio = cascade_ratio
        self.sample_from_voxel = sample_from_voxel
        self.sample_from_img = sample_from_img

        if self.cascade_ratio != 1: 
            if self.sample_from_voxel or self.sample_from_img:
                # Fix input dimension to 128 for multimodal config compatibility
                fine_mlp_input_dim = 128  # Fixed to match actual voxel features
                if sample_from_img:
                    self.img_mlp_0 = nn.Sequential(
                        nn.Conv2d(512, 128, 1, 1, 0),
                        nn.GroupNorm(16, 128),
                        nn.ReLU(inplace=True)
                    )
                    self.img_mlp = nn.Sequential(
                        nn.Linear(128, 64),
                        nn.GroupNorm(16, 64),
                        nn.ReLU(inplace=True),
                    )
                    # Add image channels since image features are now active
                    fine_mlp_input_dim += 64  # 128 (voxel) + 64 (image) = 192

                self.fine_mlp = nn.Sequential(
                    nn.Linear(fine_mlp_input_dim, 64),
                    nn.GroupNorm(16, 64),
                    nn.ReLU(inplace=True),
                    nn.Linear(64, out_channel)
            )

        if loss_weight_cfg is None:
            self.loss_weight_cfg = {
                "loss_voxel_ce_weight": 1.0,
                "loss_voxel_sem_scal_weight": 1.0,
                "loss_voxel_geo_scal_weight": 1.0,
                "loss_voxel_lovasz_weight": 1.0,
            }
        else:
            self.loss_weight_cfg = loss_weight_cfg
        
        # voxel losses
        self.loss_voxel_ce_weight = self.loss_weight_cfg.get('loss_voxel_ce_weight', 1.0)
        self.loss_voxel_sem_scal_weight = self.loss_weight_cfg.get('loss_voxel_sem_scal_weight', 1.0)
        self.loss_voxel_geo_scal_weight = self.loss_weight_cfg.get('loss_voxel_geo_scal_weight', 1.0)
        self.loss_voxel_lovasz_weight = self.loss_weight_cfg.get('loss_voxel_lovasz_weight', 1.0)
        
        # voxel-level prediction
        self.occ_convs = nn.ModuleList()
        for i in range(self.num_level):
            mid_channel = self.in_channels[i] // 2
            occ_conv = nn.Sequential(
                build_conv_layer(conv_cfg, in_channels=self.in_channels[i], 
                        out_channels=mid_channel, kernel_size=3, stride=1, padding=1),
                build_norm_layer(norm_cfg, mid_channel)[1],
                nn.ReLU(inplace=True))
            self.occ_convs.append(occ_conv)


        self.occ_pred_conv = nn.Sequential(
                build_conv_layer(conv_cfg, in_channels=mid_channel, 
                        out_channels=mid_channel//2, kernel_size=1, stride=1, padding=0),
                build_norm_layer(norm_cfg, mid_channel//2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(conv_cfg, in_channels=mid_channel//2, 
                        out_channels=out_channel, kernel_size=1, stride=1, padding=0))

        self.soft_weights = soft_weights
        self.num_img_level = num_img_level
        self.num_point_sampling_feat = self.num_level
        if self.soft_weights:
            soft_in_channel = mid_channel
            self.voxel_soft_weights = nn.Sequential(
                build_conv_layer(conv_cfg, in_channels=soft_in_channel, 
                        out_channels=soft_in_channel//2, kernel_size=1, stride=1, padding=0),
                build_norm_layer(norm_cfg, soft_in_channel//2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(conv_cfg, in_channels=soft_in_channel//2, 
                        out_channels=self.num_point_sampling_feat, kernel_size=1, stride=1, padding=0))
            
        # loss functions
        if balance_cls_weight:
            # Select appropriate class frequencies based on number of classes
            if out_channel == 17:
                # nuScenes-Occupancy format (17 classes)
                class_freqs = nusc_class_frequencies
            elif out_channel == 18:
                # Occ3D format (18 classes)
                class_freqs = occ3d_class_frequencies
            else:
                # Unknown format, use uniform weights
                print(f"Warning: out_channel ({out_channel}) is neither 17 nor 18. Using uniform weights.")
                self.class_weights = torch.ones(out_channel) / out_channel
                class_freqs = None
            
            if class_freqs is not None:
                self.class_weights = torch.from_numpy(1 / np.log(class_freqs + 0.001))
        else:
            # Use uniform weights for all classes
            self.class_weights = torch.ones(out_channel) / out_channel

        self.class_names = nusc_class_names    
        self.empty_idx = empty_idx
        
    def forward_coarse_voxel(self, voxel_feats):
        output_occs = []
        output = {}
        for feats, occ_conv in zip(voxel_feats, self.occ_convs):
            output_occs.append(occ_conv(feats))

        if self.soft_weights:
            voxel_soft_weights = self.voxel_soft_weights(output_occs[0])
            voxel_soft_weights = torch.softmax(voxel_soft_weights, dim=1)
        else:
            voxel_soft_weights = torch.ones([output_occs[0].shape[0], self.num_point_sampling_feat, 1, 1, 1],).to(output_occs[0].device) / self.num_point_sampling_feat

        out_voxel_feats = 0
        _, _, H, W, D= output_occs[0].shape
        for feats, weights in zip(output_occs, torch.unbind(voxel_soft_weights, dim=1)):
            feats = F.interpolate(feats, size=[H, W, D], mode='trilinear', align_corners=False).contiguous()
            out_voxel_feats += feats * weights.unsqueeze(1)
        output['out_voxel_feats'] = [out_voxel_feats]

        out_voxel = self.occ_pred_conv(out_voxel_feats)
        output['occ'] = [out_voxel]

        return output
     
    def forward(self, voxel_feats, img_feats=None, pts_feats=None, transform=None, **kwargs):
        assert type(voxel_feats) is list and len(voxel_feats) == self.num_level
        
        # breakpoint()

        # forward voxel 
        output = self.forward_coarse_voxel(voxel_feats)

        out_voxel_feats = output['out_voxel_feats'][0]
        coarse_occ = output['occ'][0]

        if self.cascade_ratio != 1:
            if self.sample_from_img or self.sample_from_voxel:
                coarse_occ_mask = coarse_occ.argmax(1) != self.empty_idx
                
                # 원본과 동일하게 assert 사용 (early return 제거)
                assert coarse_occ_mask.sum() > 0, 'no foreground in coarse voxel'
                
                B, W, H, D = coarse_occ_mask.shape
                coarse_coord_x, coarse_coord_y, coarse_coord_z = torch.meshgrid(torch.arange(W).to(coarse_occ.device),
                            torch.arange(H).to(coarse_occ.device), torch.arange(D).to(coarse_occ.device), indexing='ij')
                
                output['fine_output'] = []
                output['fine_coord'] = []

                if self.sample_from_img and img_feats is not None:
                    img_feats_ = img_feats[0]
                    # Handle both 4D and 5D cases safely
                    if len(img_feats_.shape) == 5:
                        B_i, N_i, C_i, W_i, H_i = img_feats_.shape
                        img_feats_ = img_feats_.reshape(-1, C_i, W_i, H_i)
                        img_feats = [self.img_mlp_0(img_feats_).reshape(B_i, N_i, -1, W_i, H_i)]
                    elif len(img_feats_.shape) == 4:
                        # Handle 4D case: [B, C, W, H] -> assume single camera view
                        B_i, C_i, W_i, H_i = img_feats_.shape
                        N_i = 1  # Single camera view
                        img_feats_ = img_feats_.reshape(-1, C_i, W_i, H_i)
                        img_feats = [self.img_mlp_0(img_feats_).reshape(B_i, N_i, -1, W_i, H_i)]
                    else:
                        # Skip img sampling if shape is unexpected
                        img_feats = None

                for b in range(B):
                    append_feats = []
                    
                    this_coarse_coord = torch.stack([coarse_coord_x[coarse_occ_mask[b]],
                                                    coarse_coord_y[coarse_occ_mask[b]],
                                                    coarse_coord_z[coarse_occ_mask[b]]], dim=0)  # 3, N
                    
                    if self.training:
                        this_fine_coord = coarse_to_fine_coordinates(this_coarse_coord, self.cascade_ratio, topk=self.fine_topk)  # 3, 8N/64N
                    else:
                        this_fine_coord = coarse_to_fine_coordinates(this_coarse_coord, self.cascade_ratio)  # 3, 8N/64N

                    output['fine_coord'].append(this_fine_coord)
                    new_coord = this_fine_coord[None].permute(0,2,1).float().contiguous()  # x y z

                    if self.sample_from_voxel:
                        this_fine_coord = this_fine_coord.float()
                        this_fine_coord[0, :] = (this_fine_coord[0, :]/(self.final_occ_size[0]-1) - 0.5) * 2
                        this_fine_coord[1, :] = (this_fine_coord[1, :]/(self.final_occ_size[1]-1) - 0.5) * 2
                        this_fine_coord[2, :] = (this_fine_coord[2, :]/(self.final_occ_size[2]-1) - 0.5) * 2
                        this_fine_coord = this_fine_coord[None,None,None].permute(0,4,1,2,3).float()
                        # 5D grid_sample input: [B, C, H, W, D]; cor: [B, N, 1, 1, 3]; output: [B, C, N, 1, 1]
                        new_feat = F.grid_sample(out_voxel_feats[b:b+1].permute(0,1,4,3,2), this_fine_coord, mode='bilinear', padding_mode='zeros', align_corners=False)
                        append_feats.append(new_feat[0,:,:,0,0].permute(1,0))
                        assert torch.isnan(new_feat).sum().item() == 0
                    
                    # image branch
                    if img_feats is not None and self.sample_from_img and transform is not None:
                        W_new, H_new, D_new = W * self.cascade_ratio, H * self.cascade_ratio, D * self.cascade_ratio
                        
                        # Handle transform[6] being torch.Size, tensor, or tuple
                        # Extract H_img and W_img from various possible formats
                        img_shape = transform[6]
                        if isinstance(img_shape, (torch.Size, tuple)):
                            H_img_val = img_shape[0]
                            W_img_val = img_shape[1]
                            # If H_img_val or W_img_val are still tuple/list, extract first element
                            if isinstance(H_img_val, (tuple, list)):
                                H_img_val = H_img_val[0]
                            if isinstance(W_img_val, (tuple, list)):
                                W_img_val = W_img_val[0]
                        else:
                            H_img_val = img_shape[0][b:b+1]
                            W_img_val = img_shape[1][b:b+1]
                        
                        # Transform matrices don't have batch dimension, so unsqueeze them
                        # transform[i] has shape [n_cam, ...], need to make it [1, n_cam, ...]
                        rots_b = transform[0].unsqueeze(0) if transform[0].dim() == 3 else transform[0][b:b+1]
                        trans_b = transform[1].unsqueeze(0) if transform[1].dim() == 2 else transform[1][b:b+1]
                        intrins_b = transform[2].unsqueeze(0) if transform[2].dim() == 3 else transform[2][b:b+1]
                        post_rots_b = transform[3].unsqueeze(0) if transform[3].dim() == 3 else transform[3][b:b+1]
                        post_trans_b = transform[4].unsqueeze(0) if transform[4].dim() == 2 else transform[4][b:b+1]
                        bda_for_projection = transform[5][None] if transform[5].dim() == 2 else transform[5][b:b+1]
                        
                        img_uv, img_mask = project_points_on_img(new_coord, rots=rots_b, trans=trans_b,
                                    intrins=intrins_b, post_rots=post_rots_b,
                                    post_trans=post_trans_b, bda_mat=bda_for_projection,
                                    W_img=W_img_val, H_img=H_img_val,
                                    pts_range=self.point_cloud_range, W_occ=W_new, H_occ=H_new, D_occ=D_new)  # [n_cam, 1, N, 2], [N, n_cam]
                        for img_feat in img_feats:
                            # img_feat[b]: [n_cam, C, H, W], img_uv: [n_cam, 1, N, 2]
                            sampled_img_feat = F.grid_sample(img_feat[b].contiguous(), img_uv.contiguous(), align_corners=True, mode='bilinear', padding_mode='zeros')
                            # sampled_img_feat: [n_cam, C, 1, N]
                            
                            # img_mask: [B, N, n_cam] -> [n_cam, B, N] -> squeeze to [n_cam, N]
                            img_mask_reshaped = img_mask.permute(2, 0, 1).squeeze(1)  # [n_cam, N]
                            
                            sampled_img_feat = sampled_img_feat * img_mask_reshaped[:, None, None, :]  # [n_cam, C, 1, N] * [n_cam, 1, 1, N]
                            sampled_img_feat = sampled_img_feat.sum(0)  # [C, 1, N]
                            sampled_img_feat = sampled_img_feat[:, 0, :]  # [C, N]
                            sampled_img_feat = self.img_mlp(sampled_img_feat.permute(1, 0))  # [N, C]
                            
                            append_feats.append(sampled_img_feat)  # N C
                            assert torch.isnan(sampled_img_feat).sum().item() == 0
                    
                    fine_output = self.fine_mlp(torch.cat(append_feats, dim=1))
                    output['fine_output'].append(fine_output)

        res = {
            'output_voxels': output['occ'],
            'output_voxels_fine': output.get('fine_output', None),
            'output_coords_fine': output.get('fine_coord', None),
        }
        
        return res

    def loss_voxel(self, output_voxels, target_voxels, tag):

        # Handle case where target_voxels might be a list
        if isinstance(target_voxels, list):
            if len(target_voxels) > 0:
                target_voxels = target_voxels[0]
            else:
                # No target voxels available, skip loss calculation
                return {}

        # Add batch dimension if missing
        if target_voxels.dim() == 3:
            target_voxels = target_voxels.unsqueeze(0)  # Add batch dimension


        # resize gt (dense: OpenOccupancy or Occ3D; both end up as dense grid here)
        B, C, H, W, D = output_voxels.shape
        ratio = target_voxels.shape[2] // H  # 원본처럼 shape[2] 사용
        if ratio != 1:
            target_voxels = target_voxels.reshape(B, H, ratio, W, ratio, D, ratio).permute(0,1,3,5,2,4,6).reshape(B, H, W, D, ratio**3)
            # Empty cells: all ratio**3 sub-voxels are empty_idx (nuScenes: 0, Occ3D: 17)
            empty_mask = (target_voxels == self.empty_idx).all(dim=-1)
            target_voxels = target_voxels.to(torch.int64)
            occ_space = target_voxels[~empty_mask]
            # Original config (nuScenes, empty_idx=0): 0 = noise → exclude from mode → 255 (ignore). Keep as-is.
            # Occ3D (empty_idx=17): 0 = others (valid class) → do not replace; 0 may win mode.
            if self.empty_idx == 0:
                occ_space[occ_space == 0] = -torch.arange(len(occ_space[occ_space == 0])).to(occ_space.device) - 1
            target_voxels[~empty_mask] = occ_space
            target_voxels = torch.mode(target_voxels, dim=-1)[0]
            target_voxels[target_voxels < 0] = 255
            target_voxels = target_voxels.long()

        assert torch.isnan(output_voxels).sum().item() == 0
        assert torch.isnan(target_voxels).sum().item() == 0

        loss_dict = {}

        # ignore 255 = ignore masked voxels
        # For Occ3D: empty_idx=17 (free class) - excluded from sem_scal_loss to align with evaluation
        loss_dict['loss_voxel_ce_{}'.format(tag)] = self.loss_voxel_ce_weight * CE_ssc_loss(output_voxels, target_voxels, self.class_weights.type_as(output_voxels), ignore_index=255)
        loss_dict['loss_voxel_sem_scal_{}'.format(tag)] = self.loss_voxel_sem_scal_weight * sem_scal_loss(output_voxels, target_voxels, ignore_index=255, empty_idx=self.empty_idx)
        loss_dict['loss_voxel_geo_scal_{}'.format(tag)] = self.loss_voxel_geo_scal_weight * geo_scal_loss(output_voxels, target_voxels, ignore_index=255, empty_idx=self.empty_idx)
        loss_dict['loss_voxel_lovasz_{}'.format(tag)] = self.loss_voxel_lovasz_weight * lovasz_softmax(torch.softmax(output_voxels, dim=1), target_voxels, ignore=255)

        return loss_dict

    def loss_point(self, fine_coord, fine_output, target_voxels, tag):

        # Handle case where target_voxels might be a list
        if isinstance(target_voxels, list):
            if len(target_voxels) > 0:
                target_voxels = target_voxels[0]
            else:
                # No target voxels available, skip loss calculation
                return {}

        
        # Check if target_voxels has the expected 4D shape [batch, x, y, z]
        if target_voxels.dim() == 3:
            target_voxels = target_voxels.unsqueeze(0)

        # Occ3D config는 cascade_ratio=2로 fine=GT=200x200x16이라 fine_coord 그대로 사용
        selected_gt = target_voxels[:, fine_coord[0,:], fine_coord[1,:], fine_coord[2,:]].long()[0]
        assert torch.isnan(selected_gt).sum().item() == 0, torch.isnan(selected_gt).sum().item()
        assert torch.isnan(fine_output).sum().item() == 0, torch.isnan(fine_output).sum().item()

        loss_dict = {}

        # ignore 255 = ignore masked voxels
        # For Occ3D: empty_idx=17 (free class) - excluded from sem_scal_loss to align with evaluation
        loss_dict['loss_voxel_ce_{}'.format(tag)] = self.loss_voxel_ce_weight * CE_ssc_loss(fine_output, selected_gt, ignore_index=255)
        loss_dict['loss_voxel_sem_scal_{}'.format(tag)] = self.loss_voxel_sem_scal_weight * sem_scal_loss(fine_output, selected_gt, ignore_index=255, empty_idx=self.empty_idx)
        loss_dict['loss_voxel_geo_scal_{}'.format(tag)] = self.loss_voxel_geo_scal_weight * geo_scal_loss(fine_output, selected_gt, ignore_index=255, empty_idx=self.empty_idx)
        loss_dict['loss_voxel_lovasz_{}'.format(tag)] = self.loss_voxel_lovasz_weight * lovasz_softmax(torch.softmax(fine_output, dim=1), selected_gt, ignore=255)


        return loss_dict

    def loss(self, output_voxels=None,
                output_coords_fine=None, output_voxels_fine=None, 
                target_voxels=None, visible_mask=None, **kwargs):
        loss_dict = {}
        for index, output_voxel in enumerate(output_voxels):
            loss_dict.update(self.loss_voxel(output_voxel, target_voxels,  tag='c_{}'.format(index)))
        if self.cascade_ratio != 1:
            loss_batch_dict = {}
            if self.sample_from_voxel or self.sample_from_img:
                for index, (fine_coord, fine_output) in enumerate(zip(output_coords_fine, output_voxels_fine)):
                    this_batch_loss = self.loss_point(fine_coord, fine_output, target_voxels, tag='fine')
                    for k, v in this_batch_loss.items():
                        if k not in loss_batch_dict:
                            loss_batch_dict[k] = v
                        else:
                            loss_batch_dict[k] = loss_batch_dict[k] + v
                for k, v in loss_batch_dict.items():
                    loss_dict[k] = v / len(output_coords_fine)
            
        return loss_dict
    
        
