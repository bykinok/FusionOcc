import os
import copy
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn

from mmdet3d.registry import MODELS
from mmdet3d.models.detectors.centerpoint import CenterPoint

from ..losses.semkitti import geo_scal_loss, sem_scal_loss
from ..losses.lovasz_softmax import lovasz_softmax

def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2]/2.0 for row in [xbound, ybound, zbound]])
    nx = torch.Tensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])
    return dx, bx, nx

@MODELS.register_module()
class STCOcc(CenterPoint):

    def __init__(self,
                 # BEVDet-Series
                 forward_projection=None,
                 # BEVFormer-Series
                 backward_projection=None,
                 # Occupancy_Head
                 occupancy_head=None,
                 # Flow Head
                 flow_head=None,
                 # Option: Temporalfusion
                 temporal_fusion=None,
                 # Other setting
                 intermediate_pred_loss_weight=(0.5, 0.25),
                 backward_num_layer=(1, 2),
                 num_stage=2,
                 bev_w=None,
                 bev_h=None,
                 bev_z=None,
                 class_weights=None,
                 empty_idx=17,
                 class_weights_group=None,
                 history_frame_num=None,
                 foreground_idx=None,
                 background_idx=None,
                 train_flow=False,
                 use_ms_feats=False,
                 train_top_k=None,
                 val_top_k=None,
                 save_results=False,
                 **kwargs):
        super(STCOcc, self).__init__(**kwargs)
        # ---------------------- init params ------------------------------
        self.bev_w = bev_w
        self.bev_h = bev_h
        self.bev_z = bev_z
        self.train_top_k = train_top_k
        self.val_top_k = val_top_k
        self.empty_idx = empty_idx
        self.num_stage = num_stage
        self.backward_num_layer = backward_num_layer
        self.intermediate_pred_loss_weight = intermediate_pred_loss_weight
        self.class_weights_group = class_weights_group
        self.history_frame_num = history_frame_num
        self.foreground_idx = foreground_idx
        self.background_idx = background_idx
        self.train_flow = train_flow
        self.use_ms_feats = use_ms_feats
        self.save_results = save_results
        self.scene_can_bus_info = dict()
        self.scene_loss = dict()
        # ---------------------- init loss ------------------------------
        self.class_weights = torch.tensor(np.array(class_weights), dtype=torch.float32, device='cuda')
        self.flow_loss = nn.L1Loss()
        self.flow_loss_weight = 1.0
        self.focal_loss_dict = self._build_focal_loss(dict(type='CustomFocalLoss', bev_h=200, bev_w=200), num_stage)
        # ---------------------- build components ------------------------------
        # BEVDet-Series
        self.forward_projection = MODELS.build(forward_projection)
        # BEVFormer-Series
        self._build_backward_projection(backward_projection, num_stage)
        # Temporal-Fsuion
        self._build_temporal_fusion(temporal_fusion, num_stage) if temporal_fusion else None
        # Simple Occupancy Head
        self.occupancy_head = MODELS.build(occupancy_head)
        # flow head
        self.flow_head = MODELS.build(flow_head) if flow_head else None


    def _build_focal_loss(self, focal_loss_config, num_stage):
        loss_dict = dict()
        focal_loss = MODELS.build(focal_loss_config)
        loss_dict['num_stage_1_1'] = focal_loss
        for i in range(0, num_stage):
            focal_loss_config = copy.deepcopy(focal_loss_config)
            focal_loss_config['bev_h'] = int(focal_loss_config['bev_h'] / 2)
            focal_loss_config['bev_w'] = int(focal_loss_config['bev_w'] / 2)
            focal_loss = MODELS.build(focal_loss_config)
            loss_dict['num_stage_1_{}'.format(2**(i+1))] = focal_loss
        return loss_dict

    def _build_backward_projection(self, backward_projection_config, num_stage):
        self.backward_projection_list = nn.ModuleList()
        backward_projection_config_dict = dict()
        backward_projection_config['transformer']['encoder']['num_layers'] = self.backward_num_layer[-1]
        backward_projection_config_dict['num_stage_0'] = copy.deepcopy(backward_projection_config)

        for index in range(num_stage):
            # first stage:
            if index == num_stage - 1:
                backward_projection_config_dict['num_stage_{}'.format(index)]['transformer']['encoder']['first_stage'] = True

            backward_projection = MODELS.build(backward_projection_config_dict['num_stage_{}'.format(index)])
            self.backward_projection_list.append(backward_projection)
            if index != num_stage-1:
                # different stage, adjust backward params, copy avoid in-place operation
                backward_projection_config_dict['num_stage_{}'.format(index+1)] = copy.deepcopy(backward_projection_config_dict['num_stage_{}'.format(index)])

                # num_layer adjust
                backward_projection_config_dict['num_stage_{}'.format(index+1)]['transformer']['encoder']['num_layers'] = self.backward_num_layer[num_stage-index-2]

                # bev shape adjust or voxel shape adjust
                backward_projection_config_dict['num_stage_{}'.format(index + 1)]['bev_h'] = int(backward_projection_config_dict['num_stage_{}'.format(index + 1)]['bev_h'] / 2)
                backward_projection_config_dict['num_stage_{}'.format(index + 1)]['bev_w'] = int(backward_projection_config_dict['num_stage_{}'.format(index + 1)]['bev_w'] / 2)
                if 'bev_z' in backward_projection_config_dict['num_stage_{}'.format(index + 1)]:
                    backward_projection_config_dict['num_stage_{}'.format(index + 1)]['bev_z'] = int(backward_projection_config_dict['num_stage_{}'.format(index + 1)]['bev_z'] / 2)

                # grid config bev shape adjust
                backward_projection_config_dict['num_stage_{}'.format(index+1)]['transformer']['encoder']['grid_config']['x'][2] = \
                    backward_projection_config_dict['num_stage_{}'.format(index+1)]['transformer']['encoder']['grid_config']['x'][2] * 2
                backward_projection_config_dict['num_stage_{}'.format(index+1)]['transformer']['encoder']['grid_config']['y'][2] = \
                    backward_projection_config_dict['num_stage_{}'.format(index+1)]['transformer']['encoder']['grid_config']['y'][2] * 2
                backward_projection_config_dict['num_stage_{}'.format(index+1)]['transformer']['encoder']['grid_config']['z'][2] = \
                    backward_projection_config_dict['num_stage_{}'.format(index+1)]['transformer']['encoder']['grid_config']['z'][2] * 2

                # transformerlayers bev shape adjust
                backward_projection_config_dict['num_stage_{}'.format(index+1)]['transformer']['encoder']['transformerlayers']['attn_cfgs'][1][
                    'deformable_attention']['num_points'] = \
                    int(backward_projection_config_dict['num_stage_{}'.format(index+1)]['transformer']['encoder']['transformerlayers']['attn_cfgs'][1][
                        'deformable_attention']['num_points'] / 2)

                backward_projection_config_dict['num_stage_{}'.format(index + 1)]['transformer']['encoder'][
                    'transformerlayers']['attn_cfgs'][0]['num_points'] = \
                    int(backward_projection_config_dict['num_stage_{}'.format(index + 1)]['transformer']['encoder'][
                            'transformerlayers']['attn_cfgs'][0]['num_points'] / 2)

                if 'num_Z_anchors' in backward_projection_config_dict['num_stage_{}'.format(index + 1)]['transformer']['encoder'][
                    'transformerlayers']['attn_cfgs'][1]['deformable_attention']:
                    backward_projection_config_dict['num_stage_{}'.format(index + 1)]['transformer']['encoder'][
                        'transformerlayers']['attn_cfgs'][1]['deformable_attention']['num_Z_anchors'] = \
                        int(backward_projection_config_dict['num_stage_{}'.format(index + 1)]['transformer']['encoder'][
                                'transformerlayers']['attn_cfgs'][1]['deformable_attention']['num_Z_anchors'] / 2)

                # positional_encoding bev shape adjust
                backward_projection_config_dict['num_stage_{}'.format(index+1)]['positional_encoding']['row_num_embed'] = \
                    int(backward_projection_config_dict['num_stage_{}'.format(index+1)]['positional_encoding']['row_num_embed'] / 2)
                backward_projection_config_dict['num_stage_{}'.format(index+1)]['positional_encoding']['col_num_embed'] = \
                    int(backward_projection_config_dict['num_stage_{}'.format(index+1)]['positional_encoding']['col_num_embed'] / 2)

    def _build_temporal_fusion(self, temporal_fusion_config, num_stage):
        self.temporal_fusion_list = nn.ModuleList()
        self.dx_list, self.bx_list, self.nx_list = [], [], []
        x_config = self.forward_projection.img_view_transformer.grid_config['x']
        y_config = self.forward_projection.img_view_transformer.grid_config['y']
        z_config = self.forward_projection.img_view_transformer.grid_config['z']
        for i in range(num_stage):
            temporal_fusion_config['history_num'] = self.history_frame_num[i]
            if self.train_top_k and self.training:
                temporal_fusion_config['top_k'] = self.train_top_k[i]
            elif self.val_top_k and not self.training:
                temporal_fusion_config['top_k'] = self.val_top_k[i]
            temporal_fusion = MODELS.build(temporal_fusion_config)
            dx, bx, nx = gen_dx_bx(x_config, y_config, z_config)

            dx = nn.Parameter(dx, requires_grad=False)
            bx = nn.Parameter(bx, requires_grad=False)
            nx = nn.Parameter(nx, requires_grad=False)
            self.temporal_fusion_list.append(temporal_fusion)
            self.dx_list.append(dx)
            self.bx_list.append(bx)
            self.nx_list.append(nx)

            x_config[2] = x_config[2] * 2
            y_config[2] = y_config[2] * 2
            z_config[2] = z_config[2] * 2
            temporal_fusion_config['bev_z'] = int(temporal_fusion_config['bev_z'] / 2)
            temporal_fusion_config['bev_h'] = int(temporal_fusion_config['bev_h'] / 2)
            temporal_fusion_config['bev_w'] = int(temporal_fusion_config['bev_w'] / 2)


    def with_specific_component(self, component_name):
        """Whether the model owns a specific component"""
        return getattr(self, component_name, None) is not None

    def get_voxel_loss(self,
                       pred_voxel_semantic,
                       target_voxel_semantic,
                       loss_weight,
                       focal_loss=None,
                       tag='c_0',
                       ):
        # change pred_voxel_semantic from [bs, w, h, z, c] -> [bs, c, w, h, z]  !!!
        pred_voxel_semantic = pred_voxel_semantic.permute(0, 4, 1, 2, 3)
        loss_dict = {}

        loss_dict['loss_voxel_ce_{}'.format(tag)] = loss_weight * focal_loss(
            pred_voxel_semantic,
            target_voxel_semantic,
            self.class_weights,
            ignore_index=255)
        loss_dict['loss_voxel_sem_scal_{}'.format(tag)] = loss_weight * sem_scal_loss(
            pred_voxel_semantic,
            target_voxel_semantic,
            ignore_index=255)
        loss_dict['loss_voxel_geo_scal_{}'.format(tag)] = loss_weight * geo_scal_loss(
            pred_voxel_semantic,
            target_voxel_semantic,
            ignore_index=255,
            empty_idx=self.empty_idx)
        loss_dict['loss_voxel_lovasz_{}'.format(tag)] = loss_weight * lovasz_softmax(
            torch.softmax(pred_voxel_semantic, dim=1),
            target_voxel_semantic,
            ignore=255,
        )

        return loss_dict

    def get_flow_loss(self, pred_flows, target_flows, target_sem, loss_weight):
        # Handle case where target_flows is a list or numpy array
        if isinstance(target_flows, list):
            # Convert list elements to tensors if needed
            flow_tensors = []
            for f in target_flows:
                if isinstance(f, np.ndarray):
                    flow_tensors.append(torch.from_numpy(f))
                elif torch.is_tensor(f):
                    flow_tensors.append(f)
                else:
                    flow_tensors.append(torch.tensor(f))
            target_flows = torch.stack(flow_tensors)
        elif isinstance(target_flows, np.ndarray):
            target_flows = torch.from_numpy(target_flows)
        
        # Handle target_sem as well
        if isinstance(target_sem, list):
            # Convert list elements to tensors if needed
            sem_tensors = []
            for s in target_sem:
                if isinstance(s, np.ndarray):
                    sem_tensors.append(torch.from_numpy(s))
                elif torch.is_tensor(s):
                    sem_tensors.append(s)
                else:
                    sem_tensors.append(torch.tensor(s))
            target_sem = torch.stack(sem_tensors)
        elif isinstance(target_sem, np.ndarray):
            target_sem = torch.from_numpy(target_sem)
        
        # Ensure tensors are on the same device as pred_flows
        if hasattr(pred_flows, 'device') and pred_flows is not None:
            if isinstance(pred_flows, (list, tuple)) and len(pred_flows) > 0:
                device = pred_flows[0].device if torch.is_tensor(pred_flows[0]) else None
            else:
                device = pred_flows.device if torch.is_tensor(pred_flows) else None
            
            if device is not None:
                target_flows = target_flows.to(device)
                target_sem = target_sem.to(device)
        
        loss_dict = {}

        loss_flow = 0
        for i in range(target_flows.shape[0]):
            foreground_mask = torch.zeros(target_flows[i].shape[:-1])
            for idx in self.foreground_idx:
                foreground_mask[target_sem[i] == idx] = 1

            pred_flow = pred_flows[i][foreground_mask!=0]
            target_flow = target_flows[i][foreground_mask!=0]

            loss_flow = loss_flow + loss_weight * self.flow_loss(pred_flow, target_flow)
        loss_dict['loss_flow'] = loss_flow

        return loss_dict

    def obtain_feats_from_images(self, points, img, img_metas, **kwargs):
        
        
        # 0、Prepare
        if self.with_specific_component('temporal_fusion_list'):
            use_temporal = True
            sequence_group_idx = torch.stack(
                [torch.tensor(img_meta['sequence_group_idx'], device=img[0].device) for img_meta in img_metas])
            start_of_sequence = torch.stack(
                [torch.tensor(img_meta['start_of_sequence'], device=img[0].device) for img_meta in img_metas])
            curr_to_prev_ego_rt = torch.stack(
                [img_meta['curr_to_prev_ego_rt'].to(img[0].device) if torch.is_tensor(img_meta['curr_to_prev_ego_rt']) else torch.tensor(np.array(img_meta['curr_to_prev_ego_rt']), device=img[0].device) for img_meta in img_metas])
            history_fusion_params = {
                'sequence_group_idx': sequence_group_idx,
                'start_of_sequence': start_of_sequence,
                'curr_to_prev_ego_rt': curr_to_prev_ego_rt
            }
            # process can_bus info
            if 'can_bus' in img_metas[0]:
                for index, start in enumerate(start_of_sequence):
                    if start:
                        can_bus = copy.deepcopy(img_metas[index]['can_bus'])
                        temp_pose = copy.deepcopy(can_bus[:3])
                        temp_angle = copy.deepcopy(can_bus[-1])
                        can_bus[:3] = 0
                        can_bus[-1] = 0
                        self.scene_can_bus_info[sequence_group_idx[index].item()] = {
                            'prev_pose':temp_pose,
                            'prev_angle':temp_angle
                        }
                        img_metas[index]['can_bus'] = can_bus
                    else:
                        can_bus = copy.deepcopy(img_metas[index]['can_bus'])
                        temp_pose = copy.deepcopy(can_bus[:3])
                        temp_angle = copy.deepcopy(can_bus[-1])
                        can_bus[:3] = can_bus[:3] - self.scene_can_bus_info[sequence_group_idx[index].item()]['prev_pose']
                        can_bus[-1] = can_bus[-1] - self.scene_can_bus_info[sequence_group_idx[index].item()]['prev_angle']
                        self.scene_can_bus_info[sequence_group_idx[index].item()] = {
                            'prev_pose': temp_pose,
                            'prev_angle': temp_angle
                        }
                        img_metas[index]['can_bus'] = can_bus

        else:
            use_temporal = False
            history_fusion_params = None

        # breakpoint()

        # 1、Forward-Projection create coarse voxel features
        if 'sequential' not in kwargs or not kwargs['sequential']:
            voxel_feats, depth, tran_feats, ms_feats, cam_params = self.forward_projection.extract_feat(points, img=img, img_metas=img_metas, **kwargs)
        else:
            voxel_feats, depth, tran_feats, ms_feats, cam_params = kwargs['voxel_feats'], kwargs['depth'], kwargs['tran_feats'], kwargs['ms_feats'],kwargs['cam_params']

        # 2、Backward-Projection Refine
        last_voxel_feat = None
        last_occ_pred = None
        voxel_feats_index = len(voxel_feats) - 1
        intermediate_occ_pred_dict = {}
        intermediate_voxel_feat = []
        for i in range(len(self.backward_projection_list)-1, -1, -1):
            voxel_feat = voxel_feats[voxel_feats_index]

            if last_voxel_feat is not None:
                voxel_feat = last_voxel_feat + voxel_feat

            # voxel_feats shape: [bs, c, z, h, w]
            bev_feat, occ_pred = self.backward_projection_list[i](
                mlvl_feats=ms_feats if self.use_ms_feats else [tran_feats],
                img_metas=img_metas,
                voxel_feats=voxel_feat,  # mean in the z direction
                cam_params=cam_params,
                pred_img_depth=depth,
                last_occ_pred=last_occ_pred,
                prev_bev=self.temporal_fusion_list[i].history_last_bev if use_temporal else None,
                prev_bev_aug=self.temporal_fusion_list[i].history_forward_augs,
                history_fusion_params=history_fusion_params,
            )

            # save for loss
            intermediate_occ_pred_dict['pred_voxel_semantic_1_{}'.format(2 ** (i + 1))] = occ_pred
            last_occ_pred = occ_pred.clone().detach()
            # bev_feats shape: [bs, c, h, w], recover to occupancy with occ weight
            if bev_feat.dim() == 4:
                bs, c, z, h, w = voxel_feat.shape
                bev_feat = bev_feat.unsqueeze(2).repeat(1, 1, z, 1, 1)
                nonempty_prob = 1 - last_occ_pred.softmax(-1)[..., -1].permute(0, 3, 2, 1)
                last_voxel_feat = voxel_feat + bev_feat * nonempty_prob.unsqueeze(1)
            else:
                nonempty_prob = 1 - last_occ_pred.softmax(-1)[..., -1].permute(0, 3, 2, 1)
                last_voxel_feat = voxel_feat + bev_feat * nonempty_prob.unsqueeze(1)

            # Option: temporal fusion
            if self.with_specific_component('temporal_fusion_list'):
                last_voxel_feat = self.temporal_fusion_list[i](
                    last_voxel_feat, cam_params, history_fusion_params, dx=self.dx_list[i], bx=self.bx_list[i],
                    history_last_bev=self.temporal_fusion_list[i+1].history_bev if i+1 < len(self.temporal_fusion_list)-1 else None,
                    last_occ_pred=last_occ_pred,
                    nonempty_prob=nonempty_prob,
                )

            # output stage don't need to upsample
            if i != 0:
                last_voxel_feat = F.interpolate(last_voxel_feat, scale_factor=2, align_corners=False, mode='trilinear')

            voxel_feats_index = voxel_feats_index - 1
            intermediate_voxel_feat.append(last_voxel_feat)

        return_dict = dict(
            voxel_feats=last_voxel_feat,
            last_occ_pred=last_occ_pred,
            depth=depth,
            tran_feats=tran_feats,
            cam_params=cam_params,
            intermediate_occ_pred_dict=intermediate_occ_pred_dict,
            history_fusion_params=history_fusion_params,
        )

        # breakpoint()
        return return_dict

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    img_inputs=None,
                    **kwargs):
        # CRITICAL: Unwrap DataContainer from points to match original format
        # Original format: points is a list of tensors: [tensor, tensor, ...]
        # MMEngine format: points is a list of DataContainers: [{'data': tensor}, {'data': tensor}, ...]
        # or nested lists: [[tensor], [tensor], ...]
        if points is not None:
            if isinstance(points, list) and len(points) > 0:
                # Handle DataContainer format: [{'data': tensor}, ...]
                if isinstance(points[0], dict) and 'data' in points[0]:
                    unwrapped_points = []
                    for item in points:
                        if isinstance(item, dict) and 'data' in item:
                            data = item['data']
                            # If data is a list, take the first element
                            if isinstance(data, list) and len(data) > 0:
                                unwrapped_points.append(data[0])
                            else:
                                unwrapped_points.append(data)
                        else:
                            unwrapped_points.append(item)
                    points = unwrapped_points
                # Handle nested list format: [[tensor], [tensor], ...] -> [tensor, tensor, ...]
                elif isinstance(points[0], list) and len(points[0]) > 0:
                    # Unwrap nested lists: take first element from each inner list
                    points = [item[0] if isinstance(item, list) and len(item) > 0 else item for item in points]
        
        # Unwrap DataContainer from img_metas to match original format
        if img_metas is not None:
            # Handle DataContainer format: {'data': [...], 'cpu_only': True}
            if isinstance(img_metas, dict) and 'data' in img_metas:
                img_metas = img_metas['data']
            # Handle list of DataContainers: [{'data': {...}, 'cpu_only': True}, ...]
            elif isinstance(img_metas, list) and len(img_metas) > 0:
                if isinstance(img_metas[0], dict) and 'data' in img_metas[0]:
                    img_metas = [item['data'] if isinstance(item, dict) and 'data' in item else item for item in img_metas]
            
            # CRITICAL: Unwrap tuple-wrapped values in img_metas lists
            # DataContainer may wrap list elements as tuples: [(tensor,), (tensor,), ...]
            # Original format: [tensor, tensor, ...]
            for img_meta in img_metas:
                if isinstance(img_meta, dict):
                    # Unwrap tuple-wrapped list elements for transformation matrices
                    for key in ['sensor2sensorego', 'sensorego2global', 'sensorego2sensor', 'global2sensorego', 'lidar2img']:
                        if key in img_meta and isinstance(img_meta[key], list):
                            # Unwrap tuples: [(tensor,), (tensor,), ...] -> [tensor, tensor, ...]
                            unwrapped_list = []
                            for item in img_meta[key]:
                                if isinstance(item, tuple) and len(item) > 0:
                                    unwrapped_list.append(item[0])
                                else:
                                    unwrapped_list.append(item)
                            img_meta[key] = unwrapped_list
                    
                    # CRITICAL: Unwrap single-element lists to match original format
                    # Original format: sample_idx is a string (token), not a list
                    # Current format: sample_idx might be [token] (single-element list)
                    for key in ['sample_idx', 'flip', 'pcd_horizontal_flip', 'pcd_vertical_flip', 
                               'pcd_scale_factor', 'sequence_group_idx', 'start_of_sequence',
                               'box_mode_3d', 'box_type_3d', 'index', 'curr_to_prev_ego_rt', 
                               'curr_to_prev_lidar_rt', 'can_bus', 'scene_name', 'ego2lidar',
                               'sensor2sensorego', 'sensorego2global', 'sensorego2sensor', 'global2sensorego'
                               ]:
                        if key in img_meta and isinstance(img_meta[key], list) and len(img_meta[key]) == 1:
                            img_meta[key] = img_meta[key][0]
        
        # ---------------------- normalize img_inputs to original format -----------------------------
        # mmengine format: img_inputs = [[(imgs,), (sensor2egos,), (ego2globals,), (cam2imgs,), (post_augs,)]]
        # original format: img_inputs = (imgs, sensor2egos, ego2globals, cam2imgs, post_augs)
        # where imgs = tensor([6, 3, 256, 704])
        
        # CRITICAL: Unwrap img_inputs to match original format
        # Original: img=img_inputs[0] (first element of tuple, which is imgs tensor)
        if img_inputs is not None:
            if isinstance(img_inputs, list) and len(img_inputs) > 0:
                # Handle nested list format: [[(imgs,), (sensor2egos,), ...]]
                if isinstance(img_inputs[0], list) and len(img_inputs[0]) > 0:
                    # Unwrap tuples: [(imgs,), (sensor2egos,), ...] -> [imgs, sensor2egos, ...]
                    unwrapped_elements = []
                    for item in img_inputs[0]:
                        if isinstance(item, tuple) and len(item) > 0:
                            unwrapped_elements.append(item[0])
                        else:
                            unwrapped_elements.append(item)
                    # CRITICAL: Add batch dimension to tensors if missing
                    # Original format: all tensors have batch dimension [1, ...]
                    if len(unwrapped_elements) > 0:
                        imgs = unwrapped_elements[0]
                        if torch.is_tensor(imgs) and imgs.dim() == 4:  # (18, 3, 256, 704) -> (1, 18, 3, 256, 704)
                            unwrapped_elements[0] = imgs.unsqueeze(0)
                    if len(unwrapped_elements) > 1:
                        sensor2egos = unwrapped_elements[1]
                        if torch.is_tensor(sensor2egos) and sensor2egos.dim() == 3:  # (18, 4, 4) -> (1, 18, 4, 4)
                            unwrapped_elements[1] = sensor2egos.unsqueeze(0)
                    if len(unwrapped_elements) > 2:
                        ego2globals = unwrapped_elements[2]
                        if torch.is_tensor(ego2globals) and ego2globals.dim() == 3:  # (18, 4, 4) -> (1, 18, 4, 4)
                            unwrapped_elements[2] = ego2globals.unsqueeze(0)
                    if len(unwrapped_elements) > 3:
                        cam2imgs = unwrapped_elements[3]
                        if torch.is_tensor(cam2imgs) and cam2imgs.dim() == 3:  # (18, 4, 4) -> (1, 18, 4, 4)
                            unwrapped_elements[3] = cam2imgs.unsqueeze(0)
                    if len(unwrapped_elements) > 4:
                        post_augs = unwrapped_elements[4]
                        if torch.is_tensor(post_augs) and post_augs.dim() == 3:  # (18, 4, 4) -> (1, 18, 4, 4)
                            unwrapped_elements[4] = post_augs.unsqueeze(0)
                    # CRITICAL: Add batch dimension to bda if missing
                    # Original format: bda has shape [1, 4, 4], not [4, 4]
                    if len(unwrapped_elements) >= 6:
                        bda = unwrapped_elements[5]
                        if torch.is_tensor(bda) and bda.dim() == 2:  # (4, 4) -> (1, 4, 4)
                            unwrapped_elements[5] = bda.unsqueeze(0)
                    # CRITICAL: Convert to list format to match original: [[imgs, sensor2egos, ego2globals, cam2imgs, post_augs, bda]]
                    # Original format: img_inputs is a list with single list element (length 6)
                    img_inputs = [list(unwrapped_elements)]
                elif isinstance(img_inputs[0], tuple):
                    # CRITICAL: Convert tuple to list and add batch dimensions
                    # Original format: img_inputs[0] is a list, not a tuple
                    tuple_elements = list(img_inputs[0])
                    # Add batch dimension to tensors if missing
                    if len(tuple_elements) > 0:
                        imgs = tuple_elements[0]
                        if torch.is_tensor(imgs) and imgs.dim() == 4:  # (18, 3, 256, 704) -> (1, 18, 3, 256, 704)
                            tuple_elements[0] = imgs.unsqueeze(0)
                    if len(tuple_elements) > 1:
                        sensor2egos = tuple_elements[1]
                        if torch.is_tensor(sensor2egos) and sensor2egos.dim() == 3:  # (18, 4, 4) -> (1, 18, 4, 4)
                            tuple_elements[1] = sensor2egos.unsqueeze(0)
                    if len(tuple_elements) > 2:
                        ego2globals = tuple_elements[2]
                        if torch.is_tensor(ego2globals) and ego2globals.dim() == 3:  # (18, 4, 4) -> (1, 18, 4, 4)
                            tuple_elements[2] = ego2globals.unsqueeze(0)
                    if len(tuple_elements) > 3:
                        cam2imgs = tuple_elements[3]
                        if torch.is_tensor(cam2imgs) and cam2imgs.dim() == 3:  # (18, 4, 4) -> (1, 18, 4, 4)
                            tuple_elements[3] = cam2imgs.unsqueeze(0)
                    if len(tuple_elements) > 4:
                        post_augs = tuple_elements[4]
                        if torch.is_tensor(post_augs) and post_augs.dim() == 3:  # (18, 4, 4) -> (1, 18, 4, 4)
                            tuple_elements[4] = post_augs.unsqueeze(0)
                    # Add batch dimension to bda if missing
                    if len(tuple_elements) >= 6:
                        bda = tuple_elements[5]
                        if torch.is_tensor(bda) and bda.dim() == 2:  # (4, 4) -> (1, 4, 4)
                            tuple_elements[5] = bda.unsqueeze(0)
                    img_inputs = [tuple_elements]
                # else: img_inputs[0] is already a list, check all tensors
                elif isinstance(img_inputs[0], list) and len(img_inputs[0]) >= 1:
                    # Add batch dimension to imgs if missing
                    imgs = img_inputs[0][0]
                    if torch.is_tensor(imgs) and imgs.dim() == 4:  # (18, 3, 256, 704) -> (1, 18, 3, 256, 704)
                        img_inputs[0][0] = imgs.unsqueeze(0)
                    # Add batch dimension to sensor2egos if missing
                    if len(img_inputs[0]) > 1:
                        sensor2egos = img_inputs[0][1]
                        if torch.is_tensor(sensor2egos) and sensor2egos.dim() == 3:  # (18, 4, 4) -> (1, 18, 4, 4)
                            img_inputs[0][1] = sensor2egos.unsqueeze(0)
                    # Add batch dimension to ego2globals if missing
                    if len(img_inputs[0]) > 2:
                        ego2globals = img_inputs[0][2]
                        if torch.is_tensor(ego2globals) and ego2globals.dim() == 3:  # (18, 4, 4) -> (1, 18, 4, 4)
                            img_inputs[0][2] = ego2globals.unsqueeze(0)
                    # Add batch dimension to cam2imgs if missing
                    if len(img_inputs[0]) > 3:
                        cam2imgs = img_inputs[0][3]
                        if torch.is_tensor(cam2imgs) and cam2imgs.dim() == 3:  # (18, 4, 4) -> (1, 18, 4, 4)
                            img_inputs[0][3] = cam2imgs.unsqueeze(0)
                    # Add batch dimension to post_augs if missing
                    if len(img_inputs[0]) > 4:
                        post_augs = img_inputs[0][4]
                        if torch.is_tensor(post_augs) and post_augs.dim() == 3:  # (18, 4, 4) -> (1, 18, 4, 4)
                            img_inputs[0][4] = post_augs.unsqueeze(0)
                    # Add batch dimension to bda if missing
                    if len(img_inputs[0]) >= 6:
                        bda = img_inputs[0][5]
                        if torch.is_tensor(bda) and bda.dim() == 2:  # (4, 4) -> (1, 4, 4)
                            img_inputs[0][5] = bda.unsqueeze(0)
            elif isinstance(img_inputs, tuple):
                # CRITICAL: Convert tuple to list and add batch dimensions
                # Original format: img_inputs is a list with single list element (length 6)
                tuple_elements = list(img_inputs)
                # Add batch dimension to imgs if missing
                if len(tuple_elements) > 0:
                    imgs = tuple_elements[0]
                    if torch.is_tensor(imgs) and imgs.dim() == 4:  # (18, 3, 256, 704) -> (1, 18, 3, 256, 704)
                        tuple_elements[0] = imgs.unsqueeze(0)
                # Add batch dimension to sensor2egos if missing
                if len(tuple_elements) > 1:
                    sensor2egos = tuple_elements[1]
                    if torch.is_tensor(sensor2egos) and sensor2egos.dim() == 3:  # (18, 4, 4) -> (1, 18, 4, 4)
                        tuple_elements[1] = sensor2egos.unsqueeze(0)
                # Add batch dimension to ego2globals if missing
                if len(tuple_elements) > 2:
                    ego2globals = tuple_elements[2]
                    if torch.is_tensor(ego2globals) and ego2globals.dim() == 3:  # (18, 4, 4) -> (1, 18, 4, 4)
                        tuple_elements[2] = ego2globals.unsqueeze(0)
                # Add batch dimension to cam2imgs if missing
                if len(tuple_elements) > 3:
                    cam2imgs = tuple_elements[3]
                    if torch.is_tensor(cam2imgs) and cam2imgs.dim() == 3:  # (18, 4, 4) -> (1, 18, 4, 4)
                        tuple_elements[3] = cam2imgs.unsqueeze(0)
                # Add batch dimension to post_augs if missing
                if len(tuple_elements) > 4:
                    post_augs = tuple_elements[4]
                    if torch.is_tensor(post_augs) and post_augs.dim() == 3:  # (18, 4, 4) -> (1, 18, 4, 4)
                        tuple_elements[4] = post_augs.unsqueeze(0)
                # Add batch dimension to bda if missing
                if len(tuple_elements) >= 6:
                    bda = tuple_elements[5]
                    if torch.is_tensor(bda) and bda.dim() == 2:  # (4, 4) -> (1, 4, 4)
                        tuple_elements[5] = bda.unsqueeze(0)
                img_inputs = [tuple_elements]
        

        # breakpoint()
            
        # ---------------------- obtain feats from images -----------------------------
        # Original: img=img_inputs[0] (imgs tensor)
        return_dict = self.obtain_feats_from_images(points, img=img_inputs[0] if img_inputs is not None else None, img_metas=img_metas, **kwargs)
        voxel_feat = return_dict['voxel_feats']
        last_occ_pred = return_dict['last_occ_pred']

        # ---------------------- forward ------------------------------
        pred_voxel_semantic, pred_voxel_feats = self.occupancy_head(voxel_feat, last_occ_pred=last_occ_pred)
        pred_voxel_semantic_cls = pred_voxel_semantic.softmax(-1).argmax(-1)
        if self.with_specific_component('flow_head'):
            pred_voxel_flows, foreground_mask = self.flow_head(voxel_feat, pred_voxel_semantic)
            return_pred_voxel_flows = torch.zeros_like(pred_voxel_flows)
            return_pred_voxel_flows[foreground_mask != 0] = pred_voxel_flows[foreground_mask != 0]
        else:
            return_pred_voxel_flows = torch.zeros(size=(pred_voxel_semantic_cls.shape + (2,)), device=pred_voxel_semantic_cls.device)

        return_dict = dict()
        return_dict['occ_results'] = pred_voxel_semantic_cls.cpu().numpy().astype(np.uint8)
        return_dict['flow_results'] = return_pred_voxel_flows.cpu().numpy().astype(np.float16)
        return_dict['index'] = [img_meta.get('index', i) for i, img_meta in enumerate(img_metas)]
        if self.save_results:
            sample_idx = [img_meta.get('sample_idx', i) for i, img_meta in enumerate(img_metas)]
            scene_name = [img_meta['scene_name'] for img_meta in img_metas]
            # check save_dir
            for name in scene_name:
                if not os.path.exists('results/{}'.format(name)):
                    os.makedirs('results/{}'.format(name))
            for i, idx in enumerate(sample_idx):
                np.savez('results/{}/{}.npz'.format(scene_name[i], idx),semantics=return_dict['occ_results'][i], flow=return_dict['flow_results'][i])
        return [return_dict]

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):

        # Extract actual data from DataContainer format if needed
        if img_metas is not None and isinstance(img_metas, dict) and 'data' in img_metas:
            img_metas = img_metas['data']

        # ---------------------- obtain feats from images -----------------------------
        return_dict = self.obtain_feats_from_images(points, img=img_inputs, img_metas=img_metas, **kwargs)

        voxel_feats = return_dict['voxel_feats']    # shape: [bs, c, z, h, w]
        last_occ_pred = return_dict['last_occ_pred']
        depth = return_dict['depth']
        intermediate_occ_pred_dict = return_dict['intermediate_occ_pred_dict']
        history_fusion_params = return_dict['history_fusion_params']

        # ---------------------- forward ------------------------------
        pred_voxel_semantic, pred_voxel_feats = self.occupancy_head(voxel_feats, last_occ_pred=last_occ_pred)
        intermediate_occ_pred_dict['pred_voxel_semantic_1_1'] = pred_voxel_semantic

        if self.with_specific_component('flow_head'):
            pred_voxel_flows, foreground_masks = self.flow_head(voxel_feats, pred_voxel_semantic)

        # ---------------------- calc loss ------------------------------
        losses = dict()

        gt_semantic_voxel_dict = dict()
        gt_semantic_voxel_dict['gt_semantic_voxel_1_1'] = kwargs['voxel_semantics']
        num_stage = self.num_stage
        for index in range(num_stage):
            gt_semantic_voxel_dict['gt_semantic_voxel_1_{}'.format(2**(index+1))] = kwargs['voxel_semantics_1_{}'.format(2**(index+1))]

        # calc forward-projection depth-loss
        loss_depth = self.forward_projection.img_view_transformer.get_depth_loss(kwargs['gt_depth'], depth)
        losses['loss_depth'] = loss_depth

        # calc voxel loss
        for index in range(num_stage+1):
            loss_occ = self.get_voxel_loss(
                intermediate_occ_pred_dict['pred_voxel_semantic_1_{}'.format(2**index)],
                gt_semantic_voxel_dict['gt_semantic_voxel_1_{}'.format(2 **index)],
                self.intermediate_pred_loss_weight[index],
                focal_loss=self.focal_loss_dict['num_stage_1_{}'.format(2 **index)],
                tag='c_1_{}'.format(2**index),
            )
            losses.update(loss_occ)

        if self.with_specific_component('flow_head'):
            losses.update(
                self.get_flow_loss(
                    pred_voxel_flows,
                    kwargs['voxel_flows'],
                    gt_semantic_voxel_dict['gt_semantic_voxel_1_1'],
                    loss_weight=0.8
                )
            )

        return losses

    def _handle_legacy_forward(self, **kwargs):
        """Handle legacy forward calls that don't match the new API."""
        # Check if this is a training call (has return_loss parameter)
        if kwargs.get('return_loss', False):
            # Remove return_loss from kwargs before calling forward_train
            kwargs.pop('return_loss', None)
            return self.forward_train(**kwargs)
        else:
            # This is likely a test/inference call
            # Extract required parameters for simple_test
            points = kwargs.get('points', None)
            img_metas = kwargs.get('img_metas', None) 
            img_inputs = kwargs.get('img_inputs', None) or kwargs.get('img', None)
            
            return self.simple_test(
                points=points,
                img_metas=img_metas,
                img_inputs=img_inputs,
                **{k: v for k, v in kwargs.items() if k not in ['points', 'img_metas', 'img_inputs', 'img']}
            )

    def forward(self, inputs=None, data_samples=None, mode='tensor', **kwargs):
        """Unified forward method for the new mmdet3d API.
        
        Args:
            inputs (dict, optional): Input data containing 'points' and 'img_inputs' keys.
            data_samples (list, optional): Data samples with annotations.
            mode (str): Mode for forward pass. One of 'loss', 'predict', or 'tensor'.
            **kwargs: Additional arguments.
            
        Returns:
            Depends on mode: losses (dict) for 'loss', predictions for 'predict',
            or raw outputs for 'tensor'.
        """
        # Handle legacy call format where inputs might be passed as kwargs
        if inputs is None:
            # Try to extract inputs from kwargs or construct from individual parameters
            if 'points' in kwargs or 'img_inputs' in kwargs:
                inputs = {
                    'points': kwargs.pop('points', None),
                    'img_inputs': kwargs.pop('img_inputs', None)
                }
            else:
                # Fallback: treat all kwargs as the old format and redirect to legacy methods
                return self._handle_legacy_forward(**kwargs)
        
        # Unwrap DataContainer from img_metas if present
        if 'img_metas' in kwargs:
            img_metas = kwargs['img_metas']
            # Handle DataContainer format: {'data': [...], 'cpu_only': True}
            if isinstance(img_metas, dict) and 'data' in img_metas:
                kwargs['img_metas'] = img_metas['data']
            # Handle list of DataContainers
            elif isinstance(img_metas, list) and len(img_metas) > 0:
                if isinstance(img_metas[0], dict) and 'data' in img_metas[0]:
                    kwargs['img_metas'] = [item['data'] if isinstance(item, dict) and 'data' in item else item for item in img_metas]
        
        if mode == 'loss':
            # Training mode - call forward_train
            # Extract traditional arguments from new format
            points = inputs.get('points', None)
            img_inputs = inputs.get('img_inputs', None)
            
            # Extract ground truth from data_samples if provided
            if data_samples is not None:
                # Convert data_samples to traditional format
                img_metas_list = []
                gt_bboxes_3d = []
                gt_labels_3d = []
                
                for data_sample in data_samples:
                    # Extract metainfo
                    img_metas_list.append(data_sample.metainfo)
                    
                    # Extract 3D ground truth if available
                    if hasattr(data_sample, 'gt_instances_3d'):
                        gt_bboxes_3d.append(data_sample.gt_instances_3d.bboxes_3d)
                        gt_labels_3d.append(data_sample.gt_instances_3d.labels_3d)
                    else:
                        gt_bboxes_3d.append(None)
                        gt_labels_3d.append(None)
                
                kwargs.update({
                    'img_metas': img_metas_list,
                    'gt_bboxes_3d': gt_bboxes_3d,
                    'gt_labels_3d': gt_labels_3d,
                })
            
            return self.forward_train(
                points=points,
                img_inputs=img_inputs,
                **kwargs
            )
        
        elif mode == 'predict':
            # Inference mode - call simple_test
            points = inputs.get('points', None)
            img_inputs = inputs.get('img_inputs', None)
            
            if data_samples is not None:
                img_metas = [data_sample.metainfo for data_sample in data_samples]
            else:
                img_metas = kwargs.get('img_metas', None)
            
            # Remove img_metas from kwargs to avoid duplicate argument
            kwargs_filtered = {k: v for k, v in kwargs.items() if k != 'img_metas'}
            
            return self.simple_test(
                points=points,
                img_metas=img_metas,
                img_inputs=img_inputs,
                **kwargs_filtered
            )
        
        else:  # mode == 'tensor'
            # Raw tensor output mode - default to simple_test for now
            points = inputs.get('points', None)
            img_inputs = inputs.get('img_inputs', None)
            
            if data_samples is not None:
                img_metas = [data_sample.metainfo for data_sample in data_samples]
            else:
                img_metas = kwargs.get('img_metas', None)
            
            # Remove img_metas from kwargs to avoid duplicate argument
            kwargs_filtered = {k: v for k, v in kwargs.items() if k != 'img_metas'}
            
            return self.simple_test(
                points=points,
                img_metas=img_metas,
                img_inputs=img_inputs,
                **kwargs_filtered
            )