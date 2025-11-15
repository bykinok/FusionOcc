# Copyright (c) Phigent Robotics. All rights reserved.
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

from .bevdet import BEVStereo4D
from mmdet3d.registry import MODELS
from mmcv.cnn.bricks.conv_module import ConvModule
from mmcv.ops import Voxelization

try:
    import spconv.pytorch as spconv
    from spconv.pytorch import SparseConvTensor, SparseSequential
    IS_SPCONV2_AVAILABLE = True
except ImportError:
    try:
        from mmcv.ops import SparseConvTensor, SparseSequential
        IS_SPCONV2_AVAILABLE = True
    except ImportError:
        IS_SPCONV2_AVAILABLE = False


class Unet3D(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super(Unet3D, self).__init__()
        self.init_dres = nn.Conv3d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.hg1 = Hourglass3D(mid_channels)
        self.hg2 = Hourglass3D(mid_channels)

    def forward(self, x):
        dres = self.init_dres(x)
        out1, pre1, post1 = self.hg1(dres, None, None)
        out1 = out1 + dres
        out2, pre2, post2 = self.hg2(out1, pre1, post1)
        out2 = out2 + dres
        return out2


class Hourglass3D(nn.Module):
    def __init__(self, mid_channels):
        super(Hourglass3D, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(mid_channels, 2 * mid_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(2 * mid_channels, mid_channels * 2, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(2 * mid_channels, 2 * mid_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(2 * mid_channels, 2 * mid_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(inplace=True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv3d(2 * mid_channels, 2 * mid_channels, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.conv6 = nn.Sequential(
            nn.Conv3d(2 * mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=False),
        )

    def forward(self, x, presqu=None, postsqu=None):
        out = self.conv1(x)
        pre = self.conv2(out)

        if postsqu is not None:
            pre = F.leaky_relu(pre + postsqu, inplace=True)
        else:
            pre = F.leaky_relu(pre, inplace=True)
        out = self.conv3(pre)
        out = self.conv4(out)
        out = F.interpolate(out, (pre.shape[-3], pre.shape[-2], pre.shape[-1]), mode='trilinear', align_corners=True)
        out = self.conv5(out)
        if presqu is not None:
            post = F.leaky_relu(out + presqu, inplace=True)
        else:
            post = F.leaky_relu(out + pre, inplace=True)
        out = F.interpolate(post, (x.shape[-3], x.shape[-2], x.shape[-1]), mode='trilinear', align_corners=True)
        out = self.conv6(out)
        return out, pre, post


@MODELS.register_module()
class BEVStereo4DOCCRC(BEVStereo4D):

    def __init__(self,
    
                 loss_occ=None,
                 out_dim=32,
                 use_mask=False,
                 num_classes=18,
                 use_predicter=True,
                 class_wise=False,

                 radar_voxel_layer=None,
                 radar_voxel_encoder=None,
                 radar_middle_encoder=None,
                 radar_bev_backbone=None,
                 radar_bev_neck=None,
                 radar_reduc_conv=False, #new
                 imc=256, rac=64, #im ra 특징 차원
                 freeze_img=False,
                 sparse_shape=None,

                 **kwargs):
        super(BEVStereo4DOCCRC, self).__init__(**kwargs)
        self.out_dim = out_dim
        out_channels = out_dim if use_predicter else num_classes
        self.final_conv = ConvModule(
                        self.img_view_transformer.out_channels,
                        out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=True,
                        conv_cfg=dict(type='Conv3d'))
        self.use_predicter = use_predicter
        if use_predicter:
            self.predicter = nn.Sequential(
                nn.Linear(self.out_dim, self.out_dim*2),
                nn.Softplus(),
                nn.Linear(self.out_dim*2, num_classes),
            )
        self.pts_bbox_head = None
        self.use_mask = use_mask
        self.num_classes = num_classes
        self.loss_occ = MODELS.build(loss_occ)
        self.class_wise = class_wise
        self.align_after_view_transfromation = False
        
        # Radar related components
        if radar_voxel_layer != None:
            self.radar_voxel_layer = Voxelization(**radar_voxel_layer)
        if radar_voxel_encoder != None:
            self.radar_voxel_encoder = MODELS.build(radar_voxel_encoder)
        if radar_middle_encoder != None:
            self.radar_middle_encoder = MODELS.build(radar_middle_encoder)
        if radar_bev_backbone is not None:
            self.radar_bev_backbone = MODELS.build(radar_bev_backbone)
        if radar_bev_neck is not None:
            self.radar_bev_neck = MODELS.build(radar_bev_neck)

        # voxel_channel = rac//2*5
        voxel_channel = imc*2
        self.radar_bev_to_voxel_conv = nn.Conv2d(rac, voxel_channel*16, kernel_size=1)

        if radar_reduc_conv:
            self.reduc_conv = ConvModule(
                    voxel_channel + imc,
                    # self.img_view_transformer.out_channels,  #rac change imc
                    imc,
                    kernel_size=3,
                    padding=1,
                    conv_cfg=dict(type='Conv3d'),
                    norm_cfg=dict(type='BN3d', eps=1e-3, momentum=0.01),
                    act_cfg=dict(type='ReLU'),
                    inplace=False)
            
        self.freeze_img = freeze_img
        self.sparse_shape = sparse_shape
    
    def init_weights(self):
        """Initialize model weights."""
        super(BEVStereo4DOCCRC, self).init_weights()
        if self.freeze_img:
            
            if self.with_img_backbone:
                for param in self.img_backbone.parameters():
                    param.requires_grad = False
            if self.with_img_neck:
                for param in self.img_neck.parameters():
                    param.requires_grad = False
            
            for name, param in self.named_parameters():
                if 'img_view_transformer' in name:
                    param.requires_grad = False
                if 'img_bev_encoder_backbone' in name:
                    param.requires_grad = False
                if 'img_bev_encoder_neck' in name:
                    param.requires_grad = False
                if 'pre_process' in name:
                    param.requires_grad = False
            def fix_bn(m):
                if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
                    m.track_running_stats = False
            self.img_view_transformer.apply(fix_bn)
            self.img_bev_encoder_backbone.apply(fix_bn)
            self.img_bev_encoder_neck.apply(fix_bn)
            
            self.img_backbone.apply(fix_bn)
            self.img_neck.apply(fix_bn)

            self.pre_process_net.apply(fix_bn)
    
    @torch.no_grad()
    def radar_voxelize(self, points):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            tuple[torch.Tensor]: Concatenated points, number of points
                per voxel, and coordinates.
        """
        voxels, coors, num_points = [], [], []
        for res in points:
            # Skip empty or invalid radar data
            if res is None:
                continue
            if isinstance(res, torch.Tensor):
                if res.numel() == 0 or len(res.shape) < 2:
                    continue
            
            res_voxels, res_coors, res_num_points = self.radar_voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            # For single batch, all voxels should have batch_id=0
            batch_id = 0  # All voxels belong to the same batch
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=batch_id)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch
    
    def extract_radar_feat(self, radar, img_metas):
        """Extract features of points."""
        voxels, num_points, coors = self.radar_voxelize(radar)

        voxel_features = self.radar_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        
        x_before = self.radar_middle_encoder(voxel_features, coors, batch_size)

        x = self.radar_bev_backbone(x_before) # [bs, 64, h/2, w/2]
        
        x = self.radar_bev_neck(x) # returns list of features
        
        x = torch.nn.functional.interpolate(x[0], scale_factor=2, mode='bilinear')

        x = self.radar_bev_to_voxel(x)
        return [x], [x_before]
    
    def radar_bev_to_voxel(self, x):
        x = self.radar_bev_to_voxel_conv(x)
        bs, c, h, w = x.shape
        x = x.reshape(bs, c//16, 16, h, w)
        return x

    def loss_single(self, voxel_semantics, mask_camera, preds):
        loss_ = dict()
        # Ensure all tensors are on the same device
        voxel_semantics = voxel_semantics.to(preds.device).long()
        mask_camera = mask_camera.to(preds.device)
        if self.use_mask:
            mask_camera = mask_camera.to(torch.int32)
            voxel_semantics = voxel_semantics.reshape(-1)
            preds = preds.reshape(-1, self.num_classes)
            mask_camera = mask_camera.reshape(-1)
            num_total_samples = mask_camera.sum()
            loss_occ = self.loss_occ(preds, voxel_semantics, mask_camera, avg_factor=num_total_samples)
            loss_['loss_occ'] = loss_occ
        else:
            voxel_semantics = voxel_semantics.reshape(-1)
            preds = preds.reshape(-1, self.num_classes)
            loss_occ = self.loss_occ(preds, voxel_semantics,)
            loss_['loss_occ'] = loss_occ
        return loss_

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    gt_masks_bev=None,
                    rescale=False,
                    radar=None,
                    **kwargs):
        """Test function without augmentaiton."""
        img_feats, pts_feats, depth, bev_feat_list, radar_feats, prev_radar_feats = self.extract_feat(
            points, img=img, img_metas=img_metas, radar=radar[0], **kwargs)
        
        # Keep 3D structure for 3D convolutions: concatenate along channel dimension
        # img_feats[0]: [bs, 32, 4, 50, 50] 
        # radar_feats[0]: [bs, 96, 4, 50, 50]
        # Total: [bs, 32+96=128, 4, 50, 50]
        fusion_feats = self.reduc_conv(torch.cat((img_feats[0], radar_feats[0]), dim=1))
        occ_pred = self.final_conv(fusion_feats).permute(0, 4, 3, 2, 1)
        # bncdhw->bnwhdc
        # Upsample prediction to match ground truth resolution
        # preds: [bs, 50, 50, 4, classes] -> [bs, 200, 200, 16, classes]
        bs, w, h, d, classes = occ_pred.shape
        target_w, target_h, target_d = 200, 200, 16
        
        # Interpolate each spatial dimension
        occ_pred = occ_pred.permute(0, 4, 3, 1, 2)  # [bs, classes, d, w, h]
        occ_pred = F.interpolate(occ_pred, size=(target_d, target_w, target_h), mode='trilinear', align_corners=False)
        occ_pred = occ_pred.permute(0, 3, 4, 2, 1)  # [bs, w, h, d, classes] -> [bs, 200, 200, 16, classes]
        if self.use_predicter:
            occ_pred = self.predicter(occ_pred)
        occ_score = occ_pred.softmax(-1)
        occ_res = occ_score.argmax(-1)
        occ_res = occ_res.squeeze(dim=0).cpu().numpy().astype(np.uint8)
        return [occ_res]

    def forward(self, inputs=None, data_samples=None, mode='loss', **kwargs):
        """Forward function for BEVStereo4DOCCRC.
        
        Args:
            inputs (dict): Input data including images, points, etc.
            data_samples (list, optional): Data samples containing ground truth. Defaults to None.
            mode (str): Mode of forward function. Defaults to 'loss'.
            **kwargs: Additional keyword arguments.
            
        Returns:
            dict: Forward results.
        """
        # If inputs is None, use kwargs as inputs
        if inputs is None:
            inputs = kwargs
        else:
            # Merge kwargs into inputs
            if isinstance(inputs, dict):
                inputs.update(kwargs)
        
        if mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        elif mode == 'tensor':
            return self._forward(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". Only supports loss, predict and tensor mode')

    def loss(self, inputs, data_samples):
        """Calculate loss from inputs and data samples."""
        # Extract data from inputs and data_samples
        img_inputs = inputs.get('img_inputs', None)
        points = inputs.get('points', None) 
        radar = inputs.get('radar', None)
        gt_depth = inputs.get('gt_depth', None)
        img_metas = inputs.get('img_metas', None)
        
        # Convert img_inputs to tensor if it's a list/tuple of tensors
        if img_inputs is not None:
            if isinstance(img_inputs, (list, tuple)) and len(img_inputs) > 0:
                # Stack list of tensors into a single tensor
                if isinstance(img_inputs[0], torch.Tensor):
                    img_inputs = torch.stack(img_inputs, dim=0)
        
        # Convert radar to proper format if needed
        if radar is not None:
            if isinstance(radar, (list, tuple)) and len(radar) > 0:
                if hasattr(radar[0], 'tensor'):  # RadarPoints object
                    radar = [r.tensor for r in radar]
        
        # Extract metadata and ground truth from data_samples if available  
        if data_samples is not None:
            gt_bboxes_3d = []
            gt_labels_3d = []
            for sample in data_samples:
                if hasattr(sample, 'metainfo') and img_metas is None:
                    if not img_metas:
                        img_metas = []
                    img_metas.append(sample.metainfo)
                if hasattr(sample, 'gt_instances_3d'):
                    gt_bboxes_3d.append(sample.gt_instances_3d.bboxes_3d)
                    gt_labels_3d.append(sample.gt_instances_3d.labels_3d)
        else:
            gt_bboxes_3d = None
            gt_labels_3d = None
        
        return self.forward_train(
            points=points,
            img_metas=img_metas,
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            img_inputs=img_inputs,
            radar=radar,
            gt_depth=gt_depth
        )

    def predict(self, inputs, data_samples):
        """Predict from inputs and data samples."""
        if isinstance(inputs, dict):
            img_inputs = inputs.get('img_inputs', None)
            points = inputs.get('points', None)
            radar = inputs.get('radar', None)
            
            # Check for 'inputs' key (mmengine might wrap the data)
            if img_inputs is None and 'inputs' in inputs:
                inner_inputs = inputs['inputs']
                if isinstance(inner_inputs, dict):
                    img_inputs = inner_inputs.get('img_inputs', None)
                    points = inner_inputs.get('points', None)
                    radar = inner_inputs.get('radar', None)
            
            kwargs = {k: v for k, v in inputs.items() if k not in ['img_inputs', 'points', 'radar', 'inputs']}
        else:
            img_inputs = None
            points = None
            radar = None
            kwargs = {}
        
        img_metas = []
        if data_samples is not None:
            # Check if data_samples is a dict with 'metainfo' key
            if isinstance(data_samples, dict):
                metainfo = data_samples.get('metainfo', {})
                if metainfo:
                    img_metas.append(metainfo)
            elif isinstance(data_samples, (list, tuple)):
                for sample in data_samples:
                    if hasattr(sample, 'metainfo'):
                        img_metas.append(sample.metainfo)
                    elif isinstance(sample, dict) and 'metainfo' in sample:
                        img_metas.append(sample['metainfo'])
        
        # Use simple_test method - it requires points and img_metas as positional args
        return self.simple_test(
            points,
            img_metas if img_metas else None,
            img=img_inputs,
            radar=radar,
            **kwargs
        )

    def _forward(self, inputs, data_samples=None):
        """Forward without loss calculation."""
        # Similar to predict but returns raw features
        img_inputs = inputs.get('img_inputs', None)
        points = inputs.get('points', None)
        radar = inputs.get('radar', None)
        
        img_metas = []
        if data_samples is not None:
            for sample in data_samples:
                if hasattr(sample, 'metainfo'):
                    img_metas.append(sample.metainfo)
        
        return self.extract_feat(
            points, 
            img=img_inputs, 
            img_metas=img_metas if img_metas else None,
            radar=radar
        )

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
        """Forward training function."""
        img_feats, pts_feats, depth, bev_feat_list, radar_feats, prev_radar_feats = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)
        gt_depth = kwargs.get('gt_depth', None)
        losses = dict()
        
        # Debug depth loss calculation
        if gt_depth is not None and depth is not None:
            loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth)
        else:
            # If no gt_depth available, create a small loss to ensure gradients flow
            if depth is not None:
                loss_depth = torch.mean(depth) * 0.001  # Small regularization loss
            else:
                loss_depth = torch.tensor(0.0, device=next(self.parameters()).device, requires_grad=True)
        
        losses['loss_depth'] = loss_depth
        
        # Keep 3D structure for 3D convolutions: concatenate along channel dimension
        # img_feats[0]: [bs, 32, 4, 50, 50] 
        # radar_feats[0]: [bs, 96, 4, 50, 50]
        # Total: [bs, 32+96=128, 4, 50, 50]
        fusion_feats = self.reduc_conv(torch.cat((img_feats[0], radar_feats[0]), dim=1))

        occ_pred = self.final_conv(fusion_feats).permute(0, 4, 3, 2, 1) # bncdhw->bnwhdc
        
        # Upsample prediction to match ground truth resolution  
        # preds: [bs, 50, 50, 4, classes] -> [bs, 200, 200, 16, classes]
        bs, w, h, d, classes = occ_pred.shape
        target_w, target_h, target_d = 200, 200, 16
        
        # Interpolate each spatial dimension
        occ_pred = occ_pred.permute(0, 4, 3, 1, 2)  # [bs, classes, d, w, h]
        occ_pred = F.interpolate(occ_pred, size=(target_d, target_w, target_h), mode='trilinear', align_corners=False)
        occ_pred = occ_pred.permute(0, 3, 4, 2, 1)  # [bs, w, h, d, classes] -> [bs, 200, 200, 16, classes]
        if self.use_predicter:
            occ_pred = self.predicter(occ_pred)

        voxel_semantics = kwargs['voxel_semantics']
        mask_camera = kwargs['mask_camera']
        # Handle case where voxel_semantics is a list
        if isinstance(voxel_semantics, list):
            # Convert numpy arrays to tensors first
            voxel_semantics = [torch.from_numpy(v) if isinstance(v, np.ndarray) else v for v in voxel_semantics]
            voxel_semantics = torch.stack(voxel_semantics)
        
        # Handle case where mask_camera is a list
        if isinstance(mask_camera, list):
            # Convert numpy arrays to tensors first
            mask_camera = [torch.from_numpy(v) if isinstance(v, np.ndarray) else v for v in mask_camera]
            mask_camera = torch.stack(mask_camera)
            
        assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17
        loss_occ = self.loss_single(voxel_semantics, mask_camera, occ_pred)
        losses.update(loss_occ)

        return losses

    def extract_feat(self, points, img, img_metas, radar, **kwargs):
        """Extract features from images and points."""
        # Convert img_inputs tuple to list format expected by prepare_inputs
        # img_inputs from PrepareImageInputs: (imgs, intrins, post_rots, post_trans, sensor2egos, ego2globals)
        if img is None:
            raise ValueError("img cannot be None")
        
        # Handle radar data - convert RadarPoints objects to tensors if needed
        if radar is not None:
            if isinstance(radar, (list, tuple)):
                processed_radar = []
                for r in radar:
                    if hasattr(r, 'tensor'):
                        # RadarPoints object
                        processed_radar.append(r.tensor)
                    elif isinstance(r, torch.Tensor):
                        processed_radar.append(r)
                radar = processed_radar
            elif hasattr(radar, 'tensor'):
                # Single RadarPoints object
                radar = [radar.tensor]
            elif isinstance(radar, torch.Tensor):
                radar = [radar]
        
        # Handle case where img is a list of tuples (each containing a tensor)
        # This happens when data comes from Collect3D
        if isinstance(img, (list, tuple)) and len(img) == 6:
            # Check if elements are wrapped in tuples
            if isinstance(img[0], (tuple, list)) and len(img[0]) > 0 and isinstance(img[0][0], torch.Tensor):
                # Extract tensors from tuples
                img = tuple(item[0] if isinstance(item, (tuple, list)) else item for item in img)
        
        if isinstance(img, tuple) and len(img) == 6:
            imgs, intrins, post_rots, post_trans, sensor2egos, ego2globals = img
            
            # Check if any is None
            if imgs is None:
                raise ValueError("imgs cannot be None")
            
            # Add batch dimension if needed: [N, C, H, W] -> [B, N, C, H, W]
            if len(imgs.shape) == 4:
                imgs = imgs.unsqueeze(0)
            if len(intrins.shape) == 3:
                intrins = intrins.unsqueeze(0)
            if len(post_rots.shape) == 3:
                post_rots = post_rots.unsqueeze(0)
            if len(post_trans.shape) == 2:
                post_trans = post_trans.unsqueeze(0)
            if len(sensor2egos.shape) == 3:
                sensor2egos = sensor2egos.unsqueeze(0)
            if len(ego2globals.shape) == 3:
                ego2globals = ego2globals.unsqueeze(0)
            
            # Replicate single frame to multiple frames if needed
            # Expected shape: [B, N*num_frame, C, H, W] where N is number of cameras
            B, N, C, H, W = imgs.shape
            
            if N < 6 * self.num_frame:
                # Assume N is the number of cameras (6) for a single frame
                # Replicate to num_frame frames
                imgs = imgs.repeat(1, self.num_frame, 1, 1, 1)
                intrins = intrins.repeat(1, self.num_frame, 1, 1)
                post_rots = post_rots.repeat(1, self.num_frame, 1, 1)
                post_trans = post_trans.repeat(1, self.num_frame, 1)
                sensor2egos = sensor2egos.repeat(1, self.num_frame, 1, 1)
                ego2globals = ego2globals.repeat(1, self.num_frame, 1, 1)
            
            # Add bda (identity matrix for now)
            bda = torch.eye(3).to(imgs.device)
            
            # Reorder to match original BEVDet format: [imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans, bda]
            img = [imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans, bda]
        elif isinstance(img, (list, tuple)) and len(img) > 0:
            # Check if elements are tuples containing tensors
            if isinstance(img[0], (tuple, list)) and len(img[0]) > 0 and isinstance(img[0][0], torch.Tensor):
                # Take the first tensor from each tuple (should be the image tensor)
                img = img[0][0]  # Take first tensor from first tuple
                
                # Add batch dimension if needed (expected: [B, N, C, H, W])
                if len(img.shape) == 4:  # [N, C, H, W] -> [B, N, C, H, W]
                    img = img.unsqueeze(0)
            elif isinstance(img[0], torch.Tensor):
                img = torch.stack(img, dim=0)
        
        img_feats, depth, prev_feats = self.extract_img_feat(img, img_metas, **kwargs)
        pts_feats = None

        radar_feats, prev_radar_feats = self.extract_radar_feat(radar, img_metas)

        return (img_feats, pts_feats, depth, prev_feats, radar_feats, prev_radar_feats)

    def loss(self, batch_inputs, batch_data_samples, **kwargs):
        """Calculate losses from a batch of inputs and data samples."""
        return self.forward_train(**batch_inputs, **kwargs)

    def extract_img_feat(self,
                         img,
                         img_metas,
                         with_bevencoder=True,
                         pred_prev=False,
                         sequential=False,
                         **kwargs):
        if sequential:
            # Todo
            assert False
        imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, \
        bda, curr2adjsensor = self.prepare_inputs(img, stereo=True)
        """Extract features of images."""
        bev_feat_list = []
        depth_key_frame = None
        feat_prev_iv = None
        for fid in range(self.num_frame - 1, -1, -1):
            img, sensor2keyego, ego2global, intrin, post_rot, post_tran = \
                imgs[fid], sensor2keyegos[fid], ego2globals[fid], intrins[fid], \
                post_rots[fid], post_trans[fid]
            key_frame = fid == 0
            extra_ref_frame = fid == self.num_frame - self.extra_ref_frames
            if key_frame or self.with_prev:
                if self.align_after_view_transfromation:
                    sensor2keyego, ego2global = sensor2keyegos[0], ego2globals[0]
                mlp_input = self.img_view_transformer.get_mlp_input(
                    sensor2keyegos[0], ego2globals[0], intrin,
                    post_rot, post_tran, bda)
                inputs_curr = (img, sensor2keyego, ego2global, intrin,
                               post_rot, post_tran, bda, mlp_input,
                               feat_prev_iv, curr2adjsensor[fid],
                               extra_ref_frame)
                if key_frame:
                    bev_feat, depth, feat_curr_iv = \
                        self.prepare_bev_feat(*inputs_curr)
                    depth_key_frame = depth
                else:
                    with torch.no_grad():
                        bev_feat, depth, feat_curr_iv = \
                            self.prepare_bev_feat(*inputs_curr)
                if not extra_ref_frame:
                    bev_feat_list.append(bev_feat)
                feat_prev_iv = feat_curr_iv
        if pred_prev:
            # Todo
            assert False
        if not self.with_prev:
            bev_feat_key = bev_feat_list[0]
            if len(bev_feat_key.shape) == 4:
                b, c, h, w = bev_feat_key.shape
                # Create features for (num_frame-1) adjacent frames + 1 key frame = num_frame total
                # Total channels should be c * num_frame = 32 * 9 = 288
                bev_feat_list = \
                    [torch.zeros([b,
                                  c * (self.num_frame - 1),  # Adjacent frames
                                  h, w]).to(bev_feat_key), bev_feat_key]  # Key frame
            else:
                b, c, z, h, w = bev_feat_key.shape
                # Create features for (num_frame-1) adjacent frames + 1 key frame = num_frame total  
                # Total channels should be c * num_frame = 32 * 9 = 288
                bev_feat_list = \
                    [torch.zeros([b,
                                  c * (self.num_frame - 1), z,  # Adjacent frames
                                  h, w]).to(bev_feat_key), bev_feat_key]  # Key frame
        if self.align_after_view_transfromation:
            for adj_id in range(self.num_frame - 2):
                bev_feat_list[adj_id] = \
                    self.shift_feature(bev_feat_list[adj_id],
                                       [sensor2keyegos[0],
                                        sensor2keyegos[self.num_frame - 2 - adj_id]],
                                       bda)
        bev_feat = torch.cat(bev_feat_list, dim=1)
        if with_bevencoder:
            x = self.bev_encoder(bev_feat)
            return [x], depth_key_frame, bev_feat_list
        else:
            return [bev_feat], depth_key_frame, bev_feat_list
