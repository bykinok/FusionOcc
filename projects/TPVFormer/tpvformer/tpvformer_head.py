import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule

from mmdet3d.registry import MODELS


@MODELS.register_module()
class TPVFormerDecoder(BaseModule):

    def __init__(self,
                 tpv_h,
                 tpv_w,
                 tpv_z,
                 num_classes=20,
                 in_dims=64,
                 hidden_dims=128,
                 out_dims=None,
                 scale_h=2,
                 scale_w=2,
                 scale_z=2,
                 ignore_index=0,
                 loss_lovasz=None,
                 loss_ce=None,
                 lovasz_input='points',
                 ce_input='voxel'):
        super().__init__()
        self.tpv_h = tpv_h
        self.tpv_w = tpv_w
        self.tpv_z = tpv_z
        self.scale_h = scale_h
        self.scale_w = scale_w
        self.scale_z = scale_z

        out_dims = in_dims if out_dims is None else out_dims
        self.in_dims = in_dims
        self.decoder = nn.Sequential(
            nn.Linear(in_dims, hidden_dims), nn.Softplus(),
            nn.Linear(hidden_dims, out_dims))

        self.classifier = nn.Linear(out_dims, num_classes)
        self.loss_lovasz = MODELS.build(loss_lovasz)
        self.loss_ce = MODELS.build(loss_ce)
        self.ignore_index = ignore_index
        self.lovasz_input = lovasz_input
        self.ce_input = ce_input

    def forward(self, tpv_list, points=None):
        """
        tpv_list[0]: bs, h*w, c
        tpv_list[1]: bs, z*h, c
        tpv_list[2]: bs, w*z, c
        """
        tpv_hw, tpv_zh, tpv_wz = tpv_list[0], tpv_list[1], tpv_list[2]
        bs, _, c = tpv_hw.shape
        tpv_hw = tpv_hw.permute(0, 2, 1).reshape(bs, c, self.tpv_h, self.tpv_w)
        tpv_zh = tpv_zh.permute(0, 2, 1).reshape(bs, c, self.tpv_z, self.tpv_h)
        tpv_wz = tpv_wz.permute(0, 2, 1).reshape(bs, c, self.tpv_w, self.tpv_z)

        if self.scale_h != 1 or self.scale_w != 1:
            tpv_hw = F.interpolate(
                tpv_hw,
                size=(self.tpv_h * self.scale_h, self.tpv_w * self.scale_w),
                mode='bilinear')
        if self.scale_z != 1 or self.scale_h != 1:
            tpv_zh = F.interpolate(
                tpv_zh,
                size=(self.tpv_z * self.scale_z, self.tpv_h * self.scale_h),
                mode='bilinear')
        if self.scale_w != 1 or self.scale_z != 1:
            tpv_wz = F.interpolate(
                tpv_wz,
                size=(self.tpv_w * self.scale_w, self.tpv_z * self.scale_z),
                mode='bilinear')

        if points is not None:
            # points: bs, n, 3
            _, n, _ = points.shape
            points = points.reshape(bs, 1, n, 3).float()
            points[...,
                   0] = points[..., 0] / (self.tpv_w * self.scale_w) * 2 - 1
            points[...,
                   1] = points[..., 1] / (self.tpv_h * self.scale_h) * 2 - 1
            points[...,
                   2] = points[..., 2] / (self.tpv_z * self.scale_z) * 2 - 1
            sample_loc = points[:, :, :, [0, 1]]
            tpv_hw_pts = F.grid_sample(tpv_hw,
                                       sample_loc).squeeze(2)  # bs, c, n
            sample_loc = points[:, :, :, [1, 2]]
            tpv_zh_pts = F.grid_sample(tpv_zh, sample_loc).squeeze(2)
            sample_loc = points[:, :, :, [2, 0]]
            tpv_wz_pts = F.grid_sample(tpv_wz, sample_loc).squeeze(2)

            tpv_hw_vox = tpv_hw.unsqueeze(-1).permute(0, 1, 3, 2, 4).expand(
                -1, -1, -1, -1, self.scale_z * self.tpv_z)
            tpv_zh_vox = tpv_zh.unsqueeze(-1).permute(0, 1, 4, 3, 2).expand(
                -1, -1, self.scale_w * self.tpv_w, -1, -1)
            tpv_wz_vox = tpv_wz.unsqueeze(-1).permute(0, 1, 2, 4, 3).expand(
                -1, -1, -1, self.scale_h * self.tpv_h, -1)

            fused_vox = (tpv_hw_vox + tpv_zh_vox + tpv_wz_vox).flatten(2)
            fused_pts = tpv_hw_pts + tpv_zh_pts + tpv_wz_pts
            fused = torch.cat([fused_vox, fused_pts], dim=-1)  # bs, c, whz+n

            fused = fused.permute(0, 2, 1)
            if hasattr(self, 'use_checkpoint') and self.use_checkpoint:
                fused = torch.utils.checkpoint.checkpoint(self.decoder, fused)
                logits = torch.utils.checkpoint.checkpoint(
                    self.classifier, fused)
            else:
                fused = self.decoder(fused)
                logits = self.classifier(fused)
            logits = logits.permute(0, 2, 1)
            logits_vox = logits[:, :, :(-n)].reshape(bs, self.num_classes,
                                                     self.scale_w * self.tpv_w,
                                                     self.scale_h * self.tpv_h,
                                                     self.scale_z * self.tpv_z)
            logits_pts = logits[:, :, (-n):].reshape(bs, self.num_classes, n, 1, 1)
            return logits_vox, logits_pts

        else:
            tpv_hw = tpv_hw.unsqueeze(-1).permute(0, 1, 3, 2, 4).expand(
                -1, -1, -1, -1, self.scale_z * self.tpv_z)
            tpv_zh = tpv_zh.unsqueeze(-1).permute(0, 1, 4, 3, 2).expand(
                -1, -1, self.scale_w * self.tpv_w, -1, -1)
            tpv_wz = tpv_wz.unsqueeze(-1).permute(0, 1, 2, 4, 3).expand(
                -1, -1, -1, self.scale_h * self.tpv_h, -1)

            fused = tpv_hw + tpv_zh + tpv_wz
            fused = fused.permute(0, 2, 3, 4, 1)
            if hasattr(self, 'use_checkpoint') and self.use_checkpoint:
                fused = torch.utils.checkpoint.checkpoint(self.decoder, fused)
                logits = torch.utils.checkpoint.checkpoint(
                    self.classifier, fused)
            else:
                fused = self.decoder(fused)
                logits = self.classifier(fused)
            logits = logits.permute(0, 4, 1, 2, 3)

            return logits

    def loss(self, tpv_list, batch_data_samples):
        tpv_hw, tpv_zh, tpv_wz = tpv_list
        bs, _, c = tpv_hw.shape
        tpv_hw = tpv_hw.permute(0, 2, 1).reshape(bs, c, self.tpv_h, self.tpv_w)
        tpv_zh = tpv_zh.permute(0, 2, 1).reshape(bs, c, self.tpv_z, self.tpv_h)
        tpv_wz = tpv_wz.permute(0, 2, 1).reshape(bs, c, self.tpv_w, self.tpv_z)

        if self.scale_h != 1 or self.scale_w != 1:
            tpv_hw = F.interpolate(
                tpv_hw,
                size=(self.tpv_h * self.scale_h, self.tpv_w * self.scale_w),
                mode='bilinear')
        if self.scale_z != 1 or self.scale_h != 1:
            tpv_zh = F.interpolate(
                tpv_zh,
                size=(self.tpv_z * self.scale_z, self.tpv_h * self.scale_h),
                mode='bilinear')
        if self.scale_w != 1 or self.scale_z != 1:
            tpv_wz = F.interpolate(
                tpv_wz,
                size=(self.tpv_w * self.scale_w, self.tpv_z * self.scale_z),
                mode='bilinear')

        batch_pts, batch_vox = [], []
        for i, data_sample in enumerate(batch_data_samples):
            point_coors = data_sample.point_coors.reshape(1, 1, -1, 3).float()
            point_coors[
                ...,
                0] = point_coors[..., 0] / (self.tpv_w * self.scale_w) * 2 - 1
            point_coors[
                ...,
                1] = point_coors[..., 1] / (self.tpv_h * self.scale_h) * 2 - 1
            point_coors[
                ...,
                2] = point_coors[..., 2] / (self.tpv_z * self.scale_z) * 2 - 1
            sample_loc = point_coors[..., [0, 1]]
            tpv_hw_pts = F.grid_sample(
                tpv_hw[i:i + 1], sample_loc, align_corners=False)
            sample_loc = point_coors[..., [1, 2]]
            tpv_zh_pts = F.grid_sample(
                tpv_zh[i:i + 1], sample_loc, align_corners=False)
            sample_loc = point_coors[..., [2, 0]]
            tpv_wz_pts = F.grid_sample(
                tpv_wz[i:i + 1], sample_loc, align_corners=False)
            fused_pts = (tpv_hw_pts + tpv_zh_pts +
                         tpv_wz_pts).squeeze(0).squeeze(1)
            batch_pts.append(fused_pts)

            tpv_hw_vox = tpv_hw.unsqueeze(-1).permute(0, 1, 3, 2, 4).expand(
                -1, -1, -1, -1, self.scale_z * self.tpv_z)
            tpv_zh_vox = tpv_zh.unsqueeze(-1).permute(0, 1, 4, 3, 2).expand(
                -1, -1, self.scale_w * self.tpv_w, -1, -1)
            tpv_wz_vox = tpv_wz.unsqueeze(-1).permute(0, 1, 2, 4, 3).expand(
                -1, -1, -1, self.scale_h * self.tpv_h, -1)
            fused_vox = tpv_hw_vox + tpv_zh_vox + tpv_wz_vox
            voxel_coors = data_sample.voxel_coors.long()
            fused_vox = fused_vox[:, :, voxel_coors[:, 0], voxel_coors[:, 1],
                                  voxel_coors[:, 2]]
            fused_vox = fused_vox.squeeze(0)
            batch_vox.append(fused_vox)
        batch_pts = torch.cat(batch_pts, dim=1)
        batch_vox = torch.cat(batch_vox, dim=1)
        num_points = batch_pts.shape[1]

        logits = self.decoder(
            torch.cat([batch_pts, batch_vox], dim=1).transpose(0, 1))
        logits = self.classifier(logits)
        pts_logits = logits[:num_points, :]
        vox_logits = logits[num_points:, :]

        pts_seg_label = torch.cat([
            data_sample.gt_pts_seg.pts_semantic_mask
            for data_sample in batch_data_samples
        ])
        voxel_seg_label = torch.cat([
            data_sample.gt_pts_seg.voxel_semantic_mask
            for data_sample in batch_data_samples
        ])
        if self.ce_input == 'voxel':
            ce_input = vox_logits
            ce_label = voxel_seg_label
        else:
            ce_input = pts_logits
            ce_label = pts_seg_label
        if self.lovasz_input == 'voxel':
            lovasz_input = vox_logits
            lovasz_label = voxel_seg_label
        else:
            lovasz_input = pts_logits
            lovasz_label = pts_seg_label

        loss = dict()
        loss['loss_ce'] = self.loss_ce(
            ce_input, ce_label, ignore_index=self.ignore_index)
        loss['loss_lovasz'] = self.loss_lovasz(
            lovasz_input, lovasz_label, ignore_index=self.ignore_index)
        return loss

    def predict(self, tpv_list, batch_data_samples):
        """
        tpv_list[0]: bs, h*w, c
        tpv_list[1]: bs, z*h, c
        tpv_list[2]: bs, w*z, c
        """
        tpv_hw, tpv_zh, tpv_wz = tpv_list
        bs, _, c = tpv_hw.shape
        tpv_hw = tpv_hw.permute(0, 2, 1).reshape(bs, c, self.tpv_h, self.tpv_w)
        tpv_zh = tpv_zh.permute(0, 2, 1).reshape(bs, c, self.tpv_z, self.tpv_h)
        tpv_wz = tpv_wz.permute(0, 2, 1).reshape(bs, c, self.tpv_w, self.tpv_z)

        if self.scale_h != 1 or self.scale_w != 1:
            tpv_hw = F.interpolate(
                tpv_hw,
                size=(self.tpv_h * self.scale_h, self.tpv_w * self.scale_w),
                mode='bilinear')
        if self.scale_z != 1 or self.scale_h != 1:
            tpv_zh = F.interpolate(
                tpv_zh,
                size=(self.tpv_z * self.scale_z, self.tpv_h * self.scale_h),
                mode='bilinear')
        if self.scale_w != 1 or self.scale_z != 1:
            tpv_wz = F.interpolate(
                tpv_wz,
                size=(self.tpv_w * self.scale_w, self.tpv_z * self.scale_z),
                mode='bilinear')

        logits = []
        for i, data_sample in enumerate(batch_data_samples):
            point_coors = data_sample.point_coors.reshape(1, 1, -1, 3).float()
            point_coors[
                ...,
                0] = point_coors[..., 0] / (self.tpv_w * self.scale_w) * 2 - 1
            point_coors[
                ...,
                1] = point_coors[..., 1] / (self.tpv_h * self.scale_h) * 2 - 1
            point_coors[
                ...,
                2] = point_coors[..., 2] / (self.tpv_z * self.scale_z) * 2 - 1
            sample_loc = point_coors[..., [0, 1]]
            tpv_hw_pts = F.grid_sample(
                tpv_hw[i:i + 1], sample_loc, align_corners=False)
            sample_loc = point_coors[..., [1, 2]]
            tpv_zh_pts = F.grid_sample(
                tpv_zh[i:i + 1], sample_loc, align_corners=False)
            sample_loc = point_coors[..., [2, 0]]
            tpv_wz_pts = F.grid_sample(
                tpv_wz[i:i + 1], sample_loc, align_corners=False)

            fused_pts = tpv_hw_pts + tpv_zh_pts + tpv_wz_pts

            fused_pts = fused_pts.squeeze(0).squeeze(1).transpose(0, 1)
            fused_pts = self.decoder(fused_pts)
            logit = self.classifier(fused_pts)
            logits.append(logit)

        return logits


@MODELS.register_module()
class TPVFormerHead(BaseModule):
    """TPVFormer Head for occupancy prediction.
    
    This class is designed to be compatible with the original tpv04_occupancy.py
    configuration and structure.
    """

    def __init__(self,
                 tpv_h,
                 tpv_w,
                 tpv_z,
                 pc_range,
                 num_feature_levels=4,
                 num_cams=6,
                 embed_dims=256,
                 positional_encoding=None,
                 encoder=None):
        super().__init__()
        self.tpv_h = tpv_h
        self.tpv_w = tpv_w
        self.tpv_z = tpv_z
        self.pc_range = pc_range
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.embed_dims = embed_dims
        
        # Build positional encoding
        if positional_encoding is not None:
            self.positional_encoding = MODELS.build(positional_encoding)
        else:
            self.encoder = None
            
        # Build encoder
        if encoder is not None:
            self.encoder = MODELS.build(encoder)
        else:
            self.encoder = None

    def forward(self, img_feats, batch_data_samples):
        """Forward function.
        
        Args:
            img_feats: Image features from backbone and neck
            batch_data_samples: Batch data samples
            
        Returns:
            TPV queries from encoder
        """
        if self.encoder is not None:
            # Forward through encoder
            tpv_queries = self.encoder(img_feats, batch_data_samples)
            
            # Add positional encoding if available
            if self.positional_encoding is not None:
                batch_size = img_feats[0].shape[0] if img_feats else 1
                device = img_feats[0].device if img_feats else 'cpu'
                pos_encodings = self.positional_encoding(batch_size, device)
                
                # Add positional encoding to TPV queries
                if len(tpv_queries) == 3 and len(pos_encodings) == 3:
                    for i in range(3):
                        if tpv_queries[i].shape == pos_encodings[i].shape:
                            tpv_queries[i] = tpv_queries[i] + pos_encodings[i]
            
            return tpv_queries
        else:
            # Return dummy queries if no encoder
            batch_size = img_feats[0].shape[0] if img_feats else 1
            device = img_feats[0].device if img_feats else 'cpu'
            
            # Create dummy TPV queries with correct dimensions
            dummy_queries = [
                torch.randn(batch_size, self.tpv_h * self.tpv_w, self.embed_dims, device=device),
                torch.randn(batch_size, self.tpv_z * self.tpv_h, self.embed_dims, device=device),
                torch.randn(batch_size, self.tpv_w * self.tpv_z, self.embed_dims, device=device)
            ]
            return dummy_queries
