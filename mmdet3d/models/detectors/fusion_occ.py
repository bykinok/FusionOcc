# Copyright (c) Zhejiang Lab. All rights reserved.
import torch
import numpy as np
from torch import nn
from mmcv.cnn.bricks.conv_module import ConvModule
from mmcv.runner import auto_fp16, force_fp32
from mmdet.models import DETECTORS
from mmdet.models.builder import build_loss

from ...models.backbones.lidar_encoder import CustomSparseEncoder
from .bevdet import BEVDepth4D


@DETECTORS.register_module()
class FusionDepthSeg(BEVDepth4D):
    def __init__(self, **kwargs):
        # Initialize data_preprocessor if not present (for compatibility with MMEngine)
        if 'data_preprocessor' not in kwargs:
            kwargs['data_preprocessor'] = None
            
        super(FusionDepthSeg, self).__init__(**kwargs)

    def prepare_img_3d_feat(self, img, sensor2keyego, ego2global, intrin,
                            post_rot, post_tran, bda, mlp_input, input_depth=None):
        x, _ = self.image_encoder(img, stereo=False)
        img_3d_feat, depth, seg = self.img_view_transformer(
            [x, sensor2keyego, ego2global, intrin, post_rot, post_tran, bda,
             mlp_input], input_depth)
        if self.pre_process:
            img_3d_feat = self.pre_process_net(img_3d_feat)[0]
        return img_3d_feat, depth, seg

    def extract_img_3d_feat(self,
                            img_inputs,
                            input_depth):
        imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, bda, _ = \
            self.prepare_inputs(img_inputs, stereo=False)
        """Extract features of images."""
        img_3d_feat_list = []
        depth_key_frame = None
        seg_key_frame = None
        for fid in range(self.num_frame - 1, -1, -1):
            img, sensor2keyego, ego2global, intrin, post_rot, post_tran = \
                imgs[fid], sensor2keyegos[fid], ego2globals[fid], intrins[fid], \
                post_rots[fid], post_trans[fid]
            curr_frame = fid == 0
            if self.align_after_view_transformation:
                sensor2keyego, ego2global = sensor2keyegos[0], ego2globals[0]
            mlp_input = self.img_view_transformer.get_mlp_input(
                sensor2keyegos[0], ego2globals[0], intrin,
                post_rot, post_tran, bda)
            inputs_curr = (img, sensor2keyego, ego2global, intrin,
                           post_rot, post_tran, bda, mlp_input, input_depth)
            if curr_frame:
                img_3d_feat, depth, pred_seg = self.prepare_img_3d_feat(*inputs_curr)
                seg_key_frame = pred_seg
                depth_key_frame = depth
            else:
                with torch.no_grad():
                    img_3d_feat, _, _ = self.prepare_img_3d_feat(*inputs_curr)
            img_3d_feat_list.append(img_3d_feat)
        if self.align_after_view_transformation:
            for adj_id in range(self.num_frame - 1):
                img_3d_feat_list[adj_id] = \
                    self.shift_feature(img_3d_feat_list[adj_id],
                                       [sensor2keyegos[0],
                                        sensor2keyegos[self.num_frame - 2 - adj_id]],
                                       bda)
        img_3d_feat_feat = torch.cat(img_3d_feat_list, dim=1)
        return img_3d_feat_feat, depth_key_frame, seg_key_frame


@DETECTORS.register_module()
class FusionOCC(FusionDepthSeg):
    def __init__(self,
                 loss_occ=None,
                 out_dim=32,
                 use_mask=False,
                 num_classes=18,
                 use_predicter=True,
                 class_wise=False,
                 point_cloud_range=[-40, -40, -1, 40, 40, 5.4],
                 voxel_size=[0.05, 0.05, 0.05],
                 lidar_in_channel=5,
                 lidar_out_channel=32,
                 fuse_loss_weight=0.1,
                 occ_encoder_backbone=None,
                 occ_encoder_neck=None,
                 **kwargs):
        # Initialize data_preprocessor if not present
        if 'data_preprocessor' not in kwargs:
            kwargs['data_preprocessor'] = None
        
        super(FusionOCC, self).__init__(
            img_bev_encoder_backbone=occ_encoder_backbone,
            img_bev_encoder_neck=occ_encoder_neck,
            **kwargs)
        self.voxel_size = voxel_size
        self.lidar_out_channel = lidar_out_channel
        self.lidar_in_channel = lidar_in_channel
        self.sparse_shape = [
            int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0]),
            int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1]),
            int((point_cloud_range[5] - point_cloud_range[2]) / voxel_size[2])
        ]
        self.point_cloud_range = point_cloud_range
        self.lidar_encoder = CustomSparseEncoder(
            in_channels=self.lidar_in_channel,
            sparse_shape=self.sparse_shape,
            point_cloud_range=self.point_cloud_range,
            voxel_size=self.voxel_size,
            output_channels=self.lidar_out_channel,
            # block_type="basicblock"
            block_type="conv_module"
        )
        self.out_dim = out_dim
        out_channels = out_dim if use_predicter else num_classes
        self.final_conv = ConvModule(
            out_dim,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            conv_cfg=dict(type='Conv3d'))
        self.use_predicter = use_predicter
        if use_predicter:
            self.predicter = nn.Sequential(
                nn.Linear(self.out_dim, self.out_dim * 2),
                nn.Softplus(),
                nn.Linear(self.out_dim * 2, num_classes),
            )
        self.pts_bbox_head = None
        self.use_mask = use_mask
        self.num_classes = num_classes
        self.loss_occ = build_loss(loss_occ)
        self.class_wise = class_wise
        self.align_after_view_transformation = False
        self.fuse_loss_weight = fuse_loss_weight

    def occ_encoder(self, fusion_feat):
        return self.bev_encoder(fusion_feat)

    def extract_feat(self, lidar_feat, img, img_metas, input_depth=None, **kwargs):
        """Extract features from images and points."""
        fusion_feats, depth, pred_segs = self.extract_fusion_feat(
            lidar_feat, img, img_metas, input_depth=input_depth, **kwargs
        )
        pts_feats = None
        return fusion_feats, pts_feats, depth, pred_segs

    def forward_train(self,
                      points=None,
                      img_inputs=None,
                      segs=None,
                      sparse_depth=None,
                      **kwargs):
        lidar_feat, x_list, x_sparse_out = self.lidar_encoder(points)
        lidar_feat = lidar_feat.permute(0, 1, 2, 4, 3).contiguous()

        input_depth = sparse_depth
        img_3d_feat_feat, depth_key_frame, seg_key_frame = self.extract_img_3d_feat(
            img_inputs=img_inputs, input_depth=input_depth)
        fusion_feat = torch.cat([img_3d_feat_feat, lidar_feat], dim=1)
        fusion_feat = self.occ_encoder(fusion_feat)

        losses = dict()
        depth_loss, seg_loss, vis_depth_pred, vis_depth_label, vis_seg_pred, vis_seg_label = \
            self.img_view_transformer.get_loss(sparse_depth, depth_key_frame, segs, seg_key_frame)
        losses['depth_loss'] = depth_loss * self.fuse_loss_weight
        losses['seg_loss'] = seg_loss * self.fuse_loss_weight

        occ_pred = self.final_conv(fusion_feat).permute(0, 4, 3, 2, 1)
        if self.use_predicter:
            occ_pred = self.predicter(occ_pred)
        voxel_semantics = kwargs['voxel_semantics']
        mask_camera = kwargs['mask_camera']

        assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17
        loss_occ = self.loss_single(voxel_semantics, mask_camera, occ_pred)
        losses.update(loss_occ)
        return losses

    def loss_single(self, voxel_semantics, mask_camera, preds):
        loss_ = dict()
        voxel_semantics = voxel_semantics.long()
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
            loss_occ = self.loss_occ(preds, voxel_semantics, )
            loss_['loss_occ'] = loss_occ
        return loss_

    def simple_test(self,
                    points=None,
                    img_metas=None,
                    img_inputs=None,
                    sparse_depth=None,
                    **kwargs):
        """Test function without augmentaiton."""

        sparse_depth = sparse_depth[0]
        lidar_feat, x_list, x_sparse_out = self.lidar_encoder(points)
        # N, C, D, H, W -> N,C,D,W,H
        lidar_feat = lidar_feat.permute(0, 1, 2, 4, 3).contiguous()
        input_depth = sparse_depth
        img_3d_feat_feat, depth_key_frame, seg_key_frame = self.extract_img_3d_feat(
            img_inputs=img_inputs, input_depth=input_depth)
        fusion_feat = torch.cat([img_3d_feat_feat, lidar_feat], dim=1)
        fusion_feat = self.occ_encoder(fusion_feat)

        occ_pred = self.final_conv(fusion_feat).permute(0, 4, 3, 2, 1)  # bncdhw->bnwhdc
        if self.use_predicter:
            occ_pred = self.predicter(occ_pred)
        occ_score = occ_pred.softmax(-1)
        occ_res = occ_score.argmax(-1)
        occ_res = occ_res.squeeze(dim=0).cpu().numpy().astype(np.uint8)
        return [occ_res]

    def show_results(self, data, result, out_dir):
        """Results visualization for occupancy prediction.

        Args:
            data (dict): Input points and the information of the sample.
            result (list): Prediction results (occupancy grids).
            out_dir (str): Output directory of visualization result.
        """
        import os
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        for batch_id in range(len(result)):
            # Get occupancy grid result
            occ_grid = result[batch_id]  # Shape: (W, H, D)
            
            # Get point cloud data for reference
            if hasattr(data['points'][0], '_data'):
                points = data['points'][0]._data[0][batch_id].numpy()
            elif hasattr(data['points'][0], 'data'):
                points = data['points'][0].data[0][batch_id].numpy()
            else:
                points = data['points'][0][batch_id].numpy()
            
            # Get metadata
            if hasattr(data['img_metas'][0], '_data'):
                img_meta = data['img_metas'][0]._data[0][batch_id]
            elif hasattr(data['img_metas'][0], 'data'):
                img_meta = data['img_metas'][0].data[0][batch_id]
            else:
                img_meta = data['img_metas'][0][batch_id]
            
            # Create filename
            pts_filename = img_meta.get('pts_filename', f'sample_{batch_id}')
            file_name = os.path.split(pts_filename)[-1].split('.')[0]
            
            # Create visualization
            fig = plt.figure(figsize=(15, 5))
            
            # Plot 1: Original point cloud (top view)
            ax1 = fig.add_subplot(131)
            ax1.scatter(points[:, 0], points[:, 1], c=points[:, 2], s=0.1, cmap='viridis')
            ax1.set_title('Original Point Cloud (Top View)')
            ax1.set_xlabel('X (m)')
            ax1.set_ylabel('Y (m)')
            ax1.axis('equal')
            
            # Plot 2: Occupancy grid slice (middle height)
            ax2 = fig.add_subplot(132)
            mid_height = occ_grid.shape[2] // 2
            occ_slice = occ_grid[:, :, mid_height]
            im = ax2.imshow(occ_slice.T, origin='lower', cmap='tab20', vmin=0, vmax=17)
            ax2.set_title(f'Occupancy Grid (Height Slice {mid_height})')
            ax2.set_xlabel('X Index')
            ax2.set_ylabel('Y Index')
            plt.colorbar(im, ax=ax2)
            
            # Plot 3: Occupancy statistics
            ax3 = fig.add_subplot(133)
            unique, counts = np.unique(occ_grid, return_counts=True)
            ax3.bar(unique, counts)
            ax3.set_title('Occupancy Class Distribution')
            ax3.set_xlabel('Class ID')
            ax3.set_ylabel('Voxel Count')
            ax3.set_yscale('log')
            
            plt.tight_layout()
            
            # Save visualization
            os.makedirs(out_dir, exist_ok=True)
            output_path = os.path.join(out_dir, f'{file_name}_occupancy.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Also save the raw occupancy data
            np.save(os.path.join(out_dir, f'{file_name}_occupancy.npy'), occ_grid)
            
            print(f"Saved occupancy visualization: {output_path}")
            print(f"Saved occupancy data: {os.path.join(out_dir, f'{file_name}_occupancy.npy')}")

    def show_results_3d(self, data, result, out_dir, show_open3d=True):
        """3D visualization for occupancy prediction using Open3D.

        Args:
            data (dict): Input points and the information of the sample.
            result (list): Prediction results (occupancy grids).
            out_dir (str): Output directory of visualization result.
            show_open3d (bool): Whether to show Open3D visualization.
        """
        try:
            import open3d as o3d
        except ImportError:
            print("Open3D not installed. Using matplotlib visualization instead.")
            return self.show_results(data, result, out_dir)
        
        import os
        
        for batch_id in range(len(result)):
            # Get occupancy grid result
            occ_grid = result[batch_id]  # Shape: (W, H, D)
            
            # Get point cloud data for reference
            if hasattr(data['points'][0], '_data'):
                points = data['points'][0]._data[0][batch_id].numpy()
            elif hasattr(data['points'][0], 'data'):
                points = data['points'][0].data[0][batch_id].numpy()
            else:
                points = data['points'][0][batch_id].numpy()
            
            # Get metadata
            if hasattr(data['img_metas'][0], '_data'):
                img_meta = data['img_metas'][0]._data[0][batch_id]
            elif hasattr(data['img_metas'][0], 'data'):
                img_meta = data['img_metas'][0].data[0][batch_id]
            else:
                img_meta = data['img_metas'][0][batch_id]
            
            # Create filename
            pts_filename = img_meta.get('pts_filename', f'sample_{batch_id}')
            file_name = os.path.split(pts_filename)[-1].split('.')[0]
            
            # Convert occupancy grid to point cloud
            occ_points, occ_colors = self._occupancy_to_pointcloud(occ_grid)
            
            # Create Open3D visualization
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name=f"Occupancy Visualization - {file_name}")
            
            # Add coordinate frame
            coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
            vis.add_geometry(coordinate_frame)
            
            # Add original point cloud
            if points is not None:
                pcd_original = o3d.geometry.PointCloud()
                pcd_original.points = o3d.utility.Vector3dVector(points[:, :3])
                pcd_original.colors = o3d.utility.Vector3dVector(
                    np.tile([0.5, 0.5, 0.5], (points.shape[0], 1)))  # Gray color
                vis.add_geometry(pcd_original)
            
            # Add occupancy voxels
            if len(occ_points) > 0:
                pcd_occ = o3d.geometry.PointCloud()
                pcd_occ.points = o3d.utility.Vector3dVector(occ_points)
                pcd_occ.colors = o3d.utility.Vector3dVector(occ_colors / 255.0)
                vis.add_geometry(pcd_occ)
            
            # Set render options
            render_option = vis.get_render_option()
            render_option.point_size = 3.0
            render_option.background_color = np.array([0.1, 0.1, 0.1])
            
            if show_open3d:
                vis.run()
            
            # Save screenshot
            os.makedirs(out_dir, exist_ok=True)
            screenshot_path = os.path.join(out_dir, f'{file_name}_3d_occupancy.png')
            vis.capture_screen_image(screenshot_path)
            vis.destroy_window()
            
            # Save occupancy data
            np.save(os.path.join(out_dir, f'{file_name}_occupancy.npy'), occ_grid)
            
            print(f"Saved 3D occupancy visualization: {screenshot_path}")

    def _occupancy_to_pointcloud(self, occ_grid):
        """Convert occupancy grid to colored point cloud.
        
        Args:
            occ_grid (np.ndarray): Occupancy grid with shape (W, H, D)
            
        Returns:
            tuple: (points, colors) where points are 3D coordinates and colors are RGB values
        """
        # Define colors for each class (same as in nuscenes_dataset_occ.py)
        colors_map = np.array([
            [0, 0, 0],           # 0 undefined
            [255, 158, 0],       # 1 car  orange
            [0, 0, 230],         # 2 pedestrian  Blue
            [47, 79, 79],        # 3 sign  Darkslategrey
            [220, 20, 60],       # 4 CYCLIST  Crimson
            [255, 69, 0],        # 5 traffic_light  Orangered
            [255, 140, 0],       # 6 pole  Darkorange
            [233, 150, 70],      # 7 construction_cone  Darksalmon
            [255, 61, 99],       # 8 bicycle  Red
            [112, 128, 144],     # 9 motorcycle  Slategrey
            [222, 184, 135],     # 10 building Burlywood
            [0, 175, 0],         # 11 vegetation  Green
            [165, 42, 42],       # 12 trunk  nuTonomy green
            [0, 207, 191],       # 13 curb, road, lane_marker, other_ground
            [75, 0, 75],         # 14 walkable, sidewalk
            [255, 0, 0],         # 15 unobserved
            [0, 0, 0],           # 16 undefined
            [0, 0, 0],           # 17 undefined
        ])
        
        # Get non-empty voxels (exclude class 0 and 17)
        valid_mask = (occ_grid > 0) & (occ_grid < 17)
        valid_indices = np.where(valid_mask)
        
        if len(valid_indices[0]) == 0:
            return np.array([]), np.array([])
        
        # Convert grid indices to world coordinates
        # Assuming point cloud range and voxel size from config
        voxel_size = np.array(self.voxel_size)  # [0.05, 0.05, 0.05]
        point_cloud_range = np.array(self.point_cloud_range)  # [-40, -40, -1, 40, 40, 5.4]
        
        points = np.stack(valid_indices, axis=1).astype(np.float32)
        points[:, 0] = points[:, 0] * voxel_size[0] + point_cloud_range[0]  # X
        points[:, 1] = points[:, 1] * voxel_size[1] + point_cloud_range[1]  # Y  
        points[:, 2] = points[:, 2] * voxel_size[2] + point_cloud_range[2]  # Z
        
        # Get colors for each valid voxel
        class_ids = occ_grid[valid_mask]
        colors = colors_map[class_ids]
        
        return points, colors
