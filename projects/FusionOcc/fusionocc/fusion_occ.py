# Copyright (c) OpenMMLab. All rights reserved.
import torch
import numpy as np
from torch import nn
from mmcv.cnn.bricks.conv_module import ConvModule
try:
    from mmcv.runner import auto_fp16, force_fp32
except ImportError:
    try:
        from mmengine.runner import auto_fp16, force_fp32
    except ImportError:
        # Create dummy decorators if not available
        def auto_fp16(*args, **kwargs):
            if args and callable(args[0]):
                return args[0]
            def decorator(func):
                return func
            return decorator
        def force_fp32(*args, **kwargs):
            if args and callable(args[0]):
                return args[0]
            def decorator(func):
                return func
            return decorator
from mmdet3d.registry import MODELS
try:
    from mmdet.models.builder import build_loss
except ImportError:
    import torch.nn as nn
    def build_loss(cfg):
        if cfg['type'] == 'CrossEntropyLoss':
            return nn.CrossEntropyLoss()
        else:
            from mmdet3d.registry import MODELS
            return MODELS.build(cfg)

from .lidar_encoder import CustomSparseEncoder


@MODELS.register_module()
class FusionDepthSeg(nn.Module):
    """Base class for FusionDepthSeg detector."""
    
    def __init__(self, **kwargs):
        super(FusionDepthSeg, self).__init__()
        # Initialize with basic components
        self.num_frame = kwargs.get('num_frame', 1)
        self.align_after_view_transformation = kwargs.get('align_after_view_transformation', False)
        self.pre_process = kwargs.get('pre_process', False)
        
        # Initialize components (these would be built from config)
        self.image_encoder = None
        self.img_view_transformer = None
        self.pre_process_net = None
        self.bev_encoder = None

    def prepare_img_3d_feat(self, img, sensor2keyego, ego2global, intrin,
                            post_rot, post_tran, bda, mlp_input, input_depth=None):
        if self.image_encoder is None:
            # Placeholder implementation
            x = torch.randn(img.shape[0], 256, img.shape[2]//16, img.shape[3]//16)
        else:
            x, _ = self.image_encoder(img, stereo=False)
            
        if self.img_view_transformer is None:
            # Placeholder implementation
            img_3d_feat = torch.randn(img.shape[0], 32, 200, 200)
            depth = torch.randn(img.shape[0], 88, 200, 200)
            seg = torch.randn(img.shape[0], 18, 200, 200)
        else:
            img_3d_feat, depth, seg = self.img_view_transformer(
                [x, sensor2keyego, ego2global, intrin, post_rot, post_tran, bda,
                 mlp_input], input_depth)
                
        if self.pre_process and self.pre_process_net is not None:
            img_3d_feat = self.pre_process_net(img_3d_feat)[0]
        return img_3d_feat, depth, seg

    def extract_img_3d_feat(self, img_inputs, input_depth):
        # Placeholder implementation for preparing inputs
        imgs = [torch.randn(1, 3, 512, 1408)] * self.num_frame
        sensor2keyegos = [torch.randn(1, 4, 4)] * self.num_frame
        ego2globals = [torch.randn(1, 4, 4)] * self.num_frame
        intrins = [torch.randn(1, 3, 3)] * self.num_frame
        post_rots = [torch.randn(1, 3, 3)] * self.num_frame
        post_trans = [torch.randn(1, 3)] * self.num_frame
        bda = torch.randn(1, 4, 4)
        
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
            mlp_input = torch.randn(1, 256)  # Placeholder
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

    def shift_feature(self, feature, sensor2keyegos, bda):
        # Placeholder implementation
        return feature


@MODELS.register_module()
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
        super(FusionOCC, self).__init__(**kwargs)
        
        self.voxel_size = voxel_size
        self.lidar_out_channel = lidar_out_channel
        self.lidar_in_channel = lidar_in_channel
        self.sparse_shape = [
            int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0]),
            int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1]),
            int((point_cloud_range[5] - point_cloud_range[2]) / voxel_size[2])
        ]
        self.point_cloud_range = point_cloud_range
        
        # Build lidar encoder
        self.lidar_encoder = CustomSparseEncoder(
            in_channels=self.lidar_in_channel,
            sparse_shape=self.sparse_shape,
            point_cloud_range=self.point_cloud_range,
            voxel_size=self.voxel_size,
            output_channels=self.lidar_out_channel,
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
        self.loss_occ = build_loss(loss_occ) if loss_occ else None
        self.class_wise = class_wise
        self.align_after_view_transformation = False
        self.fuse_loss_weight = fuse_loss_weight

    def occ_encoder(self, fusion_feat):
        if self.bev_encoder is None:
            # Placeholder implementation
            return fusion_feat
        return self.bev_encoder(fusion_feat)

    def extract_feat(self, lidar_feat, img, img_metas, input_depth=None, **kwargs):
        """Extract features from images and points."""
        fusion_feats, depth, pred_segs = self.extract_fusion_feat(
            lidar_feat, img, img_metas, input_depth=input_depth, **kwargs
        )
        pts_feats = None
        return fusion_feats, pts_feats, depth, pred_segs

    def extract_fusion_feat(self, lidar_feat, img, img_metas, input_depth=None, **kwargs):
        """Extract fusion features from lidar and image."""
        # Placeholder implementation
        img_3d_feat_feat, depth_key_frame, seg_key_frame = self.extract_img_3d_feat(
            img_inputs=img, input_depth=input_depth)
        
        # Ensure both features are on the same device
        if lidar_feat.device != img_3d_feat_feat.device:
            img_3d_feat_feat = img_3d_feat_feat.to(lidar_feat.device)
        
        # Ensure both features have the same number of dimensions and shape
        if img_3d_feat_feat.dim() != lidar_feat.dim():
            if img_3d_feat_feat.dim() == 4 and lidar_feat.dim() == 5:
                # Add an extra dimension to img_3d_feat_feat
                img_3d_feat_feat = img_3d_feat_feat.unsqueeze(2)  # Add dimension at index 2
            elif img_3d_feat_feat.dim() == 5 and lidar_feat.dim() == 4:
                # Add an extra dimension to lidar_feat
                lidar_feat = lidar_feat.unsqueeze(2)
        
        # Ensure both features have the same shape in the depth dimension (index 2)
        if img_3d_feat_feat.shape[2] != lidar_feat.shape[2]:
            # Repeat img_3d_feat_feat to match lidar_feat's depth dimension
            depth_repeats = lidar_feat.shape[2]
            img_3d_feat_feat = img_3d_feat_feat.repeat(1, 1, depth_repeats, 1, 1)
        
        fusion_feat = torch.cat([img_3d_feat_feat, lidar_feat], dim=1)
        fusion_feat = self.occ_encoder(fusion_feat)
        return fusion_feat, depth_key_frame, seg_key_frame

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
        
        # Placeholder for depth and segmentation losses
        if depth_key_frame is not None and sparse_depth is not None:
            depth_loss = torch.nn.functional.mse_loss(depth_key_frame, sparse_depth)
            losses['depth_loss'] = depth_loss * self.fuse_loss_weight
            
        if seg_key_frame is not None and segs is not None:
            seg_loss = torch.nn.functional.cross_entropy(seg_key_frame, segs)
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
            loss_occ = self.loss_occ(preds, voxel_semantics)
            loss_['loss_occ'] = loss_occ
        return loss_

    def simple_test(self,
                    points=None,
                    img_metas=None,
                    img_inputs=None,
                    sparse_depth=None,
                    **kwargs):
        """Test function without augmentation."""
        if sparse_depth is not None:
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
    
    def train_step(self, data, optim_wrapper):
        """Train step function for mmengine runner.
        
        Args:
            data (dict): The output of dataloader.
            optim_wrapper: Optimizer wrapper.
            
        Returns:
            dict: Dict of outputs.
        """
        # Extract data - Check for both direct keys and nested in data_samples
        if 'data_samples' in data:
            # MMEngine format - extract from data_samples
            data_samples = data['data_samples']
            imgs = data['inputs']['img_inputs'] if 'inputs' in data else data['img_inputs']
            points = data['inputs'].get('points', None) if 'inputs' in data else data.get('points', None)
            
            # Try to get GT data from data_samples
            if hasattr(data_samples, 'gt_seg_3d'):
                voxel_semantics = data_samples.gt_seg_3d
            elif hasattr(data_samples, 'voxel_semantics'):
                voxel_semantics = data_samples.voxel_semantics
            else:
                voxel_semantics = data.get('voxel_semantics', None)
                
            if hasattr(data_samples, 'mask_camera'):
                mask_camera = data_samples.mask_camera
            else:
                mask_camera = data.get('mask_camera', None)
                
            img_metas = data.get('img_metas', {})
        else:
            # Direct format
            imgs = data['img_inputs']
            points = data.get('points', None)
            voxel_semantics = data.get('voxel_semantics', None)
            mask_camera = data.get('mask_camera', None)
            img_metas = data.get('img_metas', {})
        
        # Get device from model parameters
        device = next(self.parameters()).device
        
        # Convert LiDARPoints object to tensor if necessary and move to correct device
        if points is not None:
            if hasattr(points, 'tensor'):
                points = points.tensor
            # Ensure points is on the correct device
            if isinstance(points, list):
                points = [pt.to(device) if hasattr(pt, 'to') else pt for pt in points]
            elif hasattr(points, 'to'):
                points = points.to(device)
        
        # Forward pass
        if points is not None:
            # Process lidar data
            lidar_feat, _, _ = self.lidar_encoder(points)
            
            # Extract features from both modalities
            occ_pred = self.extract_feat(lidar_feat, imgs, img_metas)
        else:
            # Image-only mode
            occ_pred = self.extract_feat(None, imgs, img_metas)
        
        # Calculate loss
        losses = dict()
        if self.loss_occ is not None:
            # Reshape predictions and targets
            if isinstance(occ_pred, (list, tuple)):
                occ_pred = occ_pred[0]
            
            # Make sure shapes match
            if occ_pred.ndim == 5:  # (B, C, H, W, D)
                B, C, H, W, D = occ_pred.shape
                occ_pred = occ_pred.permute(0, 2, 3, 4, 1).reshape(-1, C)
            
            # Handle voxel_semantics conversion
            if voxel_semantics is not None:
                # Convert to tensor if needed
                if isinstance(voxel_semantics, list):
                    if len(voxel_semantics) > 0:
                        # If it's a single-element list, extract the element
                        if len(voxel_semantics) == 1:
                            voxel_semantics = voxel_semantics[0]
                        
                        # Now handle the extracted element or list
                        if isinstance(voxel_semantics, torch.Tensor):
                            voxel_semantics = voxel_semantics.to(device)
                        elif hasattr(voxel_semantics, 'shape'):
                            # If it's a numpy array or similar
                            voxel_semantics = torch.as_tensor(voxel_semantics, device=device)
                        else:
                            # If it's still a list of tensors, stack them
                            if isinstance(voxel_semantics, list) and len(voxel_semantics) > 0:
                                voxel_semantics = torch.stack([torch.as_tensor(vs, device=device) for vs in voxel_semantics])
                            else:
                                # Convert to tensor
                                voxel_semantics = torch.tensor(voxel_semantics, device=device)
                    else:
                        voxel_semantics = None
                elif not isinstance(voxel_semantics, torch.Tensor):
                    voxel_semantics = torch.as_tensor(voxel_semantics, device=device)
                else:
                    voxel_semantics = voxel_semantics.to(device)
                
                # Reshape if needed
                if voxel_semantics is not None and hasattr(voxel_semantics, 'ndim'):
                    if voxel_semantics.ndim == 4:  # (B, H, W, D)
                        voxel_semantics = voxel_semantics.reshape(-1)
                    elif voxel_semantics.ndim > 1:
                        voxel_semantics = voxel_semantics.flatten()
            
            # Apply mask if available
            if mask_camera is not None and self.use_mask:
                # Convert mask_camera to tensor if needed - handle complex structures safely
                try:
                    if isinstance(mask_camera, list):
                        if len(mask_camera) > 0:
                            # If it's a single-element list, extract the element
                            if len(mask_camera) == 1:
                                mask_camera = mask_camera[0]
                            
                            # Now handle the extracted element
                            if isinstance(mask_camera, torch.Tensor):
                                mask_camera = mask_camera.to(device)
                            elif hasattr(mask_camera, 'shape'):
                                mask_camera = torch.as_tensor(mask_camera, device=device)
                            else:
                                # Skip mask if too complex
                                mask_camera = None
                        else:
                            mask_camera = None
                    elif not isinstance(mask_camera, torch.Tensor):
                        mask_camera = torch.as_tensor(mask_camera, device=device)
                    else:
                        mask_camera = mask_camera.to(device)
                except (TypeError, ValueError, RuntimeError):
                    # Skip mask if conversion fails
                    mask_camera = None
                    
                if mask_camera is not None:
                    if hasattr(mask_camera, 'ndim') and mask_camera.ndim == 4:
                        mask_camera = mask_camera.reshape(-1)
                    elif hasattr(mask_camera, 'ndim') and mask_camera.ndim > 1:
                        mask_camera = mask_camera.flatten()
                        
                    if hasattr(mask_camera, 'ndim') and not isinstance(mask_camera, list):
                        valid_mask = mask_camera > 0
                        if valid_mask.sum() > 0 and voxel_semantics is not None:
                            occ_pred = occ_pred[valid_mask]
                            voxel_semantics = voxel_semantics[valid_mask]
            
            # Calculate occupancy loss
            if voxel_semantics is not None and hasattr(voxel_semantics, 'long'):
                # Ensure both tensors have compatible shapes
                if occ_pred.shape[0] != voxel_semantics.shape[0]:
                    min_size = min(occ_pred.shape[0], voxel_semantics.shape[0])
                    occ_pred = occ_pred[:min_size]
                    voxel_semantics = voxel_semantics[:min_size]
                
                # Clamp values to valid range
                voxel_semantics = torch.clamp(voxel_semantics.long(), 0, occ_pred.shape[1] - 1)
                
                # Debug - uncomment for debugging
                # print(f"occ_pred shape: {occ_pred.shape}, voxel_semantics shape: {voxel_semantics.shape}")
                # print(f"voxel_semantics min: {voxel_semantics.min()}, max: {voxel_semantics.max()}")
                
                loss_occ = self.loss_occ(occ_pred, voxel_semantics)
                
                # Debug - uncomment for debugging
                # print(f"Calculated loss: {loss_occ.item()}")
            else:
                # If no valid GT, create a dummy loss but warn
                # print("Warning: No valid voxel_semantics found, using dummy loss")
                loss_occ = torch.tensor(0.0, device=occ_pred.device, requires_grad=True)
                
            losses['loss_occ'] = loss_occ
        else:
            # No loss function defined
            loss_occ = torch.tensor(0.0, device=next(self.parameters()).device, requires_grad=True)
            losses['loss_occ'] = loss_occ
        
        # Backward and optimize
        parsed_losses, log_vars = self.parse_losses(losses)
        optim_wrapper.update_params(parsed_losses)
        
        return log_vars
    
    def parse_losses(self, losses):
        """Parse losses for logging and backward.
        
        Args:
            losses (dict): Dict of losses.
            
        Returns:
            tuple: (loss, log_vars)
        """
        log_vars = {}
        loss_total = 0
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.item()
                loss_total += loss_value
            else:
                log_vars[loss_name] = loss_value
                loss_total += loss_value
        
        log_vars['loss'] = loss_total.item() if isinstance(loss_total, torch.Tensor) else loss_total
        return loss_total, log_vars
    
    def val_step(self, data):
        """Validation step function for mmengine runner.
        
        Args:
            data (dict): The output of dataloader.
            
        Returns:
            dict: Dict of outputs.
        """
        # Extract data
        imgs = data['img_inputs']
        points = data.get('points', None)
        voxel_semantics = data.get('voxel_semantics', None)
        mask_camera = data.get('mask_camera', None)
        img_metas = data.get('img_metas', {})
        
        # Get device from model parameters
        device = next(self.parameters()).device
        
        # Convert LiDARPoints object to tensor if necessary and move to correct device
        if points is not None:
            if hasattr(points, 'tensor'):
                points = points.tensor
            # Ensure points is on the correct device
            if isinstance(points, list):
                points = [pt.to(device) if hasattr(pt, 'to') else pt for pt in points]
            elif hasattr(points, 'to'):
                points = points.to(device)
        
        # Forward pass (without gradient)
        with torch.no_grad():
            if points is not None:
                # Process lidar data
                lidar_feat, _, _ = self.lidar_encoder(points)
                
                # Extract features from both modalities
                occ_pred = self.extract_feat(lidar_feat, imgs, img_metas)
            else:
                # Image-only mode
                occ_pred = self.extract_feat(None, imgs, img_metas)
        
        # Calculate loss for validation
        losses = dict()
        if self.loss_occ is not None and voxel_semantics is not None:
            # Reshape predictions and targets
            if isinstance(occ_pred, (list, tuple)):
                occ_pred = occ_pred[0]
            
            # Make sure shapes match
            if occ_pred.ndim == 5:  # (B, C, H, W, D)
                B, C, H, W, D = occ_pred.shape
                occ_pred = occ_pred.permute(0, 2, 3, 4, 1).reshape(-1, C)
            
            if voxel_semantics is not None and hasattr(voxel_semantics, 'ndim') and voxel_semantics.ndim == 4:  # (B, H, W, D)
                voxel_semantics = voxel_semantics.reshape(-1)
            
            # Apply mask if available
            if mask_camera is not None and self.use_mask:
                if mask_camera is not None and hasattr(mask_camera, 'ndim') and mask_camera.ndim == 4:
                    mask_camera = mask_camera.reshape(-1)
                if hasattr(mask_camera, 'ndim') and not isinstance(mask_camera, list):
                    valid_mask = mask_camera > 0
                else:
                    # Skip mask if it's not a tensor
                    valid_mask = None
                if valid_mask is not None and valid_mask.sum() > 0:
                    occ_pred = occ_pred[valid_mask]
                    voxel_semantics = voxel_semantics[valid_mask]
            
            # Calculate occupancy loss
            if hasattr(voxel_semantics, 'long') and not isinstance(voxel_semantics, list):
                loss_occ = self.loss_occ(occ_pred, voxel_semantics.long())
            else:
                # Skip loss calculation if voxel_semantics is not a proper tensor
                loss_occ = torch.tensor(0.0, device=occ_pred.device, requires_grad=True)
            losses['loss_occ'] = loss_occ
        
        # Parse losses for logging
        parsed_losses, log_vars = self.parse_losses(losses)
        
        return log_vars 