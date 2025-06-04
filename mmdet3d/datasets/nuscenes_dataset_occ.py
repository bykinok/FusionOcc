# Copyright (c) OpenMMLab. All rights reserved.
import os
import mmcv
import torch
import cv2
import numpy as np
from tqdm import tqdm

from .builder import DATASETS
from .nuscenes_dataset import NuScenesDataset
from .occ_metrics import Metric_mIoU, Metric_FScore

colors_map = np.array(
    [
        [0, 0, 0, 255],          # 0 others - Black
        [255, 165, 0, 255],      # 1 barrier - Orange (traffic safety)
        [0, 191, 255, 255],      # 2 bicycle - Deep Sky Blue
        [255, 20, 147, 255],     # 3 bus - Deep Pink (public transport)
        [255, 0, 0, 255],        # 4 car - Bright Red (main vehicle)
        [255, 140, 0, 255],      # 5 construction_vehicle - Dark Orange
        [138, 43, 226, 255],     # 6 motorcycle - Blue Violet
        [0, 0, 255, 255],        # 7 pedestrian - Bright Blue (safety)
        [255, 255, 0, 255],      # 8 traffic_cone - Bright Yellow (warning)
        [220, 20, 60, 255],      # 9 trailer - Crimson
        [178, 34, 34, 255],      # 10 truck - Fire Brick Red
        [105, 105, 105, 255],    # 11 driveable_surface - Dim Gray (road)
        [169, 169, 169, 255],    # 12 other_flat - Dark Gray
        [192, 192, 192, 255],    # 13 sidewalk - Silver (walkable)
        [160, 82, 45, 255],      # 14 terrain - Saddle Brown (earth)
        [139, 69, 19, 255],      # 15 manmade - Saddle Brown (structures)
        [34, 139, 34, 255],      # 16 vegetation - Forest Green (nature)
        [0, 0, 0, 0],            # 17 free - Transparent (empty space)
    ])


@DATASETS.register_module()
class NuScenesDatasetOccpancy(NuScenesDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # State variables for program control
        self.should_exit_program = False
    
    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        input_dict = super(NuScenesDatasetOccpancy, self).get_data_info(index)
        if "occ_path" in self.data_infos[index]:
            input_dict['occ_gt_path'] = self.data_infos[index]['occ_path']
        else:
            input_dict['occ_gt_path'] = ""
        return input_dict

    def evaluate(self, occ_results, runner=None, show_dir=None, **eval_kwargs):
        self.occ_eval_metrics = Metric_mIoU(
            num_classes=18,
            use_lidar_mask=False,
            use_image_mask=self.use_mask
        )

        print('\nStarting Evaluation...')
        for index, occ_pred in enumerate(tqdm(occ_results)):
            info = self.data_infos[index]

            occ_gt = np.load(os.path.join(info['occ_path'], 'labels.npz'))
            gt_semantics = occ_gt['semantics']
            mask_lidar = occ_gt['mask_lidar'].astype(bool)
            mask_camera = occ_gt['mask_camera'].astype(bool)
            # occ_pred = occ_pred
            self.occ_eval_metrics.add_batch(occ_pred, gt_semantics, mask_lidar, mask_camera)

            if index%100==0 and show_dir is not None:
                mask_camera = mask_camera.astype(np.int32)
                gt_semantics[mask_camera == 0] = 17   # only when use mask
                occ_pred[mask_camera == 0] = 17
            
                # 2D BEV visualization (existing)
                gt_vis = self.vis_occ(gt_semantics)
                pred_vis = self.vis_occ(occ_pred)
                mmcv.imwrite(np.concatenate([gt_vis, pred_vis], axis=1),
                             os.path.join(show_dir, f"{index}.jpg"))
                
                # 3D Open3D visualization (new)
                try:
                    self.vis_occ_3d(info, gt_semantics, occ_pred, show_dir, index)
                    # Check if user wants to exit the program
                    if self.should_exit_program:
                        print("Program terminated by user (ESC pressed)")
                        break
                except Exception as e:
                    print(f"Warning: 3D visualization failed for index {index}: {e}")
                    # Fall back to 2D visualization only

        return self.occ_eval_metrics.count_miou()

    def vis_occ(self, semantics):
        # simple visualization of result in BEV
        semantics_valid = np.logical_not(semantics == 17)
        d = np.arange(16).reshape(1, 1, 16)
        d = np.repeat(d, 200, axis=0)
        d = np.repeat(d, 200, axis=1).astype(np.float32)
        d = d * semantics_valid
        selected = np.argmax(d, axis=2)

        selected_torch = torch.from_numpy(selected)
        semantics_torch = torch.from_numpy(semantics)

        occ_bev_torch = torch.gather(semantics_torch, dim=2,
                                     index=selected_torch.unsqueeze(-1))
        occ_bev = occ_bev_torch.numpy()

        occ_bev = occ_bev.flatten().astype(np.int32)
        occ_bev_vis = colors_map[occ_bev].astype(np.uint8)
        occ_bev_vis = occ_bev_vis.reshape(200, 200, 4)[::-1, ::-1, :3]
        occ_bev_vis = cv2.resize(occ_bev_vis, (400, 400))
        return occ_bev_vis

    def vis_occ_3d(self, info, gt_semantics, pred_semantics, show_dir, index):
        """3D visualization of occupancy using Open3D.
        
        Args:
            info (dict): Data info containing file paths
            gt_semantics (np.ndarray): Ground truth occupancy grid
            pred_semantics (np.ndarray): Predicted occupancy grid  
            show_dir (str): Directory to save visualization
            index (int): Sample index
        """
        try:
            import open3d as o3d
            import matplotlib.pyplot as plt
        except ImportError:
            print("Open3D or matplotlib not installed. Skipping 3D visualization.")
            return
            
        import os
        
        # Load point cloud data
        points = None
        if 'lidar_path' in info:
            try:
                points = np.fromfile(info['lidar_path'], dtype=np.float32).reshape(-1, 5)[:, :3]
            except:
                print(f"Failed to load point cloud from {info['lidar_path']}")
        
        # Load and display camera images
        camera_fig = self._show_camera_images(info, index)
        
        # Convert occupancy grids to point clouds
        gt_points, gt_colors = self._occupancy_to_pointcloud(gt_semantics)
        pred_points, pred_colors = self._occupancy_to_pointcloud(pred_semantics)
        
        # Create visualizers for both GT and Prediction simultaneously
        visualizers = []
        
        for vis_type, (occ_points, occ_colors, title, window_pos) in [
            ("gt", (gt_points, gt_colors, "Ground Truth", (1200, 100))),
            ("pred", (pred_points, pred_colors, "Prediction", (2600, 100)))
        ]:
            if len(occ_points) == 0:
                print(f"No valid points for {title}")
                continue
                
            # Create Open3D visualizer
            vis = o3d.visualization.Visualizer()
            vis.create_window(
                window_name=f"{title} - Sample {index}", 
                width=1000, 
                height=800,
                left=window_pos[0],
                top=window_pos[1]
            )
            
            # Add coordinate frame
            coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
            vis.add_geometry(coordinate_frame)
            
            # Add original point cloud (if available)
            # if points is not None:
            #     pcd_original = o3d.geometry.PointCloud()
            #     pcd_original.points = o3d.utility.Vector3dVector(points)
            #     # Make original points semi-transparent gray
            #     pcd_original.colors = o3d.utility.Vector3dVector(
            #         np.tile([0.3, 0.3, 0.3], (points.shape[0], 1)))
            #     vis.add_geometry(pcd_original)
            
            # Add occupancy voxels
            pcd_occ = o3d.geometry.PointCloud()
            pcd_occ.points = o3d.utility.Vector3dVector(occ_points)
            pcd_occ.colors = o3d.utility.Vector3dVector(occ_colors / 255.0)
            vis.add_geometry(pcd_occ)
            
            # Set render options
            render_option = vis.get_render_option()
            render_option.point_size = 4.0
            render_option.background_color = np.array([0.05, 0.05, 0.05])
            
            # Set camera view
            ctr = vis.get_view_control()
            ctr.set_front([0.0, -1.0, -0.3])
            ctr.set_lookat([0.0, 0.0, 0.0])
            ctr.set_up([0.0, 0.0, 1.0])
            ctr.set_zoom(0.3)
            
            # Store visualizer for simultaneous display
            visualizers.append((vis, title, vis_type))
        
        if not visualizers:
            print("No valid visualizers created")
            # Close camera window if no 3D visualizers
            if camera_fig is not None:
                plt.close(camera_fig)
            return
            
        # Show all windows simultaneously
        print(f"Showing 3D GT and Prediction visualization for sample {index}")
        print("Controls: Close any window to continue to next sample, Press ESC to exit program")
        
        # Poll all visualizers until one is closed
        try:
            while True:
                all_running = True
                
                # Check if ESC was pressed
                if self.should_exit_program:
                    print("ESC detected - Terminating visualization...")
                    break
                
                # Check Open3D visualizers
                for vis, title, vis_type in visualizers:
                    # Update each visualizer
                    if not vis.poll_events():
                        all_running = False
                        break
                    vis.update_renderer()
                
                # Check matplotlib camera window
                if camera_fig is not None:
                    # Check if matplotlib window is still open
                    if not plt.fignum_exists(camera_fig.number):
                        print("Camera image window closed by user")
                        all_running = False
                    else:
                        # Update matplotlib events
                        camera_fig.canvas.flush_events()
                
                # If any window is closed, break the loop
                if not all_running:
                    break
                    
                # Small delay to prevent high CPU usage
                import time
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print("Visualization interrupted by user (Ctrl+C)")
            self.should_exit_program = True
        
        # Clean up all visualizers
        for vis, title, vis_type in visualizers:
            try:
                vis.destroy_window()
            except:
                pass  # Window might already be closed
        
        # Close camera image window if still open
        if camera_fig is not None and plt.fignum_exists(camera_fig.number):
            try:
                plt.close(camera_fig)
                print("Camera image window closed automatically")
            except:
                pass
        
        print(f"Closed all visualization windows for sample {index}")

    def _show_camera_images(self, info, index):
        """Display 6 camera images for the current scene.
        
        Args:
            info (dict): Data info containing image paths
            index (int): Sample index
        """
        try:
            import matplotlib.pyplot as plt
            from PIL import Image
        except ImportError:
            print("matplotlib or PIL not installed. Skipping camera images.")
            return
            
        # Check if camera images are available
        if 'cams' not in info or not info['cams']:
            print("No camera images available for this sample")
            return
            
        # Get camera names and paths from info['cams']
        camera_data = []
        for cam_name, cam_info in info['cams'].items():
            if 'img_path' in cam_info or 'data_path' in cam_info:
                img_path = cam_info.get('img_path') or cam_info.get('data_path')
                camera_data.append((cam_name, img_path))
        
        if not camera_data:
            print("No valid camera paths found in info['cams']")
            return
            
        # Create figure for camera images
        num_cams = len(camera_data)
        if num_cams <= 3:
            rows, cols = 1, num_cams
        elif num_cams <= 6:
            rows, cols = 2, 3
        else:
            rows, cols = 3, 3
            
        fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
        fig.suptitle(f'Camera Images - Sample {index} (Press ESC to exit program)', fontsize=16)
        
        # Add keyboard event handler for ESC key
        def on_key_press(event):
            if event.key == 'escape':
                self.should_exit_program = True
                plt.close(fig)
                print("ESC pressed - Program will terminate after closing all windows")
        
        fig.canvas.mpl_connect('key_press_event', on_key_press)
        
        # Handle single subplot case
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        # Load and display each camera image
        for i, (cam_name, img_path) in enumerate(camera_data):
            if i >= len(axes):
                break
                
            try:
                # Load image
                if os.path.exists(img_path):
                    img = Image.open(img_path)
                    img_array = np.array(img)
                    
                    # Display image
                    axes[i].imshow(img_array)
                    axes[i].set_title(cam_name, fontsize=12)
                    axes[i].axis('off')
                else:
                    # Show placeholder if image not found
                    axes[i].text(0.5, 0.5, f'{cam_name}\nImage not found\n{img_path}', 
                                      ha='center', va='center', transform=axes[i].transAxes)
                    axes[i].set_title(cam_name, fontsize=12)
                    axes[i].axis('off')
                    
            except Exception as e:
                print(f"Failed to load {cam_name}: {e}")
                axes[i].text(0.5, 0.5, f'{cam_name}\nLoad failed\n{str(e)}', 
                                  ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(cam_name, fontsize=12)
                axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(len(camera_data), len(axes)):
            axes[i].axis('off')
        
        # Adjust layout and show
        plt.tight_layout()
        
        # Position the camera window
        mngr = fig.canvas.manager
        if hasattr(mngr, 'window'):
            if hasattr(mngr.window, 'wm_geometry'):
                mngr.window.wm_geometry("+1200+600")  # Position below 3D windows
            elif hasattr(mngr.window, 'move'):
                mngr.window.move(1200, 600)
        
        plt.show(block=False)  # Non-blocking show
        print(f"Camera images displayed for sample {index} ({len(camera_data)} cameras)")
        print("Press ESC to exit program, or close any window to continue")
        
        return fig

    def _occupancy_to_pointcloud(self, occ_grid):
        """Convert occupancy grid to colored point cloud.
        
        Args:
            occ_grid (np.ndarray): Occupancy grid with shape (W, H, D)
            
        Returns:
            tuple: (points, colors) where points are 3D coordinates and colors are RGB values
        """
        # Use the same colors_map as 2D visualization (defined at top of file)
        # Remove alpha channel for Open3D (RGBA -> RGB)
        colors_map_rgb = colors_map[:, :3]  # Remove alpha channel
        
        # Get non-empty voxels (exclude class 0 and 17)
        valid_mask = (occ_grid > 0) & (occ_grid < 17)
        valid_indices = np.where(valid_mask)
        
        if len(valid_indices[0]) == 0:
            return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3)
        
        # Convert grid indices to world coordinates
        # Using typical nuScenes occupancy parameters
        voxel_size = 0.4  # 40cm voxels
        point_cloud_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
        
        points = np.stack(valid_indices, axis=1).astype(np.float32)
        points[:, 0] = points[:, 0] * voxel_size + point_cloud_range[0]  # X
        points[:, 1] = points[:, 1] * voxel_size + point_cloud_range[1]  # Y  
        points[:, 2] = points[:, 2] * voxel_size + point_cloud_range[2]  # Z
        
        # Get colors for each valid voxel using the same colors_map as 2D
        class_ids = occ_grid[valid_mask]
        colors = colors_map_rgb[class_ids]
        
        return points, colors

    def format_results(self, occ_results, submission_prefix, **kwargs):
        if submission_prefix is not None:
            mmcv.mkdir_or_exist(submission_prefix)

        for index, occ_pred in enumerate(tqdm(occ_results)):
            info = self.data_infos[index]
            sample_token = info['token']
            save_path = os.path.join(submission_prefix, '{}.npz'.format(sample_token))
            np.savez_compressed(save_path, occ_pred.astype(np.uint8))
        print('\nFinished.')
