import os

import numpy as np
import cv2 as cv
import argparse
from visualizer import OccupancyVisualizer
import open3d as o3d
import colorsys
import mmengine
from nuscenes.nuscenes import NuScenes

occ3d_colors_map = np.array(
    [
        [0, 0, 0],          # others               black
        [255, 120, 50],     # barrier              orange
        [255, 192, 203],    # bicycle              pink         √
        [255, 255, 0],      # bus                  yellow       √
        [0, 150, 245],      # car                  blue         √
        [0, 255, 255],      # construction_vehicle cyan         √
        [255, 127, 0],      # motorcycle           dark orange  √
        [255, 0, 0],        # pedestrian           red          √
        [255, 240, 150],    # traffic_cone         light yellow
        [135, 60, 0],       # trailer              brown        √
        [160, 32, 240],     # truck                purple       √
        [255, 0, 255],      # driveable_surface    dark pink
        [139, 137, 137],    # other_flat           dark red
        [75, 0, 75],        # sidewalk             dard purple
        [150, 240, 80],     # terrain              light green
        [230, 230, 250],    # manmade              white
        [0, 175, 0],        # vegetation           green
        [255, 255, 255],    # Free                 White
    ]
)
# change to BGR
occ3d_colors_map = occ3d_colors_map[:, ::-1]

openocc_colors_map = np.array(
    [
        [0, 150, 245],      # car                  blue         √
        [160, 32, 240],     # truck                purple       √
        [135, 60, 0],       # trailer              brown        √
        [255, 255, 0],      # bus                  yellow       √
        [0, 255, 255],      # construction_vehicle cyan         √
        [255, 192, 203],    # bicycle              pink         √
        [255, 127, 0],      # motorcycle           dark orange  √
        [255, 0, 0],        # pedestrian           red          √
        [255, 240, 150],    # traffic_cone         light yellow
        [255, 120, 50],     # barrier              orange
        [255, 0, 255],      # driveable_surface    dark pink
        [139, 137, 137],    # other_flat           dark red
        [75, 0, 75],        # sidewalk             dard purple
        [150, 240, 80],     # terrain              light green
        [230, 230, 250],    # manmade              white
        [0, 175, 0],        # vegetation           green
        [255, 255, 255],    # Free                 White
    ]
)

# nuScenes-Occupancy (CONet) color map
# index 0: free/unoccupied (white, ignored in vis), 1-16: semantic classes
# class ordering: 0=free, 1=barrier, 2=bicycle, 3=bus, 4=car, 5=construction_vehicle,
#                 6=motorcycle, 7=pedestrian, 8=traffic_cone, 9=trailer, 10=truck,
#                 11=driveable_surface, 12=other_flat, 13=sidewalk, 14=terrain,
#                 15=manmade, 16=vegetation
_npy_semantic_colors = np.array(
    [
        [255, 255, 255],    # 0  free              white   (ignored)
        [255, 120, 50],     # 1  barrier            orange
        [255, 192, 203],    # 2  bicycle            pink
        [255, 255, 0],      # 3  bus                yellow
        [0, 150, 245],      # 4  car                blue
        [0, 255, 255],      # 5  construction_veh   cyan
        [255, 127, 0],      # 6  motorcycle         dark orange
        [255, 0, 0],        # 7  pedestrian         red
        [255, 240, 150],    # 8  traffic_cone       light yellow
        [135, 60, 0],       # 9  trailer            brown
        [160, 32, 240],     # 10 truck              purple
        [255, 0, 255],      # 11 driveable_surface  dark pink
        [139, 137, 137],    # 12 other_flat         gray
        [75, 0, 75],        # 13 sidewalk           dark purple
        [150, 240, 80],     # 14 terrain            light green
        [230, 230, 250],    # 15 manmade            light lavender
        [0, 175, 0],        # 16 vegetation         green
    ]
)
# CONet nuScenes-Occupancy: col order [z, y, x, cls], grid 512x512x40, range [-51.2,51.2]
nusc_occ_colors_map = _npy_semantic_colors[:, ::-1].copy()

# SurroundOcc color map (same class ordering as nusc_occ)
# col order [x, y, z, cls], grid 200x200x16, range [-50,50,-50,50,-5,3]
surroundocc_colors_map = _npy_semantic_colors[:, ::-1].copy()

foreground_idx = [0, 1, 2, 3, 4, 5, 6, 7]


def parse_args():
    parse = argparse.ArgumentParser('')
    parse.add_argument('--pkl-file', type=str, default='data/nuscenes/stcocc-nuscenes_infos_val.pkl', help='path of pkl for the nuScenes dataset')
    parse.add_argument('--data-path', type=str, default='data/nuscenes', help='path of the nuScenes dataset')
    parse.add_argument('--data-version', type=str, default='v1.0-trainval', help='version of the nuScenes dataset')
    parse.add_argument('--dataset-type', type=str, default='occ3d', help='dataset type: occ3d, openocc, or nusc_occ')
    parse.add_argument('--pred-path', type=str, default='results/scene-0003', help='path of the prediction data, dir')
    parse.add_argument('--vis-scene', type=list, default=['scene-0003'], help='visualize scene list')
    parse.add_argument('--vis-path', type=str, default='demo_out', help='path of saving the visualization images')
    parse.add_argument('--car-model', type=str, default='visualizer/3d_model.obj', help='car_model path')
    parse.add_argument('--vis-single-data', type=str, default='results/scene-0107/84c24cd1d7914f72bb2fa17a6c5c41e5.npz',help='single path of the visualization data (.npz for occ3d/openocc, .npy for nusc_occ)')
    parse.add_argument('--gt-path', type=str, default=None, help='path to GT data for loading camera_mask (e.g., data/nuscenes/gts/scene-0003/token/labels.npz)')
    parse.add_argument('--use-camera-mask', action='store_true', help='only visualize voxels visible from camera (camera_mask=True)')
    parse.add_argument('--save-path', type=str, default=None, help='output image path for single data visualization (extension .png is appended automatically, e.g. demo_out/result)')
    parse.add_argument('--vis-diff', action='store_true', help='compare pred(--vis-single-data) vs GT(--gt-path) and highlight errors in density-based red')
    # nuScenes-Occupancy (CONet) specific args
    parse.add_argument('--nusc-occ-path', type=str, default=None,
                       help='[nusc_occ] path to nuScenes-Occupancy .npy file '
                            '(e.g., data/nuScenes-Occupancy/scene_<token>/occupancy/<lidar_token>.npy)')
    parse.add_argument('--nusc-occ-pc-range', type=float, nargs=6,
                       default=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                       help='[nusc_occ] point cloud range: xmin ymin zmin xmax ymax zmax')
    parse.add_argument('--nusc-occ-grid-size', type=int, nargs=3, default=[512, 512, 40],
                       help='[nusc_occ] voxel grid size: x y z')
    # SurroundOcc specific args
    parse.add_argument('--surroundocc-path', type=str, default=None,
                       help='[surroundocc] path to SurroundOcc .npy GT file '
                            '(e.g., data/nuscenes_occ/samples/<lidar_filename>.npy)')
    parse.add_argument('--surroundocc-pc-range', type=float, nargs=6,
                       default=[-50.0, -50.0, -5.0, 50.0, 50.0, 3.0],
                       help='[surroundocc] point cloud range: xmin ymin zmin xmax ymax zmax')
    parse.add_argument('--surroundocc-grid-size', type=int, nargs=3, default=[200, 200, 16],
                       help='[surroundocc] voxel grid size: x y z')
    args = parse.parse_args()
    return args

def arange_according_to_scene(infos, nusc, vis_scene):
    scenes = dict()

    for i, info in enumerate(infos):
        scene_token = nusc.get('sample', info['token'])['scene_token']
        scene_meta = nusc.get('scene', scene_token)
        scene_name = scene_meta['name']
        if not scene_name in scenes:
            scenes[scene_name] = [info]
        else:
            scenes[scene_name].append(info)

    vis_scenes = dict()
    if len(vis_scene) == 0:
        vis_scenes = scenes
    else:
        for scene_name in vis_scene:
            vis_scenes[scene_name] = scenes[scene_name]

    return vis_scenes

def vis_occ_scene_on_3d(vis_scenes_infos,
                        vis_scene,
                        vis_path,
                        pred_path,
                        dataset_type='occ3d',
                        load_camera_mask=False,
                        voxel_size=(0.4, 0.4, 0.4),
                        vis_gt=True,
                        vis_flow=False,
                        car_model=None,
                        background_color=(255, 255, 255),
                        ):
    # define free_cls
    free_cls = 16 if dataset_type == 'openocc' else 17

    # load car model
    if car_model is not None:
        car_model_mesh = o3d.io.read_triangle_mesh(car_model)
        angle = np.pi / 2  # 90 度
        R = car_model_mesh.get_rotation_matrix_from_axis_angle(np.array([angle, 0, 0]))
        car_model_mesh.rotate(R, center=car_model_mesh.get_center())
        car_model_mesh.scale(0.25, center=car_model_mesh.get_center())
        current_center = car_model_mesh.get_center()
        new_center = np.array([0, 0, 0.5])
        translation = new_center - current_center
        car_model_mesh.translate(translation)
        car_model_mesh.compute_vertex_normals()
    else:
        car_model_mesh = None

    # check vis path
    mmengine.mkdir_or_exist(vis_path)
    for scene_name in vis_scene:
        scene_infos = vis_scenes_infos[scene_name]
        vis_occ_semantics = []
        buffer_vis_path = '{}/{}'.format(vis_path, scene_name)
        # check vis path
        mmengine.mkdir_or_exist(buffer_vis_path)

        for index, info in enumerate(scene_infos):

            save_path = os.path.join(buffer_vis_path, str(index))
            # visualize the scene data
            if vis_gt:
                occ_path = info['occ_path']
                if dataset_type == 'openocc':
                    occ_path = occ_path.replace('gts', 'openocc_v2')
                occ_label_path = os.path.join(occ_path, 'labels.npz')
                print(occ_label_path)
                occ_label = np.load(occ_label_path)
                occ_semantics = occ_label['semantics']

                if load_camera_mask:
                    assert 'mask_camera' in occ_label.keys()
                    mask_camera = occ_label['mask_camera']
                    occ_semantics[mask_camera == 0] = 255
                if vis_flow:
                    occ_flow = occ_label['flow']
                else:
                    occ_flow = None

            else:
                token = info['token']
                occ_label_path = os.path.join(pred_path, token + '.npz')
                occ_label = np.load(occ_label_path)
                occ_semantics = occ_label['semantics']
                
                # Apply camera mask from GT if requested
                if load_camera_mask:
                    # Try to load mask_camera from prediction file first
                    if 'mask_camera' in occ_label.keys():
                        mask_camera = occ_label['mask_camera']
                        occ_semantics[mask_camera == 0] = 255
                        print(f"Applied camera mask from prediction: {occ_label_path}")
                    # If not found, load from GT
                    elif 'occ_path' in info:
                        occ_path = info['occ_path']
                        if dataset_type == 'openocc':
                            occ_path = occ_path.replace('gts', 'openocc_v2')
                        gt_label_path = os.path.join(occ_path, 'labels.npz')
                        
                        if os.path.exists(gt_label_path):
                            try:
                                gt_label = np.load(gt_label_path)
                                if 'mask_camera' in gt_label.keys():
                                    mask_camera = gt_label['mask_camera']
                                    occ_semantics[mask_camera == 0] = 255
                                    print(f"Applied camera mask from GT: {gt_label_path}")
                                else:
                                    print(f"Warning: 'mask_camera' not found in GT: {gt_label_path}")
                            except Exception as e:
                                print(f"Error loading GT for camera mask: {e}")
                        else:
                            print(f"Warning: GT file not found: {gt_label_path}")
                    else:
                        print(f"Warning: load_camera_mask=True but 'occ_path' not found in info for token {token}")
                
                if vis_flow:
                    # check if flow exists
                    if 'flow' in occ_label.keys():
                        occ_flow = occ_label['flow']
                    if 'flows' in occ_label.keys():
                        occ_flow = occ_label['flows']
                else:
                    occ_flow = None

            # if view json exits
            occ_visualizer = OccupancyVisualizer(color_map=occ3d_colors_map if dataset_type == 'occ3d' else openocc_colors_map,
                                                 background_color=background_color)
            if os.path.exists('view.json'):
                param = o3d.io.read_pinhole_camera_parameters('view.json')
            else:
                param = None

            occ_visualizer.vis_occ(
                occ_semantics,
                occ_flow=occ_flow,
                ignore_labels=[free_cls, 255],
                voxelSize=voxel_size,
                range=[-40.0, -40.0, -1.0, 40.0, 40.0, 5.4],
        save_path=save_path,
        wait_time=-1,  # -1 means wait until press q
                view_json=param,
                car_model_mesh=car_model_mesh,
            )

            # press top-right x to close the windows
            param = occ_visualizer.o3d_vis.get_view_control().convert_to_pinhole_camera_parameters()
            o3d.io.write_pinhole_camera_parameters('view.json', param)

            occ_visualizer.o3d_vis.destroy_window()

        # write video
        for i in range(index):
            img_path = os.path.join(buffer_vis_path, str(i) + '.png')
            img = cv.imread(img_path)
            vis_occ_semantics.append(img)
            os.remove(img_path)

        # save video
        fourcc = cv.VideoWriter_fourcc(*'XVID')
        if vis_gt:
            if vis_flow:
                video_path = vis_path + '/' + 'gt-flow_' + scene_name + '.avi'
            else:
                video_path = vis_path + '/' + 'gt-occ_' + scene_name + '.avi'
        else:
            if vis_flow:
                video_path = vis_path + '/' + 'pred-flow_' + scene_name + '.avi'
            else:
                video_path = vis_path + '/' + 'pred-occ_' + scene_name + '.avi'

        video = cv.VideoWriter(video_path, fourcc, 5, (img.shape[1], img.shape[0]))
        for img in vis_occ_semantics:
            video.write(img)
        video.release()
        print('Save video to {}'.format(video_path))

def vis_occ_single_on_3d(data_path,
                        dataset_type='occ3d',
                        voxel_size=(0.4, 0.4, 0.4),
                        car_model=None,
                        vis_flow=False,
                        use_camera_mask=False,
                        gt_path=None,
                        background_color=(255, 255, 255),
                        save_path=None,
                        ):
    # define free_cls
    free_cls = 16 if dataset_type == 'openocc' else 17

    # load car model
    if car_model is not None:
        car_model_mesh = o3d.io.read_triangle_mesh(car_model)
        angle = np.pi / 2  # 90 度
        R = car_model_mesh.get_rotation_matrix_from_axis_angle(np.array([angle, 0, 0]))
        car_model_mesh.rotate(R, center=car_model_mesh.get_center())
        car_model_mesh.scale(0.25, center=car_model_mesh.get_center())
        current_center = car_model_mesh.get_center()
        new_center = np.array([0, 0, 0.5])
        translation = new_center - current_center
        car_model_mesh.translate(translation)
        car_model_mesh.compute_vertex_normals()
    else:
        car_model_mesh = None

    # visualize the scene data
    occ_label = np.load(data_path)
    # occ_semantics = occ_label['semantics']
    occ_semantics = occ_label['semantics']
    
    # Apply camera mask if requested
    if use_camera_mask:
        mask_camera = None
        
        # Try to load mask_camera from the data file first
        if 'mask_camera' in occ_label.keys():
            mask_camera = occ_label['mask_camera']
            print(f"Loaded camera mask from data file: {data_path}")
        # If not found and gt_path is provided, load from GT
        elif gt_path is not None and os.path.exists(gt_path):
            try:
                gt_label = np.load(gt_path)
                if 'mask_camera' in gt_label.keys():
                    mask_camera = gt_label['mask_camera']
                    print(f"Loaded camera mask from GT file: {gt_path}")
                else:
                    print(f"Warning: 'mask_camera' not found in GT file: {gt_path}")
            except Exception as e:
                print(f"Error loading GT file {gt_path}: {e}")
        else:
            print(f"Warning: use_camera_mask=True but 'mask_camera' not found in {data_path}")
            if gt_path is None:
                print("Hint: Use --gt-path to specify GT file for loading camera_mask")
            elif not os.path.exists(gt_path):
                print(f"Hint: GT path does not exist: {gt_path}")
        
        # Apply mask if loaded successfully
        if mask_camera is not None:
            print(f"Applying camera mask: {mask_camera.sum()} / {mask_camera.size} voxels visible ({mask_camera.sum()/mask_camera.size*100:.2f}%)")
            # Set invisible voxels to 255 (will be filtered by ignore_labels)
            occ_semantics = np.where(mask_camera, occ_semantics, 255)
    
    if vis_flow:
        # check if flow exists
        if 'flow' in occ_label.keys():
            occ_flow = occ_label['flow']
        if 'flows' in occ_label.keys():
            occ_flow = occ_label['flows']
    else:
        occ_flow = None

    # if view json exits
    occ_visualizer = OccupancyVisualizer(color_map=occ3d_colors_map if dataset_type=='occ3d' else openocc_colors_map,
                                         background_color=background_color)

    if os.path.exists('view.json'):
        param = o3d.io.read_pinhole_camera_parameters('view.json')
    else:
        param = None

    if save_path is not None:
        mmengine.mkdir_or_exist(os.path.dirname(os.path.abspath(save_path)))

    occ_visualizer.vis_occ(
        occ_semantics,
        occ_flow=occ_flow,
        ignore_labels=[free_cls, 255],
        voxelSize=voxel_size,
        range=[-40.0, -40.0, -1.0, 40.0, 40.0, 5.4],
        save_path=save_path,
        wait_time=-1,  # -1 means wait until press q
        view_json=param,
        car_model_mesh=car_model_mesh,
    )
    # press top-right x to close the windows
    param = occ_visualizer.o3d_vis.get_view_control().convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters('view.json', param)

    occ_visualizer.o3d_vis.destroy_window()


def vis_occ_npy_on_3d(data_path,
                      pc_range=(-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
                      grid_size=(512, 512, 40),
                      xyz_order='zyx',
                      color_map=None,
                      car_model=None,
                      background_color=(255, 255, 255),
                      ):
    """
    sparse .npy GT 시각화 함수 (CONet nuScenes-Occupancy / SurroundOcc 공용).

    .npy 파일 포맷: (N, 4) sparse 배열
      - xyz_order='zyx' (nusc_occ/CONet) : 각 행 = [z_idx, y_idx, x_idx, cls]
      - xyz_order='xyz' (surroundocc)    : 각 행 = [x_idx, y_idx, z_idx, cls]

    label 규칙:
      - label 0  → noise (255로 변환, ignore)
      - label 1-16 → semantic classes
      - sparse 미포함 voxel → 0 (free, ignore)

    클래스 정의 (1-16):
      1:barrier, 2:bicycle, 3:bus, 4:car, 5:construction_vehicle,
      6:motorcycle, 7:pedestrian, 8:traffic_cone, 9:trailer, 10:truck,
      11:driveable_surface, 12:other_flat, 13:sidewalk, 14:terrain,
      15:manmade, 16:vegetation
    """
    pc_range = list(pc_range)
    grid_size = list(grid_size)

    if color_map is None:
        color_map = nusc_occ_colors_map

    # load car model
    if car_model is not None:
        car_model_mesh = o3d.io.read_triangle_mesh(car_model)
        angle = np.pi / 2
        R = car_model_mesh.get_rotation_matrix_from_axis_angle(np.array([angle, 0, 0]))
        car_model_mesh.rotate(R, center=car_model_mesh.get_center())
        car_model_mesh.scale(0.25, center=car_model_mesh.get_center())
        current_center = car_model_mesh.get_center()
        new_center = np.array([0, 0, 0.5])
        translation = new_center - current_center
        car_model_mesh.translate(translation)
        car_model_mesh.compute_vertex_normals()
    else:
        car_model_mesh = None

    # load sparse .npy  →  dense voxel grid [x, y, z]
    pcd = np.load(data_path)  # (N, 4)
    if xyz_order == 'zyx':
        # CONet/nusc_occ: col order [z, y, x, cls]
        x_idx = pcd[:, 2].astype(np.int32)
        y_idx = pcd[:, 1].astype(np.int32)
        z_idx = pcd[:, 0].astype(np.int32)
    else:
        # SurroundOcc: col order [x, y, z, cls]
        x_idx = pcd[:, 0].astype(np.int32)
        y_idx = pcd[:, 1].astype(np.int32)
        z_idx = pcd[:, 2].astype(np.int32)
    cls = pcd[:, 3].astype(np.int32)

    # noise label (0 in file) → 255 (ignore)
    cls[cls == 0] = 255

    # clip to valid range
    valid = (
        (x_idx >= 0) & (x_idx < grid_size[0]) &
        (y_idx >= 0) & (y_idx < grid_size[1]) &
        (z_idx >= 0) & (z_idx < grid_size[2])
    )
    x_idx, y_idx, z_idx, cls = x_idx[valid], y_idx[valid], z_idx[valid], cls[valid]

    # initialize dense grid with 0 (free)
    occ_semantics = np.zeros(grid_size, dtype=np.uint8)
    occ_semantics[x_idx, y_idx, z_idx] = cls.astype(np.uint8)

    voxel_size = (
        (pc_range[3] - pc_range[0]) / grid_size[0],
        (pc_range[4] - pc_range[1]) / grid_size[1],
        (pc_range[5] - pc_range[2]) / grid_size[2],
    )

    occ_visualizer = OccupancyVisualizer(
        color_map=color_map,
        background_color=background_color,
    )

    if os.path.exists('view.json'):
        param = o3d.io.read_pinhole_camera_parameters('view.json')
    else:
        param = None

    # free=0, noise=255 → ignore
    occ_visualizer.vis_occ(
        occ_semantics,
        occ_flow=None,
        ignore_labels=[0, 255],
        voxelSize=voxel_size,
        range=pc_range,
        save_path=None,
        wait_time=-1,
        view_json=param,
        car_model_mesh=car_model_mesh,
    )

    param = occ_visualizer.o3d_vis.get_view_control().convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters('view.json', param)
    occ_visualizer.o3d_vis.destroy_window()


def vis_occ_diff_on_3d(pred_path,
                        gt_path,
                        dataset_type='occ3d',
                        voxel_size=(0.4, 0.4, 0.4),
                        car_model=None,
                        use_camera_mask=False,
                        save_path=None,
                        background_color=(255, 255, 255)):
    """GT vs Pred 비교 → 이웃 정확도 비율을 RdBu pseudo-color로 표현.

    모든 occupied voxel에 대해 26-이웃 중 정확한 voxel 비율(accuracy_ratio)을 계산하고
    matplotlib RdBu 색상맵(0=빨강, 1=파랑)으로 색칠한다.

    - 이웃 중 맞은 것이 많을수록 → 파란색
    - 이웃 중 틀린 것이 많을수록 → 붉은색
    - 이웃이 없는 고립 voxel    → 자신의 정답 여부(파랑/빨강)
    """
    from scipy.ndimage import convolve as nd_convolve
    import matplotlib.cm as cm_mpl

    free_cls   = 16 if dataset_type == 'openocc' else 17
    ignore_set = {free_cls, 255}
    pc_range   = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]

    # ── 데이터 로드 ────────────────────────────────────────────────────────
    pred_label = np.load(pred_path)
    pred_sem   = pred_label['semantics'].astype(np.int32)
    gt_label   = np.load(gt_path)
    gt_sem     = gt_label['semantics'].astype(np.int32)

    if use_camera_mask and 'mask_camera' in gt_label:
        mask = gt_label['mask_camera']
        pred_sem[mask == 0] = 255
        gt_sem[mask  == 0]  = 255

    # ── 정확/오차 마스크 ───────────────────────────────────────────────────
    valid_pred   = ~np.isin(pred_sem, list(ignore_set))
    valid_gt     = ~np.isin(gt_sem,   list(ignore_set))
    occupied     = valid_gt | valid_pred
    correct_mask = valid_gt & valid_pred & (pred_sem == gt_sem)
    error_mask   = occupied & ~correct_mask

    # ── 26-이웃 정확도 비율 계산 ───────────────────────────────────────────
    kernel = np.ones((3, 3, 3), dtype=np.float32)
    kernel[1, 1, 1] = 0

    correct_nbr = nd_convolve(correct_mask.astype(np.float32), kernel,
                               mode='constant', cval=0)
    error_nbr   = nd_convolve(error_mask.astype(np.float32),   kernel,
                               mode='constant', cval=0)
    total_nbr = correct_nbr + error_nbr

    # 이웃이 없으면 자기 자신의 정확도(1.0 / 0.0)로 fallback
    own_acc       = correct_mask.astype(np.float32)
    accuracy_ratio = np.where(total_nbr > 0,
                               correct_nbr / total_nbr,
                               own_acc)

    # ── occupied voxel 좌표 & pseudo-color 계산 ───────────────────────────
    occ_idx = np.where(occupied)
    xs = occ_idx[0] * voxel_size[0] + voxel_size[0] / 2 + pc_range[0]
    ys = occ_idx[1] * voxel_size[1] + voxel_size[1] / 2 + pc_range[1]
    zs = occ_idx[2] * voxel_size[2] + voxel_size[2] / 2 + pc_range[2]
    pts_xyz = np.column_stack([xs, ys, zs]).astype(np.float32)

    ratios      = accuracy_ratio[occ_idx]                       # [0,1]
    rgba_cmap   = cm_mpl.viridis(1.0 - ratios)                   # (N,4) RGBA [0,1], 반전: 틀림→노랑, 맞음→보라
    colors_bgr  = (rgba_cmap[:, [2, 1, 0]] * 255).astype(np.uint8)  # BGR 0-255

    print(f'[vis_diff] occupied: {len(pts_xyz)}, '
          f'correct: {correct_mask.sum()}, error: {error_mask.sum()}, '
          f'accuracy: {correct_mask.sum()/max(1,occupied.sum())*100:.1f}%')

    # ── 카메라 모델 로드 ──────────────────────────────────────────────────
    if car_model is not None:
        car_model_mesh = o3d.io.read_triangle_mesh(car_model)
        angle = np.pi / 2
        R_mat = car_model_mesh.get_rotation_matrix_from_axis_angle(np.array([angle, 0, 0]))
        car_model_mesh.rotate(R_mat, center=car_model_mesh.get_center())
        car_model_mesh.scale(0.25, center=car_model_mesh.get_center())
        car_model_mesh.translate(np.array([0, 0, 0.5]) - car_model_mesh.get_center())
        car_model_mesh.compute_vertex_normals()
    else:
        car_model_mesh = None

    # ── 시각화 ────────────────────────────────────────────────────────────
    dummy_cmap = np.zeros((18, 3), dtype=np.uint8)  # vis_occ_pseudo_color에서 미사용
    occ_visualizer = OccupancyVisualizer(color_map=dummy_cmap,
                                         background_color=background_color)
    if os.path.exists('view.json'):
        param = o3d.io.read_pinhole_camera_parameters('view.json')
    else:
        param = None

    if save_path is not None:
        mmengine.mkdir_or_exist(os.path.dirname(os.path.abspath(save_path)))

    occ_visualizer.vis_occ_pseudo_color(
        pts_xyz=pts_xyz,
        colors_bgr=colors_bgr,
        voxel_size=voxel_size[0],
        view_json=param,
        save_path=save_path,
        wait_time=-1,
        car_model_mesh=car_model_mesh,
    )

    param = occ_visualizer.o3d_vis.get_view_control().convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters('view.json', param)
    occ_visualizer.o3d_vis.destroy_window()


def flow_to_color(vx, vy, max_magnitude=None):
    magnitude = np.sqrt(vx ** 2 + vy ** 2)
    angle = np.arctan2(vy, vx)

    hue = (angle + np.pi) / (2 * np.pi)

    if max_magnitude is None:
        max_magnitude = np.max(magnitude)

    saturation = np.clip(magnitude / max_magnitude, 0, 1)
    value = np.ones_like(saturation)

    hsv = np.stack((hue, saturation, value), axis=-1)
    rgb = np.apply_along_axis(lambda x: colorsys.hsv_to_rgb(*x), -1, hsv)
    rgb = (rgb * 255).astype(np.uint8)

    return rgb

def create_legend_circle(radius=1, resolution=500):
    x = np.linspace(-radius, radius, resolution)
    y = np.linspace(-radius, radius, resolution)
    X, Y = np.meshgrid(x, y)
    vx = X
    vy = Y
    magnitude = np.sqrt(vx ** 2 + vy ** 2)
    mask = magnitude <= radius

    vx = vx[mask]
    vy = vy[mask]

    colors = flow_to_color(vx, vy, max_magnitude=radius)

    legend_image = np.ones((resolution, resolution, 3), dtype=np.uint8) * 255
    legend_image[mask.reshape(resolution, resolution)] = colors

    return legend_image

if __name__ == '__main__':
    print('open3d version: {}, if you want to use viewcontrol, make sure using 0.16.0 version!!'.format(o3d.__version__))
    args = parse_args()

    # ── nuScenes-Occupancy (CONet) ──────────────────────────────────────────
    if args.dataset_type == 'nusc_occ':
        npy_path = args.nusc_occ_path if args.nusc_occ_path else args.vis_single_data
        print(f'[nusc_occ] visualizing: {npy_path}')
        vis_occ_npy_on_3d(
            data_path=npy_path,
            pc_range=args.nusc_occ_pc_range,
            grid_size=args.nusc_occ_grid_size,
            xyz_order='zyx',
            color_map=nusc_occ_colors_map,
            car_model=args.car_model,
        )
    # ── SurroundOcc ─────────────────────────────────────────────────────────
    elif args.dataset_type == 'surroundocc':
        npy_path = args.surroundocc_path if args.surroundocc_path else args.vis_single_data
        print(f'[surroundocc] visualizing: {npy_path}')
        vis_occ_npy_on_3d(
            data_path=npy_path,
            pc_range=args.surroundocc_pc_range,
            grid_size=args.surroundocc_grid_size,
            xyz_order='xyz',
            color_map=surroundocc_colors_map,
            car_model=args.car_model,
        )
    # ── occ3d / openocc ────────────────────────────────────────────────────
    else:
        if args.vis_diff:
            # GT vs Pred 오차 시각화
            if args.gt_path is None:
                raise ValueError('--vis-diff requires --gt-path')
            vis_occ_diff_on_3d(
                pred_path=args.vis_single_data,
                gt_path=args.gt_path,
                dataset_type=args.dataset_type,
                car_model=args.car_model,
                use_camera_mask=args.use_camera_mask,
                save_path=args.save_path,
            )
        else:
            # Single data visualization (NuScenes DB 불필요)
            vis_occ_single_on_3d(args.vis_single_data, dataset_type=args.dataset_type, car_model=args.car_model, vis_flow=False, use_camera_mask=args.use_camera_mask, gt_path=args.gt_path, save_path=args.save_path)
        # ── 씬 전체 시각화 시 아래 주석 해제 (NuScenes DB + pkl 필요) ──────────
        # mmengine.mkdir_or_exist(args.vis_path)
        # pkl_data = mmengine.load(args.pkl_file)
        # nusc = NuScenes(args.data_version, args.data_path)
        # vis_scenes_infos = arange_according_to_scene(pkl_data['infos'], nusc, args.vis_scene)
        # GT visualization
        # vis_occ_scene_on_3d(vis_scenes_infos, args.vis_scene, args.vis_path, args.pred_path, dataset_type=args.dataset_type, vis_gt=True, car_model=args.car_model)
        # vis_occ_scene_on_3d(vis_scenes_infos, args.vis_scene, args.vis_path, args.pred_path, load_camera_mask=False, dataset_type=args.dataset_type, vis_gt=True, car_model=args.car_model)
        # vis_occ_scene_on_3d(vis_scenes_infos, args.vis_scene, args.vis_path, args.pred_path, load_camera_mask=True, dataset_type=args.dataset_type, vis_gt=True, car_model=args.car_model)
        # Pred visualization (without camera mask)
        # vis_occ_scene_on_3d(vis_scenes_infos, args.vis_scene, args.vis_path, args.pred_path, load_camera_mask=False, dataset_type=args.dataset_type, vis_gt=False, car_model=args.car_model)
        # Pred visualization (with camera mask from GT)
        # vis_occ_scene_on_3d(vis_scenes_infos, args.vis_scene, args.vis_path, args.pred_path, load_camera_mask=True, dataset_type=args.dataset_type, vis_gt=False, car_model=args.car_model)
    # # GT Flow visualization
    # vis_occ_scene_on_3d(vis_scenes_infos, args.vis_scene, args.vis_path, args.pred_path, dataset_type=args.dataset_type, vis_gt=True, vis_flow=True, car_model=args.car_model)
    # # Pred Flow visualization (without camera mask)
    # vis_occ_scene_on_3d(vis_scenes_infos, args.vis_scene, args.vis_path, args.pred_path, load_camera_mask=False, dataset_type=args.dataset_type, vis_gt=False, vis_flow=True, car_model=args.car_model)
    # # Pred Flow visualization (with camera mask from GT)
    # vis_occ_scene_on_3d(vis_scenes_infos, args.vis_scene, args.vis_path, args.pred_path, load_camera_mask=True, dataset_type=args.dataset_type, vis_gt=False, vis_flow=True, car_model=args.car_model)
    # # Single Flow data visualization
    # vis_occ_single_on_3d(args.vis_single_data, dataset_type=args.dataset_type, car_model=args.car_model, vis_flow=True, use_camera_mask=args.use_camera_mask, gt_path=args.gt_path)


# =============================================================================
# 사용법 예시 (Usage Examples)
# =============================================================================
#
# [인자 설명]
#   --pkl-file        : nuScenes 데이터셋 pkl 파일 경로 (default: data/nuscenes/stcocc-nuscenes_infos_val.pkl)
#   --data-path       : nuScenes 데이터셋 루트 경로 (default: data/nuscenes)
#   --data-version    : nuScenes 버전 (default: v1.0-trainval)
#   --dataset-type    : 데이터셋 타입, occ3d 또는 openocc (default: occ3d)
#   --pred-path       : 예측 결과 디렉토리 경로 (default: results/scene-0003)
#   --vis-scene       : 시각화할 씬 이름 (default: scene-0003)
#   --vis-path        : 결과 이미지/비디오 저장 경로 (default: demo_out)
#   --car-model       : 차량 3D 모델(.obj) 경로 (default: visualizer/3d_model.obj)
#   --vis-single-data    : 단일 .npz 파일 경로 (occ3d/openocc) 또는 .npy 파일 경로 (nusc_occ)
#   --gt-path            : GT labels.npz 경로 (카메라 마스크 로딩 또는 --vis-diff 비교 대상)
#   --use-camera-mask    : 카메라에서 보이는 복셀만 시각화 (flag)
#   --vis-diff           : GT vs Pred 비교 모드 (--gt-path 필수)
#                          각 voxel의 26-이웃 중 정확한 예측 비율로 viridis pseudo-color 적용
#                          (맞은 이웃 많음 → 보라/남색, 틀린 이웃 많음 → 노랑)
#   --save-path          : 저장할 이미지 경로 (.png 자동 추가, e.g. demo_out/result)
#   --nusc-occ-path        : [nusc_occ] .npy 파일 경로 (--vis-single-data 대체 가능)
#   --nusc-occ-pc-range    : [nusc_occ] point cloud range 6개 값 (default: -51.2 -51.2 -5.0 51.2 51.2 3.0)
#   --nusc-occ-grid-size   : [nusc_occ] voxel grid 크기 3개 값 (default: 512 512 40)
#   --surroundocc-path     : [surroundocc] .npy 파일 경로 (--vis-single-data 대체 가능)
#   --surroundocc-pc-range : [surroundocc] point cloud range 6개 값 (default: -50 -50 -5.0 50 50 3.0)
#   --surroundocc-grid-size: [surroundocc] voxel grid 크기 3개 값 (default: 200 200 16)
#
# --------------------------------------------------------------------------
# 1. 단일 npz 파일 시각화 (occ3d 기본 동작)
#
#   python tools/vis_results.py \
#       --vis-single-data results/scene-0107/84c24cd1d7914f72bb2fa17a6c5c41e5.npz \
#       --dataset-type occ3d \
#       --car-model visualizer/3d_model.obj
#
# --------------------------------------------------------------------------
# 2. 단일 파일 + 카메라 마스크 적용
#
#   python tools/vis_results.py \
#       --vis-single-data results/scene-0107/84c24cd1d7914f72bb2fa17a6c5c41e5.npz \
#       --use-camera-mask \
#       --gt-path data/nuscenes/gts/scene-0107/<token>/labels.npz
#
# --------------------------------------------------------------------------
# 3. 씬 전체 예측 결과 시각화 (비디오 저장)
#    __main__ 블록에서 vis_occ_scene_on_3d(vis_gt=False) 호출 주석 해제 후 실행
#    결과: demo_out/pred-occ_scene-0003.avi
#
#   python tools/vis_results.py \
#       --pkl-file data/nuscenes/stcocc-nuscenes_infos_val.pkl \
#       --data-path data/nuscenes \
#       --pred-path results/scene-0003 \
#       --vis-scene scene-0003 \
#       --vis-path demo_out \
#       --dataset-type occ3d
#
# --------------------------------------------------------------------------
# 4. GT 시각화 (씬 전체)
#    __main__ 블록에서 vis_occ_scene_on_3d(vis_gt=True) 호출 주석 해제 후 실행
#    결과: demo_out/gt-occ_scene-0003.avi
#
#   python tools/vis_results.py \
#       --pkl-file data/nuscenes/stcocc-nuscenes_infos_val.pkl \
#       --data-path data/nuscenes \
#       --vis-path demo_out \
#       --dataset-type occ3d
#
# --------------------------------------------------------------------------
# 5. OpenOCC 데이터셋으로 단일 파일 시각화
#
#   python tools/vis_results.py \
#       --vis-single-data results/scene-0003/<token>.npz \
#       --dataset-type openocc \
#       --car-model visualizer/3d_model.obj
#
# --------------------------------------------------------------------------
# 6. nuScenes-Occupancy (CONet) GT .npy 파일 시각화
#    - sparse [z, y, x, cls] 포맷, grid 512x512x40, range [-51.2~51.2, -51.2~51.2, -5~3]
#    - label 0(noise)→무시, 1-16: semantic, 0(free, 미포함 voxel)→무시
#
#   python tools/vis_results.py \
#       --dataset-type nusc_occ \
#       --nusc-occ-path data/nuScenes-Occupancy/scene_<scene_token>/occupancy/<lidar_token>.npy \
#       --car-model visualizer/3d_model.obj
#
#   # --vis-single-data로도 지정 가능 (--nusc-occ-path 미지정 시 사용)
#   python tools/vis_results.py \
#       --dataset-type nusc_occ \
#       --vis-single-data data/nuScenes-Occupancy/scene_c3ab8ee2c1a54068a72d7eb4cf22e43d/occupancy/3f30536943fa4fc6a63cf8377433a9c8.npy
#
# --------------------------------------------------------------------------
# 8. GT vs Pred 비교 시각화 (--vis-diff)
#    각 voxel의 26-이웃 정확도 비율을 viridis pseudo-color로 표현
#    - 맞은 이웃이 많을수록 → 보라/남색(어두움)
#    - 틀린 이웃이 많을수록 → 밝은 노랑
#    결과: demo_out/diff_result.png
#
#   python tools/vis_results.py \
#       --vis-single-data results/scene-0268/<pred_token>.npz \
#       --gt-path data/nuscenes/gts/scene-0268/<token>/labels.npz \
#       --vis-diff \
#       --dataset-type occ3d \
#       --save-path demo_out/diff_result
#
# --------------------------------------------------------------------------
# 9. GT vs Pred 비교 시각화 + 카메라 마스크 적용
#    카메라 가시 영역만 비교 (더 공정한 평가 시각화)
#
#   python tools/vis_results.py \
#       --vis-single-data results/scene-0268/<pred_token>.npz \
#       --gt-path data/nuscenes/gts/scene-0268/<token>/labels.npz \
#       --vis-diff \
#       --use-camera-mask \
#       --save-path demo_out/diff_w_mask
#
# --------------------------------------------------------------------------
# 7. SurroundOcc GT .npy 파일 시각화
#    - sparse [x, y, z, cls] 포맷, grid 200x200x16, range [-50~50, -50~50, -5~3]
#    - label 0(noise)→무시, 1-16: semantic, 0(free, 미포함 voxel)→무시
#
#   python tools/vis_results.py \
#       --dataset-type surroundocc \
#       --surroundocc-path data/nuscenes_occ/samples/<lidar_filename>.npy \
#       --car-model visualizer/3d_model.obj
#
#   # --vis-single-data로도 지정 가능 (--surroundocc-path 미지정 시 사용)
#   python tools/vis_results.py \
#       --dataset-type surroundocc \
#       --vis-single-data "data/nuscenes_occ/samples/n015-2018-07-11-11-54-16+0800__LIDAR_TOP__1531281441800080.pcd.bin.npy"
#
# --------------------------------------------------------------------------
# [포맷 비교 요약]
#   dataset_type  | 파일 형식  | col 순서    | grid          | pc_range
#   --------------|-----------|------------|---------------|------------------------
#   occ3d         | .npz      | dense      | 200x200x16    | [-40,40,-40,40,-1,5.4]
#   openocc       | .npz      | dense      | 200x200x16    | [-40,40,-40,40,-1,5.4]
#   nusc_occ      | .npy      | [z,y,x,cls]| 512x512x40    | [-51.2,51.2,-5,3]
#   surroundocc   | .npy      | [x,y,z,cls]| 200x200x16    | [-50,50,-50,50,-5,3]
#
# --------------------------------------------------------------------------
# [동작 방식]
#   - 시각화 창이 열리면 Q 키 또는 우측 상단 X를 눌러 다음 프레임으로 이동합니다.
#   - 뷰 조정 후 view.json이 자동 저장되어 다음 실행 시 동일한 카메라 시점이 적용됩니다.
#   - --use-camera-mask 옵션 사용 시 카메라에 실제로 보이는 복셀만 렌더링합니다.
# =============================================================================

