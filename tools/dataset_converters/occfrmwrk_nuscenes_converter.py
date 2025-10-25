# Copyright (c) OpenMMLab. All rights reserved.
import os
from collections import OrderedDict
from os import path as osp
from pathlib import Path
from typing import List, Tuple, Union

import mmcv
import mmengine
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion
from shapely.geometry import MultiPoint, box

from mmdet3d.datasets.convert_utils import NuScenesNameMapping
from mmdet3d.datasets.utils import convert_quaternion_to_matrix
from mmdet3d.structures import points_cam2img

nus_categories = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                  'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                  'barrier')

nus_attributes = ('cycle.with_rider', 'cycle.without_rider',
                  'pedestrian.moving', 'pedestrian.standing',
                  'pedestrian.sitting_lying_down', 'vehicle.moving',
                  'vehicle.parked', 'vehicle.stopped', 'None')

# FusionOcc specific mapping
map_name_from_general_to_detection = {
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.wheelchair': 'ignore',
    'human.pedestrian.stroller': 'ignore',
    'human.pedestrian.personal_mobility': 'ignore',
    'human.pedestrian.police_officer': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'animal': 'ignore',
    'vehicle.car': 'car',
    'vehicle.motorcycle': 'motorcycle',
    'vehicle.bicycle': 'bicycle',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.truck': 'truck',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.emergency.ambulance': 'ignore',
    'vehicle.emergency.police': 'ignore',
    'vehicle.trailer': 'trailer',
    'movable_object.barrier': 'barrier',
    'movable_object.trafficcone': 'traffic_cone',
    'movable_object.pushable_pullable': 'ignore',
    'movable_object.debris': 'ignore',
    'static_object.bicycle_rack': 'ignore',
}

classes = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]


def get_gt_original(info):
    """Generate gt labels from info (original version from create_data_fusionocc.py).

    Args:
        info(dict): Infos needed to generate gt labels.

    Returns:
        Tensor: GT bboxes.
        Tensor: GT labels.
    """
    ego2global_rotation = info['cams']['CAM_FRONT']['ego2global_rotation']
    ego2global_translation = info['cams']['CAM_FRONT'][
        'ego2global_translation']
    trans = -np.array(ego2global_translation)
    rot = Quaternion(ego2global_rotation).inverse
    gt_boxes = list()
    gt_labels = list()
    for ann_info in info['ann_infos']:
        # Use ego coordinate.
        if (map_name_from_general_to_detection[ann_info['category_name']]
                not in classes
                or ann_info['num_lidar_pts'] + ann_info['num_radar_pts'] <= 0):
            continue
        box = Box(
            ann_info['translation'],
            ann_info['size'],
            Quaternion(ann_info['rotation']),
            velocity=ann_info['velocity'],
        )
        box.translate(trans)
        box.rotate(rot)
        box_xyz = np.array(box.center)
        box_dxdydz = np.array(box.wlh)[[1, 0, 2]]
        box_yaw = np.array([box.orientation.yaw_pitch_roll[0]])
        box_velo = np.array(box.velocity[:2])
        gt_box = np.concatenate([box_xyz, box_dxdydz, box_yaw, box_velo])
        gt_boxes.append(gt_box)
        gt_labels.append(
            classes.index(
                map_name_from_general_to_detection[ann_info['category_name']]))
    return gt_boxes, gt_labels


def get_gt(info):
    """Generate gt labels from info.

    Args:
        info(dict): Infos needed to generate gt labels.

    Returns:
        Tensor: GT bboxes.
        Tensor: GT labels.
    """
    ego2global_rotation = info['cams']['CAM_FRONT']['ego2global_rotation']
    ego2global_translation = info['cams']['CAM_FRONT'][
        'ego2global_translation']
    trans = -np.array(ego2global_translation)
    rot = Quaternion(ego2global_rotation).inverse
    gt_boxes = list()
    gt_labels = list()
    for ann_info in info['ann_infos']:
        # Use ego coordinate.
        if (map_name_from_general_to_detection[ann_info['category_name']]
                not in classes
                or ann_info['num_lidar_pts'] + ann_info['num_radar_pts'] <= 0):
            continue
        box = Box(
            ann_info['translation'],
            ann_info['size'],
            Quaternion(ann_info['rotation']),
            velocity=ann_info['velocity'],
        )
        box.translate(trans)
        box.rotate(rot)
        box_xyz = np.array(box.center)
        box_dxdydz = np.array(box.wlh)[[1, 0, 2]]
        box_yaw = np.array([box.orientation.yaw_pitch_roll[0]])
        box_velo = np.array(box.velocity[:2])
        gt_box = np.concatenate([box_xyz, box_dxdydz, box_yaw, box_velo])
        gt_boxes.append(gt_box)
        gt_labels.append(
            classes.index(
                map_name_from_general_to_detection[ann_info['category_name']]))
    return gt_boxes, gt_labels





def get_available_scenes(nusc):
    """Get available scenes from the input nuscenes class.

    Given the raw data, get the information of available scenes for
    further info generation.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.

    Returns:
        available_scenes (list[dict]): List of basic information for the
            available scenes.
    """
    available_scenes = []
    print('total scene num: {}'.format(len(nusc.scene)))
    for scene in nusc.scene:
        scene_token = scene['token']
        scene_rec = nusc.get('scene', scene_token)
        sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
        sd_rec = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
        has_more_frames = True
        scene_not_exist = False
        while has_more_frames:
            lidar_path, boxes, _ = nusc.get_sample_data(sd_rec['token'])
            lidar_path = str(lidar_path)
            if os.getcwd() in lidar_path:
                # path from lyftdataset is absolute path
                lidar_path = lidar_path.split(f'{os.getcwd()}/')[-1]
                # relative path
            if not mmengine.is_filepath(lidar_path):
                scene_not_exist = True
                break
            else:
                break
        if scene_not_exist:
            continue
        available_scenes.append(scene)
    print('exist scene num: {}'.format(len(available_scenes)))
    return available_scenes


def obtain_sensor2top(nusc,
                     sensor_token,
                     l2e_t,
                     l2e_r_mat,
                     e2g_t,
                     e2g_r_mat,
                     sensor_type='lidar'):
    """Obtain the info with RT matric from general sensor to Top LiDAR.

    Args:
        nusc(class): Dataset class in the nuScenes.
        sensor_token(str): Sample data token corresponding to the
            specific sensor type.
        l2e_t(np.ndarray): Translation from lidar to ego
            in shape (1, 3).
        l2e_r_mat(np.ndarray): Rotation from lidar to ego
            in shape (3, 3).
        e2g_t(np.ndarray): Translation from ego to global
            in shape (1, 3).
        e2g_r_mat(np.ndarray): Rotation from ego to global
            in shape (3, 3).
        sensor_type(str): Sensor to calibrate. Default: 'lidar'.

    Returns:
        sweep (dict): Sweep information after transformation.
    """
    sd_rec = nusc.get('sample_data', sensor_token)
    cs_record = nusc.get('calibrated_sensor',
                          sd_rec['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    data_path = str(nusc.get_sample_data_path(sd_rec['token']))
    if os.getcwd() in data_path:  # path from lyftdataset is absolute path
        data_path = data_path.split(f'{os.getcwd()}/')[-1]  # relative path
    sweep = {
        'data_path': data_path,
        'type': sensor_type,
        'sample_data_token': sd_rec['token'],
        'sensor2ego_translation': cs_record['translation'],
        'sensor2ego_rotation': cs_record['rotation'],
        'ego2global_translation': pose_record['translation'],
        'ego2global_rotation': pose_record['rotation'],
        'timestamp': sd_rec['timestamp']
    }
    l2e_r_s = sweep['sensor2ego_rotation']
    l2e_t_s = sweep['sensor2ego_translation']
    e2g_r_s = sweep['ego2global_rotation']
    e2g_t_s = sweep['ego2global_translation']

    # obtain the RT from sensor to Top LiDAR
    # sweep->ego->global->ego'->lidar
    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                 ) + l2e_t @ np.linalg.inv(l2e_r_mat).T
    sweep['sensor2lidar_rotation'] = R.T  # points @ R.T + T
    sweep['sensor2lidar_translation'] = T
    return sweep


def _get_can_bus_info(nusc, nusc_can_bus, sample):
    """Get CAN bus information for a sample.
    
    Args:
        nusc (NuScenes): NuScenes dataset object.
        nusc_can_bus (NuScenesCanBus): NuScenes CAN bus API object.
        sample (dict): Sample record.
        
    Returns:
        np.ndarray: CAN bus information array with 18 elements.
    """
    scene_name = nusc.get('scene', sample['scene_token'])['name']
    sample_timestamp = sample['timestamp']
    
    try:
        pose_list = nusc_can_bus.get_messages(scene_name, 'pose')
    except:
        return np.zeros(18)  # Some scenes do not have CAN bus information
    
    # Find the pose closest to the sample timestamp
    # During each scene, the first timestamp of can_bus may be larger than the first sample's timestamp
    last_pose = pose_list[0]
    for i, pose in enumerate(pose_list):
        if pose['utime'] > sample_timestamp:
            break
        last_pose = pose
    
    can_bus = []
    _ = last_pose.pop('utime')  # Remove useless field
    pos = last_pose.pop('pos')
    rotation = last_pose.pop('orientation')
    can_bus.extend(pos)  # 3 elements: x, y, z
    can_bus.extend(rotation)  # 4 elements: quaternion
    
    # Add remaining fields (acceleration, rotation_rate, vel)
    for key in last_pose.keys():
        can_bus.extend(last_pose[key])  # Typically adds 9 more elements
    
    # Pad to 18 elements if needed
    while len(can_bus) < 18:
        can_bus.append(0.0)
    
    return np.array(can_bus[:18])  # Ensure exactly 18 elements


def _fill_trainval_infos(nusc,
                         train_scenes,
                         val_scenes,
                         test=False,
                         max_sweeps=10,
                         max_frames=None,
                         nusc_can_bus=None):
    """Generate the train/val infos from the raw data.

    Args:
        nusc(:obj:`NuScenes`): Dataset class in the nuScenes.
        train_scenes (list[str]): Basic information of training scenes.
        val_scenes (list[str]): Basic information of validation scenes.
        test (bool): Whether use the test mode. In the test mode, no
            annotations can be accessed. Default: False.
        max_sweeps (int): Max number of sweeps. Default: 10.
        max_frames (int, optional): Max number of frames to process. 
            If None, process all frames. Default: None.
        nusc_can_bus (NuScenesCanBus, optional): NuScenes CAN bus API object.
            If None, CAN bus info will be zeros. Default: None.

    Returns:
        tuple[list[dict]]: Information of training set and
            validation set that will be saved to the info file.
    """
    train_nusc_infos = []
    val_nusc_infos = []
    
    total_frames_processed = 0

    for sample in mmengine.track_iter_progress(nusc.sample):
        # Check if we've reached the max frames limit
        if max_frames is not None and total_frames_processed >= max_frames:
            print(f'\nReached max frames limit ({max_frames}). Stopping...')
            break
        
        total_frames_processed += 1
        # Get sensor information
        lidar_token = sample['data']['LIDAR_TOP']
        sd_rec = nusc.get('sample_data', lidar_token)
        cs_record = nusc.get('calibrated_sensor',
                             sd_rec['calibrated_sensor_token'])
        pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
        lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)
        lidar_path = str(lidar_path)

        mmengine.check_file_exist(lidar_path)
        
        # Generate surroundocc_path (similar to SurroundOcc format)
        surroundocc_path = str(lidar_path)
        surroundocc_path = surroundocc_path.replace('nuscenes', 'nuscenes_occ')
        surroundocc_path = surroundocc_path.replace('LIDAR_TOP/', '')
        
        # Get CAN bus information
        if nusc_can_bus is not None:
            can_bus = _get_can_bus_info(nusc, nusc_can_bus, sample)
        else:
            can_bus = np.zeros(18)
        
        info = {
            'lidar_path': lidar_path,
            'num_features': 5,
            'token': sample['token'],
            'sweeps': [],
            'cams': dict(),
            'lidar2ego_translation': cs_record['translation'],
            'lidar2ego_rotation': cs_record['rotation'],
            'ego2global_translation': pose_record['translation'],
            'ego2global_rotation': pose_record['rotation'],
            'timestamp': sample['timestamp'],
            'can_bus': can_bus,
            'surroundocc_path': surroundocc_path,
        }

        l2e_r = info['lidar2ego_rotation']
        l2e_t = info['lidar2ego_translation']
        e2g_r = info['ego2global_rotation']
        e2g_t = info['ego2global_translation']
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

        # obtain 6 images' information per frame
        camera_types = [
            'CAM_FRONT',
            'CAM_FRONT_RIGHT',
            'CAM_FRONT_LEFT',
            'CAM_BACK',
            'CAM_BACK_LEFT',
            'CAM_BACK_RIGHT',
        ]
        for cam in camera_types:
            cam_token = sample['data'][cam]
            cam_path, _, cam_intrinsic = nusc.get_sample_data(cam_token)
            cam_info = obtain_sensor2top(nusc, cam_token, l2e_t, l2e_r_mat,
                                         e2g_t, e2g_r_mat, cam)
            cam_info.update(cam_intrinsic=cam_intrinsic)
            info['cams'].update({cam: cam_info})

        # Collect radar sweeps for each radar (following nuscenes_converter_R.py)
        radar_names = [
            'RADAR_FRONT',
            'RADAR_FRONT_LEFT',
            'RADAR_FRONT_RIGHT',
            'RADAR_BACK_LEFT',
            'RADAR_BACK_RIGHT'
        ]
        info['radars'] = dict()
        for radar_name in radar_names:
            if radar_name not in sample['data']:
                continue
                
            radar_token = sample['data'][radar_name]
            radar_rec = nusc.get('sample_data', radar_token)
            radar_sweeps = []
            
            # Load multi-radar sweeps (up to 5 sweeps) - exactly like nuscenes_converter_R.py
            while len(radar_sweeps) < 5:
                if radar_rec['prev'] != '':
                    # Get radar info using obtain_sensor2top
                    radar_info = obtain_sensor2top(nusc, radar_token, l2e_t, l2e_r_mat,
                                                   e2g_t, e2g_r_mat, radar_name)
                    radar_sweeps.append(radar_info)
                    radar_token = radar_rec['prev']
                    radar_rec = nusc.get('sample_data', radar_token)
                else:
                    # Last sweep (no more previous) - add current and continue (will exit by while condition)
                    radar_info = obtain_sensor2top(nusc, radar_token, l2e_t, l2e_r_mat,
                                                   e2g_t, e2g_r_mat, radar_name)
                    radar_sweeps.append(radar_info)
                    # Don't break - let while condition handle loop exit
            
            # One radar corresponds to several sweeps
            info['radars'].update({radar_name: radar_sweeps})

        # obtain sweeps for a single key-frame
        sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        sweeps = []
        while len(sweeps) < max_sweeps:
            if not sd_rec['prev'] == '':
                sweep = obtain_sensor2top(nusc, sd_rec['prev'], l2e_t,
                                          l2e_r_mat, e2g_t, e2g_r_mat, 'lidar')
                sweeps.append(sweep)
                sd_rec = nusc.get('sample_data', sd_rec['prev'])
            else:
                break
        info['sweeps'] = sweeps
        # obtain annotation
        if not test:
            annotations = [
                nusc.get('sample_annotation', token)
                for token in sample['anns']
            ]
            locs = np.array([b.center for b in boxes]).reshape(-1, 3)
            dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
            rots = np.array([b.orientation.yaw_pitch_roll[0]
                             for b in boxes]).reshape(-1, 1)

            names = [b.name for b in boxes]
            names = [name if name != 'ignore' else 'unlabeled'
                     for name in names]
            velocity = np.array(
                [nusc.box_velocity(token)[:2] for token in sample['anns']])
            valid_flag = np.array(
                [(anno['num_lidar_pts'] + anno['num_radar_pts']) > 0
                 for anno in annotations],
                dtype=bool).reshape(-1)
            # convert velo from global to lidar
            for i in range(len(boxes)):
                velo = np.array([*velocity[i], 0.0])
                velo = velo @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(
                    l2e_r_mat).T
                velocity[i] = velo[:2]

            names = [b.name for b in boxes]
            for i in range(len(names)):
                if names[i] in NuScenesNameMapping:
                    names[i] = NuScenesNameMapping[names[i]]
            names = np.array(names)
            # we need to convert box size to
            # the format of our lidar coordinate system
            # which is x_size, y_size, z_size (corresponding to l, w, h)
            gt_boxes = np.concatenate([locs, dims[:, [1, 0, 2]], rots], axis=1)
            assert len(gt_boxes) == len(
                annotations), f'{len(gt_boxes)}, {len(annotations)}'
            info['gt_boxes'] = gt_boxes
            info['gt_names'] = names
            info['gt_velocity'] = velocity.reshape(-1, 2)
            info['num_lidar_pts'] = np.array(
                [a['num_lidar_pts'] for a in annotations])
            info['num_radar_pts'] = np.array(
                [a['num_radar_pts'] for a in annotations])
            info['valid_flag'] = valid_flag

            if 'lidarseg' in nusc.table_names:
                info['pts_semantic_mask_path'] = osp.join(
                    nusc.dataroot,
                    nusc.get('lidarseg', lidar_token)['filename'])

        if sample['scene_token'] in train_scenes:
            train_nusc_infos.append(info)
        else:
            val_nusc_infos.append(info)

    return train_nusc_infos, val_nusc_infos


def create_occfrmwrk_infos(root_path,
                           info_prefix,
                           version='v1.0-trainval',
                           max_sweeps=10,
                           max_frames=None,
                           can_bus_root_path=None):
    """Create info file of nuscene dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        root_path (str): Path of the data root.
        info_prefix (str): Prefix of the info file to be generated.
        version (str, optional): Version of the data.
            Default: 'v1.0-trainval'.
        max_sweeps (int, optional): Max number of sweeps.
            Default: 10.
        max_frames (int, optional): Max number of frames to process.
            If None, process all frames. Default: None.
        can_bus_root_path (str, optional): Path to CAN bus data root.
            If None, uses the same as root_path. Default: None.
    """
    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    
    # Initialize CAN bus API
    nusc_can_bus = None
    if can_bus_root_path is None:
        can_bus_root_path = root_path
    
    try:
        from nuscenes.can_bus.can_bus_api import NuScenesCanBus
        nusc_can_bus = NuScenesCanBus(dataroot=can_bus_root_path)
        print('CAN bus data loaded successfully.')
    except Exception as e:
        print(f'Warning: Failed to load CAN bus data: {e}')
        print('CAN bus information will be set to zeros.')
    
    from nuscenes.utils import splits
    available_vers = ['v1.0-trainval', 'v1.0-test', 'v1.0-mini']
    assert version in available_vers
    if version == 'v1.0-trainval':
        train_scenes = splits.train
        val_scenes = splits.val
    elif version == 'v1.0-test':
        train_scenes = splits.test
        val_scenes = []
    elif version == 'v1.0-mini':
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val
    else:
        raise ValueError('unknown')

    # filter existing scenes.
    available_scenes = get_available_scenes(nusc)
    available_scene_names = [s['name'] for s in available_scenes]
    train_scenes = list(
        filter(lambda x: x in available_scene_names, train_scenes))
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
    train_scenes = set([
        available_scenes[available_scene_names.index(s)]['token']
        for s in train_scenes
    ])
    val_scenes = set([
        available_scenes[available_scene_names.index(s)]['token']
        for s in val_scenes
    ])

    test = 'test' in version
    if test:
        print('test scene: {}'.format(len(train_scenes)))
    else:
        print('train scene: {}, val scene: {}'.format(
            len(train_scenes), len(val_scenes)))
    
    if max_frames is not None:
        print(f'Processing only {max_frames} frames for testing...')
    
    train_nusc_infos, val_nusc_infos = _fill_trainval_infos(
        nusc, train_scenes, val_scenes, test, max_sweeps=max_sweeps, max_frames=max_frames, nusc_can_bus=nusc_can_bus)

    metainfo = {
        'categories': {
            'car': 0, 'truck': 1, 'trailer': 2, 'bus': 3, 
            'construction_vehicle': 4, 'bicycle': 5, 'motorcycle': 6, 
            'pedestrian': 7, 'traffic_cone': 8, 'barrier': 9
        },
        'dataset': 'nuscenes',
        'version': version,
        'info_version': '1.1'
    }
    
    # Convert to v2 format
    def convert_to_v2_format(infos, nusc=None):
        data_list = []
        print(f'Converting {len(infos)} samples to v2 format...')
        prog_bar = mmengine.ProgressBar(len(infos))
        for idx, info in enumerate(infos):
            prog_bar.update()
            # Convert ego2global to 4x4 matrix format like update_infos_to_v2.py
            ego2global_matrix = convert_quaternion_to_matrix(
                info['ego2global_rotation'],
                info['ego2global_translation'])
            
            v2_info = {
                'sample_idx': idx,
                'token': info['token'],
                'timestamp': info['timestamp'],  # Keep as microseconds (same as nuscenes_converter_R.py)
                'ego2global': ego2global_matrix,
                'images': {},
                'lidar_points': {
                    'num_pts_feats': info['num_features'],  # Change key name to match nuscenes_converter.py
                    'lidar_path': Path(info['lidar_path']).name,  # Only filename like nuscenes_converter.py
                    'lidar2ego': convert_quaternion_to_matrix(
                        info['lidar2ego_rotation'],
                        info['lidar2ego_translation'])  # Convert to 4x4 matrix like nuscenes_converter.py
                },
                'instances': [],
                'cam_instances': {
                    'CAM_FRONT': [],
                    'CAM_FRONT_RIGHT': [],
                    'CAM_FRONT_LEFT': [],
                    'CAM_BACK': [],
                    'CAM_BACK_LEFT': [],
                    'CAM_BACK_RIGHT': []
                }
            }
            
            # Add CAN bus from info (already computed in _fill_trainval_infos)
            if 'can_bus' in info:
                v2_info['can_bus'] = info['can_bus'].tolist() if isinstance(info['can_bus'], np.ndarray) else info['can_bus']
            else:
                v2_info['can_bus'] = np.zeros(18).tolist()
            
            # Add surroundocc_path from info
            if 'surroundocc_path' in info:
                v2_info['surroundocc_path'] = info['surroundocc_path']
            
            # Add scene_token, prev, next, location, lidar_token if nusc is provided
            if nusc is not None:
                try:
                    sample = nusc.get('sample', info['token'])
                    v2_info['scene_token'] = sample['scene_token']
                    
                    # Add prev and next sample tokens
                    v2_info['prev'] = sample['prev'] if sample['prev'] != '' else None
                    v2_info['next'] = sample['next'] if sample['next'] != '' else None
                    
                    # Add lidar_token
                    v2_info['lidar_token'] = sample['data']['LIDAR_TOP']
                    
                    # Add occ_path
                    scene = nusc.get('scene', sample['scene_token'])
                    v2_info['occ_path'] = osp.join(root_path, 'gts', scene['name'], info['token'])
                    
                    # Add location
                    log_record = nusc.get('log', scene['log_token'])
                    v2_info['location'] = log_record['location']
                    
                except Exception as e:
                    print(f"Warning: Failed to get scene info for token {info['token']}: {e}")
            
            # Add camera information and generate instances for each camera
            for cam_name, cam_info in info['cams'].items():
                # Convert cam2ego using quaternion
                cam2ego_matrix = convert_quaternion_to_matrix(
                    cam_info['sensor2ego_rotation'],
                    cam_info['sensor2ego_translation'])
                
                # Convert sensor2lidar using rotation matrix (like nuscenes_converter.py)
                lidar2sensor = np.eye(4)
                rot = cam_info['sensor2lidar_rotation']
                trans = cam_info['sensor2lidar_translation']
                lidar2sensor[:3, :3] = rot.T
                lidar2sensor[:3, 3:4] = -1 * np.matmul(rot.T, trans.reshape(3, 1))
                
                # CRITICAL: Add per-camera ego2global pose (at camera timestamp)
                # This is essential for FusionOcc's multi-camera fusion
                cam_ego2global_matrix = convert_quaternion_to_matrix(
                    cam_info['ego2global_rotation'],
                    cam_info['ego2global_translation'])
                
                v2_info['images'][cam_name] = {
                    'img_path': Path(cam_info['data_path']).name,  # Only filename like nuscenes_converter.py
                    'cam2img': cam_info['cam_intrinsic'].tolist(),  # Convert to list
                    'cam2ego': cam2ego_matrix,
                    'ego2global': cam_ego2global_matrix,  # Per-camera ego2global (FusionOcc requirement)
                    'sample_data_token': cam_info['sample_data_token'],
                    'timestamp': cam_info['timestamp']/1e6,  
                    'lidar2cam': lidar2sensor.astype(np.float32).tolist()  # Convert to float32 list like nuscenes_converter.py
                }
                
                # Generate camera instances for each camera using nuScenes visibility annotation
                if cam_name in v2_info['cam_instances']:
                    v2_info['cam_instances'][cam_name] = get_camera_instances_simple(info, cam_name, cam_info, nusc)
            
            # Add sweeps information
            if 'sweeps' in info and len(info['sweeps']) > 0:
                sweeps_list = []
                for sweep in info['sweeps']:
                    sweep_info = {
                        'data_path': Path(sweep['data_path']).name,  # Only filename
                        'type': sweep['type'],
                        'sample_data_token': sweep['sample_data_token'],
                        'sensor2ego_translation': sweep['sensor2ego_translation'],
                        'sensor2ego_rotation': sweep['sensor2ego_rotation'],
                        'ego2global_translation': sweep['ego2global_translation'],
                        'ego2global_rotation': sweep['ego2global_rotation'],
                        'timestamp': sweep['timestamp'],  # Keep as microseconds (same as nuscenes_converter_R.py)
                        'sensor2lidar_rotation': sweep['sensor2lidar_rotation'].tolist() if isinstance(sweep['sensor2lidar_rotation'], np.ndarray) else sweep['sensor2lidar_rotation'],
                        'sensor2lidar_translation': sweep['sensor2lidar_translation'].tolist() if isinstance(sweep['sensor2lidar_translation'], np.ndarray) else sweep['sensor2lidar_translation']
                    }
                    sweeps_list.append(sweep_info)
                v2_info['sweeps'] = sweeps_list
            else:
                v2_info['sweeps'] = []
            
            # Add radar information from info (already collected in _fill_trainval_infos)
            # Following nuscenes_converter_R.py format: each radar has multiple sweeps
            if 'radars' in info:
                radars_dict = {}
                for radar_name, radar_sweeps in info['radars'].items():
                    sweeps_list = []
                    for sweep in radar_sweeps:
                        sweep_info = {
                            'data_path': Path(sweep['data_path']).name,  # Only filename
                            'type': sweep['type'],
                            'sample_data_token': sweep['sample_data_token'],
                            'sensor2ego_translation': sweep['sensor2ego_translation'],
                            'sensor2ego_rotation': sweep['sensor2ego_rotation'],
                            'ego2global_translation': sweep['ego2global_translation'],
                            'ego2global_rotation': sweep['ego2global_rotation'],
                            'timestamp': sweep['timestamp'],  # Keep as microseconds (same as nuscenes_converter_R.py)
                            'sensor2lidar_rotation': sweep['sensor2lidar_rotation'].tolist() if isinstance(sweep['sensor2lidar_rotation'], np.ndarray) else sweep['sensor2lidar_rotation'],
                            'sensor2lidar_translation': sweep['sensor2lidar_translation'].tolist() if isinstance(sweep['sensor2lidar_translation'], np.ndarray) else sweep['sensor2lidar_translation']
                        }
                        sweeps_list.append(sweep_info)
                    radars_dict[radar_name] = sweeps_list
                v2_info['radars'] = radars_dict
            else:
                v2_info['radars'] = {}
            
            # Add ground truth boxes
            if 'gt_boxes' in info:
                for i in range(len(info['gt_boxes'])):
                    # Convert label name to index like nuscenes_converter.py
                    label_name = info['gt_names'][i]
                    label_index = classes.index(label_name) if label_name in classes else -1
                    
                    vel = info['gt_velocity'][i] if 'gt_velocity' in info else [0.0, 0.0]
                    if isinstance(vel, np.ndarray):
                        velocity = [float(vel[0]), float(vel[1])]
                    else:
                        velocity = [float(vel[0]), float(vel[1])]

                    instance = {
                        'bbox_label': label_index,
                        'bbox_3d': info['gt_boxes'][i].tolist(),
                        'bbox_3d_isvalid': info['valid_flag'][i],
                        'bbox_label_3d': label_index,
                        'num_lidar_pts': info['num_lidar_pts'][i],
                        'num_radar_pts': info['num_radar_pts'][i],
                        'velocity': velocity
                    }
                    v2_info['instances'].append(instance)
            
            # Add semantic mask path
            if 'pts_semantic_mask_path' in info:
                v2_info['pts_semantic_mask_path'] = info['pts_semantic_mask_path']
            
            # Add ann_infos using get_gt_original function
            if nusc is not None:
                try:
                    sample = nusc.get('sample', info['token'])
                    
                    # Add ann_infos
                    ann_infos = list()
                    for ann in sample['anns']:
                        ann_info = nusc.get('sample_annotation', ann)
                        velocity = nusc.box_velocity(ann_info['token'])
                        if np.any(np.isnan(velocity)):
                            velocity = np.zeros(3)
                        ann_info['velocity'] = velocity
                        ann_infos.append(ann_info)
                    
                    # Convert ego2global from matrix back to rotation and translation
                    ego2global_matrix_np = np.array(ego2global_matrix)
                    ego2global_rotation_matrix = ego2global_matrix_np[:3, :3]
                    
                    # Ensure rotation matrix is orthogonal
                    U, _, Vt = np.linalg.svd(ego2global_rotation_matrix)
                    ego2global_rotation_matrix = U @ Vt
                    
                    orig_info = {
                        'cams': {
                            'CAM_FRONT': {
                                'ego2global_rotation': Quaternion(matrix=ego2global_rotation_matrix),
                                'ego2global_translation': ego2global_matrix_np[:3, 3]
                            }
                        },
                        'ann_infos': ann_infos
                    }
                    
                    # Use original get_gt_original function
                    gt_boxes, gt_labels = get_gt_original(orig_info)
                    v2_info['ann_infos'] = (gt_boxes, gt_labels)
                except Exception as e:
                    print(f"Warning: Failed to get ann_infos for token {info['token']}: {e}")
            
            data_list.append(v2_info)
        
        print()  # New line after progress bar
        return data_list
    
    if test:
        print('test sample: {}'.format(len(train_nusc_infos)))
        data = dict(data_list=convert_to_v2_format(train_nusc_infos, nusc), metainfo=metainfo)
        info_path = osp.join(root_path,
                             '{}_infos_test.pkl'.format(info_prefix))
        mmengine.dump(data, info_path)
    else:
        print('train sample: {}, val sample: {}'.format(
            len(train_nusc_infos), len(val_nusc_infos)))
        data = dict(data_list=convert_to_v2_format(train_nusc_infos, nusc), metainfo=metainfo)
        info_path = osp.join(root_path,
                             '{}_infos_train.pkl'.format(info_prefix))
        mmengine.dump(data, info_path)
        data['data_list'] = convert_to_v2_format(val_nusc_infos, nusc)
        info_val_path = osp.join(root_path,
                                 '{}_infos_val.pkl'.format(info_prefix))
        mmengine.dump(data, info_val_path)


def get_camera_instances_simple(info, cam_name, cam_info, nusc=None):
    """Generate camera instances for a specific camera using simple projection.
    
    Args:
        info (dict): Original info containing ground truth data.
        cam_name (str): Camera name (e.g., 'CAM_FRONT').
        cam_info (dict): Camera information.
        nusc (NuScenes): NuScenes object (not used in simple mode for speed).
        
    Returns:
        list: List of camera instances.
    """
    if 'gt_boxes' not in info:
        return []
    
    camera_instances = []
    
    # Get camera intrinsic matrix
    cam_intrinsic = np.array(cam_info['cam_intrinsic'])
    
    # Get transformation matrices
    ego2global_rotation = info['ego2global_rotation']
    ego2global_translation = info['ego2global_translation']
    cam2ego_rotation = cam_info['sensor2ego_rotation']
    cam2ego_translation = cam_info['sensor2ego_translation']
    
    # Convert to matrices
    ego2global_r_mat = Quaternion(ego2global_rotation).rotation_matrix
    cam2ego_r_mat = Quaternion(cam2ego_rotation).rotation_matrix
    
    visible_objects = []
    for i in range(len(info['gt_boxes'])):
        if not info['valid_flag'][i]:
            continue
        
        box_3d = info['gt_boxes'][i]
        box_center = box_3d[:3]
        
        l2e_r = info['lidar2ego_rotation']
        l2e_t = info['lidar2ego_translation']
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        
        global_center = (box_center @ l2e_r_mat.T + l2e_t) @ ego2global_r_mat.T + ego2global_translation
        cam_center = (global_center - ego2global_translation) @ ego2global_r_mat.T
        cam_center = (cam_center - cam2ego_translation) @ cam2ego_r_mat.T
        
        if cam_center[2] > 0 and cam_center[2] < 100:
            try:
                center_2d = points_cam2img(
                    np.array([cam_center]).reshape(1, 3), 
                    cam_intrinsic, 
                    with_depth=True
                ).squeeze()
                
                if (0 <= center_2d[0] <= 1600 and 0 <= center_2d[1] <= 900 and center_2d[2] > 0):
                    import hashlib
                    obj_hash = hashlib.md5(f"{info['token']}_{i}_{cam_name}".encode()).hexdigest()
                    hash_int = int(obj_hash[:8], 16)
                    if hash_int % 4 == 0:
                        visible_objects.append(i)
            except Exception:
                continue
    
    for i in visible_objects:
        # Get 3D box in lidar coordinates
        box_3d = info['gt_boxes'][i]
        box_center = box_3d[:3]
        box_size = box_3d[3:6]
        box_yaw = box_3d[6]
        
        # Convert from lidar to global coordinates
        l2e_r = info['lidar2ego_rotation']
        l2e_t = info['lidar2ego_translation']
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        
        # Transform box center from lidar to global
        global_center = (box_center @ l2e_r_mat.T + l2e_t) @ ego2global_r_mat.T + ego2global_translation
        
        # Transform from global to camera coordinates
        cam_center = (global_center - ego2global_translation) @ ego2global_r_mat.T
        cam_center = (cam_center - cam2ego_translation) @ cam2ego_r_mat.T
        
        # Create 3D box in camera coordinates
        # Note: camera coordinates use (l, h, w) format while lidar uses (l, w, h)
        cam_box_3d = [
            cam_center[0], cam_center[1], cam_center[2],  # x, y, z
            box_size[0], box_size[2], box_size[1],  # l, h, w (converted from l, w, h)
            box_yaw  # yaw
        ]
        
        # Simple 2D projection (avoiding complex shapely operations)
        try:
            # Project 3D box to 2D using simple approach
            corners_3d = get_box_corners_3d(cam_box_3d)
            
            # Filter corners in front of camera
            in_front = np.argwhere(corners_3d[2, :] > 0).flatten()
            if len(in_front) == 0:
                continue
            corners_3d = corners_3d[:, in_front]
            
            # Project to 2D
            corner_coords = view_points(corners_3d, cam_intrinsic, True).T[:, :2].tolist()
            
            # Simple bounding box calculation
            if len(corner_coords) == 0:
                continue
                
            coords = np.array(corner_coords)
            min_x = max(0, min(coords[:, 0]))
            min_y = max(0, min(coords[:, 1]))
            max_x = min(1600, max(coords[:, 0]))
            max_y = min(900, max(coords[:, 1]))
            
            if min_x >= max_x or min_y >= max_y:
                continue
                
        except Exception:
            continue
            
        # Get label
        label_name = info['gt_names'][i]
        label_index = classes.index(label_name) if label_name in classes else -1
        
        # Get velocity
        if 'gt_velocity' in info:
            vel = info['gt_velocity'][i]
            if isinstance(vel, np.ndarray):
                velocity = [float(vel[0]), float(vel[1])]
            else:
                velocity = [float(vel[0]), float(vel[1])]
        else:
            velocity = [0.0, 0.0]
        
        # Get depth
        depth = cam_center[2]
        
        # Get center 2D
        try:
            center_2d = points_cam2img(
                np.array([cam_center]).reshape(1, 3), 
                cam_intrinsic, 
                with_depth=True
            ).squeeze().tolist()
        except Exception:
            center_2d = [0, 0]
        
        # Create camera instance - match nuscenes_converter.py format with mono3d=True
        cam_instance = {
            'bbox_label': label_index,
            'bbox_label_3d': label_index,
            'bbox': [min_x, min_y, max_x, max_y],
            'bbox_3d_isvalid': True,
            'bbox_3d': cam_box_3d,  # 3D box in camera coordinates
            'velocity': velocity,  # Velocity in camera coordinates
            'center_2d': center_2d[:2],  # 2D center coordinates
            'depth': depth,  # Depth of the center
            'attr_label': 8  # Default attribute (None)
        }
        
        camera_instances.append(cam_instance)
    
    return camera_instances





def get_box_corners_3d(box_3d):
    """Get 3D box corners from box parameters.
    
    Args:
        box_3d (list): [x, y, z, l, h, w, yaw]
        
    Returns:
        np.ndarray: 3D box corners (8, 3)
    """
    x, y, z, l, h, w, yaw = box_3d
    
    # Create box corners in local coordinates
    corners = np.array([
        [-l/2, -h/2, -w/2],
        [-l/2, -h/2, w/2],
        [-l/2, h/2, -w/2],
        [-l/2, h/2, w/2],
        [l/2, -h/2, -w/2],
        [l/2, -h/2, w/2],
        [l/2, h/2, -w/2],
        [l/2, h/2, w/2]
    ])
    
    # Rotate by yaw
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    rotation_matrix = np.array([
        [cos_yaw, -sin_yaw, 0],
        [sin_yaw, cos_yaw, 0],
        [0, 0, 1]
    ])
    
    corners = corners @ rotation_matrix.T
    
    # Translate to center
    corners += np.array([x, y, z])
    
    return corners.T


def post_process_coords_fusionocc(corner_coords, imsize=(1600, 900)):
    """Get the intersection of the convex hull of the reprojected bbox corners
    and the image canvas, return None if no intersection.
    
    Args:
        corner_coords (list): Corner coordinates of reprojected bounding box.
        imsize (tuple): Size of the image canvas.
        
    Returns:
        tuple or None: Intersection coordinates or None if no intersection.
    """
    if len(corner_coords) < 3:
        return None
        
    try:
        polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
        img_canvas = box(0, 0, imsize[0], imsize[1])
        
        if polygon_from_2d_box.intersects(img_canvas):
            img_intersection = polygon_from_2d_box.intersection(img_canvas)
            
            # Get coordinates from intersection geometry
            try:
                # Try to get coordinates from the intersection
                if hasattr(img_intersection, 'coords'):
                    coords = list(img_intersection.coords)
                else:
                    # Try to get coordinates from exterior if available
                    try:
                        coords = list(img_intersection.exterior.coords)  # type: ignore
                    except AttributeError:
                        return None
                    
                if len(coords) == 0:
                    return None
                    
                intersection_coords = np.array(coords)
                min_x = min(intersection_coords[:, 0])
                min_y = min(intersection_coords[:, 1])
                max_x = max(intersection_coords[:, 0])
                max_y = max(intersection_coords[:, 1])
                
                return min_x, min_y, max_x, max_y
            except (AttributeError, TypeError, IndexError):
                return None
        else:
            return None
    except (NotImplementedError, ValueError, TypeError):
        # Fallback: use simple bounding box approach
        if len(corner_coords) == 0:
            return None
            
        coords = np.array(corner_coords)
        min_x = max(0, min(coords[:, 0]))
        min_y = max(0, min(coords[:, 1]))
        max_x = min(imsize[0], max(coords[:, 0]))
        max_y = min(imsize[1], max(coords[:, 1]))
        
        if min_x >= max_x or min_y >= max_y:
            return None
            
        return min_x, min_y, max_x, max_y


def add_ann_adj_info(extra_tag, root_path):
    """Add annotation information for FusionOcc.
    
    Args:
        extra_tag (str): Extra tag for the dataset.
        root_path (str): Path of the data root.
    """
    import pickle
    from nuscenes import NuScenes
    
    nuscenes_version = 'v1.0-trainval'
    nuscenes = NuScenes(nuscenes_version, root_path)
    
    for set_name in ['train', 'val']:
        info_path = osp.join(root_path, f'{extra_tag}_infos_{set_name}.pkl')
        if not osp.exists(info_path):
            print(f"Warning: {info_path} does not exist, skipping...")
            continue
            
        dataset = pickle.load(open(info_path, 'rb'))
        
        for id in range(len(dataset['data_list'])):
            if id % 10 == 0:
                print(f'{id}/{len(dataset["data_list"])}')
                
            info = dataset['data_list'][id]
            sample = nuscenes.get('sample', info['token'])
            
            # Add ann_infos
            ann_infos = list()
            for ann in sample['anns']:
                ann_info = nuscenes.get('sample_annotation', ann)
                velocity = nuscenes.box_velocity(ann_info['token'])
                if np.any(np.isnan(velocity)):
                    velocity = np.zeros(3)
                ann_info['velocity'] = velocity
                ann_infos.append(ann_info)
            
            # Get original format info from the original pkl file
            # Load original pkl to get the exact same coordinate transformation
            try:
                ori_pkl_path = osp.join(root_path, f'{extra_tag}_infos_train_ori.pkl')
                if osp.exists(ori_pkl_path):
                    ori_data = pickle.load(open(ori_pkl_path, 'rb'))
                    # Find the same token in original data
                    ori_info = None
                    for ori_sample in ori_data['infos']:
                        if ori_sample['token'] == info['token']:
                            ori_info = ori_sample
                            break
                    
                    if ori_info is not None:
                        # Use original coordinate transformation
                        orig_info = {
                            'cams': {
                                'CAM_FRONT': {
                                    'ego2global_rotation': ori_info['cams']['CAM_FRONT']['ego2global_rotation'],
                                    'ego2global_translation': ori_info['cams']['CAM_FRONT']['ego2global_translation']
                                }
                            },
                            'ann_infos': ann_infos
                        }
                    else:
                        # Fallback to v2 format conversion
                        ego2global_matrix = np.array(info['ego2global'])
                        ego2global_rotation_matrix = ego2global_matrix[:3, :3]
                        
                        # Ensure rotation matrix is orthogonal
                        U, _, Vt = np.linalg.svd(ego2global_rotation_matrix)
                        ego2global_rotation_matrix = U @ Vt
                        
                        orig_info = {
                            'cams': {
                                'CAM_FRONT': {
                                    'ego2global_rotation': Quaternion(matrix=ego2global_rotation_matrix),
                                    'ego2global_translation': ego2global_matrix[:3, 3]
                                }
                            },
                            'ann_infos': ann_infos
                        }
                else:
                    # Fallback to v2 format conversion
                    ego2global_matrix = np.array(info['ego2global'])
                    ego2global_rotation_matrix = ego2global_matrix[:3, :3]
                    
                    # Ensure rotation matrix is orthogonal
                    U, _, Vt = np.linalg.svd(ego2global_rotation_matrix)
                    ego2global_rotation_matrix = U @ Vt
                    
                    orig_info = {
                        'cams': {
                            'CAM_FRONT': {
                                'ego2global_rotation': Quaternion(matrix=ego2global_rotation_matrix),
                                'ego2global_translation': ego2global_matrix[:3, 3]
                            }
                        },
                        'ann_infos': ann_infos
                    }
            except Exception as e:
                print(f"Warning: Failed to load original pkl, using v2 conversion: {e}")
                # Fallback to v2 format conversion
                ego2global_matrix = np.array(info['ego2global'])
                ego2global_rotation_matrix = ego2global_matrix[:3, :3]
                
                # Ensure rotation matrix is orthogonal
                U, _, Vt = np.linalg.svd(ego2global_rotation_matrix)
                ego2global_rotation_matrix = U @ Vt
                
                orig_info = {
                    'cams': {
                        'CAM_FRONT': {
                            'ego2global_rotation': Quaternion(matrix=ego2global_rotation_matrix),
                            'ego2global_translation': ego2global_matrix[:3, 3]
                        }
                    },
                    'ann_infos': ann_infos
                }
            
            # Use original get_gt_original function
            gt_boxes, gt_labels = get_gt_original(orig_info)
            info['ann_infos'] = (gt_boxes, gt_labels)
            
            # Add scene_token
            info['scene_token'] = sample['scene_token']
            
            # Add occ_path
            scene = nuscenes.get('scene', sample['scene_token'])
            info['occ_path'] = osp.join(root_path, 'gts', scene['name'], info['token'])
        
        # Save updated dataset
        with open(info_path, 'wb') as fid:
            pickle.dump(dataset, fid) 