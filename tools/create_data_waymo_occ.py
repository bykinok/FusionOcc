#!/usr/bin/env python3
"""Build an STCOcc-compatible (nuScenes-info-schema-mimicking) pkl for
Occ3D-Waymo, restricted to whichever scenes are locally available.

Why mimic the nuScenes info schema instead of writing a new Dataset class:
STCOcc's existing `NuScenesDatasetOccpancy` + `STCOccPrepareImageInputs` only
care about specific dict field NAMES (cams[...]['sensor2ego_translation'] etc,
top-level 'token'/'scene_token'/'prev'/'can_bus'/'occ_path'), not about the
dataset actually being nuScenes. If we populate those exact fields correctly
from Waymo sources, the existing architecture-unchanged pipeline just works --
this is the "data adapter, not architecture change" principle from
WAYMO_EXTENSION_REQUIREMENTS_KO.md section 2.1/7.

Calibration math (verified empirically this session via LiDAR->image
projection on real data, see research_waymo/dataset_schema.md / camera_temporal_audit.md):
  - Waymo's `cam_infos.pkl` per-camera `intrinsics` (4x4) is a composite
    "sensor2img" matrix that assumes a VEHICLE-style camera frame
    (X=forward/depth, Y=left, Z=up), NOT the standard CV camera convention
    (x=right, y=down, z=forward) that nuScenes' `cam_intrinsic` (plain 3x3 K)
    and `sensor2ego_rotation` assume.
  - M below is the fixed rotation that converts CV-frame coords into that
    vehicle-style frame: X=z_cv, Y=-x_cv, Z=-y_cv. Decomposing
    `intrinsics = K_cv_applied_after(M)` yields a clean pinhole
    K=[[fx,0,cx],[0,fy,cy],[0,0,1]] with fx=fy=-intrinsics[0,1],
    cx=intrinsics[0,0], cy=intrinsics[1,0] -- cross-verified this session
    against the independent KITTI-style `calib.P0` field (exact match).
  - camera pose index <-> actual image folder is SWAPPED for indices 2/3
    (a documented upstream bug, see CVT-Occ's waymo_temporal_zlt.py comment
    "pose info idx dismatch the image data file"). Verified empirically this
    session via LiDAR point in-bounds-rate per camera: pose index 2's
    calibration only makes geometric sense paired with the image_3 folder,
    and pose index 3 with image_2 (100% in-bounds for a tight forward cone
    vs 80.7% for the naive non-swapped pairing).
  - lidar2ego = sensor2ego_cam0_cv @ Tr_velo_to_cam (Tr_velo_to_cam taken
    from the KITTI-style waymo_infos_{train,val}.pkl `calib` block, which is
    directly LiDAR->camera-0-CV-frame -- verified empirically: with this
    composition, a tight forward LiDAR cone projects 100% inside the actual
    image bounds).

Scope restriction: only scenes whose Occ3D-Waymo voxel04 GT directory exists
locally are included (241/798 train, 202/202 val as of this session -- see
research_waymo/remaining_issues.md item #8). This is NOT a full-dataset info
file.
"""
import argparse
import os
import pickle

import mmengine
import numpy as np
from pyquaternion import Quaternion

# vehicle-style camera frame (X=depth/forward, Y=left, Z=up) <- CV camera frame
# (x=right, y=down, z=forward): X=z_cv, Y=-x_cv, Z=-y_cv
M_CV_TO_VEH = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float64)

# pose index (key into cam_infos.pkl[scene][frame][idx]) -> (STCOcc camera
# name, ACTUAL image-folder index). Indices 2/3 are intentionally swapped --
# see module docstring.
POSE_IDX_TO_CAM = {
    0: ('CAM_FRONT', 0),
    1: ('CAM_FRONT_LEFT', 1),
    2: ('CAM_SIDE_LEFT', 3),
    3: ('CAM_FRONT_RIGHT', 2),
    4: ('CAM_SIDE_RIGHT', 4),
}
CAM_NAMES = [POSE_IDX_TO_CAM[i][0] for i in range(5)]


def decompose_cam(cam):
    """cam_infos.pkl per-camera dict -> (K (3,3), sensor2ego_cv (4,4))."""
    sensor2ego = np.asarray(cam['sensor2ego'], dtype=np.float64)
    intrinsics = np.asarray(cam['intrinsics'], dtype=np.float64)
    fx = fy = -intrinsics[0, 1]
    cx, cy = intrinsics[0, 0], intrinsics[1, 0]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    R_cv = sensor2ego[:3, :3] @ M_CV_TO_VEH
    T = np.eye(4)
    T[:3, :3] = R_cv
    T[:3, 3] = sensor2ego[:3, 3]
    return K, T


def mat_to_quat_wxyz(R):
    return list(Quaternion(matrix=R, atol=1e-6, rtol=1e-6).elements)


def build_entries(split, waymo_infos_path, cam_infos_path, kitti_image_root,
                   occ_root, available_scenes):
    with open(waymo_infos_path, 'rb') as f:
        kitti_infos = pickle.load(f)
    with open(cam_infos_path, 'rb') as f:
        pose_all = pickle.load(f)

    # index KITTI-style infos by (scene_idx, frame_idx) for calib/timestamp lookup
    kitti_by_sf = {}
    for e in kitti_infos:
        image_idx = e['image']['image_idx']
        s = image_idx % 1000000 // 1000
        fr = image_idx % 1000000 % 1000
        kitti_by_sf[(s, fr)] = e

    entries = []
    n_missing_kitti = 0
    for scene_idx in sorted(available_scenes):
        if scene_idx not in pose_all:
            continue
        frames = pose_all[scene_idx]
        for frame_idx in sorted(frames.keys()):
            key = (scene_idx, frame_idx)
            if key not in kitti_by_sf:
                n_missing_kitti += 1
                continue
            kinfo = kitti_by_sf[key]
            Tr = np.asarray(kinfo['calib']['Tr_velo_to_cam'], dtype=np.float64)
            timestamp = kinfo.get('timestamp', 0)

            cam0 = frames[frame_idx][0]
            K0, s2e0_cv = decompose_cam(cam0)
            lidar2ego = s2e0_cv @ Tr
            ego2global = np.asarray(cam0['ego2global'], dtype=np.float64)

            cams = {}
            for pose_idx in range(5):
                cam_name, img_folder = POSE_IDX_TO_CAM[pose_idx]
                cam = frames[frame_idx][pose_idx]
                K, s2e_cv = decompose_cam(cam)
                sensor2lidar = np.linalg.inv(lidar2ego) @ s2e_cv
                img_path = os.path.join(
                    kitti_image_root, 'training', f'image_{img_folder}',
                    f'{scene_idx * 1000 + frame_idx:07d}.jpg')
                cams[cam_name] = dict(
                    data_path=img_path,
                    type=cam_name,
                    sample_data_token=f'waymo_{split}_{scene_idx:03d}_{frame_idx:03d}_{cam_name}',
                    sensor2ego_translation=s2e_cv[:3, 3].tolist(),
                    sensor2ego_rotation=mat_to_quat_wxyz(s2e_cv[:3, :3]),
                    ego2global_translation=ego2global[:3, 3].tolist(),
                    ego2global_rotation=mat_to_quat_wxyz(ego2global[:3, :3]),
                    timestamp=timestamp,
                    sensor2lidar_rotation=sensor2lidar[:3, :3],
                    sensor2lidar_translation=sensor2lidar[:3, 3],
                    cam_intrinsic=K,
                )

            token = f'waymo_{split}_{scene_idx:03d}_{frame_idx:03d}'
            scene_token = f'waymo_{split}_{scene_idx:03d}'
            prev = '' if frame_idx == 0 else f'waymo_{split}_{scene_idx:03d}_{frame_idx - 1:03d}'
            occ_prefix = os.path.join(occ_root, split, f'{scene_idx:03d}', f'{frame_idx:03d}')

            lidar_points_path = os.path.join(
                kitti_image_root, 'training', 'velodyne',
                f'{scene_idx * 1000 + frame_idx:07d}.bin')

            entries.append(dict(
                # STCOcc still uses LiDAR points to build the depth supervision
                # map consumed by the forward projection branch.
                lidar_path=lidar_points_path,
                token=token,
                can_bus=np.zeros(18),  # Waymo has no can_bus signal; STCOcc only overwrites [:3]/[3:7]/[-2:] from ego pose, see nuscenes_dataset_occ.py
                sweeps=[],
                cams=cams,
                lidar2ego_translation=lidar2ego[:3, 3].tolist(),
                lidar2ego_rotation=mat_to_quat_wxyz(lidar2ego[:3, :3]),
                ego2global_translation=ego2global[:3, 3].tolist(),
                ego2global_rotation=mat_to_quat_wxyz(ego2global[:3, :3]),
                timestamp=timestamp,
                gt_boxes=np.zeros((0, 9)),
                gt_names=np.array([], dtype='<U1'),
                gt_velocity=np.zeros((0, 2)),
                num_lidar_pts=np.zeros(0, dtype=np.int64),
                num_radar_pts=np.zeros(0, dtype=np.int64),
                valid_flag=np.zeros(0, dtype=bool),
                pts_semantic_mask_path=None,
                ann_infos=([], []),
                scene_token=scene_token,
                prev=prev,
                occ_path=occ_prefix,
                # Waymo-specific extras (not read by the nuScenes-schema
                # pipeline, kept for the Waymo GT loader / bookkeeping):
                waymo_scene_idx=scene_idx,
                waymo_frame_idx=frame_idx,
                lidar_points_path=lidar_points_path,
                velo_to_cam0=Tr,
            ))
    return entries, n_missing_kitti


def list_available_scenes(occ_root, split):
    d = os.path.join(occ_root, split)
    if not os.path.isdir(d):
        return set()
    return {int(name) for name in os.listdir(d) if name.isdigit()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--info-root', default='data/waymo/occ3d',
                     help='dir containing waymo_infos_{train,val}.pkl and cam_infos*.pkl')
    ap.add_argument('--occ-root', default='data/waymo/occ3d/voxel04',
                     help='dir containing training/validation voxel04 scene folders')
    ap.add_argument('--kitti-image-root', default='data/waymo/kitti_format')
    ap.add_argument('--out-dir', default='data/waymo')
    args = ap.parse_args()

    for split, waymo_infos_name, cam_infos_name in [
        ('training', 'waymo_infos_train.pkl', 'cam_infos.pkl'),
        ('validation', 'waymo_infos_val.pkl', 'cam_infos_vali.pkl'),
    ]:
        available = list_available_scenes(args.occ_root, split)
        print(f'[{split}] locally available scenes with voxel04 GT: {len(available)}')
        entries, n_missing = build_entries(
            split,
            os.path.join(args.info_root, waymo_infos_name),
            os.path.join(args.info_root, cam_infos_name),
            args.kitti_image_root,
            args.occ_root,
            available)
        print(f'[{split}] built {len(entries)} sample entries '
              f'({n_missing} frames had voxel GT but no matching KITTI-pkl calib entry, skipped)')
        out = dict(infos=entries, metadata=dict(
            version='occ3d-waymo-v1', cam_names=CAM_NAMES,
            available_scenes=sorted(available)))
        out_name = 'stcocc-waymo_infos_train.pkl' if split == 'training' else 'stcocc-waymo_infos_val.pkl'
        out_path = os.path.join(args.out_dir, out_name)
        mmengine.dump(out, out_path)
        print(f'[{split}] wrote {out_path}')


if __name__ == '__main__':
    main()
