import os
from multiprocessing import Pool
import pickle
import traceback

import numpy as np
from PIL import Image
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
import copy

# https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/nuscenes.py#L834
def map_pointcloud_to_image(
    pc,
    im,
    lidar2ego_translation,
    lidar2ego_rotation,
    ego2global_translation,
    ego2global_rotation,
    sensor2ego_translation, 
    sensor2ego_rotation,
    cam_ego2global_translation,
    cam_ego2global_rotation,
    cam_intrinsic,
    min_dist: float = 0.0,
):

    # Points live in the point sensor frame. So they need to be
    # transformed via global to the image plane.
    # First step: transform the pointcloud to the ego vehicle
    # frame for the timestamp of the sweep.

    pc = LidarPointCloud(pc.T)
    pc.rotate(Quaternion(lidar2ego_rotation).rotation_matrix)
    pc.translate(np.array(lidar2ego_translation))

    # Second step: transform from ego to the global frame.
    pc.rotate(Quaternion(ego2global_rotation).rotation_matrix)
    pc.translate(np.array(ego2global_translation))

    # Third step: transform from global into the ego vehicle
    # frame for the timestamp of the image.
    pc.translate(-np.array(cam_ego2global_translation))
    pc.rotate(Quaternion(cam_ego2global_rotation).rotation_matrix.T)

    # Fourth step: transform from ego into the camera.
    pc.translate(-np.array(sensor2ego_translation))
    pc.rotate(Quaternion(sensor2ego_rotation).rotation_matrix.T)

    # Fifth step: actually take a "picture" of the point cloud.
    # Grab the depths (camera frame z axis points away from the camera).
    depths = pc.points[2, :]
    coloring = depths

    # Take the actual picture (matrix multiplication with camera-matrix
    # + renormalization).
    points = view_points(pc.points[:3, :],
                         cam_intrinsic,
                         normalize=True)

    # Remove points that are either outside or behind the camera.
    # Leave a margin of 1 pixel for aesthetic reasons. Also make
    # sure points are at least 1m in front of the camera to avoid
    # seeing the lidar points on the camera casing for non-keyframes
    # which are slightly out of sync.
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > min_dist)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < im.shape[1] - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < im.shape[0] - 1)
    points = points[:, mask]
    coloring = coloring[mask]

    return points, coloring


data_root = './data/nuscenes'
info_path_train = './data/nuscenes/nuscenes_occ_infos_train.pkl'
info_path_val = './data/nuscenes/nuscenes_occ_infos_val.pkl'

# data3d_nusc = NuscMVDetData()

lidar_key = 'LIDAR_TOP'
cam_keys = [
    'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT',
    'CAM_BACK', 'CAM_BACK_LEFT'
]


def worker(info):
    try:
        lidar_path = info['lidar_path']
        if not os.path.exists(lidar_path):
            print(f"LiDAR 파일이 존재하지 않습니다: {lidar_path}")
            return False
            
        points = np.fromfile(lidar_path,
                             dtype=np.float32,
                             count=-1).reshape(-1, 5)[..., :4]
        
        lidar2ego_translation = info['lidar2ego_translation']
        lidar2ego_rotation = info['lidar2ego_rotation']
        ego2global_translation = info['ego2global_translation']
        ego2global_rotation = info['ego2global_rotation']
        
        processed_cams = 0
        for i, cam_key in enumerate(cam_keys):
            try:
                sensor2ego_translation = info['cams'][cam_key]['sensor2ego_translation']
                sensor2ego_rotation = info['cams'][cam_key]['sensor2ego_rotation']
                cam_ego2global_translation = info['cams'][cam_key]['ego2global_translation']
                cam_ego2global_rotation = info['cams'][cam_key]['ego2global_rotation']
                cam_intrinsic = info['cams'][cam_key]['cam_intrinsic']
                
                img_path = info['cams'][cam_key]['data_path']
                if not os.path.exists(img_path):
                    print(f"이미지 파일이 존재하지 않습니다: {img_path}")
                    continue
                    
                img = Image.open(img_path)
                img = np.array(img)
                pts_img, depth = map_pointcloud_to_image(
                    points.copy(), img, 
                    copy.deepcopy(lidar2ego_translation), 
                    copy.deepcopy(lidar2ego_rotation), 
                    copy.deepcopy(ego2global_translation),
                    copy.deepcopy(ego2global_rotation),
                    copy.deepcopy(sensor2ego_translation), 
                    copy.deepcopy(sensor2ego_rotation), 
                    copy.deepcopy(cam_ego2global_translation), 
                    copy.deepcopy(cam_ego2global_rotation),
                    copy.deepcopy(cam_intrinsic))
                
                file_name = os.path.split(info['cams'][cam_key]['data_path'])[-1]
                output_path = os.path.join('./data', 'depth_gt', f'{file_name}.bin')
                np.concatenate([pts_img[:2, :].T, depth[:, None]],
                               axis=1).astype(np.float32).flatten().tofile(output_path)
                processed_cams += 1
                
            except Exception as e:
                print(f"카메라 {cam_key} 처리 중 오류: {e}")
                continue
        
        print(f"토큰 {info.get('token', 'unknown')}에서 {processed_cams}/{len(cam_keys)}개 카메라 처리 완료")
        return True
        
    except Exception as e:
        print(f"Worker 함수에서 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False

def error_callback(error):
    print(f"작업 처리 중 오류 발생: {error}")

if __name__ == '__main__':
    print("Depth GT 생성을 시작합니다...")
    
    # 출력 디렉토리 생성
    output_dir = os.path.join('./data', 'depth_gt')
    os.makedirs(output_dir, exist_ok=True)
    print(f"출력 디렉토리 생성됨: {output_dir}")
    
    # 훈련 데이터 처리
    print(f"\n훈련 데이터 파일 로딩 중: {info_path_train}")
    if not os.path.exists(info_path_train):
        print(f"오류: 훈련 데이터 파일이 존재하지 않습니다: {info_path_train}")
        exit(1)
        
    with open(info_path_train, 'rb') as f:
        train_data = pickle.load(f)
        infos = train_data['infos']
    print(f"훈련 데이터 로딩 완료: {len(infos)}개 샘플")
    
    po = Pool(12)
    results = []
    for i, info in enumerate(infos):
        result = po.apply_async(func=worker, args=(info, ), error_callback=error_callback)
        results.append(result)
        if (i + 1) % 100 == 0:
            print(f"훈련 데이터: {i + 1}/{len(infos)} 작업 제출됨")
    
    po.close()
    print("훈련 데이터 처리 중... (완료될 때까지 기다리세요)")
    po.join()
    
    # 결과 확인
    success_count = sum(1 for r in results if r.get())
    print(f"훈련 데이터 처리 완료: {success_count}/{len(infos)} 성공")
    
    # 검증 데이터 처리
    print(f"\n검증 데이터 파일 로딩 중: {info_path_val}")
    if not os.path.exists(info_path_val):
        print(f"오류: 검증 데이터 파일이 존재하지 않습니다: {info_path_val}")
        exit(1)
        
    with open(info_path_val, 'rb') as f:
        val_data = pickle.load(f)
        infos = val_data['infos']
    print(f"검증 데이터 로딩 완료: {len(infos)}개 샘플")
    
    po2 = Pool(12)
    results2 = []
    for i, info in enumerate(infos):
        result = po2.apply_async(func=worker, args=(info, ), error_callback=error_callback)
        results2.append(result)
        if (i + 1) % 100 == 0:
            print(f"검증 데이터: {i + 1}/{len(infos)} 작업 제출됨")
    
    po2.close()
    print("검증 데이터 처리 중... (완료될 때까지 기다리세요)")
    po2.join()
    
    # 결과 확인
    success_count2 = sum(1 for r in results2 if r.get())
    print(f"검증 데이터 처리 완료: {success_count2}/{len(infos)} 성공")
    
    print(f"\n전체 작업 완료! 출력 파일들이 {output_dir}에 저장되었습니다.")
