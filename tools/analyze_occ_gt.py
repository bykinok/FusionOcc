#!/usr/bin/env python
"""
OCC3D GT 클래스별 갯수 분석 스크립트

Usage:
    python tools/analyze_occ_gt.py --pkl-path <path_to_pkl_file>
    
Example:
    python tools/analyze_occ_gt.py \
        --pkl-path data/nuscenes/nuscenes_infos_train_occ.pkl
"""

import argparse
import os
import pickle
import numpy as np
from tqdm import tqdm
from collections import defaultdict


# OCC3D 클래스 정의 (NuScenes occupancy)
CLASS_NAMES = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk',
    'terrain', 'manmade', 'vegetation', 'free'
]


def load_pkl(pkl_path):
    """pkl 파일 로드"""
    print(f"Loading pkl file: {pkl_path}")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data


def analyze_occ_gt(pkl_data, pkl_dir, max_samples=None):
    """
    OCC GT 데이터의 클래스별 갯수 분석
    
    Args:
        pkl_data: pkl 파일에서 로드한 데이터
        pkl_dir: pkl 파일이 위치한 디렉토리 경로
        max_samples: 분석할 최대 샘플 수 (None이면 전체)
    """
    # pkl 데이터에서 infos 추출
    if isinstance(pkl_data, dict):
        # dict의 키 출력하여 구조 확인
        print(f"pkl 파일의 키: {list(pkl_data.keys())}")
        
        if 'infos' in pkl_data:
            data_infos = pkl_data['infos']
            metadata = pkl_data.get('metadata', {})
            print(f"Metadata: {metadata}")
        elif 'data_list' in pkl_data:
            data_infos = pkl_data['data_list']
            metadata = pkl_data.get('metadata', {})
            print(f"Metadata: {metadata}")
        else:
            # dict 자체가 infos일 수도 있음 - 키 개수가 적으면 dict 전체를 확인
            if len(pkl_data) < 10:
                print(f"pkl_data 샘플 (처음 몇 개): {list(pkl_data.keys())[:5]}")
                raise ValueError(f"'infos' 또는 'data_list' 키를 찾을 수 없습니다. 사용 가능한 키: {list(pkl_data.keys())}")
            else:
                # 키가 많으면 dict의 값들이 각 샘플일 가능성
                data_infos = list(pkl_data.values())
                print(f"dict의 값들을 data_infos로 사용 (총 {len(data_infos)}개)")
    elif isinstance(pkl_data, list):
        data_infos = pkl_data
    else:
        raise ValueError(f"Unexpected pkl format: {type(pkl_data)}")
    
    print(f"Total samples in pkl: {len(data_infos)}")
    
    # 분석할 샘플 수 결정
    if max_samples is not None:
        data_infos = data_infos[:max_samples]
        print(f"Analyzing first {len(data_infos)} samples")
    
    # 클래스별 voxel 갯수 누적
    class_counts_total = defaultdict(int)
    class_counts_camera_mask = defaultdict(int)
    class_counts_lidar_mask = defaultdict(int)
    class_counts_per_sample = []
    samples_with_gt = 0
    samples_without_gt = 0
    
    print("\nAnalyzing OCC GT data...")
    
    # 프로젝트 루트 찾기 (pkl_dir의 상위 2단계)
    # pkl_dir이 /path/to/project/data/nuscenes 형태라면, project_root는 /path/to/project
    project_root = os.path.dirname(os.path.dirname(pkl_dir))
    print(f"프로젝트 루트: {project_root}")
    
    for idx, info in enumerate(tqdm(data_infos)):
        # occ_gt_path 확인 (occ_path 또는 occ_gt_path 키 모두 지원)
        occ_gt_path = info.get('occ_path') or info.get('occ_gt_path')
        if occ_gt_path is None:
            samples_without_gt += 1
            continue
        
        # 경로 처리: ./ 로 시작하거나 상대 경로이면 프로젝트 루트 기준으로 처리
        if occ_gt_path.startswith('./'):
            # ./ 제거하고 프로젝트 루트 기준으로
            occ_gt_path = occ_gt_path[2:]
            full_path = os.path.join(project_root, occ_gt_path)
        elif occ_gt_path.startswith('/'):
            # 절대 경로
            full_path = occ_gt_path
        else:
            # 상대 경로 - 프로젝트 루트 기준으로 시도
            full_path = os.path.join(project_root, occ_gt_path)
            # 존재하지 않으면 pkl_dir 기준으로도 시도
            if not os.path.exists(full_path):
                full_path = os.path.join(pkl_dir, occ_gt_path)
        
        # occ_path가 디렉토리인 경우 labels.npz 파일 추가
        if os.path.isdir(full_path):
            full_path = os.path.join(full_path, 'labels.npz')
        
        # GT 파일 존재 확인
        if not os.path.exists(full_path):
            if idx == 0:  # 첫 번째 샘플에서만 경로 출력
                print(f"\nWarning: GT file not found: {full_path}")
            samples_without_gt += 1
            continue
        
        # GT 로드
        try:
            occ_gt = np.load(full_path)
            gt_semantics = occ_gt['semantics']
            
            # mask 로드 (있는 경우)
            mask_camera = occ_gt.get('mask_camera', None)
            mask_lidar = occ_gt.get('mask_lidar', None)
            
            samples_with_gt += 1
            
            # 1. 원본 클래스별 갯수 계산
            sample_counts = {}
            for class_id in range(len(CLASS_NAMES)):
                count = np.sum(gt_semantics == class_id)
                class_counts_total[class_id] += count
                sample_counts[class_id] = count
            
            # 2. camera_mask 적용 후 클래스별 갯수 계산
            if mask_camera is not None:
                mask_camera_bool = mask_camera.astype(bool)
                gt_semantics_camera = gt_semantics[mask_camera_bool]
                for class_id in range(len(CLASS_NAMES)):
                    count = np.sum(gt_semantics_camera == class_id)
                    class_counts_camera_mask[class_id] += count
            
            # 3. lidar_mask 적용 후 클래스별 갯수 계산
            if mask_lidar is not None:
                mask_lidar_bool = mask_lidar.astype(bool)
                gt_semantics_lidar = gt_semantics[mask_lidar_bool]
                for class_id in range(len(CLASS_NAMES)):
                    count = np.sum(gt_semantics_lidar == class_id)
                    class_counts_lidar_mask[class_id] += count
            
            class_counts_per_sample.append({
                'sample_idx': idx,
                'token': info.get('token', 'unknown'),
                'counts': sample_counts
            })
            
        except Exception as e:
            print(f"\nError loading {full_path}: {e}")
            samples_without_gt += 1
            continue
    
    return {
        'class_counts_total': class_counts_total,
        'class_counts_camera_mask': class_counts_camera_mask,
        'class_counts_lidar_mask': class_counts_lidar_mask,
        'class_counts_per_sample': class_counts_per_sample,
        'samples_with_gt': samples_with_gt,
        'samples_without_gt': samples_without_gt,
        'total_samples': len(data_infos)
    }


def print_statistics(results):
    """분석 결과 출력"""
    class_counts_total = results['class_counts_total']
    class_counts_camera_mask = results['class_counts_camera_mask']
    class_counts_lidar_mask = results['class_counts_lidar_mask']
    samples_with_gt = results['samples_with_gt']
    samples_without_gt = results['samples_without_gt']
    total_samples = results['total_samples']
    
    print("\n" + "="*80)
    print("OCC3D GT 클래스별 통계")
    print("="*80)
    print(f"전체 샘플 수: {total_samples}")
    print(f"GT가 있는 샘플: {samples_with_gt}")
    print(f"GT가 없는 샘플: {samples_without_gt}")
    print("="*80)
    
    # 1. 원본 통계
    total_voxels = sum(class_counts_total.values())
    total_voxels_no_free = sum(class_counts_total.get(i, 0) for i in range(len(CLASS_NAMES)) if i != 17)
    
    print(f"\n[1] 원본 (마스크 미적용)")
    print(f"{'Class ID':<10} {'Class Name':<25} {'Count':>15} {'Percentage':>12}")
    print("-"*80)
    
    for class_id in range(len(CLASS_NAMES)):
        class_name = CLASS_NAMES[class_id]
        count = class_counts_total.get(class_id, 0)
        percentage = (count / total_voxels * 100) if total_voxels > 0 else 0
        print(f"{class_id:<10} {class_name:<25} {count:>15,} {percentage:>11.2f}%")
    
    print("-"*80)
    print(f"{'Total':<10} {'':<25} {total_voxels:>15,} {100.0:>11.2f}%")
    print("="*80)
    
    # 1-1. 원본 통계 (free 제외)
    print(f"\n[1-1] 원본 (free 제외)")
    print(f"{'Class ID':<10} {'Class Name':<25} {'Count':>15} {'Percentage':>12}")
    print("-"*80)
    
    for class_id in range(len(CLASS_NAMES)):
        if class_id == 17:  # free class 건너뛰기
            continue
        class_name = CLASS_NAMES[class_id]
        count = class_counts_total.get(class_id, 0)
        percentage = (count / total_voxels_no_free * 100) if total_voxels_no_free > 0 else 0
        print(f"{class_id:<10} {class_name:<25} {count:>15,} {percentage:>11.2f}%")
    
    print("-"*80)
    print(f"{'Total':<10} {'':<25} {total_voxels_no_free:>15,} {100.0:>11.2f}%")
    print("="*80)
    
    # 2. camera_mask 적용 통계
    if sum(class_counts_camera_mask.values()) > 0:
        total_voxels_camera = sum(class_counts_camera_mask.values())
        total_voxels_camera_no_free = sum(class_counts_camera_mask.get(i, 0) for i in range(len(CLASS_NAMES)) if i != 17)
        
        print(f"\n[2] camera_mask 적용")
        print(f"{'Class ID':<10} {'Class Name':<25} {'Count':>15} {'Percentage':>12}")
        print("-"*80)
        
        for class_id in range(len(CLASS_NAMES)):
            class_name = CLASS_NAMES[class_id]
            count = class_counts_camera_mask.get(class_id, 0)
            percentage = (count / total_voxels_camera * 100) if total_voxels_camera > 0 else 0
            print(f"{class_id:<10} {class_name:<25} {count:>15,} {percentage:>11.2f}%")
        
        print("-"*80)
        print(f"{'Total':<10} {'':<25} {total_voxels_camera:>15,} {100.0:>11.2f}%")
        print("="*80)
        
        # 2-1. camera_mask 적용 (free 제외)
        print(f"\n[2-1] camera_mask 적용 (free 제외)")
        print(f"{'Class ID':<10} {'Class Name':<25} {'Count':>15} {'Percentage':>12}")
        print("-"*80)
        
        for class_id in range(len(CLASS_NAMES)):
            if class_id == 17:  # free class 건너뛰기
                continue
            class_name = CLASS_NAMES[class_id]
            count = class_counts_camera_mask.get(class_id, 0)
            percentage = (count / total_voxels_camera_no_free * 100) if total_voxels_camera_no_free > 0 else 0
            print(f"{class_id:<10} {class_name:<25} {count:>15,} {percentage:>11.2f}%")
        
        print("-"*80)
        print(f"{'Total':<10} {'':<25} {total_voxels_camera_no_free:>15,} {100.0:>11.2f}%")
        print("="*80)
    
    # 3. lidar_mask 적용 통계
    if sum(class_counts_lidar_mask.values()) > 0:
        total_voxels_lidar = sum(class_counts_lidar_mask.values())
        total_voxels_lidar_no_free = sum(class_counts_lidar_mask.get(i, 0) for i in range(len(CLASS_NAMES)) if i != 17)
        
        print(f"\n[3] lidar_mask 적용")
        print(f"{'Class ID':<10} {'Class Name':<25} {'Count':>15} {'Percentage':>12}")
        print("-"*80)
        
        for class_id in range(len(CLASS_NAMES)):
            class_name = CLASS_NAMES[class_id]
            count = class_counts_lidar_mask.get(class_id, 0)
            percentage = (count / total_voxels_lidar * 100) if total_voxels_lidar > 0 else 0
            print(f"{class_id:<10} {class_name:<25} {count:>15,} {percentage:>11.2f}%")
        
        print("-"*80)
        print(f"{'Total':<10} {'':<25} {total_voxels_lidar:>15,} {100.0:>11.2f}%")
        print("="*80)
        
        # 3-1. lidar_mask 적용 (free 제외)
        print(f"\n[3-1] lidar_mask 적용 (free 제외)")
        print(f"{'Class ID':<10} {'Class Name':<25} {'Count':>15} {'Percentage':>12}")
        print("-"*80)
        
        for class_id in range(len(CLASS_NAMES)):
            if class_id == 17:  # free class 건너뛰기
                continue
            class_name = CLASS_NAMES[class_id]
            count = class_counts_lidar_mask.get(class_id, 0)
            percentage = (count / total_voxels_lidar_no_free * 100) if total_voxels_lidar_no_free > 0 else 0
            print(f"{class_id:<10} {class_name:<25} {count:>15,} {percentage:>11.2f}%")
        
        print("-"*80)
        print(f"{'Total':<10} {'':<25} {total_voxels_lidar_no_free:>15,} {100.0:>11.2f}%")
        print("="*80)


def save_detailed_results(results, output_path):
    """상세 결과를 파일로 저장"""
    print(f"\n상세 결과를 저장 중: {output_path}")
    
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("OCC3D GT 샘플별 클래스 분포\n")
        f.write("="*80 + "\n\n")
        
        for sample_info in results['class_counts_per_sample'][:10]:  # 처음 10개만
            f.write(f"Sample {sample_info['sample_idx']} (token: {sample_info['token']})\n")
            f.write("-"*80 + "\n")
            
            counts = sample_info['counts']
            for class_id in range(len(CLASS_NAMES)):
                class_name = CLASS_NAMES[class_id]
                count = counts.get(class_id, 0)
                f.write(f"  {class_id:2d} {class_name:<25}: {count:>10,}\n")
            f.write("\n")
    
    print(f"저장 완료: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='OCC3D GT 클래스별 갯수 분석')
    parser.add_argument('--pkl-path', type=str, required=True,
                        help='pkl 파일 경로 (예: data/nuscenes/nuscenes_infos_train_occ.pkl)')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='분석할 최대 샘플 수 (기본값: 전체)')
    parser.add_argument('--output', type=str, default=None,
                        help='상세 결과를 저장할 파일 경로 (옵션)')
    
    args = parser.parse_args()
    
    # pkl 파일 존재 확인
    if not os.path.exists(args.pkl_path):
        raise FileNotFoundError(f"pkl 파일을 찾을 수 없습니다: {args.pkl_path}")
    
    # pkl 파일이 위치한 디렉토리 경로 추출
    pkl_dir = os.path.dirname(os.path.abspath(args.pkl_path))
    print(f"pkl 파일 디렉토리: {pkl_dir}")
    
    # pkl 로드
    pkl_data = load_pkl(args.pkl_path)
    
    # 분석 수행
    results = analyze_occ_gt(pkl_data, pkl_dir, args.max_samples)
    
    # 결과 출력
    print_statistics(results)
    
    # 상세 결과 저장 (옵션)
    if args.output:
        save_detailed_results(results, args.output)
    
    print("\n분석 완료!")


if __name__ == '__main__':
    main()
