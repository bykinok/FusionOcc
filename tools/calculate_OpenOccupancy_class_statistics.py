"""
nuScenes-Occupancy 데이터셋의 GT 클래스별 통계 계산 스크립트 (pkl 파일 기반)

사용법:
    python tools/calculate_class_statistics.py --pkl-file ./data/nuscenes/nuscenes_occ_infos_train.pkl --output-file class_statistics.txt

데이터 형식:
    - 각 .npy 파일은 (N, 4) shape의 sparse occupancy 데이터
    - 각 행: [x, y, z, class_label]
    - class_label: 0-16 (17개 클래스)
"""

import os
import numpy as np
import argparse
import pickle
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict


# nuScenes-Occupancy 클래스 이름 (17개 클래스)
nusc_class_names = [
    "empty",           # 0
    "barrier",         # 1
    "bicycle",         # 2
    "bus",             # 3
    "car",             # 4
    "construction",    # 5
    "motorcycle",      # 6
    "pedestrian",      # 7
    "trafficcone",     # 8
    "trailer",         # 9
    "truck",           # 10
    "driveable_surface", # 11
    "other",           # 12
    "sidewalk",        # 13
    "terrain",         # 14
    "mannade",         # 15
    "vegetation",      # 16
]


def load_pkl_infos(pkl_file):
    """pkl 파일에서 샘플 정보 로드"""
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    
    if isinstance(data, dict) and 'infos' in data:
        return data['infos']
    elif isinstance(data, list):
        return data
    else:
        raise ValueError(f"pkl 파일 형식이 올바르지 않습니다: {type(data)}")


def get_occ_file_path(sample_info, occ_path_base=None, use_occ3d=False):
    """
    loading.py의 로직을 참고하여 occ 파일 경로 생성
    
    Args:
        sample_info: pkl에서 로드한 샘플 정보 (dict)
        occ_path_base: occupancy 데이터 루트 경로
        use_occ3d: Occ3D 형식 사용 여부
        
    Returns:
        occ 파일 경로 (str 또는 Path)
    """
    # Occ3D 형식: pkl 파일의 occ_path 키 사용
    if use_occ3d and 'occ_path' in sample_info:
        occ_path = sample_info['occ_path']
        # .npy 확장자가 없으면 추가
        if not occ_path.endswith('.npy'):
            occ_path = occ_path + '.npy'
        return occ_path
    
    # 기존 nuScenes-Occupancy 형식
    if 'scene_token' not in sample_info or 'lidar_token' not in sample_info:
        raise ValueError(f"샘플 정보에 scene_token 또는 lidar_token이 없습니다: {sample_info.keys()}")
    
    rel_path = 'scene_{0}/occupancy/{1}.npy'.format(
        sample_info['scene_token'], 
        sample_info['lidar_token']
    )
    
    if occ_path_base is None:
        raise ValueError("occ_path_base must be provided for nuScenes-Occupancy format")
    
    return os.path.join(occ_path_base, rel_path)


def calculate_statistics(pkl_file, occ_path_base=None, use_occ3d=False, verbose=True):
    """
    pkl 파일을 읽어서 각 샘플의 occupancy 데이터로 클래스별 통계 계산
    
    Args:
        pkl_file: nuscenes_occ_infos_train.pkl 파일 경로
        occ_path_base: occupancy 데이터 루트 경로 (nuScenes-Occupancy 형식용)
        use_occ3d: Occ3D 형식 사용 여부
        verbose: 진행 상황 출력 여부
        
    Returns:
        class_counts: 클래스별 voxel 개수 딕셔너리
        total_voxels: 전체 voxel 개수
        file_count: 처리한 파일 개수
    """
    # pkl 파일 로드
    sample_infos = load_pkl_infos(pkl_file)
    
    if verbose:
        print(f"총 {len(sample_infos)}개의 샘플 발견")
    
    # 클래스별 카운트 초기화 (0-16)
    class_counts = defaultdict(int)
    total_voxels = 0
    file_count = 0
    error_count = 0
    
    # 모든 샘플 처리
    iterator = tqdm(sample_infos, desc="Processing samples") if verbose else sample_infos
    
    for sample_info in iterator:
        try:
            # 샘플의 occupancy 파일 경로 생성
            occ_file = get_occ_file_path(sample_info, occ_path_base, use_occ3d)
            
            # 파일 존재 여부 확인
            if not os.path.exists(occ_file):
                if verbose and error_count < 5:  # 처음 5개 에러만 출력
                    print(f"\n경고: 파일이 존재하지 않습니다: {occ_file}")
                error_count += 1
                continue
            
            # npy 파일 로드
            data = np.load(occ_file)
            
            # 데이터 검증
            if data.ndim != 2 or data.shape[1] < 4:
                if verbose and error_count < 5:
                    print(f"\n경고: {occ_file} 파일의 형식이 올바르지 않습니다. Shape: {data.shape}")
                error_count += 1
                continue
            
            # 클래스 레이블 추출 (마지막 열)
            class_labels = data[:, -1]
            
            # 클래스별 카운트 업데이트
            unique_classes, counts = np.unique(class_labels, return_counts=True)
            for cls, count in zip(unique_classes, counts):
                class_counts[int(cls)] += int(count)
                total_voxels += int(count)
            
            file_count += 1
            
        except Exception as e:
            if verbose and error_count < 5:
                print(f"\n에러 발생 ({sample_info.get('token', 'unknown')}): {e}")
            error_count += 1
            continue
    
    if verbose and error_count > 0:
        print(f"\n총 {error_count}개의 파일 처리 실패")
    
    return class_counts, total_voxels, file_count


def print_statistics(class_counts, total_voxels, file_count, output_file=None):
    """통계 결과 출력 및 저장"""
    
    lines = []
    lines.append("=" * 80)
    lines.append("nuScenes-Occupancy GT 클래스별 통계")
    lines.append("=" * 80)
    lines.append(f"\n처리한 파일 개수: {file_count}")
    lines.append(f"전체 Voxel 개수: {total_voxels:,}\n")
    lines.append("-" * 80)
    lines.append(f"{'클래스 ID':<10} {'클래스 이름':<20} {'Voxel 개수':<20} {'비율 (%)':<10}")
    lines.append("-" * 80)
    
    # 클래스 ID 순으로 정렬하여 출력
    for class_id in sorted(class_counts.keys()):
        count = class_counts[class_id]
        percentage = (count / total_voxels) * 100 if total_voxels > 0 else 0
        class_name = nusc_class_names[class_id] if class_id < len(nusc_class_names) else "unknown"
        
        line = f"{class_id:<10} {class_name:<20} {count:<20,} {percentage:<10.4f}"
        lines.append(line)
    
    lines.append("-" * 80)
    lines.append(f"{'합계':<10} {'':<20} {total_voxels:<20,} {100.0:<10.4f}")
    lines.append("=" * 80)
    
    # 누락된 클래스 확인
    missing_classes = []
    for i in range(len(nusc_class_names)):
        if i not in class_counts:
            missing_classes.append(f"{i} ({nusc_class_names[i]})")
    
    if missing_classes:
        lines.append(f"\n주의: 다음 클래스는 데이터셋에 존재하지 않습니다:")
        lines.append(", ".join(missing_classes))
    
    # 클래스 빈도 배열 생성 (nusc_param.py 형식)
    lines.append("\n" + "=" * 80)
    lines.append("클래스 빈도 배열 (nusc_param.py 형식):")
    lines.append("=" * 80)
    
    freq_array = []
    for i in range(len(nusc_class_names)):
        freq_array.append(class_counts.get(i, 0))
    
    lines.append(f"nusc_class_frequencies = np.array([")
    # 한 줄에 6개씩 출력
    for i in range(0, len(freq_array), 6):
        chunk = freq_array[i:i+6]
        chunk_str = ", ".join([f"{x}" for x in chunk])
        if i + 6 < len(freq_array):
            lines.append(f"    {chunk_str},")
        else:
            lines.append(f"    {chunk_str}])")
    
    # 화면 출력
    output_text = "\n".join(lines)
    print(output_text)
    
    # 파일 저장
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(output_text)
        print(f"\n결과가 {output_file}에 저장되었습니다.")


def main():
    parser = argparse.ArgumentParser(
        description="nuScenes-Occupancy 데이터셋의 GT 클래스별 통계 계산 (pkl 파일 기반)"
    )
    parser.add_argument(
        "--pkl-file",
        type=str,
        default="./data/nuscenes/nuscenes_occ_infos_train.pkl",
        help="nuscenes_occ_infos pkl 파일 경로 (기본값: ./data/nuscenes/nuscenes_occ_infos_train.pkl)"
    )
    parser.add_argument(
        "--occ-path",
        type=str,
        default="data/nuScenes-Occupancy",
        help="occupancy 데이터 루트 경로 (nuScenes-Occupancy 형식용, 기본값: data/nuScenes-Occupancy)"
    )
    parser.add_argument(
        "--use-occ3d",
        action="store_true",
        help="Occ3D 형식 사용 (pkl의 occ_path 키 사용)"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="class_statistics.txt",
        help="통계 결과를 저장할 파일 경로 (기본값: class_statistics.txt)"
    )
    parser.add_argument(
        "--no-verbose",
        action="store_true",
        help="진행 상황 출력 비활성화"
    )
    
    args = parser.parse_args()
    
    # pkl 파일 경로 확인
    pkl_file = Path(args.pkl_file)
    if not pkl_file.exists():
        print(f"에러: pkl 파일이 존재하지 않습니다: {pkl_file}")
        return
    
    print(f"pkl 파일: {pkl_file}")
    if args.use_occ3d:
        print(f"형식: Occ3D (pkl의 occ_path 키 사용)")
    else:
        print(f"형식: nuScenes-Occupancy")
        print(f"occupancy 루트: {args.occ_path}")
    print(f"출력 파일: {args.output_file}\n")
    
    # 통계 계산
    class_counts, total_voxels, file_count = calculate_statistics(
        pkl_file=args.pkl_file,
        occ_path_base=args.occ_path if not args.use_occ3d else None,
        use_occ3d=args.use_occ3d,
        verbose=not args.no_verbose
    )
    
    # 결과 출력 및 저장
    print_statistics(class_counts, total_voxels, file_count, args.output_file)


if __name__ == "__main__":
    main()
