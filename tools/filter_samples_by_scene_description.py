#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NuScenes 데이터셋의 pkl 파일에서 scene description에 따라 sample을 필터링하여 저장하는 스크립트

사용 예시:
    python tools/filter_samples_by_scene_description.py \
        --input-pkl ./data/nuscenes/occfrmwrk-nuscenes_infos_val.pkl \
        --dataroot ./data/nuscenes \
        --version v1.0-trainval \
        --output-dir ./data/nuscenes/filtered
"""

import argparse
import pickle
import os
from pathlib import Path
from nuscenes.nuscenes import NuScenes


def parse_args():
    parser = argparse.ArgumentParser(
        description='Filter NuScenes samples by scene description')
    parser.add_argument(
        '--input-pkl',
        type=str,
        required=True,
        help='Input pkl file path')
    parser.add_argument(
        '--dataroot',
        type=str,
        default='./data/nuscenes',
        help='NuScenes dataset root directory')
    parser.add_argument(
        '--version',
        type=str,
        default='v1.0-trainval',
        help='NuScenes dataset version')
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for filtered pkl files (default: same as input)')
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed information')
    
    return parser.parse_args()


def load_pkl(pkl_path):
    """pkl 파일 로드"""
    print(f"Loading pkl file: {pkl_path}")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data


def save_pkl(data, output_path):
    """pkl 파일 저장"""
    print(f"Saving pkl file: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved {len(data.get('data_list', data))} samples")


def get_sample_tokens_from_pkl(data):
    """pkl 데이터에서 sample token 추출"""
    # pkl 구조 파악
    if isinstance(data, dict) and 'data_list' in data:
        infos = data['data_list']
    elif isinstance(data, list):
        infos = data
    else:
        raise RuntimeError(
            f"예상치 못한 pkl 구조: {type(data)}, "
            f"keys: {getattr(data, 'keys', lambda: [])()}")
    
    # sample token과 info 매핑
    sample_token_to_info = {}
    for info in infos:
        if isinstance(info, dict):
            if 'sample_token' in info:
                token = info['sample_token']
            elif 'token' in info:
                token = info['token']
            else:
                continue
            sample_token_to_info[token] = info
    
    return sample_token_to_info


def get_scene_description(nusc, sample_token):
    """sample token으로부터 scene description 가져오기"""
    sample = nusc.get('sample', sample_token)
    scene = nusc.get('scene', sample['scene_token'])
    return scene['description'].lower()


def filter_samples_by_description(nusc, sample_token_to_info, verbose=False):
    """
    Scene description에 따라 sample 필터링
    
    Returns:
        dict: 4가지 카테고리로 분류된 sample token to info 딕셔너리
            - 'not_night_not_rain'
            - 'not_night_rain'
            - 'night_not_rain'
            - 'night_rain'
    """
    filtered = {
        'not_night_not_rain': {},
        'not_night_rain': {},
        'night_not_rain': {},
        'night_rain': {}
    }
    
    print("\n=== Filtering samples by scene description ===")
    
    for sample_token, info in sample_token_to_info.items():
        try:
            desc = get_scene_description(nusc, sample_token)
            
            has_night = "night" in desc
            has_rain = "rain" in desc
            
            # 1) not_night + not rain
            if not has_night and not has_rain:
                filtered['not_night_not_rain'][sample_token] = info
            
            # 2) not_night + rain
            if not has_night and has_rain:
                filtered['not_night_rain'][sample_token] = info
            
            # 3) night + not rain
            if has_night and not has_rain:
                filtered['night_not_rain'][sample_token] = info
            
            # 4) night + rain
            if has_night and has_rain:
                filtered['night_rain'][sample_token] = info
                
            if verbose and (has_night or has_rain):
                print(f"Token: {sample_token[:8]}... | "
                      f"Night: {has_night}, Rain: {has_rain} | "
                      f"Description: {desc}")
        
        except Exception as e:
            print(f"Warning: Failed to process sample {sample_token}: {e}")
            continue
    
    # 통계 출력
    print("\n=== Filtering Results ===")
    print(f"Total samples: {len(sample_token_to_info)}")
    print(f"Day_Clear samples: {len(filtered['not_night_not_rain'])}")
    print(f"Day_Rain samples: {len(filtered['not_night_rain'])}")
    print(f"Night_Clear samples: {len(filtered['night_not_rain'])}")
    print(f"Night_Rain samples: {len(filtered['night_rain'])}")
    
    return filtered


def create_filtered_pkl_data(original_data, filtered_info_dict):
    """필터링된 정보로 새로운 pkl 데이터 구조 생성"""
    # 원본 데이터의 구조를 유지하면서 data_list만 필터링
    if isinstance(original_data, dict):
        # 딕셔너리 구조인 경우 (data_list 포함)
        new_data = original_data.copy()
        new_data['data_list'] = list(filtered_info_dict.values())
    else:
        # 리스트 구조인 경우
        new_data = list(filtered_info_dict.values())
    
    return new_data


def main():
    args = parse_args()
    
    # pkl 파일 로드
    data = load_pkl(args.input_pkl)
    
    # NuScenes 초기화
    print(f"\nInitializing NuScenes dataset...")
    print(f"Version: {args.version}")
    print(f"Dataroot: {args.dataroot}")
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)
    
    # sample token 추출
    sample_token_to_info = get_sample_tokens_from_pkl(data)
    print(f"\nTotal samples in pkl: {len(sample_token_to_info)}")
    
    # scene description으로 필터링
    filtered = filter_samples_by_description(
        nusc, sample_token_to_info, verbose=args.verbose)
    
    # 출력 디렉토리 설정
    if args.output_dir is None:
        output_dir = os.path.dirname(args.input_pkl)
    else:
        output_dir = args.output_dir
    
    # 입력 파일명에서 base name 추출
    input_basename = os.path.basename(args.input_pkl)
    base_name = input_basename.replace('.pkl', '')
    
    # 필터링된 데이터를 pkl로 저장
    print("\n=== Saving filtered pkl files ===")
    
    categories = [
        ('not_night_not_rain', 'not_night_not_rain'),
        ('not_night_rain', 'not_night_rain'),
        ('night_not_rain', 'night_not_rain'),
        ('night_rain', 'night_rain')
    ]
    
    for key, suffix in categories:
        filtered_data = create_filtered_pkl_data(data, filtered[key])
        output_path = os.path.join(output_dir, f"{base_name}_{suffix}.pkl")
        save_pkl(filtered_data, output_path)
    
    print("\n=== Done! ===")
    print(f"Output directory: {output_dir}")


if __name__ == '__main__':
    main()
