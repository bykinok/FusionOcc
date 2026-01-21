#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pkl 파일에 포함된 NuScenes 샘플들을 렌더링하는 스크립트

사용 예시:
    # 모든 샘플 렌더링 (기본)
    python tools/render_samples_from_pkl.py \
        --input-pkl ./data/nuscenes/occfrmwrk-nuscenes_infos_val_night_not_rain.pkl \
        --dataroot ./data/nuscenes \
        --version v1.0-trainval \
        --output-dir ./rendered_images
    
    # 처음 10개 샘플만 렌더링
    python tools/render_samples_from_pkl.py \
        --input-pkl ./data/nuscenes/occfrmwrk-nuscenes_infos_val_night_not_rain.pkl \
        --dataroot ./data/nuscenes \
        --version v1.0-trainval \
        --output-dir ./rendered_images \
        --max-samples 10
    
    # 특정 카메라만 렌더링
    python tools/render_samples_from_pkl.py \
        --input-pkl ./data/nuscenes/occfrmwrk-nuscenes_infos_val_night_not_rain.pkl \
        --dataroot ./data/nuscenes \
        --version v1.0-trainval \
        --output-dir ./rendered_images \
        --cameras CAM_FRONT CAM_FRONT_LEFT CAM_FRONT_RIGHT
    
    # 특정 sample token만 렌더링 (pkl 파일 없이)
    python tools/render_samples_from_pkl.py \
        --sample-tokens fd8420396768425eabec9bdddf7e64b6 67d2d74087714e4994a68c7347acf55a \
        --dataroot ./data/nuscenes \
        --version v1.0-trainval \
        --output-dir ./rendered_images
"""

import argparse
import pickle
import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # GUI 없이 이미지 저장
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from nuscenes.nuscenes import NuScenes
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description='Render NuScenes samples from pkl file')
    parser.add_argument(
        '--input-pkl',
        type=str,
        default=None,
        help='Input pkl file path')
    parser.add_argument(
        '--sample-tokens',
        type=str,
        nargs='+',
        default=None,
        help='Specific sample tokens to render (alternative to --input-pkl)')
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
        default='./rendered_images',
        help='Output directory for rendered images')
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum number of samples to render (default: all)')
    parser.add_argument(
        '--cameras',
        type=str,
        nargs='+',
        default=['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
                 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'],
        help='Camera names to render')
    parser.add_argument(
        '--render-mode',
        type=str,
        choices=['separate', 'combined', 'both'],
        default='both',
        help='Render mode: separate (각 카메라별), combined (6개 카메라 한번에), both (둘 다)')
    parser.add_argument(
        '--dpi',
        type=int,
        default=100,
        help='DPI for saved images')
    
    return parser.parse_args()


def load_pkl(pkl_path):
    """pkl 파일 로드"""
    print(f"Loading pkl file: {pkl_path}")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data


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
    
    # sample token 리스트 추출
    sample_tokens = []
    for info in infos:
        if isinstance(info, dict):
            if 'sample_token' in info:
                sample_tokens.append(info['sample_token'])
            elif 'token' in info:
                sample_tokens.append(info['token'])
    
    return sample_tokens


def render_sample_separate(nusc, sample_token, cameras, output_dir, dpi=100):
    """각 카메라별로 개별 이미지 렌더링"""
    sample = nusc.get('sample', sample_token)
    
    # 샘플별 디렉토리 생성
    sample_dir = os.path.join(output_dir, 'separate', sample_token)
    os.makedirs(sample_dir, exist_ok=True)
    
    for camera in cameras:
        if camera not in sample['data']:
            continue
        
        try:
            # 카메라 데이터 가져오기
            sample_data_token = sample['data'][camera]
            img_path = nusc.get_sample_data_path(sample_data_token)
            
            # 이미지 로드
            img = Image.open(img_path)
            img_array = np.array(img)
            
            # 이미지 저장
            plt.figure(figsize=(16, 9))
            plt.imshow(img_array)
            plt.title(f'{camera} - Sample: {sample_token[:8]}...', fontsize=14)
            plt.axis('off')
            plt.tight_layout()
            
            output_path = os.path.join(sample_dir, f'{camera}.png')
            plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Warning: Failed to render {camera} for sample {sample_token}: {e}")


def render_sample_combined(nusc, sample_token, cameras, output_dir, dpi=100):
    """6개 카메라를 한 이미지에 렌더링 (2x3 grid)"""
    sample = nusc.get('sample', sample_token)
    
    # 출력 디렉토리 생성
    combined_dir = os.path.join(output_dir, 'combined')
    os.makedirs(combined_dir, exist_ok=True)
    
    # 카메라 배치 순서 (2x3)
    # Row 1: FRONT_LEFT, FRONT, FRONT_RIGHT
    # Row 2: BACK_LEFT, BACK, BACK_RIGHT
    camera_layout = [
        ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT'],
        ['CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(24, 12))
    fig.suptitle(f'Sample: {sample_token}', fontsize=16, y=0.98)
    
    for row_idx, row in enumerate(camera_layout):
        for col_idx, camera in enumerate(row):
            ax = axes[row_idx, col_idx]
            
            if camera not in cameras or camera not in sample['data']:
                ax.axis('off')
                ax.text(0.5, 0.5, f'{camera}\nNot Available', 
                       ha='center', va='center', fontsize=12)
                continue
            
            try:
                # 카메라 데이터 가져오기
                sample_data_token = sample['data'][camera]
                img_path = nusc.get_sample_data_path(sample_data_token)
                
                # 이미지 로드 및 표시
                img = Image.open(img_path)
                img_array = np.array(img)
                
                ax.imshow(img_array)
                ax.set_title(camera, fontsize=12)
                ax.axis('off')
                
            except Exception as e:
                ax.axis('off')
                ax.text(0.5, 0.5, f'{camera}\nError: {str(e)[:30]}...', 
                       ha='center', va='center', fontsize=10)
    
    plt.tight_layout()
    
    # 이미지 저장
    output_path = os.path.join(combined_dir, f'{sample_token}.png')
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()


def render_samples(nusc, sample_tokens, args):
    """모든 샘플 렌더링"""
    print(f"\n=== Rendering {len(sample_tokens)} samples ===")
    print(f"Output directory: {args.output_dir}")
    print(f"Render mode: {args.render_mode}")
    print(f"Cameras: {', '.join(args.cameras)}")
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 최대 샘플 수 제한
    if args.max_samples:
        sample_tokens = sample_tokens[:args.max_samples]
        print(f"Limiting to first {args.max_samples} samples")
    
    # 각 샘플 렌더링
    for idx, sample_token in enumerate(tqdm(sample_tokens, desc="Rendering")):
        try:
            if args.render_mode in ['separate', 'both']:
                render_sample_separate(
                    nusc, sample_token, args.cameras, args.output_dir, args.dpi)
            
            if args.render_mode in ['combined', 'both']:
                render_sample_combined(
                    nusc, sample_token, args.cameras, args.output_dir, args.dpi)
                
        except Exception as e:
            print(f"\nWarning: Failed to render sample {sample_token}: {e}")
            continue
    
    print(f"\n=== Rendering complete! ===")
    print(f"Images saved to: {args.output_dir}")


def main():
    args = parse_args()
    
    # sample token 결정 (--sample-tokens 또는 --input-pkl 사용)
    if args.sample_tokens:
        # 직접 입력된 sample token 사용
        sample_tokens = args.sample_tokens
        print(f"Using provided sample tokens: {len(sample_tokens)} samples")
        for token in sample_tokens:
            print(f"  - {token}")
    elif args.input_pkl:
        # pkl 파일에서 sample token 추출
        data = load_pkl(args.input_pkl)
        sample_tokens = get_sample_tokens_from_pkl(data)
        print(f"Total samples in pkl: {len(sample_tokens)}")
    else:
        print("Error: Either --input-pkl or --sample-tokens must be provided!")
        return
    
    if len(sample_tokens) == 0:
        print("Error: No samples to render!")
        return
    
    # NuScenes 초기화
    print(f"\nInitializing NuScenes dataset...")
    print(f"Version: {args.version}")
    print(f"Dataroot: {args.dataroot}")
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)
    
    # 샘플 렌더링
    render_samples(nusc, sample_tokens, args)


if __name__ == '__main__':
    main()
