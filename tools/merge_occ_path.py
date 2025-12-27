#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
occfrmwrk-nuscenes_infos pkl 파일에 surroundocc pkl 파일의 occ_path 정보를 추가하는 스크립트

사용법:
    python tools/merge_occ_path.py
"""

import pickle
import os.path as osp
from tqdm import tqdm


def load_pkl(file_path):
    """pkl 파일 로드"""
    print(f"Loading {file_path}...")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def save_pkl(data, file_path):
    """pkl 파일 저장"""
    print(f"Saving to {file_path}...")
    # 임시 파일에 먼저 저장
    temp_file = file_path + '.tmp'
    with open(temp_file, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    # 저장 성공 후 원본 파일로 교체
    import shutil
    shutil.move(temp_file, file_path)
    print(f"Saved successfully!")


def merge_occ_paths(occfrmwrk_file, surroundocc_file, output_file=None, key_name='occ_path_surroundocc'):
    """
    occfrmwrk pkl 파일에 surroundocc pkl 파일의 occ_path 정보를 추가
    
    Args:
        occfrmwrk_file (str): mmdetection3d v2 형식의 pkl 파일 경로
        surroundocc_file (str): surroundocc 형식의 pkl 파일 경로 (occ_path 포함)
        output_file (str): 출력 파일 경로. None이면 원본 파일을 백업하고 덮어씀
        key_name (str): 추가할 키 이름. 기본값: 'occ_path_surroundocc'
    """
    
    # 파일 존재 확인
    if not osp.exists(occfrmwrk_file):
        raise FileNotFoundError(f"File not found: {occfrmwrk_file}")
    if not osp.exists(surroundocc_file):
        raise FileNotFoundError(f"File not found: {surroundocc_file}")
    
    # 데이터 로드
    occfrmwrk_data = load_pkl(occfrmwrk_file)
    surroundocc_data = load_pkl(surroundocc_file)
    
    # data_list 추출
    occfrmwrk_list = occfrmwrk_data.get('data_list', occfrmwrk_data.get('infos', []))
    surroundocc_list = surroundocc_data.get('data_list', surroundocc_data.get('infos', []))
    
    print(f"\nOccfrmwrk data_list length: {len(occfrmwrk_list)}")
    print(f"Surroundocc data_list length: {len(surroundocc_list)}")
    
    # token을 키로 하는 딕셔너리 생성 (빠른 검색을 위해)
    surroundocc_dict = {}
    for item in surroundocc_list:
        token = item.get('token', item.get('sample_idx', None))
        if token and 'occ_path' in item:
            surroundocc_dict[token] = item['occ_path']
    
    print(f"\nSurroundocc items with occ_path: {len(surroundocc_dict)}")
    
    # occ_path 병합
    matched_count = 0
    missing_count = 0
    
    print(f"\nMerging occ_path information...")
    for item in tqdm(occfrmwrk_list):
        token = item.get('token', item.get('sample_idx', None))
        
        if token and token in surroundocc_dict:
            item[key_name] = surroundocc_dict[token]
            matched_count += 1
        else:
            missing_count += 1
            if missing_count <= 5:  # 처음 5개만 출력
                print(f"Warning: No occ_path found for token: {token}")
    
    print(f"\nMatched: {matched_count} / {len(occfrmwrk_list)}")
    print(f"Missing: {missing_count} / {len(occfrmwrk_list)}")
    
    # 출력 파일 결정
    if output_file is None:
        # 원본 파일 백업
        backup_file = occfrmwrk_file + '.backup'
        if not osp.exists(backup_file):
            print(f"\nCreating backup: {backup_file}")
            save_pkl(occfrmwrk_data, backup_file)
        output_file = occfrmwrk_file
    
    # 병합된 데이터 저장
    save_pkl(occfrmwrk_data, output_file)
    
    print(f"\n✓ Merge completed successfully!")
    print(f"  - Output file: {output_file}")
    print(f"  - Added key: '{key_name}'")
    print(f"  - Matched items: {matched_count}")
    
    return matched_count, missing_count


def main():
    """메인 함수"""
    
    # 파일 경로 설정
    data_root = 'data/nuscenes'
    
    files_to_merge = [
        ('occfrmwrk-nuscenes_infos_train.pkl', 'surroundocc-nuscenes_infos_train.pkl'),
        ('occfrmwrk-nuscenes_infos_val.pkl', 'surroundocc-nuscenes_infos_val.pkl'),
    ]
    
    for occfrmwrk_file, surroundocc_file in files_to_merge:
        print(f"\n{'='*80}")
        print(f"Processing: {occfrmwrk_file}")
        print(f"{'='*80}")
        
        occfrmwrk_path = osp.join(data_root, occfrmwrk_file)
        surroundocc_path = osp.join(data_root, surroundocc_file)
        
        # 파일 존재 확인
        if not osp.exists(occfrmwrk_path):
            print(f"Skip: {occfrmwrk_path} not found")
            continue
        
        if not osp.exists(surroundocc_path):
            print(f"Skip: {surroundocc_path} not found")
            continue
        
        try:
            merge_occ_paths(
                occfrmwrk_file=occfrmwrk_path,
                surroundocc_file=surroundocc_path,
                output_file=None,  # 원본 파일 덮어쓰기 (백업 자동 생성)
                key_name='occ_path_surroundocc'
            )
        except Exception as e:
            print(f"Error processing {occfrmwrk_file}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*80}")
    print("All files processed!")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()

