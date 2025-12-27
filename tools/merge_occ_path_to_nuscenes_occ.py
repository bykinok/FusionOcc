#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
nuscenes_occ pkl 파일에 occfrmwrk pkl 파일의 occ_path 정보를 추가하는 스크립트

scene_token을 기준으로 매칭하여 occ_path를 추가합니다.

사용법:
    python tools/merge_occ_path_to_nuscenes_occ.py
"""

import pickle
import os.path as osp
from tqdm import tqdm
import shutil


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
    shutil.move(temp_file, file_path)
    print(f"Saved successfully!")


def merge_occ_paths(target_file, occfrmwrk_file, output_file=None, key_name='occ_path'):
    """
    nuscenes_occ pkl 파일에 occfrmwrk pkl 파일의 occ_path 정보를 추가
    
    Args:
        target_file (str): 대상 pkl 파일 경로 (nuscenes_occ_infos)
        occfrmwrk_file (str): occfrmwrk 형식의 pkl 파일 경로 (occ_path 소스)
        output_file (str, optional): 출력 파일 경로. None이면 target_file을 덮어씀
        key_name (str): 추가할 키 이름
    
    Returns:
        dict: 병합 통계
    """
    # 파일 로드
    target_data = load_pkl(target_file)
    occfrmwrk = load_pkl(occfrmwrk_file)
    
    # data_list 추출 (두 형식 모두 지원)
    target_list = target_data.get('data_list', target_data.get('infos', []))
    occfrmwrk_list = occfrmwrk.get('data_list', occfrmwrk.get('infos', []))
    
    print(f"\nTarget items: {len(target_list)}")
    print(f"Occfrmwrk items: {len(occfrmwrk_list)}")
    
    # scene_token 기반 매칭 딕셔너리 생성
    print("\nBuilding scene_token mapping from occfrmwrk...")
    scene_to_occ_path = {}
    
    for item in tqdm(occfrmwrk_list, desc="Processing occfrmwrk"):
        scene_token = item.get('scene_token')
        occ_path = item.get('occ_path')
        
        if scene_token and occ_path:
            # scene_token을 키로 사용
            # 동일 scene_token이 여러 개 있을 수 있으므로 리스트로 저장
            if scene_token not in scene_to_occ_path:
                scene_to_occ_path[scene_token] = []
            scene_to_occ_path[scene_token].append({
                'token': item.get('token', item.get('sample_idx')),
                'occ_path': occ_path
            })
    
    print(f"Found {len(scene_to_occ_path)} unique scene_tokens in occfrmwrk")
    
    # 대상 파일에 occ_path 추가
    print("\nMerging occ_path to target file...")
    matched = 0
    not_matched = 0
    multiple_matches = 0
    
    for item in tqdm(target_list, desc="Merging"):
        scene_token = item.get('scene_token')
        
        if scene_token and scene_token in scene_to_occ_path:
            candidates = scene_to_occ_path[scene_token]
            
            if len(candidates) == 1:
                # 하나만 매칭되면 바로 사용
                item[key_name] = candidates[0]['occ_path']
                matched += 1
            else:
                # 여러 개 매칭되면 token으로 추가 매칭 시도
                item_token = item.get('token', item.get('sample_idx'))
                found = False
                
                for candidate in candidates:
                    if candidate['token'] == item_token:
                        item[key_name] = candidate['occ_path']
                        matched += 1
                        found = True
                        break
                
                if not found:
                    # token도 매칭 안되면 첫 번째 것 사용
                    item[key_name] = candidates[0]['occ_path']
                    matched += 1
                    multiple_matches += 1
        else:
            not_matched += 1
    
    # 통계 출력
    print(f"\n{'='*60}")
    print(f"Merge Statistics:")
    print(f"{'='*60}")
    print(f"Total items:          {len(target_list)}")
    print(f"Matched:              {matched} ({matched/len(target_list)*100:.1f}%)")
    print(f"Not matched:          {not_matched} ({not_matched/len(target_list)*100:.1f}%)")
    if multiple_matches > 0:
        print(f"Multiple matches:     {multiple_matches} (used first match)")
    print(f"{'='*60}")
    
    # 결과 저장
    if output_file is None:
        output_file = target_file
    
    save_pkl(target_data, output_file)
    
    # 검증
    print("\nVerifying merged data...")
    verified = load_pkl(output_file)
    verified_list = verified.get('data_list', verified.get('infos', []))
    
    count_with_key = sum(1 for item in verified_list if key_name in item)
    print(f"✓ Items with '{key_name}': {count_with_key}/{len(verified_list)}")
    
    if count_with_key > 0:
        # 첫 번째 아이템 예시 출력
        for item in verified_list:
            if key_name in item:
                print(f"✓ Example: {item[key_name]}")
                break
    
    return {
        'total': len(target_list),
        'matched': matched,
        'not_matched': not_matched,
        'multiple_matches': multiple_matches
    }


def main():
    """메인 함수"""
    # 파일 경로 설정
    data_root = 'data/nuscenes'
    
    files_to_process = [
        ('nuscenes_occ_infos_train.pkl', 'occfrmwrk-nuscenes_infos_train.pkl'),
        ('nuscenes_occ_infos_val.pkl', 'occfrmwrk-nuscenes_infos_val.pkl'),
    ]
    
    for target_file, occfrmwrk_file in files_to_process:
        target_path = osp.join(data_root, target_file)
        occfrmwrk_path = osp.join(data_root, occfrmwrk_file)
        
        print(f"\n{'#'*70}")
        print(f"# Processing: {target_file}")
        print(f"{'#'*70}")
        
        # 파일 존재 확인
        if not osp.exists(target_path):
            print(f"✗ Error: {target_path} not found!")
            continue
        
        if not osp.exists(occfrmwrk_path):
            print(f"✗ Error: {occfrmwrk_path} not found!")
            continue
        
        # 백업 생성
        backup_path = target_path + '.backup'
        if not osp.exists(backup_path):
            print(f"Creating backup: {backup_path}")
            shutil.copy2(target_path, backup_path)
        else:
            print(f"Backup already exists: {backup_path}")
        
        try:
            # 병합 수행
            stats = merge_occ_paths(
                target_file=target_path,
                occfrmwrk_file=occfrmwrk_path,
                output_file=target_path,
                key_name='occ_path'
            )
            
            print(f"\n✓ Merge completed successfully!")
            print(f"  - Added key: 'occ_path'")
            print(f"  - Matched: {stats['matched']}/{stats['total']}")
            
        except Exception as e:
            print(f"\n✗ Error processing {target_file}: {e}")
            import traceback
            traceback.print_exc()
            
            # 에러 발생시 백업에서 복원
            if osp.exists(backup_path):
                print(f"Restoring from backup...")
                shutil.copy2(backup_path, target_path)
            continue
    
    print(f"\n{'='*70}")
    print("All done!")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()



