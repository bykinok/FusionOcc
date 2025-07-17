#!/usr/bin/env python3
"""
NuScenes 데이터 무결성 검증 실행 스크립트

사용 예제:
python run_integrity_check.py --data-root /path/to/nuscenes --version v1.0-trainval
"""

import os
import sys
from pathlib import Path

# 현재 디렉토리를 Python 경로에 추가
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from nuscenes_integrity_checker import NuScenesIntegrityChecker


def main():
    """메인 함수"""
    # NuScenes 데이터 경로 설정 (실제 경로로 변경 필요)
    data_root = "/path/to/nuscenes"  # 실제 NuScenes 데이터 경로로 변경하세요
    
    # 환경 변수에서 경로를 가져오거나 기본값 사용
    if 'NUSCENES_DATA_ROOT' in os.environ:
        data_root = os.environ['NUSCENES_DATA_ROOT']
    
    # 데이터 경로가 존재하는지 확인
    if not os.path.exists(data_root):
        print(f"오류: NuScenes 데이터 경로를 찾을 수 없습니다: {data_root}")
        print("다음 중 하나를 확인하세요:")
        print("1. NUSCENES_DATA_ROOT 환경 변수 설정")
        print("2. data_root 변수를 올바른 경로로 수정")
        return
    
    # 검증 실행
    print(f"NuScenes 데이터 무결성 검증 시작...")
    print(f"데이터 경로: {data_root}")
    
    checker = NuScenesIntegrityChecker(data_root, version='v1.0-trainval')
    results = checker.run_full_check(max_samples=5)  # 테스트용으로 5개 샘플만 검증
    
    # 리포트 생성
    report = checker.generate_report(results)
    print("\n" + "="*60)
    print("검증 완료!")
    print("="*60)
    print(report)
    
    # 결과 파일 저장
    output_file = "nuscenes_integrity_report.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n리포트가 {output_file}에 저장되었습니다.")


if __name__ == '__main__':
    main() 