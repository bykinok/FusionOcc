#!/usr/bin/env python3
"""
NuScenes 데이터셋에서 손상된 파일과 크기가 0인 파일을 확인하는 스크립트
"""

import os
import sys
import argparse
from pathlib import Path
import pickle
import json
from typing import List, Dict, Tuple
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('corrupted_files_check.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class NuScenesFileChecker:
    def __init__(self, data_root: str):
        self.data_root = Path(data_root)
        self.corrupted_files = []
        self.zero_size_files = []
        self.total_files = 0
        self.checked_files = 0
        
    def check_file_size(self, file_path: Path) -> bool:
        """파일 크기가 0인지 확인"""
        try:
            if file_path.stat().st_size == 0:
                self.zero_size_files.append(str(file_path))
                logger.warning(f"Zero size file found: {file_path}")
                return False
            return True
        except Exception as e:
            logger.error(f"Error checking file size for {file_path}: {e}")
            return False
    
    def check_pickle_file(self, file_path: Path) -> bool:
        """pickle 파일이 손상되었는지 확인"""
        try:
            with open(file_path, 'rb') as f:
                pickle.load(f)
            return True
        except Exception as e:
            self.corrupted_files.append(str(file_path))
            logger.error(f"Corrupted pickle file: {file_path} - {e}")
            return False
    
    def check_json_file(self, file_path: Path) -> bool:
        """JSON 파일이 손상되었는지 확인"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                json.load(f)
            return True
        except Exception as e:
            self.corrupted_files.append(str(file_path))
            logger.error(f"Corrupted JSON file: {file_path} - {e}")
            return False
    
    def check_image_file(self, file_path: Path) -> bool:
        """이미지 파일이 손상되었는지 확인"""
        try:
            from PIL import Image
            with Image.open(file_path) as img:
                img.verify()  # 이미지 파일 무결성 검사
            return True
        except Exception as e:
            self.corrupted_files.append(str(file_path))
            logger.error(f"Corrupted image file: {file_path} - {e}")
            return False
    
    def check_bin_file(self, file_path: Path) -> bool:
        """bin 파일이 손상되었는지 확인"""
        try:
            # bin 파일의 기본적인 바이너리 무결성 검사
            with open(file_path, 'rb') as f:
                # 파일 헤더 읽기 (최소 16바이트)
                header = f.read(16)
                if len(header) < 16:
                    # 파일이 너무 작으면 의심스러움
                    self.corrupted_files.append(str(file_path))
                    logger.error(f"Bin file too small: {file_path} - size: {len(header)} bytes")
                    return False
                
                # 파일 끝까지 읽어서 전체 무결성 확인
                f.seek(0)
                data = f.read()
                
                # 빈 데이터나 너무 작은 파일 체크
                if len(data) < 32:  # 최소 32바이트 이상이어야 함
                    self.corrupted_files.append(str(file_path))
                    logger.error(f"Bin file too small: {file_path} - size: {len(data)} bytes")
                    return False
                
                # 모든 바이트가 읽을 수 있는지 확인
                if not data:
                    self.corrupted_files.append(str(file_path))
                    logger.error(f"Empty bin file: {file_path}")
                    return False
                    
            return True
        except Exception as e:
            self.corrupted_files.append(str(file_path))
            logger.error(f"Corrupted bin file: {file_path} - {e}")
            return False
    
    def check_file_integrity(self, file_path: Path) -> bool:
        """파일 무결성 검사"""
        # 파일 크기 확인
        if not self.check_file_size(file_path):
            return False
        
        # 파일 확장자에 따른 특별 검사
        ext = file_path.suffix.lower()
        
        if ext == '.pkl':
            return self.check_pickle_file(file_path)
        elif ext == '.json':
            return self.check_json_file(file_path)
        elif ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
            return self.check_image_file(file_path)
        elif ext == '.bin':
            return self.check_bin_file(file_path)
        
        return True
    
    def scan_directory(self, directory: Path, file_extensions: List[str] = None) -> None:
        """디렉토리를 재귀적으로 스캔하여 파일 검사"""
        if not directory.exists():
            logger.error(f"Directory does not exist: {directory}")
            return
        
        logger.info(f"Scanning directory: {directory}")
        
        for root, dirs, files in os.walk(directory):
            root_path = Path(root)
            
            for file in files:
                file_path = root_path / file
                self.total_files += 1
                
                # 특정 확장자만 검사하는 경우
                if file_extensions:
                    if file_path.suffix.lower() not in file_extensions:
                        continue
                
                try:
                    self.check_file_integrity(file_path)
                    self.checked_files += 1
                    
                    # 진행 상황 출력 (1000개 파일마다)
                    if self.checked_files % 1000 == 0:
                        logger.info(f"Checked {self.checked_files} files...")
                        
                except Exception as e:
                    logger.error(f"Unexpected error checking {file_path}: {e}")
    
    def generate_report(self) -> Dict:
        """검사 결과 리포트 생성"""
        report = {
            'total_files': self.total_files,
            'checked_files': self.checked_files,
            'corrupted_files': len(self.corrupted_files),
            'zero_size_files': len(self.zero_size_files),
            'corrupted_file_list': self.corrupted_files,
            'zero_size_file_list': self.zero_size_files
        }
        
        logger.info("=" * 50)
        logger.info("검사 결과 리포트")
        logger.info("=" * 50)
        logger.info(f"총 파일 수: {self.total_files}")
        logger.info(f"검사한 파일 수: {self.checked_files}")
        logger.info(f"손상된 파일 수: {len(self.corrupted_files)}")
        logger.info(f"크기가 0인 파일 수: {len(self.zero_size_files)}")
        
        if self.corrupted_files:
            logger.info("\n손상된 파일 목록:")
            for file in self.corrupted_files:
                logger.info(f"  - {file}")
        
        if self.zero_size_files:
            logger.info("\n크기가 0인 파일 목록:")
            for file in self.zero_size_files:
                logger.info(f"  - {file}")
        
        return report

def main():
    parser = argparse.ArgumentParser(description='NuScenes 데이터셋 파일 무결성 검사')
    parser.add_argument('--data_root', type=str, default='data/nuscenes',
                       help='NuScenes 데이터 루트 경로')
    parser.add_argument('--extensions', type=str, nargs='+',
                       default=['.pkl', '.json', '.jpg', '.jpeg', '.png', '.bin'],
                       help='검사할 파일 확장자 목록')
    parser.add_argument('--output', type=str, default='corrupted_files_report.json',
                       help='결과 리포트 저장 파일')
    
    args = parser.parse_args()
    
    # 데이터 루트 경로 확인
    data_root = Path(args.data_root)
    if not data_root.exists():
        logger.error(f"Data root directory does not exist: {data_root}")
        sys.exit(1)
    
    logger.info(f"Starting file integrity check for: {data_root}")
    logger.info(f"Checking file extensions: {args.extensions}")
    
    # 파일 검사기 생성 및 실행
    checker = NuScenesFileChecker(str(data_root))
    
    # 주요 디렉토리들 검사
    important_dirs = [
        'samples', 'sweeps', 'maps', 'gts', 'imgseg', 'lidarseg',
        'v1.0-trainval', 'v1.0-test'
    ]
    
    for dir_name in important_dirs:
        dir_path = data_root / dir_name
        if dir_path.exists():
            logger.info(f"\n검사 중: {dir_name}")
            checker.scan_directory(dir_path, args.extensions)
    
    # 루트 디렉토리의 파일들도 검사
    logger.info("\n검사 중: 루트 디렉토리 파일들")
    for file_path in data_root.glob('*'):
        if file_path.is_file():
            checker.total_files += 1
            if file_path.suffix.lower() in args.extensions:
                checker.check_file_integrity(file_path)
                checker.checked_files += 1
    
    # 리포트 생성
    report = checker.generate_report()
    
    # 결과를 JSON 파일로 저장
    import json
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n결과가 {args.output}에 저장되었습니다.")
    
    # 종료 코드 설정
    if report['corrupted_files'] > 0 or report['zero_size_files'] > 0:
        logger.warning("손상된 파일이나 크기가 0인 파일이 발견되었습니다!")
        sys.exit(1)
    else:
        logger.info("모든 파일이 정상입니다!")
        sys.exit(0)

if __name__ == "__main__":
    main() 