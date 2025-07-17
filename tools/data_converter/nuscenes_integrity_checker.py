#!/usr/bin/env python3
"""
NuScenes 데이터 무결성 검증 스크립트

이 스크립트는 NuScenes 데이터셋의 무결성을 확인합니다:
- 메타데이터 파일 존재 여부
- 센서 데이터 파일 존재 여부
- 데이터 일관성 검증
- 파일 크기 및 형식 검증
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import argparse
from datetime import datetime

from nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
import numpy as np


class NuScenesIntegrityChecker:
    """NuScenes 데이터 무결성 검증 클래스"""
    
    def __init__(self, data_root: str, version: str = 'v1.0-trainval'):
        """
        Args:
            data_root: NuScenes 데이터 루트 디렉토리
            version: NuScenes 버전
        """
        self.data_root = Path(data_root)
        self.version = version
        self.nusc = None
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """로거 설정"""
        logger = logging.getLogger('NuScenesIntegrityChecker')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def initialize_nuscenes(self) -> bool:
        """NuScenes 객체 초기화"""
        try:
            self.nusc = NuScenes(version=self.version, dataroot=str(self.data_root))
            self.logger.info(f"NuScenes 초기화 성공: {self.version}")
            return True
        except Exception as e:
            self.logger.error(f"NuScenes 초기화 실패: {e}")
            return False
    
    def check_metadata_files(self) -> Dict[str, bool]:
        """메타데이터 파일 존재 여부 확인"""
        self.logger.info("메타데이터 파일 검증 시작...")
        
        required_files = [
            'attribute.json',
            'category.json', 
            'visibility.json',
            'instance.json',
            'sensor.json',
            'calibrated_sensor.json',
            'ego_pose.json',
            'map.json',
            'scene.json',
            'sample_annotation.json',
            'sample_data.json',
            'sample.json',
            'map.json'
        ]
        
        results = {}
        for file_name in required_files:
            file_path = self.data_root / file_name
            exists = file_path.exists()
            results[file_name] = exists
            status = "✓" if exists else "✗"
            self.logger.info(f"{status} {file_name}")
            
        return results
    
    def check_sensor_data_files(self) -> Dict[str, List[str]]:
        """센서 데이터 파일 존재 여부 확인"""
        self.logger.info("센서 데이터 파일 검증 시작...")
        
        if not self.nusc:
            self.logger.error("NuScenes 객체가 초기화되지 않았습니다.")
            return {}
        
        missing_files = {
            'camera': [],
            'lidar': [],
            'radar': []
        }
        
        # 모든 sample_data 확인
        for sample_data in self.nusc.sample_data:
            file_path = self.data_root / sample_data['filename']
            
            if not file_path.exists():
                sensor_type = sample_data['sensor_modality']
                missing_files[sensor_type].append(sample_data['filename'])
                
        # 결과 로깅
        for sensor_type, files in missing_files.items():
            if len(files) > 0:
                self.logger.warning(f"{sensor_type} 센서에서 {len(files)}개 파일 누락")
                for file in files[:5]:  # 처음 5개만 표시
                    self.logger.warning(f"  - {file}")
                if len(files) > 5:
                    self.logger.warning(f"  ... 및 {len(files) - 5}개 더")
            else:
                self.logger.info(f"{sensor_type} 센서 파일 모두 존재 ✓")
                
        return missing_files
    
    def check_data_consistency(self) -> Dict[str, Any]:
        """데이터 일관성 검증"""
        self.logger.info("데이터 일관성 검증 시작...")
        
        if not self.nusc:
            self.logger.error("NuScenes 객체가 초기화되지 않았습니다.")
            return {}
        
        consistency_checks = {
            'scene_count': len(self.nusc.scene),
            'sample_count': len(self.nusc.sample),
            'sample_data_count': len(self.nusc.sample_data),
            'sample_annotation_count': len(self.nusc.sample_annotation),
            'instance_count': len(self.nusc.instance),
            'category_count': len(self.nusc.category),
            'sensor_count': len(self.nusc.sensor),
            'calibrated_sensor_count': len(self.nusc.calibrated_sensor),
            'ego_pose_count': len(self.nusc.ego_pose),
        }
        
        # 각 씬의 샘플 수 확인
        scene_sample_counts = []
        for scene in self.nusc.scene:
            # 씬의 첫 번째 샘플을 찾아서 해당 씬의 샘플들을 카운트
            first_sample = self.nusc.get('sample', scene['first_sample_token'])
            sample_count = 0
            current_sample = first_sample
            while current_sample is not None:
                sample_count += 1
                if current_sample['next']:
                    current_sample = self.nusc.get('sample', current_sample['next'])
                else:
                    break
            scene_sample_counts.append(sample_count)
        
        # scene_sample_counts는 리스트이므로, 타입 오류를 피하기 위해 Any 타입으로 명시
        consistency_checks['scene_sample_counts'] = scene_sample_counts  # type: ignore

        # 평균 샘플 수는 float이므로, 타입 오류를 피하기 위해 Any 타입으로 명시
        if scene_sample_counts:
            consistency_checks['avg_samples_per_scene'] = float(np.mean(scene_sample_counts))  # type: ignore
        else:
            consistency_checks['avg_samples_per_scene'] = 0.0  # type: ignore
        
        # 결과 로깅
        for key, value in consistency_checks.items():
            if key != 'scene_sample_counts':
                self.logger.info(f"{key}: {value}")
                
        return consistency_checks
    
    def check_lidar_data_integrity(self, max_samples: int = 10) -> Dict[str, Any]:
        """LiDAR 데이터 무결성 검증"""
        self.logger.info("LiDAR 데이터 무결성 검증 시작...")
        
        if not self.nusc:
            self.logger.error("NuScenes 객체가 초기화되지 않았습니다.")
            return {}
        
        lidar_checks = {
            'total_lidar_samples': 0,
            'successful_loads': 0,
            'failed_loads': 0,
            'point_count_stats': [],
            'file_size_stats': [],
            'failed_files': []  # 실패한 파일 경로 저장
        }
        
        # LiDAR 샘플 데이터 찾기
        lidar_samples = [sd for sd in self.nusc.sample_data 
                        if sd['sensor_modality'] == 'lidar']
        
        lidar_checks['total_lidar_samples'] = len(lidar_samples)
        
        # 일부 샘플만 검증 (성능상 이유)
        test_samples = lidar_samples[:max_samples]
        
        for i, sample_data in enumerate(test_samples):
            try:
                # LiDAR 데이터 로드
                pc = LidarPointCloud.from_file(
                    self.data_root / sample_data['filename']
                )
                
                # 포인트 수 확인
                point_count = pc.points.shape[1]
                lidar_checks['point_count_stats'].append(point_count)
                
                # 파일 크기 확인
                file_path = self.data_root / sample_data['filename']
                file_size = file_path.stat().st_size
                lidar_checks['file_size_stats'].append(file_size)
                
                lidar_checks['successful_loads'] += 1
                self.logger.info(f"LiDAR 샘플 {i+1}/{len(test_samples)}: "
                               f"{point_count} 포인트, {file_size/1024/1024:.2f}MB")
                
            except Exception as e:
                lidar_checks['failed_loads'] += 1
                failed_file_path = str(self.data_root / sample_data['filename'])
                lidar_checks['failed_files'].append(failed_file_path)
                self.logger.error(f"LiDAR 샘플 {i+1} 로드 실패: {e}")
                self.logger.error(f"실패한 파일 경로: {failed_file_path}")
                
        # 통계 계산
        if lidar_checks['point_count_stats']:
            lidar_checks['avg_points'] = np.mean(lidar_checks['point_count_stats'])
            lidar_checks['min_points'] = np.min(lidar_checks['point_count_stats'])
            lidar_checks['max_points'] = np.max(lidar_checks['point_count_stats'])
            
        if lidar_checks['file_size_stats']:
            lidar_checks['avg_file_size_mb'] = np.mean(lidar_checks['file_size_stats']) / 1024 / 1024
            
        return lidar_checks
    
    def check_camera_data_integrity(self, max_samples: int = 10) -> Dict[str, Any]:
        """카메라 데이터 무결성 검증"""
        self.logger.info("카메라 데이터 무결성 검증 시작...")
        
        if not self.nusc:
            self.logger.error("NuScenes 객체가 초기화되지 않았습니다.")
            return {}
        
        camera_checks = {
            'total_camera_samples': 0,
            'successful_loads': 0,
            'failed_loads': 0,
            'file_size_stats': [],
            'failed_files': []  # 실패한 파일 경로 저장
        }
        
        # 카메라 샘플 데이터 찾기
        camera_samples = [sd for sd in self.nusc.sample_data 
                         if sd['sensor_modality'] == 'camera']
        
        camera_checks['total_camera_samples'] = len(camera_samples)
        
        # 일부 샘플만 검증
        test_samples = camera_samples[:max_samples]
        
        for i, sample_data in enumerate(test_samples):
            try:
                file_path = self.data_root / sample_data['filename']
                
                if file_path.exists():
                    file_size = file_path.stat().st_size
                    camera_checks['file_size_stats'].append(file_size)
                    camera_checks['successful_loads'] += 1
                    
                    self.logger.info(f"카메라 샘플 {i+1}/{len(test_samples)}: "
                                   f"{file_size/1024/1024:.2f}MB")
                else:
                    camera_checks['failed_loads'] += 1
                    failed_file_path = str(self.data_root / sample_data['filename'])
                    camera_checks['failed_files'].append(failed_file_path)
                    self.logger.error(f"카메라 파일 없음: {sample_data['filename']}")
                    self.logger.error(f"실패한 파일 경로: {failed_file_path}")
                    
            except Exception as e:
                camera_checks['failed_loads'] += 1
                failed_file_path = str(self.data_root / sample_data['filename'])
                camera_checks['failed_files'].append(failed_file_path)
                self.logger.error(f"카메라 샘플 {i+1} 검증 실패: {e}")
                self.logger.error(f"실패한 파일 경로: {failed_file_path}")
                
        # 통계 계산
        if camera_checks['file_size_stats']:
            camera_checks['avg_file_size_mb'] = np.mean(camera_checks['file_size_stats']) / 1024 / 1024
            
        return camera_checks
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """검증 결과 리포트 생성"""
        report = []
        report.append("=" * 60)
        report.append("NuScenes 데이터 무결성 검증 리포트")
        report.append("=" * 60)
        report.append(f"검증 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"데이터 루트: {self.data_root}")
        report.append(f"버전: {self.version}")
        report.append("")
        
        # 메타데이터 검증 결과
        if 'metadata' in results:
            report.append("메타데이터 파일 검증:")
            metadata_results = results['metadata']
            all_good = all(metadata_results.values())
            status = "✓ 모든 파일 존재" if all_good else "✗ 일부 파일 누락"
            report.append(f"  {status}")
            
        # 센서 데이터 검증 결과
        if 'sensor_data' in results:
            report.append("\n센서 데이터 파일 검증:")
            sensor_results = results['sensor_data']
            for sensor_type, missing_files in sensor_results.items():
                if missing_files:
                    report.append(f"  {sensor_type}: {len(missing_files)}개 파일 누락")
                else:
                    report.append(f"  {sensor_type}: ✓ 모든 파일 존재")
                    
        # 데이터 일관성 결과
        if 'consistency' in results:
            report.append("\n데이터 일관성:")
            consistency = results['consistency']
            report.append(f"  씬 수: {consistency.get('scene_count', 'N/A')}")
            report.append(f"  샘플 수: {consistency.get('sample_count', 'N/A')}")
            report.append(f"  평균 샘플/씬: {consistency.get('avg_samples_per_scene', 'N/A'):.1f}")
            
        # LiDAR 검증 결과
        if 'lidar_integrity' in results:
            report.append("\nLiDAR 데이터 무결성:")
            lidar = results['lidar_integrity']
            report.append(f"  총 샘플: {lidar.get('total_lidar_samples', 'N/A')}")
            report.append(f"  성공 로드: {lidar.get('successful_loads', 'N/A')}")
            report.append(f"  실패 로드: {lidar.get('failed_loads', 'N/A')}")
            if 'avg_points' in lidar:
                report.append(f"  평균 포인트 수: {lidar['avg_points']:.0f}")
            if 'avg_file_size_mb' in lidar:
                report.append(f"  평균 파일 크기: {lidar['avg_file_size_mb']:.2f}MB")
            
            # 실패한 파일 경로 출력
            if 'failed_files' in lidar and lidar['failed_files']:
                report.append(f"  실패한 파일들:")
                for failed_file in lidar['failed_files']:
                    report.append(f"    - {failed_file}")
                
        # 카메라 검증 결과
        if 'camera_integrity' in results:
            report.append("\n카메라 데이터 무결성:")
            camera = results['camera_integrity']
            report.append(f"  총 샘플: {camera.get('total_camera_samples', 'N/A')}")
            report.append(f"  성공 로드: {camera.get('successful_loads', 'N/A')}")
            report.append(f"  실패 로드: {camera.get('failed_loads', 'N/A')}")
            if 'avg_file_size_mb' in camera:
                report.append(f"  평균 파일 크기: {camera['avg_file_size_mb']:.2f}MB")
            
            # 실패한 파일 경로 출력
            if 'failed_files' in camera and camera['failed_files']:
                report.append(f"  실패한 파일들:")
                for failed_file in camera['failed_files']:
                    report.append(f"    - {failed_file}")
                
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)
    
    def run_full_check(self, max_samples: int = 10) -> Dict[str, Any]:
        """전체 무결성 검증 실행"""
        self.logger.info("NuScenes 데이터 무결성 검증 시작")
        
        results = {}
        
        # 1. NuScenes 초기화
        if not self.initialize_nuscenes():
            return {'error': 'NuScenes 초기화 실패'}
            
        # 2. 메타데이터 파일 검증
        results['metadata'] = self.check_metadata_files()
        
        # 3. 센서 데이터 파일 검증
        results['sensor_data'] = self.check_sensor_data_files()
        
        # 4. 데이터 일관성 검증
        results['consistency'] = self.check_data_consistency()
        
        # 5. LiDAR 데이터 무결성 검증
        results['lidar_integrity'] = self.check_lidar_data_integrity(max_samples)
        
        # 6. 카메라 데이터 무결성 검증
        results['camera_integrity'] = self.check_camera_data_integrity(max_samples)
        
        self.logger.info("무결성 검증 완료")
        return results


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='NuScenes 데이터 무결성 검증')
    parser.add_argument('--data-root', required=True, 
                       help='NuScenes 데이터 루트 디렉토리')
    parser.add_argument('--version', default='v1.0-trainval',
                       help='NuScenes 버전 (기본값: v1.0-trainval)')
    parser.add_argument('--max-samples', type=int, default=10,
                       help='검증할 최대 샘플 수 (기본값: 10)')
    parser.add_argument('--output', default='nuscenes_integrity_report.txt',
                       help='리포트 출력 파일 (기본값: nuscenes_integrity_report.txt)')
    
    args = parser.parse_args()
    
    # 검증 실행
    checker = NuScenesIntegrityChecker(args.data_root, args.version)
    results = checker.run_full_check(args.max_samples)
    
    # 리포트 생성 및 저장
    report = checker.generate_report(results)
    
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(report)
        
    print(report)
    print(f"\n리포트가 {args.output}에 저장되었습니다.")


if __name__ == '__main__':
    main() 