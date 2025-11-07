"""
데이터 로더 성능 테스트 스크립트

이 스크립트는 데이터 로딩 속도를 측정하여 최적화 효과를 확인합니다.

사용 방법:
    python projects/BEVFormer/test_dataloader_speed.py

예상 결과:
    - 최적화 전: ~2-3초/샘플
    - 최적화 후: ~0.5-0.8초/샘플
"""

import time
import torch
from mmengine.config import Config
from mmdet3d.registry import DATASETS


def test_dataloader_speed(config_path, num_samples=10):
    """데이터 로더 속도 테스트"""
    
    print("=" * 80)
    print("BEVFormer 데이터 로더 성능 테스트")
    print("=" * 80)
    
    # 설정 로드
    print(f"\n1. 설정 파일 로드: {config_path}")
    cfg = Config.fromfile(config_path)
    
    # 데이터셋 생성
    print("\n2. 데이터셋 생성 중...")
    dataset = DATASETS.build(cfg.train_dataloader.dataset)
    print(f"   - 총 샘플 수: {len(dataset)}")
    print(f"   - Queue length: {dataset.queue_length}")
    print(f"   - 프레임 수/샘플: {dataset.queue_length + 1}")
    
    # 워밍업 (첫 로딩은 캐시 미스로 느릴 수 있음)
    print("\n3. 워밍업 (첫 샘플 로딩)...")
    _ = dataset[0]
    
    # 성능 측정
    print(f"\n4. 성능 측정 ({num_samples}개 샘플)...")
    print("-" * 80)
    
    times = []
    for i in range(num_samples):
        start_time = time.time()
        data = dataset[i]
        elapsed = time.time() - start_time
        times.append(elapsed)
        
        # 이미지 정보 출력
        if i == 0:
            print(f"\n   샘플 #{i}:")
            print(f"   - 로딩 시간: {elapsed:.3f}초")
            if 'img' in data:
                if hasattr(data['img'], 'data'):
                    imgs = data['img'].data
                    if isinstance(imgs, torch.Tensor):
                        print(f"   - 이미지 shape: {imgs.shape}")
                        print(f"   - 프레임 수: {imgs.shape[0]}")
        else:
            print(f"   샘플 #{i}: {elapsed:.3f}초")
    
    # 통계 출력
    print("\n" + "=" * 80)
    print("결과 요약")
    print("=" * 80)
    print(f"평균 로딩 시간: {sum(times) / len(times):.3f}초/샘플")
    print(f"최소 로딩 시간: {min(times):.3f}초")
    print(f"최대 로딩 시간: {max(times):.3f}초")
    print(f"총 시간: {sum(times):.3f}초")
    
    # 예상 학습 시간 계산
    total_samples = len(dataset)
    estimated_epoch_time = (sum(times) / len(times)) * total_samples / 3600
    print(f"\n예상 1 epoch 데이터 로딩 시간: {estimated_epoch_time:.2f}시간")
    print("   (실제 학습 시간 = 데이터 로딩 + 모델 forward + backward + 최적화)")
    
    # 성능 평가
    avg_time = sum(times) / len(times)
    print("\n" + "=" * 80)
    print("성능 평가")
    print("=" * 80)
    if avg_time < 1.0:
        print("✅ 우수: 데이터 로딩이 매우 빠릅니다!")
        print("   최적화가 성공적으로 적용되었습니다.")
    elif avg_time < 2.0:
        print("⚠️  보통: 데이터 로딩이 적당한 속도입니다.")
        print("   추가 최적화를 고려할 수 있습니다:")
        print("   - workers_per_gpu를 증가시키기")
        print("   - SSD를 사용하여 데이터 저장")
    else:
        print("❌ 느림: 데이터 로딩이 느립니다!")
        print("   다음을 확인하세요:")
        print("   - persistent_workers=False로 설정되었는지")
        print("   - 커스텀 LoadMultiViewImageFromFiles가 사용되고 있는지")
        print("   - 디스크 I/O 성능")
    
    print("=" * 80)


if __name__ == '__main__':
    import sys
    import os
    
    # 프로젝트 루트를 Python path에 추가
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, project_root)
    
    # 커스텀 모듈을 명시적으로 import (설정 파일 로드 전에 필요)
    print("커스텀 모듈 로딩 중...")
    try:
        import projects.BEVFormer  # noqa: F401
        print("✅ 커스텀 모듈 로딩 완료")
    except Exception as e:
        print(f"⚠️  커스텀 모듈 로딩 경고: {e}")
    
    # 설정 파일 경로
    config_path = 'projects/BEVFormer/configs/bevformer_base_occ.py'
    
    try:
        test_dataloader_speed(config_path, num_samples=10)
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        print("\n다음을 확인하세요:")
        print("1. 데이터가 올바른 경로에 있는지")
        print("2. 필요한 패키지가 모두 설치되어 있는지")
        print("3. 설정 파일이 올바른지")
        import traceback
        traceback.print_exc()

