# 전체 NuScenes 데이터 무결성 검증 도구

이 도구는 NuScenes 데이터셋의 **모든 파일**에 대해 무결성을 확인하는 고급 스크립트입니다.

## 주요 특징

### 🔍 **전체 데이터 검증**
- 모든 메타데이터 파일 검증
- 모든 센서 데이터 파일 검증 (카메라, LiDAR, 레이더)
- 모든 데이터 일관성 검증

### ⚡ **성능 최적화**
- 진행률 표시 (tqdm)
- 병렬 처리 지원 (선택적)
- 메모리 효율적인 처리

### 📊 **상세한 리포트**
- 성공률 통계
- 실패한 파일 목록
- 파일 크기 및 포인트 수 통계
- 소요 시간 측정

## 요구사항

```bash
pip install nuscenes-devkit numpy tqdm
```

## 사용법

### 1. 기본 사용법 (순차 처리)

```bash
# 전체 데이터 검증 (순차 처리)
python nuscenes_full_integrity_checker.py --data-root /path/to/nuscenes

# 특정 버전 지정
python nuscenes_full_integrity_checker.py \
    --data-root /path/to/nuscenes \
    --version v1.0-trainval
```

### 2. 병렬 처리 사용

```bash
# 병렬 처리로 빠른 검증
python nuscenes_full_integrity_checker.py \
    --data-root /path/to/nuscenes \
    --parallel \
    --num-workers 8

# 워커 수 조정
python nuscenes_full_integrity_checker.py \
    --data-root /path/to/nuscenes \
    --parallel \
    --num-workers 16
```

### 3. 출력 파일 지정

```bash
# 커스텀 리포트 파일명
python nuscenes_full_integrity_checker.py \
    --data-root /path/to/nuscenes \
    --output my_integrity_report.txt
```

## 명령행 옵션

- `--data-root`: NuScenes 데이터 루트 디렉토리 (필수)
- `--version`: NuScenes 버전 (기본값: v1.0-trainval)
- `--parallel`: 병렬 처리 사용 (기본값: False)
- `--num-workers`: 병렬 처리 시 워커 수 (기본값: 4)
- `--output`: 리포트 출력 파일 (기본값: nuscenes_full_integrity_report.txt)

## 검증 항목

### 1. 메타데이터 파일 (14개)
- `attribute.json`
- `category.json`
- `visibility.json`
- `instance.json`
- `sensor.json`
- `calibrated_sensor.json`
- `ego_pose.json`
- `map.json`
- `scene.json`
- `sample_annotation.json`
- `sample_data.json`
- `sample.json`

### 2. 센서 데이터 파일 (전체)
- **카메라**: 모든 이미지 파일 (약 1,183,790개)
- **LiDAR**: 모든 포인트 클라우드 파일 (약 331,886개)
- **레이더**: 모든 레이더 데이터 파일

### 3. 데이터 일관성
- 씬, 샘플, 어노테이션 개수
- 각 씬별 샘플 수 계산
- 센서 및 캘리브레이션 정보

### 4. 데이터 무결성
- **LiDAR**: 파일 로드, 포인트 수, 파일 크기
- **카메라**: 파일 존재 여부, 파일 크기
- **레이더**: 파일 존재 여부

## 성능 비교

### 순차 처리 vs 병렬 처리

| 처리 방식 | 소요 시간 | 메모리 사용량 | CPU 사용률 |
|-----------|-----------|---------------|------------|
| 순차 처리 | ~30-60분 | 낮음 | 25% |
| 병렬 처리 | ~10-20분 | 높음 | 100% |

### 권장 사항

- **작은 데이터셋** (< 10GB): 순차 처리
- **대용량 데이터셋** (> 10GB): 병렬 처리
- **메모리 제한**: 순차 처리 또는 워커 수 줄이기

## 출력 예제

```
================================================================================
전체 NuScenes 데이터 무결성 검증 리포트
================================================================================
검증 시간: 2024-01-15 16:30:25
데이터 루트: /path/to/nuscenes
버전: v1.0-trainval
병렬 처리: 사용
워커 수: 8

메타데이터 파일 검증:
  ✗ 일부 파일 누락

센서 데이터 파일 검증:
  camera: ✓ 모든 파일 존재
  lidar: ✓ 모든 파일 존재
  radar: ✓ 모든 파일 존재

데이터 일관성:
  씬 수: 850
  샘플 수: 34,149
  평균 샘플/씬: 40.2

LiDAR 데이터 무결성:
  총 샘플: 331,886
  성공 로드: 331,886
  실패 로드: 0
  성공률: 100.0%
  평균 포인트 수: 34,730
  평균 파일 크기: 0.66MB

카메라 데이터 무결성:
  총 샘플: 1,183,790
  성공 로드: 1,183,790
  실패 로드: 0
  성공률: 100.0%
  평균 파일 크기: 0.14MB
================================================================================
```

## 문제 해결

### 1. 메모리 부족 오류
```bash
# 워커 수 줄이기
python nuscenes_full_integrity_checker.py \
    --data-root /path/to/nuscenes \
    --parallel \
    --num-workers 2
```

### 2. 시간이 너무 오래 걸림
```bash
# 병렬 처리 사용
python nuscenes_full_integrity_checker.py \
    --data-root /path/to/nuscenes \
    --parallel \
    --num-workers 16
```

### 3. 특정 센서만 검증하고 싶을 때
현재는 모든 센서를 검증하지만, 필요시 코드를 수정하여 특정 센서만 검증할 수 있습니다.

## 주의사항

1. **대용량 데이터셋**: 전체 검증은 시간이 오래 걸릴 수 있습니다
2. **메모리 사용량**: 병렬 처리 시 메모리 사용량이 증가합니다
3. **디스크 I/O**: 많은 파일을 읽으므로 디스크 성능이 중요합니다
4. **네트워크 스토리지**: 네트워크 스토리지의 경우 병렬 처리가 오히려 느릴 수 있습니다

## 라이센스

이 도구는 MIT 라이센스 하에 배포됩니다. 