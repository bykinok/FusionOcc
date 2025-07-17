# NuScenes 데이터 무결성 검증 도구

이 도구는 NuScenes 데이터셋의 무결성을 확인하는 Python 스크립트입니다.

## 기능

- **메타데이터 파일 검증**: 필수 JSON 파일들의 존재 여부 확인
- **센서 데이터 파일 검증**: 카메라, LiDAR, 레이더 파일들의 존재 여부 확인
- **데이터 일관성 검증**: 씬, 샘플, 어노테이션 등의 개수 및 관계 확인
- **LiDAR 데이터 무결성 검증**: 포인트 클라우드 파일 로드 및 포인트 수 확인
- **카메라 데이터 무결성 검증**: 이미지 파일 크기 및 존재 여부 확인

## 요구사항

```bash
pip install nuscenes-devkit numpy
```

## 사용법

### 1. 명령행에서 직접 실행

```bash
# 기본 사용법
python nuscenes_integrity_checker.py --data-root /path/to/nuscenes

# 옵션 지정
python nuscenes_integrity_checker.py \
    --data-root /path/to/nuscenes \
    --version v1.0-trainval \
    --max-samples 20 \
    --output integrity_report.txt
```

### 2. Python 스크립트에서 사용

```python
from nuscenes_integrity_checker import NuScenesIntegrityChecker

# 검증기 초기화
checker = NuScenesIntegrityChecker(
    data_root="/path/to/nuscenes",
    version="v1.0-trainval"
)

# 전체 검증 실행
results = checker.run_full_check(max_samples=10)

# 리포트 생성
report = checker.generate_report(results)
print(report)
```

### 3. 예제 스크립트 실행

```bash
# 환경 변수 설정
export NUSCENES_DATA_ROOT=/path/to/nuscenes

# 예제 스크립트 실행
python run_integrity_check.py
```

## 명령행 옵션

- `--data-root`: NuScenes 데이터 루트 디렉토리 (필수)
- `--version`: NuScenes 버전 (기본값: v1.0-trainval)
- `--max-samples`: 검증할 최대 샘플 수 (기본값: 10)
- `--output`: 리포트 출력 파일 (기본값: nuscenes_integrity_report.txt)

## 검증 항목

### 1. 메타데이터 파일
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

### 2. 센서 데이터 파일
- 카메라 이미지 파일
- LiDAR 포인트 클라우드 파일
- 레이더 데이터 파일

### 3. 데이터 일관성
- 씬, 샘플, 어노테이션 개수
- 각 씬별 샘플 수
- 센서 및 캘리브레이션 정보

### 4. 데이터 무결성
- LiDAR 파일 로드 가능 여부
- 포인트 클라우드 포인트 수
- 파일 크기 통계
- 카메라 이미지 파일 크기

## 출력 예제

```
============================================================
NuScenes 데이터 무결성 검증 리포트
============================================================
검증 시간: 2024-01-15 14:30:25
데이터 루트: /path/to/nuscenes
버전: v1.0-trainval

메타데이터 파일 검증:
  ✓ 모든 파일 존재

센서 데이터 파일 검증:
  camera: ✓ 모든 파일 존재
  lidar: ✓ 모든 파일 존재
  radar: ✓ 모든 파일 존재

데이터 일관성:
  씬 수: 1000
  샘플 수: 40000
  평균 샘플/씬: 40.0

LiDAR 데이터 무결성:
  총 샘플: 40000
  성공 로드: 10
  실패 로드: 0
  평균 포인트 수: 32000
  평균 파일 크기: 15.23MB

카메라 데이터 무결성:
  총 샘플: 160000
  성공 로드: 10
  실패 로드: 0
  평균 파일 크기: 2.45MB
============================================================
```

## 문제 해결

### 1. NuScenes 초기화 실패
- 데이터 경로가 올바른지 확인
- NuScenes 버전이 데이터와 일치하는지 확인
- `nuscenes-devkit`이 설치되어 있는지 확인

### 2. 파일 누락 오류
- 데이터 다운로드가 완료되었는지 확인
- 파일 권한 문제가 없는지 확인
- 디스크 공간이 충분한지 확인

### 3. 메모리 부족
- `--max-samples` 옵션으로 검증할 샘플 수를 줄이기
- 대용량 데이터셋의 경우 일부만 검증

## 라이센스

이 도구는 MIT 라이센스 하에 배포됩니다. 