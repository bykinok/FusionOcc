# NuScenes 데이터셋 파일 무결성 검사 도구

이 도구는 NuScenes 데이터셋에서 손상된 파일이나 크기가 0인 파일을 찾아내는 Python 스크립트입니다.

## 📋 기능

- **파일 크기 검사**: 크기가 0인 파일 탐지
- **파일 무결성 검사**: 
  - Pickle 파일 (.pkl) 손상 여부 확인
  - JSON 파일 (.json) 손상 여부 확인  
  - 이미지 파일 (.jpg, .jpeg, .png, .bmp, .tiff, .tif) 손상 여부 확인
  - 바이너리 파일 (.bin) 손상 여부 확인
- **재귀적 디렉토리 스캔**: 모든 하위 디렉토리 검사
- **상세한 로깅**: 진행 상황과 오류 로그
- **JSON 리포트 생성**: 검사 결과를 구조화된 형태로 저장

## 🚀 사용법

### 1. 간단한 실행 (권장)

```bash
./run_file_check.sh
```

이 스크립트는 자동으로 환경을 확인하고 검사를 실행합니다.

### 2. 직접 Python 스크립트 실행

```bash
# 기본 설정으로 실행
python check_corrupted_files.py

# 사용자 정의 설정으로 실행
python check_corrupted_files.py \
    --data_root data/nuscenes \
    --extensions .pkl .json .jpg .jpeg .png .bin \
    --output my_report.json
```

### 3. 명령행 옵션

```bash
python check_corrupted_files.py --help
```

**옵션 설명:**
- `--data_root`: NuScenes 데이터 루트 경로 (기본값: `data/nuscenes`)
- `--extensions`: 검사할 파일 확장자 목록 (기본값: `.pkl .json .jpg .jpeg .png .bin`)
- `--output`: 결과 리포트 저장 파일 (기본값: `corrupted_files_report.json`)

## 📊 출력 파일

### 1. 로그 파일 (`corrupted_files_check.log`)
- 검사 진행 상황
- 발견된 오류들
- 상세한 디버깅 정보

### 2. 리포트 파일 (`corrupted_files_report.json`)
```json
{
  "total_files": 1000,
  "checked_files": 950,
  "corrupted_files": 2,
  "zero_size_files": 1,
  "corrupted_file_list": [
    "data/nuscenes/samples/CAM_FRONT/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151603512460.jpg",
    "data/nuscenes/v1.0-trainval/meta.json"
  ],
  "zero_size_file_list": [
    "data/nuscenes/samples/CAM_FRONT/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151603512461.jpg"
  ]
}
```

## 🔍 검사 대상 디렉토리

스크립트는 다음 디렉토리들을 자동으로 검사합니다:

- `samples/` - 카메라 이미지 파일들
- `sweeps/` - 라이다 스윕 데이터
- `maps/` - 지도 데이터
- `gts/` - 그라운드 트루스 데이터
- `imgseg/` - 이미지 세그멘테이션 데이터
- `lidarseg/` - 라이다 세그멘테이션 데이터
- `v1.0-trainval/` - 훈련/검증 메타데이터
- `v1.0-test/` - 테스트 메타데이터

## ⚠️ 주의사항

1. **대용량 데이터셋**: NuScenes 데이터셋은 매우 크므로 검사에 시간이 오래 걸릴 수 있습니다.
2. **메모리 사용량**: 이미지 파일 검사 시 메모리를 사용합니다.
3. **PIL 패키지**: 이미지 파일 검사를 위해서는 `Pillow` 패키지가 필요합니다.

## 🛠️ 설치 요구사항

### 필수 패키지
```bash
# 기본 Python 패키지들 (대부분 기본 설치됨)
# - pathlib
# - pickle  
# - json
# - logging

# 이미지 파일 검사를 위한 패키지
pip install Pillow
```

## 📈 성능 최적화

### 1. 특정 확장자만 검사
```bash
# pickle 파일만 검사
python check_corrupted_files.py --extensions .pkl

# 이미지 파일만 검사  
python check_corrupted_files.py --extensions .jpg .jpeg .png

# 바이너리 파일만 검사
python check_corrupted_files.py --extensions .bin
```

### 2. 특정 디렉토리만 검사
스크립트를 수정하여 `important_dirs` 리스트를 변경하거나, 직접 Python 코드를 수정하여 특정 디렉토리만 검사할 수 있습니다.

## 🔧 문제 해결

### 1. PIL 패키지 오류
```bash
pip install Pillow
```

### 2. 권한 오류
```bash
chmod +x check_corrupted_files.py
chmod +x run_file_check.sh
```

### 3. 메모리 부족
- 이미지 파일 검사를 비활성화하려면 `--extensions`에서 이미지 확장자를 제외하세요
- 또는 스크립트를 수정하여 이미지 검사 부분을 주석 처리하세요

## 📝 예시 출력

```
==========================================
검사 결과 리포트
==========================================
총 파일 수: 15420
검사한 파일 수: 15420
손상된 파일 수: 0
크기가 0인 파일 수: 0

✅ 모든 파일이 정상입니다!
```

## 🤝 기여

버그 리포트나 기능 제안은 이슈를 통해 제출해주세요.

## 📄 라이선스

이 스크립트는 MIT 라이선스 하에 배포됩니다. 