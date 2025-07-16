#!/bin/bash

# NuScenes 데이터셋 파일 무결성 검사 실행 스크립트

echo "=========================================="
echo "NuScenes 데이터셋 파일 무결성 검사"
echo "=========================================="

# 기본 설정
DATA_ROOT="data/nuscenes"
OUTPUT_FILE="corrupted_files_report.json"
LOG_FILE="corrupted_files_check.log"

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 데이터 디렉토리 존재 확인
if [ ! -d "$DATA_ROOT" ]; then
    echo -e "${RED}오류: $DATA_ROOT 디렉토리가 존재하지 않습니다.${NC}"
    exit 1
fi

echo -e "${YELLOW}데이터 루트: $DATA_ROOT${NC}"
echo -e "${YELLOW}출력 파일: $OUTPUT_FILE${NC}"
echo -e "${YELLOW}로그 파일: $LOG_FILE${NC}"
echo ""

# Python 환경 확인
if ! command -v python &> /dev/null; then
    echo -e "${RED}오류: Python이 설치되지 않았습니다.${NC}"
    exit 1
fi

# 필요한 Python 패키지 확인
echo "Python 패키지 확인 중..."
python -c "import pathlib, pickle, json, logging" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${RED}오류: 필요한 Python 패키지가 설치되지 않았습니다.${NC}"
    exit 1
fi

# PIL 패키지 확인 (이미지 파일 검사용)
python -c "from PIL import Image" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}경고: PIL 패키지가 설치되지 않았습니다. 이미지 파일 검사가 제한됩니다.${NC}"
    echo "PIL 설치: pip install Pillow"
    echo ""
fi

echo -e "${GREEN}모든 준비가 완료되었습니다.${NC}"
echo ""

# 사용자에게 실행 확인
read -p "파일 무결성 검사를 시작하시겠습니까? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "검사가 취소되었습니다."
    exit 0
fi

echo ""
echo "검사를 시작합니다..."
echo "=========================================="

# 스크립트 실행
python check_corrupted_files.py \
    --data_root "$DATA_ROOT" \
    --output "$OUTPUT_FILE" \
    --extensions .pkl .json .jpg .jpeg .png .bmp .tiff .tif .bin

# 실행 결과 확인
EXIT_CODE=$?

echo ""
echo "=========================================="
echo "검사 완료"
echo "=========================================="

if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✅ 모든 파일이 정상입니다!${NC}"
else
    echo -e "${RED}❌ 손상된 파일이나 크기가 0인 파일이 발견되었습니다!${NC}"
    echo ""
    echo "자세한 내용은 다음 파일들을 확인하세요:"
    echo "- $OUTPUT_FILE (상세 리포트)"
    echo "- $LOG_FILE (로그 파일)"
fi

echo ""
echo "결과 파일:"
if [ -f "$OUTPUT_FILE" ]; then
    echo "- $OUTPUT_FILE"
fi
if [ -f "$LOG_FILE" ]; then
    echo "- $LOG_FILE"
fi

echo ""
echo "스크립트 실행 완료." 