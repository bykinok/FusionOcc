# FusionOcc Docker 환경

이 문서는 FusionOcc 프로젝트를 위한 Docker 환경 설정 방법을 설명합니다.

## 요구사항

- Docker
- Docker Compose
- NVIDIA Docker (GPU 사용 시)
- NVIDIA 드라이버 (GPU 사용 시)

## 환경 정보

- **Python**: 3.10
- **CUDA**: 12.1
- **PyTorch**: 2.1.0
- **MMDetection3D**: 1.3.0
- **MMCV**: 2.1.0
- **MMEngine**: 0.10.3

## 사용 방법

### 1. Docker 이미지 빌드

```bash
# Docker Compose를 사용하여 빌드 및 실행
docker-compose up --build

# 또는 Docker 명령어로 직접 빌드
docker build -t fusionocc .
```

### 2. 컨테이너 실행

```bash
# Docker Compose 사용
docker-compose up -d

# 또는 Docker 명령어로 직접 실행
docker run --gpus all -it \
  -v $(pwd):/workspace/FusionOcc \
  -p 8888:8888 \
  -p 6006:6006 \
  fusionocc
```

### 3. 컨테이너 접속

```bash
# Docker Compose 사용
docker-compose exec fusionocc bash

# 또는 Docker 명령어로 직접 접속
docker exec -it fusionocc-dev bash
```

### 4. 컨테이너 중지

```bash
# Docker Compose 사용
docker-compose down

# 또는 Docker 명령어로 직접 중지
docker stop fusionocc-dev
```

## 포트 설정

- **8888**: Jupyter Notebook
- **6006**: TensorBoard

## 볼륨 마운트

- 현재 디렉토리가 `/workspace/FusionOcc`에 마운트됩니다
- pip 캐시와 torch 캐시가 호스트와 공유됩니다

## GPU 사용

GPU를 사용하려면 NVIDIA Docker가 설치되어 있어야 합니다:

```bash
# NVIDIA Docker 설치 (Ubuntu)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

## 개발 환경 설정

컨테이너 내부에서 다음 명령어로 개발 환경을 설정할 수 있습니다:

```bash
# 프로젝트 설치 (개발 모드)
pip install -e .

# 테스트 실행
python -m pytest tests/

# 코드 포맷팅
black .
isort .

# 린팅
flake8 .
```

## 문제 해결

### GPU 인식 문제

```bash
# GPU 상태 확인
nvidia-smi

# PyTorch에서 CUDA 사용 가능 여부 확인
python -c "import torch; print(torch.cuda.is_available())"
```

### 메모리 부족 문제

Docker Compose 파일에서 메모리 제한을 설정할 수 있습니다:

```yaml
services:
  fusionocc:
    deploy:
      resources:
        limits:
          memory: 16G
```

### 캐시 문제

```bash
# Docker 빌드 캐시 삭제
docker-compose build --no-cache

# pip 캐시 삭제
docker-compose exec fusionocc pip cache purge
```

## 추가 정보

- 이 Docker 환경은 MMDetection3D 기반의 3D 객체 탐지 프로젝트를 위한 것입니다
- CUDA 12.1과 PyTorch 2.1.0을 지원합니다
- 모든 필요한 의존성이 포함되어 있습니다 