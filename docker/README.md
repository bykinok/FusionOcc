# Docker 환경 사용법

이 폴더는 FusionOcc 프로젝트의 Docker 환경 설정 파일들을 포함합니다.

## 파일 구조

- `Dockerfile` - Docker 이미지 빌드 설정
- `docker-compose.yml` - Docker Compose 설정
- `.dockerignore` - Docker 빌드 시 제외할 파일들
- `README_Docker.md` - 상세한 Docker 사용법 가이드
- `Dockerfile_old` - 이전 버전의 Dockerfile (참고용)

## 빠른 시작

### 1. Docker 환경 빌드 및 실행

```bash
# docker 폴더에서 실행
cd docker
docker-compose up --build
```

### 2. 컨테이너 접속

```bash
# docker 폴더에서 실행
docker-compose exec fusionocc bash
```

### 3. 컨테이너 중지

```bash
# docker 폴더에서 실행
docker-compose down
```

## 주요 특징

- **상위 디렉토리 컨텍스트**: `context: ..`로 설정하여 프로젝트 루트의 모든 파일에 접근
- **볼륨 마운트**: 프로젝트 루트가 컨테이너의 `/workspace/FusionOcc`에 마운트
- **GPU 지원**: NVIDIA Docker를 통한 GPU 가속 지원
- **포트 포워딩**: Jupyter Notebook (8888), TensorBoard (6006)

## 환경 정보

- Python 3.10
- CUDA 12.1
- PyTorch 2.1.0
- MMDetection3D 1.3.0

## 상세 가이드

더 자세한 사용법은 `README_Docker.md` 파일을 참조하세요. 