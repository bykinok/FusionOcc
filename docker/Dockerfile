# 1. CUDA 11.3 + Ubuntu 20.04 베이스 이미지 사용 (devel 버전으로 변경)
FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

# 2. DNS 설정 및 네트워크 문제 해결
RUN echo "nameserver 8.8.8.8" > /etc/resolv.conf && \
    echo "nameserver 8.8.4.4" >> /etc/resolv.conf

# 3. 환경 변수 설정 (비대화형 설치)
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

# 4. 패키지 소스 목록 업데이트 및 재시도 로직
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get update --fix-missing || \
    (sleep 10 && apt-get update --fix-missing) || \
    (sleep 30 && apt-get update --fix-missing)

# 5. 시스템 필수 패키지 및 Python 설치 (재시도 로직 포함)
RUN apt-get install -y \
    python3.8 \
    python3.8-dev \
    python3.8-distutils \
    python3-pip \
    wget \
    curl \
    git \
    vim \
    build-essential \
    libgl1-mesa-glx \
    ca-certificates \
    software-properties-common \
    net-tools \
    iputils-ping \
    || (apt-get update && apt-get install -y \
    python3.8 \
    python3.8-dev \
    python3.8-distutils \
    python3-pip \
    wget \
    curl \
    git \
    vim \
    build-essential \
    libgl1-mesa-glx \
    ca-certificates \
    software-properties-common \
    net-tools \
    iputils-ping) \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# 6. Python 3.8을 기본 python으로 설정
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1

# 7. pip 업그레이드
RUN python -m pip install --upgrade pip setuptools wheel

# 8. requirements.txt 복사 및 패키지 설치
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113

# 9. torch-scatter를 PyTorch Geometric 저장소에서 직접 설치
RUN pip install torch-scatter==2.0.9 -f https://data.pyg.org/whl/torch-1.10.1+cu113.html

# 10. 작업 디렉토리 설정
WORKDIR /workspace

# 11. 컨테이너 시작 시 기본 명령어
CMD ["python"]
