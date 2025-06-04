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
    ninja-build \
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
    iputils-ping \
    ninja-build) \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# 5.5. GUI Visualization을 위한 X11 및 OpenGL 패키지 설치
RUN apt-get update && apt-get install -y \
    x11-apps \
    python3-tk \
    libglu1-mesa-dev \
    freeglut3-dev \
    mesa-utils \
    libice6 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libfontconfig1 \
    libgtk-3-0 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# 6. Python 3.8을 기본 python으로 설정
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1

# 6.5. CUDA 환경 변수 설정
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV LIBRARY_PATH=${CUDA_HOME}/lib64:${LIBRARY_PATH}
ENV CPATH=${CUDA_HOME}/include:${CPATH}
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6"

# 6.6. GUI Visualization을 위한 환경 변수 설정
ENV XDG_RUNTIME_DIR=/tmp/runtime-root
ENV MESA_GL_VERSION_OVERRIDE=3.3
ENV MESA_GLSL_VERSION_OVERRIDE=330
ENV LIBGL_ALWAYS_SOFTWARE=1

# 6.7. X11 및 GUI 지원을 위한 디렉토리 생성
RUN mkdir -p /tmp/runtime-root && \
    chmod 755 /tmp/runtime-root

# 7. pip 업그레이드
RUN python -m pip install --upgrade pip setuptools wheel

# 8. requirements.txt 복사 및 패키지 설치
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113

# 9. mmcv-full 별도 설치
RUN pip install mmcv-full==1.5.3 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.1/index.html

# 10. torch-scatter를 PyTorch Geometric 저장소에서 직접 설치
RUN pip install torch-scatter==2.0.9 -f https://data.pyg.org/whl/torch-1.10.1+cu113.html

# 11. 작업 디렉토리 설정
WORKDIR /workspace

# 12. FusionOcc 리포지토리 클론 및 설치
RUN git clone https://github.com/bykinok/FusionOcc.git && \
    cd FusionOcc && \
    pip install -v -e .

# 13. 컨테이너 시작 시 기본 명령어
CMD ["python"]

# GUI Visualization 사용을 위한 Docker 실행 명령어 예시:
# docker run -it --gpus all \
#   -e DISPLAY=$DISPLAY \
#   -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
#   -v $(pwd):/workspace \
#   --name fusionocc_container \
#   fusionocc:latest /bin/bash
