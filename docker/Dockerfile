# Use NVIDIA CUDA base image with Python 3.10
FROM nvidia/cuda:12.1-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV FORCE_CUDA="1"

# Set working directory
WORKDIR /workspace/FusionOcc

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    vim \
    build-essential \
    cmake \
    ninja-build \
    libglib2.0-0 \
    libsm6 \
    libxrender-dev \
    libxext6 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install PyTorch with CUDA support
RUN pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# Install MMCV and MMDetection
RUN pip install openmim && \
    mim install "mmengine>=0.10.3" "mmcv>=2.1.0" "mmdet>=3.2.0"

# Install additional dependencies
RUN pip install \
    lyft_dataset_sdk \
    networkx>=2.5 \
    numba \
    numpy<2.0 \
    nuscenes-devkit \
    open3d \
    plyfile \
    scikit-image \
    tensorboard \
    trimesh \
    torch-scatter==2.1.2

# Install optional dependencies
RUN pip install \
    black==20.8b1 \
    typing-extensions \
    waymo-open-dataset-tf-2-11-0

# Install test dependencies
RUN pip install \
    codecov \
    flake8 \
    interrogate \
    isort \
    kwarray \
    parameterized \
    pytest \
    pytest-cov \
    pytest-runner \
    ubelt \
    xdoctest>=0.10.0 \
    yapf

# Copy the current project files
COPY . /workspace/FusionOcc/

# Install the project in development mode
RUN pip install -e .

# Set default command
CMD ["/bin/bash"] 