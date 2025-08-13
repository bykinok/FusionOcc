# Use NVIDIA CUDA base image with Python 3.10
FROM nvidia/cuda:12.4.1-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6 9.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV FORCE_CUDA="1"

# Install the required packages
RUN apt-get update \
    && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


# Miniconda 설치
RUN apt-get update && apt-get install -y wget bzip2 && \
    wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    /opt/conda/bin/conda clean -afy

ENV PATH="/opt/conda/bin:$PATH"

# conda 최신화 및 pytorch 설치
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

RUN conda update -n base -c defaults conda -y 

# Miniconda base 환경의 Python 버전을 3.10으로 맞춤
RUN conda install -y python=3.10

# Install PyTorch with CUDA support 
RUN pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124


# Install MMCV and MMDetection (skip for now due to network issues)
RUN pip install openmim && \
    mim install "mmengine==0.10.3" "mmcv==2.1.0" "mmdet==3.2.0"

# Install additional dependencies (skip for now due to network issues)
RUN pip install \
    lyft_dataset_sdk \
    "networkx>=2.5" \
    "numba" \
    "numpy<2.0" \
    "nuscenes-devkit" \
    "open3d" \
    plyfile \
    scikit-image \
    tensorboard \
    trimesh \
    torch-scatter -f https://data.pyg.org/whl/torch-2.4.0+cu124.html

# Install MMDetection3D
RUN conda clean --all \
&& git clone https://github.com/bykinok/FusionOcc.git -b fusionocc-integration /workspace/FusionOcc \
&& cd /workspace/FusionOcc \
&& pip install --no-cache-dir -e .

# Set working directory
WORKDIR /workspace/FusionOcc

# Set default command
CMD ["/bin/bash"] 