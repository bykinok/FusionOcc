# STCOcc

This directory contains STCOcc model implementation for MMDetection3D framework.

## Configs

- `stcocc_r50_704x256_16f_openocc_12e.py`: Configuration for OpenOcc dataset
- `stcocc_r50_704x256_16f_occ3d_36e.py`: Configuration for Occ3D dataset

## Components

The STCOcc model consists of:
- Forward projection module
- Backward projection module  
- Temporal fusion module
- Occupancy head
- Flow head (for OpenOcc)

## Usage

```bash
# Training
python tools/train.py projects/STCOcc/configs/stcocc_r50_704x256_16f_openocc_12e.py

# Testing
python tools/test.py projects/STCOcc/configs/stcocc_r50_704x256_16f_openocc_12e.py work_dirs/stcocc_openocc/latest.pth
```
