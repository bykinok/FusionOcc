#!/bin/bash

# FusionOcc 데이터 디렉토리 및 심볼릭 링크 설정 스크립트

echo "FusionOcc 데이터 디렉토리 설정을 시작합니다..."

# 데이터 디렉토리 생성
echo "데이터 디렉토리 생성 중..."
mkdir -p /workspace/FusionOcc/data/nuscenes

# 심볼릭 링크 생성
echo "심볼릭 링크 생성 중..."

echo "  - gts 디렉토리 링크 생성..."
ln -s /NAS/mmDataset/nuscenes_fusionocc/gts/ /workspace/FusionOcc/data/nuscenes/gts

echo "  - imgseg 디렉토리 링크 생성..."
ln -s /NAS/mmDataset/nuscenes_fusionocc/imgseg/ /workspace/FusionOcc/data/nuscenes/imgseg

echo "  - 훈련 정보 파일 링크 생성..."
ln -s /NAS/mmDataset/nuscenes_fusionocc/fusionocc-nuscenes_infos_train.pkl /workspace/FusionOcc/data/nuscenes/fusionocc-nuscenes_infos_train.pkl

echo "  - 검증 정보 파일 링크 생성..."
ln -s /NAS/mmDataset/nuscenes_fusionocc/fusionocc-nuscenes_infos_val.pkl /workspace/FusionOcc/data/nuscenes/fusionocc-nuscenes_infos_val.pkl

echo "  - lidarseg 디렉토리 링크 생성..."
ln -s /NAS/mmDataset/nuscenes/lidarseg/ /workspace/FusionOcc/data/nuscenes/lidarseg

echo "  - maps 디렉토리 링크 생성..."
ln -s /NAS/mmDataset/nuscenes/maps/ /workspace/FusionOcc/data/nuscenes/maps

echo "  - samples 디렉토리 링크 생성..."
ln -s /NAS/mmDataset/nuscenes/samples /workspace/FusionOcc/data/nuscenes/samples

echo "  - sweeps 디렉토리 링크 생성..."
ln -s /NAS/mmDataset/nuscenes/sweeps/ /workspace/FusionOcc/data/nuscenes/sweeps

echo "  - v1.0-test 디렉토리 링크 생성..."
ln -s /NAS/mmDataset/nuscenes/v1.0-test/ /workspace/FusionOcc/data/nuscenes/v1.0-test

echo "  - v1.0-trainval 디렉토리 링크 생성..."
ln -s /NAS/mmDataset/nuscenes/v1.0-trainval/ /workspace/FusionOcc/data/nuscenes/v1.0-trainval

echo "  - 체크포인트 디렉토리 링크 생성..."
ln -s /NAS/mmDataset/nuscenes_fusionocc/ckpt/ /workspace/FusionOcc/ckpt

echo "  - results 디렉토리 링크 생성..."
ln -s /NAS/mmDataset/nuscenes_fusionocc/results /workspace/FusionOcc/results

echo "설정이 완료되었습니다!"
echo "생성된 링크를 확인하려면 'ls -la /workspace/FusionOcc/data/nuscenes/' 명령어를 사용하세요." 