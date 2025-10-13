#!/bin/bash

echo "occfrmwrk 데이터 디렉토리 설정을 시작합니다."

mkdir -p /workspace/FusionOcc/data/nuscenes

ln -s /NAS/mmDataset/nuscenes_fusionocc/gts/ /workspace/FusionOcc/data/nuscenes/gts
ln -s /NAS/mmDataset/nuscenes_fusionocc/imgseg/ /workspace/FusionOcc/data/nuscenes/imgseg
ln -s /NAS/mmDataset/nuscenes_fusionocc_h100/fusionocc-nuscenes_infos_train.pkl /workspace/FusionOcc/data/nuscenes/fusionocc-nuscenes_infos_train.pkl
ln -s /NAS/mmDataset/nuscenes_fusionocc_h100/fusionocc-nuscenes_infos_val.pkl /workspace/FusionOcc/data/nuscenes/fusionocc-nuscenes_infos_val.pkl
ln -s /NAS/mmDataset/nuscenes_fusionocc_h100/nuscenes_dbinfos_train.pkl /workspace/FusionOcc/data/nuscenes/nuscenes_dbinfos_train.pkl
ln -s /NAS/mmDataset/nuscenes_fusionocc_h100/nuscenes_infos_train.pkl /workspace/FusionOcc/data/nuscenes/nuscenes_infos_train.pkl
ln -s /NAS/mmDataset/nuscenes_fusionocc_h100/nuscenes_infos_val.pkl /workspace/FusionOcc/data/nuscenes/nuscenes_infos_val.pkl
ln -s /NAS/mmDataset/nuscenes_fusionocc_h100/nuscenes_infos_test.pkl /workspace/FusionOcc/data/nuscenes/nuscenes_infos_test.pkl
ln -s /NAS/mmDataset/nuscenes_fusionocc_h100/nuscenes_gt_database/ /workspace/FusionOcc/data/nuscenes/nuscenes_gt_database
ln -s /NAS/mmDataset/nuscenes/lidarseg/ /workspace/FusionOcc/data/nuscenes/lidarseg
ln -s /NAS/mmDataset/nuscenes/maps/ /workspace/FusionOcc/data/nuscenes/maps
ln -s /NAS/mmDataset/nuscenes/sweeps/ /workspace/FusionOcc/data/nuscenes/sweeps
ln -s /NAS/mmDataset/nuscenes/samples/ /workspace/FusionOcc/data/nuscenes/samples
ln -s /NAS/mmDataset/nuscenes/v1.0-test/ /workspace/FusionOcc/data/nuscenes/v1.0-test
ln -s /NAS/mmDataset/nuscenes/v1.0-trainval/ /workspace/FusionOcc/data/nuscenes/v1.0-trainval

echo "설정이 완료되었습니다."
