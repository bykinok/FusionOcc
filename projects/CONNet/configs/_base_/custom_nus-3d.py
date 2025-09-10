# Copyright (c) OpenMMLab. All rights reserved.

# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
metainfo = dict(classes=class_names)
dataset_type = 'NuscOCCDataset'
data_root = 'data/nuscenes/'

# Input modality configuration
input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

# Occupancy configuration
occ_path = "./data/nuScenes-Occupancy"
occ_size = [512, 512, 40]
empty_idx = 0  # noise 0-->255
num_cls = 17  # 0 free, 1-16 obj
visible_mask = False

backend_args = dict(backend='disk')

train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        occ_root=occ_path,
        ann_file='nuscenes_occ_infos_train.pkl',
        pipeline=[],  # To be filled in specific configs
        metainfo=metainfo,
        modality=input_modality,
        test_mode=False,
        occ_size=occ_size,
        pc_range=point_cloud_range,
        box_type_3d='LiDAR',
        backend_args=backend_args))

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        occ_root=occ_path,
        ann_file='nuscenes_occ_infos_val.pkl',
        pipeline=[],  # To be filled in specific configs
        metainfo=metainfo,
        modality=input_modality,
        test_mode=True,
        occ_size=occ_size,
        pc_range=point_cloud_range,
        box_type_3d='LiDAR',
        backend_args=backend_args))

test_dataloader = val_dataloader

# Evaluation configuration will be defined in specific configs
val_evaluator = dict(
    type='OccMetric',  # Custom metric for occupancy
    save_best='SSC_mean',
    rule='greater')

test_evaluator = val_evaluator
