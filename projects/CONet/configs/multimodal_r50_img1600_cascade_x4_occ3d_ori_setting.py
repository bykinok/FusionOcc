# Copyright (c) OpenMMLab. All rights reserved.
_base_ = [
    '_base_/custom_nus-3d.py',
    '_base_/default_runtime.py'
]

# Plugin configuration
custom_imports = dict(
    imports=[
        'projects.CONet.mmdet3d_plugin',
    ],
    allow_failed_imports=False)

# Input modality
input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

# Data configuration
# === Occ3D 평가 설정 ===
dataset_name = 'occ3d'  # 'occ3d' for occ3d GT format
use_occ3d = True  # True: occ3d 형식 (labels.npz), False: nuScenes-Occupancy 형식 (.npy)
occ_path = "./data/nuScenes-Occupancy"  # use_occ3d=False일 때 사용
depth_gt_path = './data/depth_gt'

# Annotation files (각 샘플의 'occ_path' 키에 GT 경로가 포함되어 있어야 함)
train_ann_file = "nuscenes_occ_infos_train.pkl"
val_ann_file = "nuscenes_occ_infos_val.pkl"

# === Occ3D Class Configuration ===
# Occ3D uses 18 classes: 0=others, 1-16=semantic classes, 17=free
# For 3D detection (not used in occupancy but kept for compatibility)
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

# Occ3D occupancy class names (18 classes total)
occ_class_names = [
    'others',              # 0
    'barrier',             # 1
    'bicycle',             # 2
    'bus',                 # 3
    'car',                 # 4
    'construction_vehicle', # 5
    'motorcycle',          # 6
    'pedestrian',          # 7
    'traffic_cone',        # 8
    'trailer',             # 9
    'truck',               # 10
    'driveable_surface',   # 11
    'other_flat',          # 12
    'sidewalk',            # 13
    'terrain',             # 14
    'manmade',             # 15
    'vegetation',          # 16
    'free'                 # 17 (empty/free space)
]

# === Occ3D Grid Configuration ===
# Occ3D 데이터셋 사양: 200x200x16 grid, 18 classes
point_cloud_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]  # Occ3D range
occ_size = [200, 200, 16]  # Occ3D grid size [X, Y, Z]
# coarse 100x100x8 유지 (50x50x4보다 표현력 좋음). fine 400x400x32 → loss_point에서 GT 200x200x16로 스케일
lss_downsample = [2, 2, 2]  # [100, 100, 8] for LSS → fine 400x400x32
voxel_x = (point_cloud_range[3] - point_cloud_range[0]) / occ_size[0]  # 0.4
voxel_y = (point_cloud_range[4] - point_cloud_range[1]) / occ_size[1]  # 0.4
voxel_z = (point_cloud_range[5] - point_cloud_range[2]) / occ_size[2]  # 0.4
voxel_channels = [80, 160, 320, 640]
empty_idx = 17  # Occ3D: class 17 is free/empty
num_cls = 18  # Occ3D: 0-17 (18 classes total)
visible_mask = True  # Occ3D provides mask_camera

# Occ3D: coarse 100x100x8 * 2 = fine 200x200x16 = GT (원본 nuScenes는 cascade_ratio=4)
cascade_ratio = 2
sample_from_voxel = True
sample_from_img = True

dataset_type = 'NuscOCCDataset'
data_root = 'data/nuscenes/'
backend_args = dict(backend='disk')

data_config = {
    'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
             'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
    'Ncams': 6,
    'input_size': (896, 1600),
    'src_size': (900, 1600),
    # image-view augmentation
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

# downsample ratio in [x, y, z] when generating 3D volumes in LSS
grid_config = {
    'xbound': [point_cloud_range[0], point_cloud_range[3], voxel_x*lss_downsample[0]],
    'ybound': [point_cloud_range[1], point_cloud_range[4], voxel_y*lss_downsample[1]],
    'zbound': [point_cloud_range[2], point_cloud_range[5], voxel_z*lss_downsample[2]],
    'dbound': [2.0, 58.0, 0.5],
}
numC_Trans = 80
voxel_out_channel = 256
voxel_out_indices = (0, 1, 2, 3)

# Model configuration
model = dict(
    type='OccNet',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=10,
            point_cloud_range=point_cloud_range,
            voxel_size=[0.1, 0.1, 0.1],
            max_voxels=(90000, 120000))),
    loss_norm=True,
    # Occ3D configuration
    num_cls=num_cls,  # Pass num_cls to model for correct class handling
    empty_idx=empty_idx,  # Pass empty_idx to model
    img_backbone=dict(
        pretrained='torchvision://resnet50',
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=0,
        with_cp=True,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=False,
        style='pytorch'),
    img_neck=dict(
        type='SECONDFPN',
        in_channels=[256, 512, 1024, 2048],
        upsample_strides=[0.25, 0.5, 1, 2],
        out_channels=[128, 128, 128, 128]),
    img_view_transformer=dict(
        type='ViewTransformerLiftSplatShootVoxel',
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        loss_depth_weight=3.,
        loss_depth_type='kld',
        grid_config=grid_config,
        data_config=data_config,
        numC_Trans=numC_Trans,
        vp_megvii=False),
    pts_voxel_layer=dict(
        max_num_points=10, 
        point_cloud_range=point_cloud_range,
        voxel_size=[0.1, 0.1, 0.1],  # xy size follow centerpoint
        max_voxels=(90000, 120000)),
    pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=5),
    pts_middle_encoder=dict(
        type='SparseLiDAREnc8x',
        input_channel=4,
        base_channel=16,
        out_channel=numC_Trans,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        sparse_shape_xyz=[800, 800, 64],  # Occ3D: 8x downsample -> [100, 100, 8] to match LSS output
        ),
    occ_fuser=dict(
        type='VisFuser',
        in_channels=numC_Trans,
        out_channels=numC_Trans,
    ),
    occ_encoder_backbone=dict(
        type='CustomResNet3D',
        depth=18,
        n_input_channels=numC_Trans,
        block_inplanes=voxel_channels,
        out_indices=voxel_out_indices,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
    ),
    occ_encoder_neck=dict(
        type='FPN3D',
        with_cp=True,
        in_channels=voxel_channels,
        out_channels=voxel_out_channel,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
    ),
    pts_bbox_head=dict(
        type='OccHead',
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        soft_weights=True,
        # balance_cls_weight=True (default) - uses occ3d_class_frequencies for 18 classes
        cascade_ratio=cascade_ratio,
        sample_from_voxel=sample_from_voxel,
        sample_from_img=sample_from_img,
        final_occ_size=occ_size,
        fine_topk=10000,  # Note: Different from camera-only config
        empty_idx=empty_idx,
        num_level=len(voxel_out_indices),
        in_channels=[voxel_out_channel] * len(voxel_out_indices),
        out_channel=num_cls,
        point_cloud_range=point_cloud_range,
        loss_weight_cfg=dict(
            loss_voxel_ce_weight=1.0,
            loss_voxel_sem_scal_weight=1.0,
            loss_voxel_geo_scal_weight=1.0,
            loss_voxel_lovasz_weight=1.0,
        ),
    ),
)

bda_aug_conf = dict(
    rot_lim=(-0, 0),
    scale_lim=(0.95, 1.05),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5)

# Data pipeline
train_pipeline = [
    dict(type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5),
    dict(type='LoadPointsFromMultiSweeps',
        sweeps_num=10),
    dict(type='LoadMultiViewImageFromFiles_BEVDet', 
         is_train=True, 
         data_config=data_config,
         sequential=False, 
         aligned=True, 
         trans_only=False, 
         depth_gt_path=depth_gt_path,
         mmlabnorm=True, 
         load_depth=True, 
         img_norm_cfg=None,
         use_ego_frame=True),  # Occ3D GT is Ego → output Ego volume
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        input_modality=input_modality),
    dict(type='LoadOccupancy', 
         to_float32=True, 
         use_semantic=True, 
         occ_path=occ_path, 
         grid_size=occ_size, 
         use_vel=False,
         unoccupied=empty_idx, 
         pc_range=point_cloud_range, 
         cal_visible=visible_mask,
         use_occ3d=use_occ3d,
         use_camera_mask=visible_mask),  # Occ3D: visible only → invisible=255 for loss
    dict(type='OccDefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['img_inputs', 'gt_occ', 'points']),
]

test_pipeline = [
    dict(type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5),
    dict(type='LoadPointsFromMultiSweeps',
        sweeps_num=10),
    dict(type='LoadMultiViewImageFromFiles_BEVDet', 
         data_config=data_config, 
         depth_gt_path=depth_gt_path,
         sequential=False, 
         aligned=True, 
         trans_only=False, 
         mmlabnorm=True, 
         img_norm_cfg=None,
         use_ego_frame=True),  # Occ3D: Ego-frame output for eval
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        input_modality=input_modality,
        is_train=False),
    dict(type='LoadOccupancy', 
         to_float32=True, 
         use_semantic=True, 
         occ_path=occ_path, 
         grid_size=occ_size, 
         use_vel=False,
         unoccupied=empty_idx, 
         pc_range=point_cloud_range, 
         cal_visible=visible_mask,
         use_occ3d=use_occ3d,
         use_camera_mask=visible_mask),  # Occ3D: visible only for eval
    dict(type='OccDefaultFormatBundle3D', class_names=class_names, with_label=False), 
    dict(type='Collect3D', 
         keys=['img_inputs', 'gt_occ', 'points'],
         meta_keys=['pc_range', 'occ_size', 'scene_token', 'lidar_token']),
]

# Data loaders
train_dataloader = dict(
    batch_size=1,
    num_workers=0,#4,
    persistent_workers=False,#True,
    sampler=dict(type='DistributedGroupSampler', seed=0),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        occ_root=occ_path,
        ann_file=train_ann_file,
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        occ_size=occ_size,
        pc_range=point_cloud_range,
        box_type_3d='LiDAR',
        backend_args=backend_args))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,  # 0 → 4: Enable parallel data loading (4-8x speedup expected)
    persistent_workers=True,  # Keep workers alive to avoid restart overhead
    drop_last=False,
    sampler=dict(type='DistributedSampler', shuffle=False, seed=0),
    dataset=dict(
        type=dataset_type,
        occ_root=occ_path,
        data_root=data_root,
        ann_file=val_ann_file,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        occ_size=occ_size,
        pc_range=point_cloud_range,
        box_type_3d='LiDAR',
        backend_args=backend_args))

test_dataloader = val_dataloader

# Optimizer configuration  
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=3e-4,
        weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    clip_grad=dict(max_norm=35, norm_type=2))

# Learning rate scheduler
param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        T_max=15,
        eta_min=3e-7,
        by_epoch=True,
        begin=0,
        end=15),
    dict(
        type='LinearLR',
        start_factor=1.0 / 3,
        by_epoch=False,
        begin=0,
        end=500)
]

# Training configuration
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=15,
    val_interval=16)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Evaluation configuration
# Use OccupancyMetricHybrid which supports both occ3d and original GT formats
# (following TPVFormer and SurroundOcc pattern)

# Occ3D class names (18 classes: 0=others, 1-16=semantic, 17=free)
occ_class_names = ['others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle', 
                   'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck', 
                   'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade', 
                   'vegetation', 'free']

val_evaluator = dict(
    type='OccupancyMetricHybrid',  # Hybrid metric for occ3d support
    dataset_name=dataset_name,  # 'occ3d'
    num_classes=18,  # occ3d uses 18 classes (0=others, 1-16=semantic, 17=free)
    use_lidar_mask=False,
    use_image_mask=True,  # occ3d provides mask_camera
    ann_file=data_root + val_ann_file,
    data_root=data_root,
    class_names=occ_class_names,
    eval_metric='miou')
test_evaluator = val_evaluator

#load_from = 'projects/CONet/ckpt/multi-modal-CONet.pth'