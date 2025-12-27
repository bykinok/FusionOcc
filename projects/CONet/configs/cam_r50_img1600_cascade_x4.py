# Copyright (c) OpenMMLab. All rights reserved.
_base_ = [
    '_base_/custom_nus-3d.py',
    '_base_/default_runtime.py'
]

# Plugin configuration
custom_imports = dict(
    imports=['projects.CONet.mmdet3d_plugin'],
    allow_failed_imports=False)

# Input modality configuration
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

img_norm_cfg = None
occ_path = "./data/nuScenes-Occupancy"
depth_gt_path = './data/depth_gt'
train_ann_file = "nuscenes_occ_infos_train.pkl"
val_ann_file = "nuscenes_occ_infos_val.pkl"

# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
occ_size = [512, 512, 40]
lss_downsample = [4, 4, 4]  # [128 128 10]
voxel_x = (point_cloud_range[3] - point_cloud_range[0]) / occ_size[0]  # 0.4
voxel_y = (point_cloud_range[4] - point_cloud_range[1]) / occ_size[1]
voxel_z = (point_cloud_range[5] - point_cloud_range[2]) / occ_size[2]
voxel_channels = [80, 160, 320, 640]
empty_idx = 0  # noise 0-->255
num_cls = 17  # 0 free, 1-16 obj
visible_mask = False

cascade_ratio = 4
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
        voxel=False,
        mean=[103.530, 116.280, 123.675],
        std=[1.0, 1.0, 1.0],
        bgr_to_rgb=False,
        pad_size_divisor=32),
    loss_norm=True,
    # nuScenes-Occupancy configuration
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
        use_bev_pool=True,
        vp_megvii=False),
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
        cascade_ratio=cascade_ratio,
        sample_from_voxel=sample_from_voxel,
        sample_from_img=sample_from_img,
        final_occ_size=occ_size,
        fine_topk=15000,
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
    empty_idx=empty_idx,
)

bda_aug_conf = dict(
    rot_lim=(-0, 0),
    scale_lim=(0.95, 1.05),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5)

# Data pipeline
train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles_BEVDet', 
         is_train=True, 
         data_config=data_config,
         sequential=False, 
         aligned=True, 
         trans_only=False, 
         depth_gt_path=depth_gt_path,
         mmlabnorm=True, 
         load_depth=True, 
         img_norm_cfg=img_norm_cfg),
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
         cal_visible=visible_mask),
    dict(type='OccDefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['img_inputs', 'gt_occ']),
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles_BEVDet', 
         data_config=data_config, 
         depth_gt_path=depth_gt_path,
         sequential=False, 
         aligned=True, 
         trans_only=False, 
         mmlabnorm=True, 
         img_norm_cfg=img_norm_cfg),
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
         cal_visible=visible_mask),
    dict(type='OccDefaultFormatBundle3D', class_names=class_names, with_label=False), 
    dict(type='Collect3D', 
         keys=['img_inputs', 'gt_occ'],
         meta_keys=['pc_range', 'occ_size', 'scene_token', 'lidar_token']),
]

# Data loaders
train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
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
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
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
        end=15,
        ),
    dict(
        type='LinearLR',
        start_factor=1.0 / 3,
        by_epoch=False,
        begin=0,
        end=500,
        )
]

# Training configuration
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=15,
    val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Evaluation configuration
val_evaluator = dict(
    type='OccMetric',
    save_best='SSC_mean',
    rule='greater')
test_evaluator = val_evaluator