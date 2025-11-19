_base_ = ['../../../configs/_base_/datasets/nus-3d.py', '../../../configs/_base_/default_runtime.py']

# Custom imports
custom_imports = dict(imports=['projects.TEOcc'], allow_failed_imports=False)

# Global
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

data_config = {
    'cams': [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    'Ncams':
    6,
    # 'input_size': (256, 704),
    'input_size': (384, 704),
    'src_size': (900, 1600),

    # Augmentation
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

# Model
grid_config = {
    'x': [-40, 40, 0.4],
    'y': [-40, 40, 0.4],
    'z': [-1, 5.4, 0.4],
    'depth': [1.0, 45.0, 0.5],
}

point_cloud_range = [-40, -40, -5, 40, 40, 3]

# radar_voxel_size = [0.4, 0.4, 8]
radar_voxel_size = [0.2, 0.2, 8]
# x y z vx_comp vy_comp rcs 
radar_use_dims = [0, 1, 2, 8, 9, 5, 18]

voxel_size = [0.1, 0.1, 0.2]

numC_Trans = 32
numRadar_Trans = 64

multi_adj_frame_id_cfg = (1, 1+8, 1)  # 8 adjacent frames like original

with_cp=False
model = dict(
    type='BEVStereo4DOCCRC',    
    freeze_img=False,
    align_after_view_transfromation=False,
    num_adj=len(range(*multi_adj_frame_id_cfg)),  # 8 adjacent frames
    # extra_ref_frames=1 by default: num_frame=9, temporal_frame=8, concat 8 frames
    img_backbone=dict(
        _scope_='mmdet',
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        with_cp=False,
        style='pytorch'),
    img_neck=dict(
        type='CustomFPN',
        in_channels=[1024, 2048],
        out_channels=256,
        num_outs=1,
        start_level=0,
        out_ids=[0]),
    img_view_transformer=dict(
        type='LSSViewTransformerBEVStereo',
        grid_config=grid_config,
        input_size=data_config['input_size'],
        in_channels=256,
        out_channels=numC_Trans,
        sid=False,
        collapse_z=False,
        loss_depth_weight=0.05,
        depthnet_cfg=dict(use_dcn=False,
                          aspp_mid_channels=96,
                          stereo=True,
                          bias=5.),
        downsample=16),
    img_bev_encoder_backbone=dict(
        type='CustomResNet3D',
        numC_input=numC_Trans * (len(range(*multi_adj_frame_id_cfg))+1),  # 32 * 9 = 288 (원본과 동일)
        num_layer=[1, 2, 4],
        with_cp=False,
        num_channels=[numC_Trans,numC_Trans*2,numC_Trans*4],
        stride=[1,2,2],
        backbone_output_ids=[0,1,2]),
    img_bev_encoder_neck=dict(type='LSSFPN3D',
                              in_channels=numC_Trans*7,  # 32+64+128 = 224 channels from concat
                              out_channels=numC_Trans),
    pre_process=dict(
        type='CustomResNet3D',
        numC_input=numC_Trans,
        with_cp=False,
        num_layer=[1,],
        num_channels=[numC_Trans,],
        stride=[1,],
        backbone_output_ids=[0,]),
    
    radar_voxel_layer=dict(
        max_num_points=10,
        voxel_size=radar_voxel_size,
        max_voxels=(90000, 120000),
        point_cloud_range=point_cloud_range),
    pts_voxel_encoder=dict(
        type='HardSimpleVFE',
        num_features=5,
    ),
    radar_voxel_encoder=dict(
        type='RadarEncoder',
        in_channels=6+1,
        feat_channels=[32, 64],
        with_distance=False,
        point_cloud_range=point_cloud_range,
        voxel_size=radar_voxel_size,
        norm_cfg=dict(
            type='BN1d',
            eps=1.0e-3,
            momentum=0.01),
        with_pos_embed=True,
        permute_injection_extraction=True,
    ),
    radar_middle_encoder=dict(
        type='PointPillarsScatter',
        in_channels=64,
        output_shape=[400, 400],
    ),
    radar_bev_backbone=dict(
        type='SECOND',
        in_channels=64,
        out_channels=[64, 128, 256],
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    radar_bev_neck=dict(
        type='SECONDFPN',
        in_channels=[64, 128, 256],
        out_channels=[128, 128, 128],
        upsample_strides=[0.5, 1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),

    # sparse_shape = [16, 200, 200],
    imc=numC_Trans, rac=sum([128, 128, 128]),
    radar_reduc_conv=True,
    
    loss_occ=dict(
        _scope_='mmdet',
        type='CrossEntropyLoss',
        use_sigmoid=False,
        loss_weight=1.0),
    use_mask=True,
    )

# Data
dataset_type = 'NuScenesDatasetOccpancy'
data_root = 'data/nuscenes/'
file_client_args = None
occ_gt_data_root='data/nuscenes'

bda_aug_conf = dict(
    rot_lim=(-0., 0.),
    scale_lim=(1., 1.),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5)

train_pipeline = [
    dict(
        type='PrepareImageInputs',
        is_train=True,
        data_config=data_config,
        sequential=True),
    dict( #load radar
        type='LoadRadarPointsMultiSweeps',
        load_dim=18,
        sweeps_num=8,
        use_dim=radar_use_dims,
        max_num=1200, ),
    dict(type='LoadOccGTFromFile',data_root=occ_gt_data_root),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        is_train=True),
    
    dict(type='GlobalRotScaleTrans_radar'),

    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        backend_args=file_client_args),
    dict(type='PointToMultiViewDepth', downsample=1, grid_config=grid_config),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D', keys=['img_inputs', 'gt_depth', 'voxel_semantics',
                                'mask_lidar','mask_camera',
                                # 'hop_voxel_semantics','hop_mask_camera',
                                'radar'
                                ])
]

test_pipeline = [
    dict(_scope_='mmdet3d', type='PrepareImageInputs', data_config=data_config, sequential=True),
    dict(
        _scope_='mmdet3d',
        type='LoadRadarPointsMultiSweeps',
        load_dim=18,
        sweeps_num=8,
        use_dim=radar_use_dims,
        max_num=1200, ),
    dict(
        _scope_='mmdet3d',
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        is_train=False),
    dict(_scope_='mmdet3d', type='GlobalRotScaleTrans_radar'),
    dict(
        _scope_='mmdet3d',
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        backend_args=file_client_args),
    dict(
        _scope_='mmdet3d',
        type='DefaultFormatBundle3D',
        class_names=class_names,
        with_label=False),
    dict(_scope_='mmdet3d', type='Collect3D', keys=['img_inputs', 'points', 'radar'],
            meta_keys=[
                'filename', 'ori_shape', 'img_shape', 'lidar2img',
                'depth2img', 'cam2img', 'pad_shape', 'scale_factor',
                'flip', 'pcd_horizontal_flip', 'pcd_vertical_flip',
                'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'pcd_trans',
                'sample_idx', 'prev_idx', 'next_idx', 'scene_token',
                'can_bus', 'frame_idx', 'ego_id', 'sweeps_idx',
                'post_trans', 'post_rots', 'frustum_size', 'bda_rot',
                'sequence_idx', 'curr2ego_rot', 'curr2ego_tran',
                'prev2ego_rot', 'prev2ego_tran', 'cam_sweep_ts',
                'cam_type', 'lidar_path', 'radar_path', 'timestamp',
                'occ_path'  # Critical for ground truth loading
            ])
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=True,
    use_map=False,
    use_external=False)

share_data_config = dict(
    type=dataset_type,
    classes=class_names,
    modality=input_modality,
    stereo=True,
    filter_empty_gt=False,
    img_info_prototype='bevdet4d',
    multi_adj_frame_id_cfg=multi_adj_frame_id_cfg,
    use_rays=False
)

test_data_config = dict(
    pipeline=test_pipeline,
    ann_file='teocc-nuscenes_R_infos_val.pkl')

train_data_config=dict(
        data_root=data_root,
        ann_file='teocc-nuscenes_R_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        test_mode=False,
        use_valid_flag=True,
        load_adj_occ_labels=False,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR')

train_data_config.update(share_data_config)
test_data_config.update(share_data_config)

train_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=train_data_config)

val_dataloader = dict(
    batch_size=1,
    num_workers=0,#4,
    persistent_workers=False,#True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=test_data_config)

test_dataloader = val_dataloader

# Evaluator for occupancy prediction
val_evaluator = dict(
    type='OccupancyMetric',
    num_classes=18,
    use_lidar_mask=False,
    use_image_mask=True,
    ann_file='data/nuscenes/teocc-nuscenes_R_infos_val.pkl'
)
test_evaluator = val_evaluator

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=1e-2),
    clip_grad=dict(max_norm=5, norm_type=2))

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-3,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        type='CosineAnnealingLR',
        T_max=12,
        by_epoch=True,
        begin=0,
        end=12)
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=12)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

custom_hooks = []

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'))

# load_from = 'work_dirs/teocc_r50_9kf_stereo_384x704.pth'
