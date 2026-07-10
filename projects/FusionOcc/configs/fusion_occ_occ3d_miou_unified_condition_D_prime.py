_base_ = ['../../../configs/_base_/datasets/nus-3d.py', '../../../configs/_base_/default_runtime.py']

# Custom imports for FusionOCC
custom_imports = dict(
    imports=['projects.FusionOcc.fusionocc'],
    allow_failed_imports=False)

# -----------------------------------------------------------------------
# Ablation: condition_D_prime (free-only extended distance mask)
#   - Free voxels < 35 m from ego: always supervised (force mask_camera=1)
#     regardless of camera visibility.
#   - Free voxels >= 35 m: obey original mask_camera.
#   - Occupied voxels: obey original mask_camera (unchanged).
#
# FusionOcc uses mask_camera as a per-voxel binary weight in loss_single,
# so condition_D_prime is implemented by forcing mask_camera=1 for free
# voxels within 35 m in LoadOccGTFromFile.
#
# Extends condition_D (0–20 m free supervision) to 0–35 m, verifying
# whether mid-range (20–35 m) free voxels also contribute to RayIoU.
# If RayIoU recovers further vs condition_D → Hypothesis 2 confirmed.
# Eval: unchanged (evaluator loads mask_camera independently).
# -----------------------------------------------------------------------

point_cloud_range = [-40, -40, -1, 40, 40, 5.4]
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

data_config = {
    'cams': [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    'Ncams': 6,
    'input_size': (512, 1408),
    'src_size': (900, 1600),
    'resize': (0.0, 0.0),
    'rot': (0.0, 0.0),
    'flip': False,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

grid_config = {
    'x': [-40, 40, 0.4],
    'y': [-40, 40, 0.4],
    'z': [-1, 5.4, 0.4],
    'depth': [1.0, 45.0, 0.5],
}

use_mask = True

voxel_size = [0.05, 0.05, 0.05]

img_backbone_out_channel = 256
feature_channel = 32
lidar_out_channel = 32
img_channels = feature_channel
numC_Trans = img_channels + lidar_out_channel
num_classes = 18

occ_class_names = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
    'vegetation', 'free'
]

dataset_name = 'occ3d'
eval_metric = 'miou'

multi_adj_frame_id_cfg = (1, 1 + 1, 1)
multi_adj_frame_id_cfg_lidar = (1, 7 + 1, 1)

model = dict(
    type='FusionOCC',
    save_results=False,
    data_preprocessor=None,
    lidar_in_channel=5,
    point_cloud_range=point_cloud_range,
    voxel_size=voxel_size,
    lidar_out_channel=lidar_out_channel,
    align_after_view_transformation=True,
    num_adj=len(range(*multi_adj_frame_id_cfg)),
    fuse_loss_weight=0.1,
    img_backbone=dict(
        type='SwinTransformer',
        pretrain_img_size=224,
        patch_size=4,
        window_size=12,
        mlp_ratio=4,
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        strides=(4, 2, 2, 2),
        out_indices=(2, 3),
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        use_abs_pos_embed=False,
        return_stereo_feat=True,
        act_cfg=dict(type='GELU'),
        norm_cfg=dict(type='LN', requires_grad=True),
        pretrain_style='official',
        output_missing_index_as_none=False),
    img_neck=dict(
        type='FPN_LSS',
        in_channels=512 + 1024,
        out_channels=img_backbone_out_channel,
        extra_upsample=None,
        input_feature_index=(0, 1),
        scale_factor=2),
    img_view_transformer=dict(
        type='CrossModalLSS',
        feature_channels=feature_channel,
        seg_num_classes=num_classes,
        grid_config=grid_config,
        input_size=data_config['input_size'],
        in_channels=img_backbone_out_channel,
        mid_channels=128,
        depth_channels=88,
        is_train=True,
        out_channels=img_channels,
        sid=False,
        collapse_z=False,
        depthnet_cfg=dict(aspp_mid_channels=96, ),
        downsample=16),
    pre_process=dict(
        type='CustomResNet3D',
        numC_input=img_channels,
        with_cp=False,
        num_layer=[1, ],
        num_channels=[img_channels, ],
        stride=[1, ],
        backbone_output_ids=[0, ]),
    img_bev_encoder_backbone=dict(
        type='CustomResNet3D',
        numC_input=img_channels * (len(range(*multi_adj_frame_id_cfg)) + 1) + lidar_out_channel,
        num_layer=[1, 2, 3],
        with_cp=False,
        num_channels=[numC_Trans, numC_Trans * 2, numC_Trans * 4],
        stride=[1, 2, 2],
        backbone_output_ids=[0, 1, 2]),
    img_bev_encoder_neck=dict(type='LSSFPN3D',
                          in_channels=numC_Trans * 7,
                          out_channels=numC_Trans),
    out_dim=numC_Trans,
    loss_occ=dict(
        type='CrossEntropyLoss',
        use_sigmoid=False,
        loss_weight=1.0),
    use_mask=use_mask,
)

dataset_type = 'NuScenesDatasetOccpancy'
data_root = 'data/nuscenes/'
img_seg_dir = 'data/nuscenes/imgseg/samples'

bda_aug_conf = dict(
    rot_lim=(-0., 0.),
    scale_lim=(1., 1.),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5)

# Condition D': free voxels within 35 m force mask_camera=1 (always supervised);
# occupied voxels and far free voxels (>= 35 m) obey original mask_camera.
train_pipeline = [
    dict(
        type='PrepareImageSeg',
        downsample=1,
        is_train=True,
        data_config=data_config,
        sequential=True,
        img_seg_dir=img_seg_dir
    ),
    dict(
        type='LoadOccGTFromFile',
        mask_mode='condition_D_prime',
        free_class_id=occ_class_names.index('free'),  # 17
        dist_threshold_d_prime=35.0,
        pc_range_x=80.0,
    ),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5),
    dict(
        type='FuseAdjacentSweeps',
        load_dim=5,
        use_dim=5),
    dict(type='PointsLidar2Ego'),
    dict(type='FusionOccPointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(
        type='LoadAnnotationsAll',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        is_train=True),
    dict(type='PointToMultiViewDepth', downsample=1, grid_config=grid_config),
    dict(type='FormatDataSamples'),
]

# Eval pipeline is unchanged (evaluator loads mask_camera independently)
test_pipeline = [
    dict(
        type='PrepareImageSeg',
        restore_upsample=8,
        downsample=1,
        data_config=data_config,
        sequential=True,
        img_seg_dir=img_seg_dir
    ),
    dict(type='LoadOccGTFromFile'),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5),
    dict(
        type='FuseAdjacentSweeps',
        load_dim=5,
        use_dim=5),
    dict(type='PointsLidar2Ego'),
    dict(type='FusionOccPointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(
        type='LoadAnnotationsAll',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        is_train=False),
    dict(type='PointToMultiViewDepth', downsample=1, grid_config=grid_config),
    dict(type='FormatDataSamples'),
]

input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

share_data_config = dict(
    use_mask=use_mask,
    type=dataset_type,
    classes=class_names,
    modality=input_modality,
    stereo=False,
    filter_empty_gt=False,
    img_info_prototype='fusionocc',
    multi_adj_frame_id_cfg=multi_adj_frame_id_cfg,
    multi_adj_frame_id_cfg_lidar=multi_adj_frame_id_cfg_lidar,
)

test_data_config = dict(
    pipeline=test_pipeline,
    ann_file='data/nuscenes/fusionocc-nuscenes_infos_val.pkl')

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,
    train=dict(
        data_root=data_root,
        ann_file='data/nuscenes/fusionocc-nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        test_mode=False,
        use_valid_flag=True,
        box_type_3d='LiDAR'),
    val=test_data_config,
    test=test_data_config
)
for key in ['val', 'train', 'test']:
    data[key].update(share_data_config)

train_samples = 28130
num_gpus = 2
num_iters_per_epoch = train_samples // (num_gpus * data['samples_per_gpu'])

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=2e-4,
        weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
            'img_view_transformer': dict(lr_mult=0.1),
        }),
    clip_grad=dict(max_norm=35, norm_type=2),
    accumulative_counts=8)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.05,
        end_factor=1.0,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        type='CosineAnnealingLR',
        begin=500,
        end=24 * num_iters_per_epoch,
        by_epoch=False,
        eta_min=1e-6)
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=24, val_interval=999999)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

val_evaluator = dict(
    type='OccupancyMetricHybrid',
    num_classes=num_classes,
    use_lidar_mask=False,
    use_image_mask=use_mask,
    data_root='data/nuscenes/',
    ann_file='data/nuscenes/fusionocc-nuscenes_infos_val.pkl',
    dataset_name=dataset_name,
    class_names=occ_class_names,
    eval_metric=eval_metric,
    sort_by_timestamp=True,
    prefix='val'
)

test_evaluator = dict(
    type='OccupancyMetricHybrid',
    num_classes=num_classes,
    use_lidar_mask=False,
    use_image_mask=use_mask,
    data_root='data/nuscenes/',
    ann_file='data/nuscenes/fusionocc-nuscenes_infos_val.pkl',
    dataset_name=dataset_name,
    class_names=occ_class_names,
    eval_metric=eval_metric,
    sort_by_timestamp=True,
    prefix='test'
)

train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root='',
        ann_file='data/nuscenes/fusionocc-nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        use_mask=use_mask,
        classes=class_names,
        modality=input_modality,
        stereo=False,
        filter_empty_gt=False,
        img_info_prototype='fusionocc',
        multi_adj_frame_id_cfg=multi_adj_frame_id_cfg,
        multi_adj_frame_id_cfg_lidar=multi_adj_frame_id_cfg_lidar,
        test_mode=False,
        use_valid_flag=True,
        box_type_3d='LiDAR'
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root='',
        ann_file='data/nuscenes/fusionocc-nuscenes_infos_val.pkl',
        pipeline=test_pipeline,
        use_mask=use_mask,
        classes=class_names,
        modality=input_modality,
        stereo=False,
        filter_empty_gt=False,
        img_info_prototype='fusionocc',
        multi_adj_frame_id_cfg=multi_adj_frame_id_cfg,
        multi_adj_frame_id_cfg_lidar=multi_adj_frame_id_cfg_lidar,
        test_mode=False,
        box_type_3d='LiDAR'
    )
)

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root='',
        ann_file='data/nuscenes/fusionocc-nuscenes_infos_val.pkl',
        pipeline=test_pipeline,
        use_mask=use_mask,
        classes=class_names,
        modality=input_modality,
        stereo=False,
        filter_empty_gt=False,
        img_info_prototype='fusionocc',
        multi_adj_frame_id_cfg=multi_adj_frame_id_cfg,
        multi_adj_frame_id_cfg_lidar=multi_adj_frame_id_cfg_lidar,
        test_mode=False,
        box_type_3d='LiDAR'
    )
)

custom_hooks = []

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='Det3DVisualizationHook'))

visualizer = dict(
    type='Det3DLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer'
)
vis_backends = [dict(type='LocalVisBackend')]

work_dir = './work_dirs/fusion_occ_occ3d_miou_unified_condition_D_prime'

voxel_size = [0.05, 0.05, 0.05]

load_from = "./projects/FusionOcc/pretrain/bevdet-occ-stbase-4d-stereo-512x1408-24e.pth"

randomness = dict(seed=0, deterministic=False)
