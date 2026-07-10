# Removed _base_ to avoid lazy_import conflicts
# _base_ = [
#     '../../../mmdet3d/configs/_base_/datasets/nus_3d.py',
#     '../../../mmdet3d/configs/_base_/default_runtime.py'
# ]

# Enable project imports
custom_imports = dict(
    imports=['projects.BEVFormer'],
    allow_failed_imports=False
)

custom_imports = dict(
    imports=['projects.BEVFormer.datasets.samplers'],
    allow_failed_imports=False)

# -----------------------------------------------------------------------
# Ablation: condition_D_full (free-only full-range supervision)
#   - ALL free voxels at every distance: always supervised (force mask_camera=1)
#     regardless of camera visibility.
#   - Occupied voxels: obey original mask_camera (unchanged).
#
# BEVFormer uses mask_camera as a per-voxel binary weight in the loss,
# so condition_D_full is implemented by forcing mask_camera=1 for ALL
# free voxels in LoadOccGTFromFile, regardless of distance.
#
# Fully restores free voxel supervision across all distances (0 m ~ inf).
# Compare with:
#   condition_D       → free supervised 0–20 m only
#   condition_D_prime → free supervised 0–35 m only
#   condition_D_full  → free supervised 0–∞  (this config)
# Eval: unchanged (evaluator loads mask_camera independently).
# -----------------------------------------------------------------------

# Set default scope to mmdet3d for all modules
default_scope = 'mmdet3d'

# CRITICAL: Configure default_hooks to load checkpoint in test mode
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

point_cloud_range = [-40, -40, -1.0, 40, 40, 5.4]
voxel_size = [0.2, 0.2, 8]

img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

occ_class_names = ['others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
                   'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
                   'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
                   'vegetation', 'free']

dataset_name = 'occ3d'
eval_metric = 'miou'

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)

_dim_ = 256
_pos_dim_ = _dim_//2
_ffn_dim_ = _dim_*2
_num_levels_ = 4
bev_h_ = 200
bev_w_ = 200
queue_length = 4

model = dict(
    type='mmdet3d.BEVFormerOcc',
    use_grid_mask=True,
    video_test_mode=True,
    save_results=False,
    img_backbone=dict(
        type='mmdet.ResNet',
        depth=101,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        style='caffe',
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True)),
    img_neck=dict(
        type='mmdet.FPN',
        in_channels=[512, 1024, 2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=4,
        relu_before_extra_convs=True),
    pts_bbox_head=dict(
        type='mmdet3d.BEVFormerOccHead',
        pc_range=point_cloud_range,
        bev_h=bev_h_,
        bev_w=bev_w_,
        num_classes=18,
        in_channels=_dim_,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        use_mask=True,
        loss_occ=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0),
        transformer=dict(
                type='mmdet3d.TransformerOcc',
            pillar_h=16,
            num_classes=18,
            norm_cfg=dict(type='BN', ),
            norm_cfg_3d=dict(type='BN3d', ),
            use_3d=True,
            use_conv=False,
            rotate_prev_bev=True,
            use_shift=True,
            use_can_bus=True,
            embed_dims=_dim_,
            encoder=dict(
                type='mmdet3d.BEVFormerEncoder',
                num_layers=4,
                pc_range=point_cloud_range,
                num_points_in_pillar=8,
                return_intermediate=False,
                transformerlayers=dict(
                    type='mmdet3d.BEVFormerLayer',
                    attn_cfgs=[
                        dict(
                            type='mmdet3d.TemporalSelfAttention',
                            embed_dims=_dim_,
                            num_levels=1),
                        dict(
                            type='mmdet3d.SpatialCrossAttention',
                            pc_range=point_cloud_range,
                            deformable_attention=dict(
                                type='mmdet3d.MSDeformableAttention3D',
                                embed_dims=_dim_,
                                num_points=8,
                                num_levels=_num_levels_),
                            embed_dims=_dim_,
                        )
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'))),
        ),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_,
        ),
    ),
    train_cfg=dict(pts=dict(
        grid_size=[512, 512, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=4,
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='IoUCost', weight=0.0),
            pc_range=point_cloud_range))))

dataset_type = 'mmdet3d.NuSceneOcc'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')
occ_gt_data_root = 'data/nuscenes/'

bda_aug_conf = dict(
    rot_lim=(0, 0),
    scale_lim=(1., 1.),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5
)

# condition_D_full: ALL free voxels force mask_camera=1 (always supervised);
# occupied voxels obey original mask_camera (no distance threshold needed).
train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(
        type='LoadOccGTFromFile',
        data_root=occ_gt_data_root,
        mask_mode='condition_D_full',
        free_class_id=occ_class_names.index('free'),  # 17
        pc_range_x=80.0,
    ),
    dict(type='BEVAug', bda_aug_conf=bda_aug_conf, is_train=True),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='CustomCollect3D', keys=['img', 'voxel_semantics', 'mask_lidar', 'mask_camera'])
]

# Eval pipeline is unchanged (evaluator loads mask_camera independently)
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadOccGTFromFile', data_root=occ_gt_data_root),
    dict(type='BEVAug', bda_aug_conf=bda_aug_conf, is_train=False),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1600, 900),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='CustomCollect3D', keys=['img', 'voxel_semantics', 'mask_lidar', 'mask_camera'])
        ])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='occ_infos_temporal_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        bev_size=(bev_h_, bev_w_),
        queue_length=queue_length,
        box_type_3d='LiDAR'),
    val=dict(type=dataset_type,
             data_root=data_root,
             ann_file='occ_infos_temporal_val.pkl',
             pipeline=test_pipeline, bev_size=(bev_h_, bev_w_),
             classes=class_names, modality=input_modality, samples_per_gpu=1,
             test_mode=True),
    test=dict(type=dataset_type,
              data_root=data_root,
              ann_file='occ_infos_temporal_val.pkl',
              pipeline=test_pipeline, bev_size=(bev_h_, bev_w_),
              classes=class_names, modality=input_modality,
              test_mode=True),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
)

optimizer = dict(type='AdamW', lr=2e-4, weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

total_epochs = 24
train_samples = 28130
num_gpus = 2
num_iters_per_epoch = train_samples // (num_gpus * data['samples_per_gpu'])
evaluation = dict(interval=1, pipeline=test_pipeline)

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)

train_dataloader = dict(
    batch_size=data['samples_per_gpu'],
    num_workers=data['workers_per_gpu'],
    persistent_workers=True,
    sampler=dict(
        type='DistributedGroupSampler',
        samples_per_gpu=data['samples_per_gpu'],
        seed=0
    ),
    dataset=data['train']
)

val_dataloader = dict(
    batch_size=data['samples_per_gpu'],
    num_workers=data['workers_per_gpu'],
    persistent_workers=True,
    sampler=dict(
        type='DistributedSampler',
        shuffle=False,
    ),
    dataset=data['val']
)

test_dataloader = dict(
    batch_size=data['samples_per_gpu'],
    num_workers=data['workers_per_gpu'],
    persistent_workers=True,
    sampler=dict(
        type='DistributedSampler',
        shuffle=False,
    ),
    dataset=data['test']
)

val_evaluator = dict(
    type='mmdet3d.OccupancyMetricHybrid',
    dataset_name=dataset_name,
    num_classes=18,
    use_lidar_mask=False,
    use_image_mask=True,
    ann_file='data/nuscenes/occ_infos_temporal_val.pkl',
    data_root='data/nuscenes/',
    class_names=occ_class_names,
    eval_metric=eval_metric,
    sort_by_timestamp=True,
    prefix='val'
)

test_evaluator = dict(
    type='mmdet3d.OccupancyMetricHybrid',
    dataset_name=dataset_name,
    num_classes=18,
    use_lidar_mask=False,
    use_image_mask=True,
    ann_file='data/nuscenes/occ_infos_temporal_val.pkl',
    data_root='data/nuscenes/',
    class_names=occ_class_names,
    eval_metric=eval_metric,
    sort_by_timestamp=True,
    prefix='test'
)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=2e-4,
        weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    clip_grad=dict(max_norm=35, norm_type=2),
    accumulative_counts=8)

train_cfg = dict(
    by_epoch=True,
    max_epochs=total_epochs,
    val_interval=999999
)

val_cfg = dict()
test_cfg = dict()

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.05,
        end_factor=1.0,
        by_epoch=False,
        begin=0,
        end=500
    ),
    dict(
        type='CosineAnnealingLR',
        begin=500,
        end=total_epochs * num_iters_per_epoch,
        by_epoch=False,
        eta_min=1e-6
    )
]

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

load_from = './projects/BEVFormer/pretrain/r101_dcn_fcos3d_pretrain.pth'

checkpoint_config = dict(interval=1)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

model_wrapper_cfg = dict(
    type='MMDistributedDataParallel',
    find_unused_parameters=True,
    broadcast_buffers=False,
)

work_dir = './work_dirs/bevformer_base_occ_w_cam_mask_unified_condition_D_full'

randomness = dict(seed=0, deterministic=False)
