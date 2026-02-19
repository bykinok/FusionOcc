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

# Set default scope to mmdet3d for all modules
default_scope = 'mmdet3d'

# CRITICAL: Configure default_hooks to load checkpoint in test mode
# mmengine Runner loads checkpoint automatically from cfg.load_from
# This ensures checkpoint is loaded even in test-only mode
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-40, -40, -1.0, 40, 40, 5.4]
voxel_size = [0.2, 0.2, 8]

img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

# Occ3D class names (18 classes: 0=others, 1-16=semantic, 17=free)
occ_class_names = ['others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle', 
                   'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck', 
                   'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade', 
                   'vegetation', 'free']

# Dataset configuration for evaluation metric
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
queue_length = 4 # each sequence contains `queue_length` frames.

model = dict(
    type='mmdet3d.BEVFormerOcc',
    use_grid_mask=True,
    video_test_mode=True,  # 원본과 동일하게 temporal 정보 사용
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
        loss_occ= dict(
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
    # model training and testing settings
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

dataset_type = 'mmdet3d.NuSceneOcc'  # This matches the registered name in datasets/__init__.py
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')
occ_gt_data_root='data/nuscenes/'

# BEV augmentation (Flip X/Y), same as STCOcc
bda_aug_conf = dict(
    rot_lim=(0, 0),
    scale_lim=(1., 1.),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5
)

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadOccGTFromFile',data_root=occ_gt_data_root),
    dict(type='BEVAug', bda_aug_conf=bda_aug_conf, is_train=True),
    # BEV aug만 사용, PhotoMetricDistortion 비활성
    # dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='CustomCollect3D', keys=[ 'img','voxel_semantics','mask_lidar','mask_camera'] )
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadOccGTFromFile', data_root=occ_gt_data_root),  # GT 로드 (평가용)
    dict(type='BEVAug', bda_aug_conf=bda_aug_conf, is_train=False),  # No augmentation in test, but adds identity bda_mat
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    # CRITICAL: Follow original BEVFormer's test pipeline with MultiScaleFlipAug3D
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
    workers_per_gpu=4,  # 원본과 동일하게 4로 설정
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
             pipeline=test_pipeline,  bev_size=(bev_h_, bev_w_),
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

# Legacy (optim_wrapper below overrides; kept for reference)
optimizer = dict(type='AdamW', lr=2e-4, weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# learning policy: 0-500 iter linear 1e-5 -> 2e-4, 이후 epoch 24까지 cosine 2e-4 -> 1e-6
total_epochs = 24
evaluation = dict(interval=1, pipeline=test_pipeline)

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)

# Convert old-style data config to new-style dataloader config for mmengine
# 원본 BEVFormer와 동일한 설정 사용 (persistent_workers, prefetch_factor 제거)
# CRITICAL: Use DistributedGroupSampler to match original BEVFormer sampling order
train_dataloader = dict(
    batch_size=data['samples_per_gpu'],
    num_workers=data['workers_per_gpu'],
    persistent_workers=True, #False,
    sampler=dict(
        type='DistributedGroupSampler',
        # shuffle=False
        samples_per_gpu=data['samples_per_gpu'],
        seed=0  # 원본과 동일한 seed 사용
    ),
    dataset=data['train']
)

# Enable validation and testing with OccupancyMetric
# CRITICAL: Use DistributedGroupSampler for validation/test to match original BEVFormer
val_dataloader = dict(
    batch_size=data['samples_per_gpu'],
    num_workers=data['workers_per_gpu'],  # Changed from 4 to 0 to match CONet and avoid multiprocessing issues
    persistent_workers=True, #False,
    sampler=dict(
        type='DistributedSampler',
        shuffle=False,
        # samples_per_gpu=1,  # validation batch size
        # seed=0  # 원본과 동일한 seed 사용
    ),
    # Removed collate_fn to use default behavior like CONet
    dataset=data['val']
)

test_dataloader = dict(
    batch_size=data['samples_per_gpu'],
    num_workers=data['workers_per_gpu'],  # Changed from 4 to 0 to match CONet and avoid multiprocessing issues
    persistent_workers=True, #False,
    sampler=dict(
        type='DistributedSampler',
        shuffle=False,
        # samples_per_gpu=1,  # test batch size
        # seed=0  # 원본과 동일한 seed 사용
    ),
    # Removed collate_fn to use default behavior like CONet
    dataset=data['test']
)

# Use OccupancyMetricHybrid which uses STCOcc metric internally
# This ensures evaluation metrics are calculated the same way as SurroundOcc, CONet, and STCOcc
val_evaluator = dict(
    type='mmdet3d.OccupancyMetricHybrid',  # Hybrid metric using STCOcc's implementation
    dataset_name=dataset_name,  # 'occ3d'
    num_classes=18,  # occ3d uses 18 classes (0=others, 1-16=semantic, 17=free)
    use_lidar_mask=False,
    use_image_mask=True,  # occ3d provides mask_camera
    ann_file='data/nuscenes/occ_infos_temporal_val.pkl',
    data_root='data/nuscenes/',
    class_names=occ_class_names,
    eval_metric=eval_metric,  # 'miou'
    sort_by_timestamp=True,  # BEVFormer dataset sorts by timestamp (line 100 in nuscenes_occ.py)
    prefix='val'  # prefix 설정하여 경고 제거
)

test_evaluator = dict(
    type='mmdet3d.OccupancyMetricHybrid',  # Hybrid metric using STCOcc's implementation
    dataset_name=dataset_name,  # 'occ3d'
    num_classes=18,  # occ3d uses 18 classes (0=others, 1-16=semantic, 17=free)
    use_lidar_mask=False,
    use_image_mask=True,  # occ3d provides mask_camera
    ann_file='data/nuscenes/occ_infos_temporal_val.pkl',
    data_root='data/nuscenes/',
    class_names=occ_class_names,
    eval_metric=eval_metric,  # 'miou'
    sort_by_timestamp=True,  # BEVFormer dataset sorts by timestamp (line 100 in nuscenes_occ.py)
    prefix='test'
)

# SurroundOcc와 동일한 optimizer 세팅 (MMEngine OptimWrapper)
# Gradient accumulation: effective batch = batch_size * num_gpus * accumulative_counts
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

# Add train_cfg for mmengine
train_cfg = dict(
    by_epoch=True,
    max_epochs=total_epochs,
    val_interval=999999 # No validation
)

# Enable val_cfg and test_cfg
val_cfg = dict()
test_cfg = dict()

# Param scheduler: 0-500 iter linear 1e-5->2e-4, then cosine 2e-4->1e-6 to end of epoch 24
# steps_per_epoch ≈ len(train) / (batch_size * num_gpus) e.g. 28130/8 ≈ 3516
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.05,  # 1e-5 / 2e-4
        end_factor=1.0,
        by_epoch=False,
        begin=0,
        end=500
    ),
    dict(
        type='CosineAnnealingLR',
        begin=500,
        end=24 * 28130,  # total_epochs * steps_per_epoch (approx for nuscenes train)
        by_epoch=False,
        eta_min=1e-6
    )
]
# Add default_hooks for mmengine
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),#, max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

# Pretrained checkpoint - download if needed: https://github.com/open-mmlab/mmdetection3d/tree/master/configs/fcos3d
load_from = './projects/BEVFormer/pretrain/r101_dcn_fcos3d_pretrain.pth'
#load_from = './projects/BEVFormer/ckpt/epoch_24.pth'

checkpoint_config = dict(interval=1)

# CRITICAL: Environment configuration for distributed training
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

# CRITICAL: Distributed training configuration for mmengine
# This ensures proper handling of DDP and gradient synchronization
# find_unused_parameters=True is needed because obtain_history_bev uses torch.no_grad()
# which means some parameters don't receive gradients in every iteration
model_wrapper_cfg = dict(
    type='MMDistributedDataParallel',
    find_unused_parameters=True,
    broadcast_buffers=False,
)

# Reproducibility: seed=0. EMA/SyncBN/AMP 미사용 (optim_wrapper=OptimWrapper, model norm=BN)
randomness = dict(seed=0, deterministic=False)
