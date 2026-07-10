_base_ = [
    '../datasets/custom_nus-3d.py',
    '../_base_/default_runtime.py'
]

custom_imports = dict(
    imports=['projects.SurroundOcc.surroundocc'],
    allow_failed_imports=False)

# -----------------------------------------------------------------------
# Ablation: condition_C_full (occupied-only full-range supervision)
#   - ALL occupied voxels: always supervised (gt label kept) regardless
#     of camera visibility, at every distance from ego.
#   - Free voxels: obey original mask_camera (invisible free → 255).
#
# SurroundOcc sets non-supervised voxels to 255 (ignore) in gt_occ.
# condition_C_full is implemented in LoadOccupancy._apply_mask.
# Eval: unchanged (evaluator loads mask_camera independently).
#
# Comparison table (occupied voxel invisible supervision):
#   condition_C_20m  → occupied supervised 0–20 m invisible + all visible
#   condition_C      → occupied supervised 0–35 m invisible + all visible
#   condition_C_full → occupied supervised 0–∞  invisible + all visible
# -----------------------------------------------------------------------

dataset_name = 'occ3d'

point_cloud_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
occ_size = [200, 200, 16]
use_semantic = True

use_grid_mask = False

use_mask_camera = True
use_mask_camera_1_2 = False
use_mask_camera_1_4 = False
use_mask_camera_1_8 = False

img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

class_names = ['others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
               'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
               'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
               'vegetation', 'free']

bda_aug_conf = dict(
    rot_lim=(0, 0),
    scale_lim=(1., 1.),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5,
    tran_lim=0)

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)

_dim_ = [128, 256, 512]
_ffn_dim_ = [256, 512, 1024]
volume_h_ = [100, 50, 25]
volume_w_ = [100, 50, 25]
volume_z_ = [8, 4, 2]
_num_points_ = [2, 4, 8]
_num_layers_ = [1, 3, 6]

model = dict(
    type='SurroundOcc',
    use_grid_mask=use_grid_mask,
    use_semantic=use_semantic,
    dataset_name=dataset_name,
    save_results=False,
    img_backbone=dict(
       type='ResNet',
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
        type='FPN',
        in_channels=[512, 1024, 2048],
        out_channels=512,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=3,
        relu_before_extra_convs=True),
    pts_bbox_head=dict(
        type='OccHead',
        volume_h=volume_h_,
        volume_w=volume_w_,
        volume_z=volume_z_,
        num_query=900,
        num_classes=18,
        conv_input=[_dim_[2], 256, _dim_[1], 128, _dim_[0], 64, 64],
        conv_output=[256, _dim_[1], 128, _dim_[0], 64, 64, 32],
        out_indices=[0, 2, 4, 6],
        upsample_strides=[1, 2, 1, 2, 1, 2, 1],
        embed_dims=_dim_,
        img_channels=[512, 512, 512],
        use_semantic=use_semantic,
        dataset_name=dataset_name,
        transformer_template=dict(
            type='PerceptionTransformer',
            embed_dims=_dim_,
            encoder=dict(
                type='OccEncoder',
                num_layers=_num_layers_,
                pc_range=point_cloud_range,
                return_intermediate=False,
                transformerlayers=dict(
                    type='OccLayer',
                    attn_cfgs=[
                        dict(
                            type='SpatialCrossAttention',
                            pc_range=point_cloud_range,
                            deformable_attention=dict(
                                type='MSDeformableAttention3D',
                                embed_dims=_dim_,
                                num_points=_num_points_,
                                num_levels=1),
                            embed_dims=_dim_,
                        )
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    embed_dims=_dim_,
                    conv_num=2,
                    operation_order=('cross_attn', 'norm',
                                     'ffn', 'norm', 'conv')))),
    ),
)

dataset_type = 'CustomNuScenesOccDataset'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')

# condition_C_full: ALL occupied voxels supervised; invisible free voxels → 255.
# Multi-scale scales (1_2/1_4/1_8) remain with baseline_with_mask behavior.
train_pipeline = [
    dict(type='LoadMultiViewImageFromFilesFullRes', to_float32=True),
    dict(type='LoadOccupancy',
         use_semantic=use_semantic,
         use_occ3d=True if dataset_name == 'occ3d' else False,
         pc_range=point_cloud_range if dataset_name == 'occ3d' else None,
         use_mask_camera=use_mask_camera,
         use_mask_camera_1_2=use_mask_camera_1_2,
         use_mask_camera_1_4=use_mask_camera_1_4,
         use_mask_camera_1_8=use_mask_camera_1_8,
         mask_mode='condition_C_full',
         free_class_id=class_names.index('free')),  # 17
    dict(type='BEVAug', bda_aug_conf=bda_aug_conf, is_train=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='OccDefaultFormatBundle3D')
]

# Eval pipeline is unchanged (evaluator uses mask_camera independently)
test_pipeline = [
    dict(type='LoadMultiViewImageFromFilesFullRes', to_float32=True),
    dict(type='LoadOccupancy',
         use_semantic=use_semantic,
         use_occ3d=True if dataset_name == 'occ3d' else False,
         pc_range=point_cloud_range if dataset_name == 'occ3d' else None,
         use_mask_camera=use_mask_camera,
         use_mask_camera_1_2=use_mask_camera_1_2,
         use_mask_camera_1_4=use_mask_camera_1_4,
         use_mask_camera_1_8=use_mask_camera_1_8),
    dict(type='BEVAug', bda_aug_conf=bda_aug_conf, is_train=False),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='OccDefaultFormatBundle3D')
]

train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(
        type='DistributedGroupSampler',
        samples_per_gpu=1,
        seed=0
    ),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='surroundocc-nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        occ_size=occ_size,
        pc_range=point_cloud_range,
        use_semantic=use_semantic,
        use_ego_frame=True,
        classes=class_names,
        box_type_3d='LiDAR'))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(
        type='DistributedSampler',
        shuffle=False,
        seed=0
    ),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='surroundocc-nuscenes_infos_val.pkl',
        pipeline=test_pipeline,
        occ_size=occ_size,
        pc_range=point_cloud_range,
        use_semantic=use_semantic,
        use_ego_frame=True,
        classes=class_names,
        modality=input_modality,
        test_mode=True))

test_dataloader = val_dataloader

train_samples = 28130
num_gpus = 2
samples_per_gpu = 1
num_iters_per_epoch = train_samples // (num_gpus * samples_per_gpu)

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

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=24,
    val_interval=9999)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

val_evaluator = dict(
    type='OccupancyMetricHybrid',
    dataset_name=dataset_name,
    num_classes=18,
    use_lidar_mask=False,
    use_image_mask=True if dataset_name == 'occ3d' else False,
    ann_file='data/nuscenes/surroundocc-nuscenes_infos_val.pkl',
    data_root='data/nuscenes',
    class_names=class_names,
    eval_metric='miou',
    sort_by_timestamp=True)

test_evaluator = val_evaluator

load_from = 'projects/SurroundOcc/pretrain/r101_dcn_fcos3d_pretrain.pth'

model_wrapper_cfg = dict(
    type='MMDistributedDataParallel',
    find_unused_parameters=True)

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'))

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

checkpoint_config = dict(interval=1)

randomness = dict(seed=0, deterministic=False)
