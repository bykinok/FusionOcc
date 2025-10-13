_base_ = [
    '../datasets/custom_nus-3d.py',
    '../_base_/default_runtime.py'
]
#
custom_imports = dict(
    imports=['projects.SurroundOcc.surroundocc'],
    allow_failed_imports=False)

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-50, -50, -5.0, 50, 50, 3.0]
occ_size = [200, 200, 16]
use_semantic = True


img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

class_names =  ['barrier','bicycle', 'bus', 'car', 'construction_vehicle', 'motorcycle',
                'pedestrian', 'traffic_cone', 'trailer', 'truck', 'driveable_surface',
                'other_flat', 'sidewalk', 'terrain', 'manmade','vegetation']

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
    use_grid_mask=True,
    use_semantic=use_semantic,
    img_backbone=dict(
       type='ResNet',
       depth=101,
       num_stages=4,
       out_indices=(1,2,3),
       frozen_stages=1,
       norm_cfg=dict(type='BN2d', requires_grad=False),
       norm_eval=True,
       style='caffe',
       #with_cp=True, # using checkpoint to save GPU memory
       dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False), # original DCNv2 will print log when perform load_state_dict
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
        num_classes=17,
        conv_input=[_dim_[2], 256, _dim_[1], 128, _dim_[0], 64, 64],
        conv_output=[256, _dim_[1], 128, _dim_[0], 64, 64, 32],
        out_indices=[0, 2, 4, 6],
        upsample_strides=[1,2,1,2,1,2,1],
        embed_dims=_dim_,
        img_channels=[512, 512, 512],
        use_semantic=use_semantic,
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


train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='LoadOccupancy', use_semantic=use_semantic),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
    dict(type='CustomCollect3D', keys=['img', 'gt_occ'])
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadOccupancy', use_semantic=use_semantic),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
    dict(type='CustomCollect3D', keys=['img','gt_occ'])
]

# New style data loaders
train_dataloader = dict(
    batch_size=1,  # 원본과 동일. GPU 메모리 충분하면 2로 증가 가능
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
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
        classes=class_names,
        box_type_3d='LiDAR'))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='surroundocc-nuscenes_infos_val.pkl',
        pipeline=test_pipeline,  
        occ_size=occ_size,
        pc_range=point_cloud_range,
        use_semantic=use_semantic,
        classes=class_names,
        modality=input_modality,
        test_mode=True))

test_dataloader = val_dataloader

# New style config for MMEngine
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
    clip_grad=dict(max_norm=35, norm_type=2))

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0 / 3,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        type='CosineAnnealingLR',
        begin=500,
        end=24*28130,  # total_epochs * steps_per_epoch (approximate)
        by_epoch=False,
        eta_min_ratio=1e-3)
]

# Training config
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=24,
    val_interval=1)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

val_evaluator = dict(
    type='OccupancyMetric')

test_evaluator = val_evaluator
# load_from = 'ckpts/r101_dcn_fcos3d_pretrain.pth'  # 사전 훈련된 가중치 파일이 없으면 주석 처리

# MMEngine style logging configuration
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),  # 50개 iteration마다 로그 출력
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'))

# Legacy log_config (for backward compatibility)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

checkpoint_config = dict(interval=1)
