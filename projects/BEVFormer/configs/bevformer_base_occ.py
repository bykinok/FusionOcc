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

# Set default scope to mmdet3d for all modules
default_scope = 'mmdet3d'

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
    video_test_mode=True,
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
        use_mask=False,
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
data_root = 'data/occ3d-nus/'
file_client_args = dict(backend='disk')
occ_gt_data_root='data/occ3d-nus'

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadOccGTFromFile',data_root=occ_gt_data_root),
    dict(type='PhotoMetricDistortionMultiViewImage'),
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
    dict(type='LoadOccGTFromFile',data_root=occ_gt_data_root),
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
            dict(type='CustomCollect3D', keys=['img'])
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
             pipeline=test_pipeline,  bev_size=(bev_h_, bev_w_),
             classes=class_names, modality=input_modality, samples_per_gpu=1),
    test=dict(type=dataset_type,
              data_root=data_root,
              ann_file='occ_infos_temporal_val.pkl',
              pipeline=test_pipeline, bev_size=(bev_h_, bev_w_),
              classes=class_names, modality=input_modality),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
)

optimizer = dict(
    type='AdamW',
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)

total_epochs = 24
evaluation = dict(interval=1, pipeline=test_pipeline)

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)

# Convert old-style data config to new-style dataloader config for mmengine
train_dataloader = dict(
    batch_size=data['samples_per_gpu'],
    num_workers=data['workers_per_gpu'],
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=data['train']
)

# Disable validation and testing for now (OccMetric not registered yet)
val_dataloader = None
test_dataloader = None
val_evaluator = None
test_evaluator = None

# Convert old-style optimizer config to new-style optim_wrapper
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type=optimizer['type'],
        lr=optimizer['lr'],
        weight_decay=optimizer.get('weight_decay', 0)
    ),
    paramwise_cfg=optimizer.get('paramwise_cfg', None),
    clip_grad=optimizer_config.get('grad_clip', None)
)

# Add train_cfg for mmengine (without validation)
train_cfg = dict(
    by_epoch=True,
    max_epochs=total_epochs
)

# Disable val_cfg and test_cfg
val_cfg = None
test_cfg = None

# Add param_scheduler for mmengine (from lr_config)
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=lr_config.get('warmup_ratio', 0.001),
        by_epoch=False,
        begin=0,
        end=lr_config.get('warmup_iters', 500)
    ),
    dict(
        type='CosineAnnealingLR',
        by_epoch=True,
        begin=1,
        end=total_epochs,
        eta_min=lr_config['min_lr_ratio'] * optimizer['lr']
    )
]
# Pretrained checkpoint - download if needed: https://github.com/open-mmlab/mmdetection3d/tree/master/configs/fcos3d
# load_from = 'ckpts/r101_dcn_fcos3d_pretrain.pth'
load_from = None  # Train from scratch for now

checkpoint_config = dict(interval=1)
