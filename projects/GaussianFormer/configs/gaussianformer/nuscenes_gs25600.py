# Ported from Ref/GaussianFormer_ori/config/prob/nuscenes_gs25600.py (+ its
# config/_base_/{misc,model,surroundocc}.py bases). All hyperparameters below
# are the exact values `mmengine.Config.fromfile` resolves the original
# 4-file config chain to (verified by dumping `cfg.pretty_text` from the
# original repo) -- flattened into one file and re-expressed in this
# framework's modern Runner config style (train_dataloader/optim_wrapper/
# param_scheduler/val_evaluator) instead of GaussianFormer_ori's bespoke
# train.py loop. See projects/GaussianFormer/README.md for the full list of
# porting deviations.
default_scope = 'mmdet3d'
custom_imports = dict(
    imports=['projects.GaussianFormer.gaussianformer'],
    allow_failed_imports=False)

# =========== data config ==============
data_root = 'data/nuscenes/'
anno_root = 'projects/GaussianFormer/data/nuscenes_cam/'
occ_path = 'data/nuscenes_occ/samples'
input_shape = (1600, 864)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

data_aug_conf = {
    "resize_lim": (1.0, 1.0),
    "final_dim": input_shape[::-1],
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (0.0, 0.0),
    "H": 900,
    "W": 1600,
    "rand_flip": True,
}

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadOccupancySurroundOcc', occ_path=occ_path, semantic=True, use_ego=False),
    dict(type='ResizeCropFlipImage'),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='NuScenesAdaptor', use_ego=False, num_cams=6),
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadOccupancySurroundOcc', occ_path=occ_path, semantic=True, use_ego=False),
    dict(type='ResizeCropFlipImage'),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='NuScenesAdaptor', use_ego=False, num_cams=6),
]

train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    drop_last=True,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='custom_collate_fn_temporal'),
    dataset=dict(
        type='GaussianFormerNuScenesDataset',
        data_root=data_root,
        imageset=anno_root + 'nuscenes_infos_train_sweeps_occ.pkl',
        data_aug_conf=data_aug_conf,
        pipeline=train_pipeline,
        phase='train'))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    collate_fn=dict(type='custom_collate_fn_temporal'),
    dataset=dict(
        type='GaussianFormerNuScenesDataset',
        data_root=data_root,
        imageset=anno_root + 'nuscenes_infos_val_sweeps_occ.pkl',
        data_aug_conf=data_aug_conf,
        pipeline=test_pipeline,
        phase='val'))
test_dataloader = val_dataloader

val_evaluator = dict(type='GaussianFormerOccupancyMetric')
test_evaluator = val_evaluator

# =========== training config ==============
max_epochs = 20
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=4e-4, weight_decay=0.01),
    paramwise_cfg=dict(custom_keys={'img_backbone': dict(lr_mult=0.1)}),
    clip_grad=dict(max_norm=35, norm_type=2))

# Original GaussianFormer_ori/train.py drives an iteration-based
# timm.scheduler.CosineLRScheduler(t_initial=len(train_loader)*max_epochs,
# lr_min=lr*0.1, warmup_t=500, warmup_lr_init=1e-6, t_in_epochs=False).
# Reproduced as a 500-iter linear warmup followed by an epoch-wise cosine
# decay to lr*0.1 -- the same warmup+cosine-to-lr*0.1 shape, following this
# framework's existing convention (see projects/CONet, projects/SurroundOcc)
# of expressing the post-warmup decay by_epoch=True rather than needing the
# exact iters-per-epoch count upfront.
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6 / 4e-4, by_epoch=False, begin=0, end=500),
    dict(type='CosineAnnealingLR', by_epoch=True, T_max=max_epochs, begin=0, end=max_epochs, eta_min=4e-5),
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_scope = 'mmdet3d'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, save_best='mIoU', rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
load_from = 'projects/GaussianFormer/pretrain/r101_dcn_fcos3d_pretrain.pth'
resume = False

# =========== loss config ==============
loss = dict(
    type='MultiLoss',
    loss_cfgs=[
        dict(
            type='OccupancyLoss',
            weight=1.0,
            empty_label=17,
            num_classes=18,
            use_focal_loss=False,
            use_dice_loss=False,
            balance_cls_weight=True,
            multi_loss_weights=dict(
                loss_voxel_ce_weight=10.0,
                loss_voxel_lovasz_weight=1.0),
            use_sem_geo_scal_loss=False,
            use_lovasz_loss=True,
            lovasz_ignore=17,
            manual_class_weight=[
                1.01552756, 1.06897009, 1.30013094, 1.07253735, 0.94637502, 1.10087012,
                1.26960524, 1.06258364, 1.189019,   1.06217292, 1.00595144, 0.85706115,
                1.03923299, 0.90867526, 0.8936431,  0.85486129, 0.8527829,  0.5],
            ignore_empty=False,
            lovasz_use_softmax=False),
        dict(
            type='PixelDistributionLoss',
            weight=1.0,
            use_sigmoid=False),
    ])

loss_input_convertion = dict(
    pred_occ='pred_occ',
    sampled_xyz='sampled_xyz',
    sampled_label='sampled_label',
    occ_mask='occ_mask',
    bin_logits='bin_logits',
    density='density',
    pixel_logits='pixel_logits',
    pixel_gt='pixel_gt')

# ========= model config ===============
embed_dims = 128
num_decoder = 4
pc_range = [-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]
scale_range = [0.01, 1.8]
xyz_coordinate = 'cartesian'
phi_activation = 'sigmoid'
include_opa = True
semantics = True
semantic_dim = 17

model = dict(
    type='BEVSegmentor',
    loss=loss,
    loss_input_convertion=loss_input_convertion,
    freeze_lifter=True,
    img_backbone_out_indices=[0, 1, 2, 3],
    img_backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        style='caffe',
        with_cp=True,
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True)),
    img_neck=dict(
        type='FPN',
        num_outs=4,
        start_level=1,
        out_channels=embed_dims,
        add_extra_convs='on_output',
        relu_before_extra_convs=True,
        in_channels=[256, 512, 1024, 2048]),
    lifter=dict(
        type='GaussianLifterV2',
        num_anchor=19200,
        embed_dims=embed_dims,
        anchor_grad=False,
        feat_grad=False,
        phi_activation='loop',
        semantics=semantics,
        semantic_dim=semantic_dim,
        include_opa=include_opa,
        num_samples=128,
        anchors_per_pixel=1,
        random_sampling=False,
        projection_in=None,
        initializer=dict(
            type='ResNetSecondFPN',
            img_backbone_out_indices=[0, 1, 2, 3],
            img_backbone_config=dict(
                type='ResNet',
                depth=101,
                num_stages=4,
                out_indices=(0, 1, 2, 3),
                frozen_stages=1,
                norm_cfg=dict(type='BN2d', requires_grad=False),
                norm_eval=True,
                style='caffe',
                with_cp=True,
                dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
                stage_with_dcn=(False, False, True, True)),
            neck_confifg=dict(
                type='SECONDFPN',
                in_channels=[256, 512, 1024, 2048],
                out_channels=[embed_dims] * 4,
                upsample_strides=[0.5, 1, 2, 4])),
        initializer_img_downsample=None,
        pretrained_path='projects/GaussianFormer/pretrain/lifter_10.pth',
        deterministic=False,
        random_samples=6400),
    encoder=dict(
        type='GaussianOccEncoder',
        anchor_encoder=dict(
            type='SparseGaussian3DEncoder',
            embed_dims=embed_dims,
            include_opa=include_opa,
            semantics=semantics,
            semantic_dim=semantic_dim),
        norm_layer=dict(type='LN', normalized_shape=embed_dims),
        ffn=dict(
            type='AsymmetricFFN',
            in_channels=embed_dims,
            embed_dims=embed_dims,
            feedforward_channels=embed_dims * 4,
            ffn_drop=0.1,
            add_identity=False),
        deformable_model=dict(
            type='DeformableFeatureAggregation',
            embed_dims=embed_dims,
            num_groups=4,
            num_levels=4,
            num_cams=6,
            attn_drop=0.15,
            use_deformable_func=True,
            use_camera_embed=True,
            residual_mode='none',
            kps_generator=dict(
                type='SparseGaussian3DKeyPointsGenerator',
                embed_dims=embed_dims,
                phi_activation=phi_activation,
                xyz_coordinate=xyz_coordinate,
                num_learnable_pts=6,
                fix_scale=[
                    [0, 0, 0],
                    [0.45, 0, 0],
                    [-0.45, 0, 0],
                    [0, 0.45, 0],
                    [0, -0.45, 0],
                    [0, 0, 0.45],
                    [0, 0, -0.45],
                ],
                pc_range=pc_range,
                scale_range=scale_range,
                learnable_fixed_scale=6.0)),
        refine_layer=dict(
            type='SparseGaussian3DRefinementModuleV2',
            embed_dims=embed_dims,
            pc_range=pc_range,
            scale_range=scale_range,
            unit_xyz=[4.0, 4.0, 1.0],
            semantics=semantics,
            semantic_dim=semantic_dim,
            include_opa=include_opa,
            xyz_coordinate=xyz_coordinate,
            semantics_activation='identity'),
        spconv_layer=dict(
            type='SparseConv3D',
            in_channels=embed_dims,
            embed_channels=embed_dims,
            pc_range=pc_range,
            grid_size=[1.0, 1.0, 1.0],
            phi_activation=phi_activation,
            xyz_coordinate=xyz_coordinate,
            use_out_proj=True,
            use_multi_layer=True),
        num_decoder=num_decoder,
        operation_order=[
            'identity', 'deformable', 'add', 'norm',
            'identity', 'ffn', 'add', 'norm',
            'identity', 'spconv', 'add', 'norm',
            'identity', 'ffn', 'add', 'norm',
            'refine',
        ] * num_decoder),
    head=dict(
        type='GaussianHead',
        apply_loss_type='random_1',
        num_classes=semantic_dim + 1,
        empty_args=dict(
            mean=[0, 0, -1.0],
            scale=[100, 100, 8.0]),
        with_empty=False,
        use_localaggprob=True,
        use_localaggprob_fast=False,
        combine_geosem=True,
        cuda_kwargs=dict(
            scale_multiplier=4,
            H=200, W=200, D=16,
            pc_min=[-50.0, -50.0, -5.0],
            grid_size=0.5)))
