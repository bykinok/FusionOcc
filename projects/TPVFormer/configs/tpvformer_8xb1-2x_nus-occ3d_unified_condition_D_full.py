_base_ = ['../../../configs/_base_/default_runtime.py']

custom_imports = dict(
    imports=['projects.TPVFormer.tpvformer'], allow_failed_imports=False)

# -----------------------------------------------------------------------
# Ablation: condition_D_full (free-only full-range supervision)
#   - ALL free voxels at every distance: always supervised (gt label kept)
#     regardless of camera visibility.
#   - Occupied voxels: obey original mask_camera (invisible occupied → 255).
#
# TPVFormer sets non-supervised voxels to 255 (ignore_label) in
# voxel_semantic_mask. condition_D_full is implemented in LoadOccupancy
# by forcing supervision for all free voxels regardless of distance.
# Eval: unchanged (evaluator loads mask_camera independently).
#
# Comparison table (free voxel invisible supervision):
#   condition_D       → free supervised 0–20 m only
#   condition_D_prime → free supervised 0–35 m only
#   condition_D_full  → free supervised 0–∞  (this config)
# -----------------------------------------------------------------------

occupancy = True
lovasz_input = 'voxel'
ce_input = 'voxel'

dataset_name = 'occ3d'
use_grid_mask = False
use_camera_mask = True  # kept for eval evaluator; train masking overridden by mask_mode

dataset_params = dict(
    version = "v1.0-trainval",
    ignore_label = 255,
    fill_label = 17,
    fixed_volume_space = True,
    label_mapping = "./config/label_mapping/nuscenes-noIgnore.yaml",
    max_volume_space = [40, 40, 5.4],
    min_volume_space = [-40, -40, -1],
)

dataset_type = 'NuScenesOccupancyDataset'
data_root = 'data/nuscenes/'
data_prefix = dict(
    pts='samples/LIDAR_TOP',
    pts_semantic_mask='lidarseg/v1.0-trainval',
    CAM_FRONT='samples/CAM_FRONT',
    CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
    CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
    CAM_BACK='samples/CAM_BACK',
    CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
    CAM_BACK_LEFT='samples/CAM_BACK_LEFT')

backend_args = None

bda_aug_conf = dict(
    rot_lim=(0, 0),
    scale_lim=(1., 1.),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5
)

# condition_D_full: ALL free voxels supervised; invisible occupied voxels → 255.
train_pipeline = [
    dict(
        type='BEVLoadMultiViewImageFromFiles',
        to_float32=True,
        color_type='unchanged',
        num_views=6,
        backend_args=backend_args,
        use_ego_frame=True),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=3,
        backend_args=backend_args),
    dict(
        type='LoadOccupancy',
        use_occ3d=True,
        pc_range=[-40.0, -40.0, -1.0, 40.0, 40.0, 5.4],
        use_camera_mask=use_camera_mask,
        mask_mode='condition_D_full',
        free_class_id=17,
    ),
    dict(type='BEVAug', bda_aug_conf=bda_aug_conf, is_train=True),
    dict(
        type='MultiViewImageNormalize',
        mean=[103.530, 116.280, 123.675],
        std=[1.0, 1.0, 1.0],
        to_rgb=False),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='TPVPack3DDetInputs',
        keys=['img', 'points', 'pts_semantic_mask', 'voxel_semantic_mask', 'occ_3d', 'occ_3d_masked'],
        meta_keys=['lidar2img', 'lidar_path', 'sample_idx', 'pts_filename', 'img_shape', 'token', 'scene_name', 'scene_token',
                   'pcd_horizontal_flip', 'pcd_vertical_flip', 'bda_mat'])
]

# Eval pipeline is unchanged (evaluator uses mask_camera independently)
val_pipeline = [
    dict(
        type='BEVLoadMultiViewImageFromFiles',
        to_float32=True,
        color_type='unchanged',
        num_views=6,
        backend_args=backend_args,
        use_ego_frame=True),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=3,
        backend_args=backend_args),
    dict(
        type='LoadOccupancy',
        use_occ3d=True,
        pc_range=[-40.0, -40.0, -1.0, 40.0, 40.0, 5.4],
        use_camera_mask=use_camera_mask),
    dict(type='BEVAug', bda_aug_conf=bda_aug_conf, is_train=False),
    dict(
        type='MultiViewImageNormalize',
        mean=[103.530, 116.280, 123.675],
        std=[1.0, 1.0, 1.0],
        to_rgb=False),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='TPVPack3DDetInputs',
        keys=['img', 'points', 'pts_semantic_mask', 'voxel_semantic_mask', 'occ_3d', 'occ_3d_masked'],
        meta_keys=['lidar2img', 'lidar_path', 'sample_idx', 'pts_filename', 'img_shape', 'token', 'scene_name', 'scene_token',
                   'pcd_horizontal_flip', 'pcd_vertical_flip', 'bda_mat'])
]

test_pipeline = val_pipeline

train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=data_prefix,
        ann_file='occfrmwrk-nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        test_mode=False))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=data_prefix,
        ann_file='occfrmwrk-nuscenes_infos_val.pkl',
        pipeline=val_pipeline,
        test_mode=True))

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=data_prefix,
        ann_file='occfrmwrk-nuscenes_infos_val.pkl',
        pipeline=val_pipeline,
        test_mode=True))

val_evaluator = dict(
    type='OccupancyMetricHybrid',
    dataset_name=dataset_name,
    num_classes=18,
    use_lidar_mask=False,
    use_image_mask=use_camera_mask if dataset_name == 'occ3d' else False,
    ann_file='data/nuscenes/occfrmwrk-nuscenes_infos_val.pkl',
    data_root='data/nuscenes/',
    eval_metric='miou',
    sort_by_timestamp=True)

test_evaluator = val_evaluator

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=2e-4, weight_decay=0.01),
    paramwise_cfg=dict(custom_keys={
        'img_backbone': dict(lr_mult=0.1),
    }),
    clip_grad=dict(max_norm=35, norm_type=2),
    accumulative_counts=8)

train_samples = 28130
num_gpus = 2
samples_per_gpu = 1
num_iters_per_epoch = train_samples // (num_gpus * samples_per_gpu)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.05, end_factor=1.0, by_epoch=False, begin=0, end=500),
    dict(
        type='CosineAnnealingLR',
        begin=500,
        end=24 * num_iters_per_epoch,
        by_epoch=False,
        eta_min=1e-6)
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=24, val_interval=999)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=1))

randomness = dict(seed=0, deterministic=False, diff_rank_seed=False)

find_unused_parameters = False

load_from = './projects/TPVFormer/pretrain/r101_dcn_fcos3d_pretrain.pth'
resume = False

point_cloud_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
_dim_ = 256
_pos_dim_ = _dim_//2
_ffn_dim_ = _dim_*2
_num_levels_ = 4
_num_cams_ = 6
tpv_h_ = 100
tpv_w_ = 100
tpv_z_ = 8
scale_h = 2
scale_w = 2
scale_z = 2
grid_size = [tpv_h_*scale_h, tpv_w_*scale_w, tpv_z_*scale_z]
nbr_class = 18
num_points_in_pillar = [4, 32, 32]
num_points = [8, 64, 64]

model = dict(
    type='TPVFormer',
    dataset_name=dataset_name,
    save_results=False,
    data_preprocessor=dict(
        type='TPVFormerDataPreprocessor',
        mean=None,
        std=None,
        bgr_to_rgb=False,
        pad_size_divisor=1,
        voxel=True,
        voxel_type='cylindrical',
        voxel_layer=dict(
            grid_shape=grid_size,
            point_cloud_range=point_cloud_range,
            max_num_points=-1,
            max_voxels=-1,
        )),
    use_grid_mask=use_grid_mask,
    ignore_label=255,
    lovasz_input=lovasz_input,
    ce_input=ce_input,
    tpv_aggregator=dict(
        type='TPVAggregator',
        tpv_h=tpv_h_,
        tpv_w=tpv_w_,
        tpv_z=tpv_z_,
        nbr_classes=nbr_class,
        in_dims=_dim_,
        hidden_dims=2*_dim_,
        out_dims=_dim_,
        scale_h=scale_h,
        scale_w=scale_w,
        scale_z=scale_z,
        use_checkpoint=True
    ),
    img_backbone=dict(
        type='mmdet.ResNet',
        depth=101,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        style='caffe',
        dcn=dict(type='mmdet.DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True)),
    img_neck=dict(
        type='mmdet.FPN',
        in_channels=[512, 1024, 2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=4,
        relu_before_extra_convs=True),
    tpv_head=dict(
        type='TPVFormerHead',
        tpv_h=tpv_h_,
        tpv_w=tpv_w_,
        tpv_z=tpv_z_,
        pc_range=point_cloud_range,
        num_feature_levels=_num_levels_,
        num_cams=_num_cams_,
        embed_dims=_dim_,
        positional_encoding=dict(
            type='mmdet.LearnedPositionalEncoding',
            num_feats=_pos_dim_,
            row_num_embed=tpv_h_,
            col_num_embed=tpv_w_),
        encoder=dict(
            type='TPVFormerEncoder',
            tpv_h=tpv_h_,
            tpv_w=tpv_w_,
            tpv_z=tpv_z_,
            num_layers=3,
            pc_range=point_cloud_range,
            num_points_in_pillar=num_points_in_pillar,
            return_intermediate=False,
            transformerlayers=dict(
                type='TPVFormerLayer',
                attn_cfgs=[
                    dict(
                        type='TPVCrossViewHybridAttention',
                        embed_dims=_dim_,
                        num_levels=1),
                    dict(
                        type='TPVImageCrossAttention',
                        pc_range=point_cloud_range,
                        deformable_attention=dict(
                            type='TPVMSDeformableAttention3D',
                            embed_dims=_dim_,
                            num_points=num_points,
                            num_z_anchors=num_points_in_pillar,
                            num_levels=_num_levels_,
                            floor_sampling_offset=False,
                            tpv_h=tpv_h_,
                            tpv_w=tpv_w_,
                            tpv_z=tpv_z_,
                        ),
                        embed_dims=_dim_,
                        tpv_h=tpv_h_,
                        tpv_w=tpv_w_,
                        tpv_z=tpv_z_,
                    )
                ],
                feedforward_channels=_ffn_dim_,
                ffn_dropout=0.1,
                operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                 'ffn', 'norm')))))
