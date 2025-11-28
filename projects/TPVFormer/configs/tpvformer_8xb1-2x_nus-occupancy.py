_base_ = ['../../../configs/_base_/default_runtime.py']

custom_imports = dict(
    imports=['projects.TPVFormer.tpvformer'], allow_failed_imports=False)

# Occupancy task 설정
occupancy = True
lovasz_input = 'voxel'
ce_input = 'voxel'

dataset_params = dict(
    version = "v1.0-trainval",
    ignore_label = 0,  # ✅ Class 0 (noise/background) ignore (원본과 동일)
    fill_label = 17,   # ✅ Class 17 (empty)로 빈 voxel 채우기 (원본과 동일)
    fixed_volume_space = True,
    label_mapping = "./config/label_mapping/nuscenes-noIgnore.yaml",
    max_volume_space = [51.2, 51.2, 3],
    min_volume_space = [-51.2, -51.2, -5],
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

train_pipeline = [
    dict(
        type='BEVLoadMultiViewImageFromFiles',
        to_float32=True,  # 원본과 동일하게 float32로 변환
        color_type='unchanged',
        num_views=6,
        backend_args=backend_args),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=3,
        backend_args=backend_args),
    dict(
        type='LoadOccupancyAnnotations',  # 새로운 occupancy annotation 로딩
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        with_attr_label=False,
        seg_3d_dtype='np.uint8'),
    # ✅ 원본과 동일한 PhotoMetric Augmentation 추가 (train only)
    dict(
        type='PhotoMetricDistortionMultiViewImage',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    # 원본과 동일한 정규화 값 적용
    dict(
        type='MultiViewImageNormalize',
        mean=[103.530, 116.280, 123.675],
        std=[1.0, 1.0, 1.0],
        to_rgb=False),
    # 원본과 동일하게 32로 나누어떨어지도록 패딩
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='SegLabelMapping'),
    dict(
        type='Pack3DDetInputs',
        keys=['img', 'points', 'pts_semantic_mask', 'voxel_semantic_mask', 'voxel_coords'],
        meta_keys=['lidar2img', 'lidar_path', 'sample_idx', 'pts_filename', 'img_shape'])
]

val_pipeline = [
    dict(
        type='BEVLoadMultiViewImageFromFiles',
        to_float32=True,  # 원본과 동일하게 float32로 변환
        color_type='unchanged',
        num_views=6,
        backend_args=backend_args),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=3,
        backend_args=backend_args),
    dict(
        type='LoadOccupancyAnnotations',  # 새로운 occupancy annotation 로딩
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        with_attr_label=False,
        seg_3d_dtype='np.uint8'),
    # 원본과 동일한 정규화 값 적용
    dict(
        type='MultiViewImageNormalize',
        mean=[103.530, 116.280, 123.675],
        std=[1.0, 1.0, 1.0],
        to_rgb=False),
    # 원본과 동일하게 32로 나누어떨어지도록 패딩
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='SegLabelMapping'),
    dict(
        type='Pack3DDetInputs',
        keys=['img', 'points', 'pts_semantic_mask', 'voxel_semantic_mask', 'voxel_coords'],
        meta_keys=['lidar2img', 'lidar_path', 'sample_idx', 'pts_filename', 'img_shape'])
]

test_pipeline = val_pipeline

train_dataloader = dict(
    batch_size=1,  # ✅ 원본과 동일하게 batch_size=1 per GPU
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
    type='OccupancyMetric', 
    class_indices=list(range(1, 17)),  # ✅ 1-16 classes만 평가 (원본과 동일)
    ignore_label=0)  # ✅ Class 0 (noise) ignore

test_evaluator = val_evaluator

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=2e-4, weight_decay=0.01),
    paramwise_cfg=dict(custom_keys={
        'img_backbone': dict(lr_mult=0.1),  # 원본과 동일하게 img_backbone으로 변경
    }),
    clip_grad=dict(max_norm=35, norm_type=2),
)

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-5, by_epoch=False, begin=0, end=500),
    dict(
        type='CosineAnnealingLR',
        begin=0,
        T_max=24,
        by_epoch=True,
        eta_min=1e-6,
        convert_to_iter_based=True)
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=24, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=1))

# Reproducibility settings (for consistent results across runs)
randomness = dict(seed=0, deterministic=False, diff_rank_seed=False)

# DDP settings (following original TPVFormer)
find_unused_parameters = False

# Load pretrained backbone weights (following original TPVFormer)
# Only load img_backbone weights, ignore other keys
load_from = './projects/TPVFormer/pretrain/r101_dcn_fcos3d_pretrain.pth'
resume = False  # Don't resume training, just load backbone weights

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
_dim_ = 256
_pos_dim_ = _dim_//2
_ffn_dim_ = _dim_*2
_num_levels_ = 4
_num_cams_ = 6
tpv_h_ = 100
tpv_w_ = 100
tpv_z_ = 8
scale_h = 1
scale_w = 1
scale_z = 1
grid_size = [tpv_h_*scale_h, tpv_w_*scale_w, tpv_z_*scale_z]
nbr_class = 18
num_points_in_pillar = [4, 32, 32]
num_points = [8, 64, 64]



model = dict(
    type='TPVFormer',
    # 원본과 동일한 전처리: data pipeline에서 정규화 수행, preprocessor에서는 수행하지 않음
    data_preprocessor=dict(
        type='TPVFormerDataPreprocessor',
        mean=None,  # 정규화 비활성화 (pipeline에서 이미 수행됨)
        std=None,
        bgr_to_rgb=False,  # BGR 유지
        pad_size_divisor=1,
        # Voxel layer 설정 (원본과 동일한 voxel 크기 사용)
        voxel=True,
        voxel_type='cylindrical',
        voxel_layer=dict(
            grid_shape=grid_size,  # [100, 100, 8]
            point_cloud_range=point_cloud_range,  # [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
            max_num_points=-1,
            max_voxels=-1,
        )),
    use_grid_mask=True,
    # Loss configuration (following original TPVFormer)
    ignore_label=0,  # ✅ Class 0 (noise/background) ignore (원본과 동일)
    lovasz_input=lovasz_input,  # 'voxel' or 'points'
    ce_input=ce_input,  # 'voxel' or 'points'
    tpv_aggregator = dict(
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
        use_checkpoint=True  # 원본과 동일
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
            type='mmdet.LearnedPositionalEncoding',  # mmdet의 LearnedPositionalEncoding 사용 (원본과 동일)
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
