"""SparseOcc NuScenes 256x704 설정 (새 mmengine 형식).

원본: Ref/SparseOcc_cvpr_ori/projects/configs/sparseocc/sparseocc_nusc_256.py
변경점:
  - plugin/plugin_dir → custom_imports
  - mmcv 1.x runner → mmengine 스타일 train_cfg / optim_wrapper
  - data dict → train_dataloader / val_dataloader / test_dataloader
  - 모델 로직 / 파이프라인은 원본과 동일하게 유지
"""

_base_ = [
    '../_base_/default_runtime.py',
]

# 플러그인 로드 (새 mmengine custom_imports 방식)
custom_imports = dict(
    imports=['projects.SparseOcc_cvpr.sparseocc_cvpr'],
    allow_failed_imports=False)

work_dir = 'work_dirs/sparseocc_nusc_256'

# ── 기본 파라미터 ──────────────────────────────────────────────────────────────
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

class_names = [
    'empty', 'barrier', 'bicycle', 'bus', 'car',
    'construction_vehicle', 'motorcycle', 'pedestrian',
    'traffic_cone', 'trailer', 'truck', 'driveable_surface',
    'other_flat', 'sidewalk', 'terrain', 'manmade', 'vegetation',
]
num_class = len(class_names)

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
occ_size = [512, 512, 40]
lss_downsample = [4, 4, 4]

voxel_x = (point_cloud_range[3] - point_cloud_range[0]) / occ_size[0]
voxel_y = (point_cloud_range[4] - point_cloud_range[1]) / occ_size[1]
voxel_z = (point_cloud_range[5] - point_cloud_range[2]) / occ_size[2]
voxel_size = [voxel_x, voxel_y, voxel_z]

data_config = dict(
    cams=['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
          'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
    Ncams=6,
    input_size=(256, 704),
    src_size=(900, 1600),
    resize=(-0.06, 0.11),
    rot=(-5.4, 5.4),
    flip=True,
    crop_h=(0.0, 0.0),
    resize_test=0.00,
)

grid_config = dict(
    xbound=[point_cloud_range[0], point_cloud_range[3], voxel_x * lss_downsample[0]],
    ybound=[point_cloud_range[1], point_cloud_range[4], voxel_y * lss_downsample[1]],
    zbound=[point_cloud_range[2], point_cloud_range[5], voxel_z * lss_downsample[2]],
    dbound=[2.0, 58.0, 0.5],
)

numC_Trans = 128
voxel_channels = [128, 256, 512, 1024]
voxel_out_channels = 192
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)

mask2former_num_queries = 100
mask2former_feat_channel = voxel_out_channels
mask2former_output_channel = voxel_out_channels
mask2former_pos_channel = mask2former_feat_channel // 3
mask2former_num_heads = voxel_out_channels // 32

empty_idx = 0
visible_mask = False

# ── 모델 ───────────────────────────────────────────────────────────────────────
model = dict(
    type='SparseOcc',
    img_backbone=dict(
        pretrained='projects/SparseOcc_cvpr/pretrain/resnet50-0676ba61.pth',
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=0,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        with_cp=True,
        style='pytorch'),
    img_neck=dict(
        type='SECONDFPN',
        in_channels=[256, 512, 1024, 2048],
        upsample_strides=[0.25, 0.5, 1, 2],
        out_channels=[128, 128, 128, 128]),
    img_view_transformer=dict(
        type='ViewTransformerLiftSplatShootVoxel',
        loss_depth_weight=1.0,
        grid_config=grid_config,
        data_config=data_config,
        numC_Trans=numC_Trans,
        vp_megvii=False),
    img_bev_encoder_backbone=dict(
        type='SparseLatentDiffuser',
        output_shape=[512 // 4, 512 // 4, 40 // 4],
        in_channels=numC_Trans,
        num_layers=[3, 2, 2, 1]),
    img_bev_encoder_neck=dict(
        type='SparseFeaturePyramid',
        in_channels=voxel_channels,
        feat_channels=voxel_out_channels,
        out_channels=voxel_out_channels,
        norm_cfg=norm_cfg,
        up_kernel_size=(2, 2, 3),
        up_stride=(2, 2, 2),
        up_padding=(0, 0, 1),
    ),
    pts_bbox_head=dict(
        type='SparseMask2FormerOpenOccHead',
        feat_channels=mask2former_feat_channel,
        out_channels=mask2former_output_channel,
        num_queries=mask2former_num_queries,
        final_occ_size=occ_size,
        num_occupancy_classes=num_class,
        pooling_attn_mask=True,
        sample_weight_gamma=0.25,
        empty_idx=0,
        positional_encoding=dict(
            type='SinePositionalEncoding3D',
            num_feats=mask2former_pos_channel,
            normalize=True),
        transformer_decoder=dict(
            type='DetrTransformerDecoder',
            return_intermediate=True,
            num_layers=9,
            transformerlayers=dict(
                type='DetrTransformerDecoderLayer',
                attn_cfgs=dict(
                    type='MultiheadAttention',
                    embed_dims=mask2former_feat_channel,
                    num_heads=mask2former_num_heads,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=False),
                ffn_cfgs=dict(
                    embed_dims=mask2former_feat_channel,
                    num_fcs=2,
                    act_cfg=dict(type='ReLU', inplace=True),
                    ffn_drop=0.0,
                    dropout_layer=None,
                    add_identity=True),
                feedforward_channels=mask2former_feat_channel * 8,
                operation_order=('cross_attn', 'norm', 'self_attn', 'norm', 'ffn', 'norm')),
            init_cfg=None),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean',
            class_weight=[1.0] * num_class + [0.1]),
        loss_mask=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=5.0),
        loss_dice=dict(
            type='DiceLoss',
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=5.0),
        point_cloud_range=point_cloud_range,
    ),
    train_cfg=dict(
        pts=dict(
            num_points=12544 * 4,
            oversample_ratio=3.0,
            importance_sample_ratio=0.75,
            assigner=dict(
                type='MaskHungarianAssigner',
                cls_cost=dict(type='ClassificationCost', weight=2.0),
                mask_cost=dict(type='CrossEntropyLossCost', weight=5.0, use_sigmoid=True),
                dice_cost=dict(type='DiceCost', weight=5.0, pred_act=True, eps=1.0)),
            sampler=dict(type='MaskPseudoSampler'),
        )),
    test_cfg=dict(
        pts=dict(
            semantic_on=True,
            panoptic_on=False,
            instance_on=False)),
)

# ── 데이터셋 ───────────────────────────────────────────────────────────────────
dataset_type = 'CustomNuScenesOccLSSDataset'
data_root = 'data/nuscenes'
occ_path = './data/nuScenes-Occupancy'
depth_gt_path = './data/depth_gt'
train_ann_file = 'nuscenes_occ_infos_train_ori.pkl'
val_ann_file = 'nuscenes_occ_infos_val_ori.pkl'

bda_aug_conf = dict(
    rot_lim=(0, 0),
    scale_lim=(0.95, 1.05),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5,
    flip_dz_ratio=0.5,
)

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles_BEVDet', is_train=True,
         data_config=data_config, sequential=False, aligned=True,
         trans_only=False, depth_gt_path=depth_gt_path,
         mmlabnorm=True, load_depth=True, img_norm_cfg=img_norm_cfg),
    dict(type='LoadAnnotationsBEVDepth',
         bda_aug_conf=bda_aug_conf,
         classes=class_names,
         input_modality=input_modality,
         is_train=False),
    dict(type='LoadOccupancy', to_float32=True, use_semantic=True,
         occ_path=occ_path, grid_size=occ_size, use_vel=False,
         unoccupied=empty_idx, pc_range=point_cloud_range, cal_visible=visible_mask),
    dict(type='OccDefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['img_inputs', 'gt_occ'],
         meta_keys=['pc_range', 'occ_size']),
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles_BEVDet',
         data_config=data_config, depth_gt_path=depth_gt_path,
         sequential=False, aligned=True, trans_only=False,
         mmlabnorm=True, img_norm_cfg=img_norm_cfg),
    dict(type='LoadAnnotationsBEVDepth',
         bda_aug_conf=bda_aug_conf,
         classes=class_names,
         input_modality=input_modality,
         is_train=False),
    dict(type='LoadOccupancy', to_float32=True, use_semantic=True,
         occ_path=occ_path, grid_size=occ_size, use_vel=False,
         unoccupied=empty_idx, pc_range=point_cloud_range, cal_visible=visible_mask),
    dict(type='OccDefaultFormatBundle3D', class_names=class_names, with_label=False),
    dict(type='Collect3D', keys=['img_inputs', 'gt_occ'],
         meta_keys=['pc_range', 'occ_size', 'scene_token', 'lidar_token']),
]

# ── DataLoader (새 mmengine 형식) ──────────────────────────────────────────────
train_dataloader = dict(
    batch_size=1,
    num_workers=0,#4,
    persistent_workers=False,#True,
    sampler=dict(type='DistributedGroupSampler', seed=0),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=train_ann_file,
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        occ_size=occ_size,
        pc_range=point_cloud_range,
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=0,#4,
    persistent_workers=False,#True,
    drop_last=False,
    sampler=dict(type='DistributedSampler', shuffle=False, seed=0),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=val_ann_file,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        occ_size=occ_size,
        pc_range=point_cloud_range,
    ),
)

test_dataloader = val_dataloader

val_evaluator = dict(type='NuScenesOccMetric')
test_evaluator = val_evaluator

# ── Optimizer & Scheduler (새 mmengine 형식) ───────────────────────────────────
embed_multi = dict(lr_mult=1.0, decay_mult=0.0)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0001,
        weight_decay=0.01,
        eps=1e-8,
        betas=(0.9, 0.999),
    ),
    clip_grad=dict(max_norm=35, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'query_embed': embed_multi,
            'query_feat': embed_multi,
            'level_embed': embed_multi,
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
        },
        norm_decay_mult=0.0,
    ),
)

# 학습 설정
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=24, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# LR Scheduler (StepLR → mmengine MultiStepLR)
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=24,
        by_epoch=True,
        milestones=[20, 23],
        gamma=0.1,
    )
]

# 구버전 config 호환 (tools/train.py 등에서 사용하는 경우)
# 이 필드들은 사용하지 않지만 구버전 스크립트와의 호환을 위해 남겨둠
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    weight_decay=0.01,
    eps=1e-8,
    betas=(0.9, 0.999),
    paramwise_cfg=dict(
        custom_keys={
            'query_embed': embed_multi,
            'query_feat': embed_multi,
            'level_embed': embed_multi,
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
        },
        norm_decay_mult=0.0),
)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(policy='step', step=[20, 23])
runner = dict(type='EpochBasedRunner', max_epochs=24)
checkpoint_config = dict(max_keep_ckpts=1, interval=1)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'), dict(type='TensorboardLoggerHook')],
)

# sync_bn
sync_bn = 'torch'
