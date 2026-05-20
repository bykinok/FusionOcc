"""SparseOcc_unified r50 NuScenes 704x256 1frame 설정 (mmengine 형식) — temporal ablation.

원본: projects/SparseOcc_eccv/configs/r50_nuimg_704x256_8f_wo_cam_mask_unified_miou.py
변경 사항 (8f → 1f, temporal ablation):
  1. [_num_frames_] 8 → 1
     → SparseVoxelDecoder / MaskFormerOccDecoder의 multi-frame sampling 비활성화
     → AdaptiveMixing의 in_points = num_points * 1 (8f 대비 축소)
  2. [pipeline/LoadMultiViewImageFromMultiSweeps] sweeps_num=_num_frames_-1 → 0
     (자동 적용 — 별도 수정 불필요)
  3. [work_dir] sparseocc_eccv_r50_256x704_1f_wo_cam_mask_unified_miou

Temporal ablation policy:
  - 현재 프레임(t=0)만 사용 → multi-frame temporal 정보 완전 제거
  - 모델 가중치 shape이 변경되므로 8f checkpoint와 호환 불가 (재학습 필요)
"""

# 플러그인 로드 (새 mmengine custom_imports 방식)
custom_imports = dict(
    imports=['projects.SparseOcc_eccv.sparseocc_eccv'],
    allow_failed_imports=False)

default_scope = 'mmdet3d'

# ── 기본 파라미터 ─────────────────────────────────────────────────────────────
dataset_type = 'NuSceneOcc'
dataset_root = 'data/nuscenes/'
occ_gt_root = 'data/nuscenes/gts'

point_cloud_range = [-40, -40, -1.0, 40, 40, 5.4]
occ_size = [200, 200, 16]

img_norm_cfg = dict(
    mean=[123.675, 116.280, 103.530],
    std=[58.395, 57.120, 57.375],
    to_rgb=True
)

det_class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

occ_class_names = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk',
    'terrain', 'manmade', 'vegetation', 'free'
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False
)

_dim_ = 256
_num_points_ = 4
_num_groups_ = 4
_num_layers_ = 4
_num_frames_ = 1  # [변경] 8 → 1 (temporal ablation: 현재 프레임만 사용)
_num_queries_ = 100
_topk_training_ = [4000, 16000, 64000]
_topk_testing_ = [2000, 8000, 32000]
_topk_training_ = _topk_testing_

# ── BEV 증강 설정 ─────────────────────────────────────────────────────────────
bda_aug_conf = dict(
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5,
)

# ── LR 스케줄 계산용 변수 ─────────────────────────────────────────────────────
train_samples = 28130
num_gpus = 2                # 기본 GPU 수
batch_size = 8              # GPU당 배치 크기
_total_epochs_ = 24
_num_iters_per_epoch_ = train_samples // (num_gpus * batch_size)
_warmup_iters_ = max(1, round(500 * 2 / (num_gpus * batch_size)))

# ── 모델 ──────────────────────────────────────────────────────────────────────
model = dict(
    type='SparseOcc',
    data_aug=dict(
        img_color_aug=True,
        img_norm_cfg=img_norm_cfg,
        img_pad_cfg=dict(size_divisor=32)),
    use_grid_mask=False,
    use_mask_camera=False,
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        with_cp=True),
    img_neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=_dim_,
        num_outs=4),
    pts_bbox_head=dict(
        type='SparseOccHead',
        class_names=occ_class_names,
        embed_dims=_dim_,
        occ_size=occ_size,
        pc_range=point_cloud_range,
        transformer=dict(
            type='SparseOccTransformer',
            embed_dims=_dim_,
            num_layers=_num_layers_,
            num_frames=_num_frames_,
            num_points=_num_points_,
            num_groups=_num_groups_,
            num_queries=_num_queries_,
            num_levels=4,
            num_classes=len(occ_class_names),
            pc_range=point_cloud_range,
            occ_size=occ_size,
            topk_training=_topk_training_,
            topk_testing=_topk_testing_),
        loss_cfgs=dict(
            loss_mask2former=dict(
                type='Mask2FormerLoss',
                num_classes=len(occ_class_names),
                no_class_weight=0.1,
                loss_cls_weight=2.0,
                loss_mask_weight=5.0,
                loss_dice_weight=5.0,
            ),
        ),
    ),
)

# ── 데이터 파이프라인 ─────────────────────────────────────────────────────────
ida_aug_conf = {
    'resize_lim': (0.38, 0.55),
    'final_dim': (256, 704),
    'bot_pct_lim': (0.0, 0.0),
    'rot_lim': (0.0, 0.0),
    'H': 900, 'W': 1600,
    'rand_flip': False,
}

# sweeps_num=_num_frames_-1=0 → 현재 프레임만 로드 (자동 적용)
train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=False, color_type='color'),
    dict(type='LoadMultiViewImageFromMultiSweeps', sweeps_num=_num_frames_ - 1),
    dict(type='LoadOccGTFromFile', num_classes=len(occ_class_names)),
    dict(type='BEVAug', bda_aug_conf=bda_aug_conf, is_train=True),
    dict(type='RandomTransformImage', ida_aug_conf=ida_aug_conf, training=True),
    dict(type='DefaultFormatBundle3D', class_names=det_class_names),
    dict(type='Collect3D',
         keys=['img', 'voxel_semantics', 'voxel_instances', 'instance_class_ids'],
         meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape',
                    'lidar2img', 'img_timestamp', 'ego2lidar'))
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=False, color_type='color'),
    dict(type='LoadMultiViewImageFromMultiSweeps', sweeps_num=_num_frames_ - 1, test_mode=True),
    dict(type='LoadOccGTFromFile', num_classes=len(occ_class_names)),
    dict(type='BEVAug', bda_aug_conf=bda_aug_conf, is_train=False),
    dict(type='RandomTransformImage', ida_aug_conf=ida_aug_conf, training=False),
    dict(type='DefaultFormatBundle3D', class_names=det_class_names),
    dict(type='Collect3D',
         keys=['img', 'voxel_semantics', 'voxel_instances', 'instance_class_ids'],
         meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape',
                    'lidar2img', 'img_timestamp', 'ego2lidar', 'index'))
]

# ── DataLoader ────────────────────────────────────────────────────────────────
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DistributedGroupSampler', seed=0),
    dataset=dict(
        type=dataset_type,
        data_root=dataset_root,
        occ_gt_root=occ_gt_root,
        ann_file=dataset_root + 'nuscenes_infos_train_sweep.pkl',
        pipeline=train_pipeline,
        classes=det_class_names,
        modality=input_modality,
        test_mode=False,
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DistributedSampler', shuffle=False, seed=0),
    dataset=dict(
        type=dataset_type,
        data_root=dataset_root,
        occ_gt_root=occ_gt_root,
        ann_file=dataset_root + 'nuscenes_infos_val_sweep.pkl',
        pipeline=test_pipeline,
        classes=det_class_names,
        modality=input_modality,
        test_mode=True,
    ),
)

test_dataloader = val_dataloader

val_evaluator = dict(
    type='OccupancyMetricHybrid',
    occ_gt_root=occ_gt_root,
    ann_file=dataset_root + 'nuscenes_infos_val_sweep.pkl',
    data_root=dataset_root,
    num_classes=18,
    use_lidar_mask=False,
    use_image_mask=True,
    dataset_name='occ3d',
    eval_metric='miou',
    sort_by_timestamp=True,
)
test_evaluator = val_evaluator

# ── Optimizer ─────────────────────────────────────────────────────────────────
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=2e-4,
        weight_decay=0.01,
    ),
    clip_grad=dict(max_norm=35, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
            'sampling_offset': dict(lr_mult=0.1),
        }),
    accumulative_counts=4,   # effective batch = batch_size(8) × num_gpus(2) × 4 = 64
)

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=_total_epochs_, val_interval=_total_epochs_)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# ── LR Scheduler (iteration 기반 cosine annealing) ───────────────────────────
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.05,
        end_factor=1.0,
        by_epoch=False,
        begin=0,
        end=_warmup_iters_,
    ),
    dict(
        type='CosineAnnealingLR',
        begin=_warmup_iters_,
        end=_total_epochs_ * _num_iters_per_epoch_,
        by_epoch=False,
        eta_min=1e-6,
    ),
]

# ── 체크포인트 & 로그 ──────────────────────────────────────────────────────────
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1),
    logger=dict(type='LoggerHook', interval=50),
)

# ── 사전 학습 가중치 ────────────────────────────────────────────────────────────
load_from = 'projects/SparseOcc_eccv/pretrain/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth'

custom_hooks = [
    dict(type='ReviseCheckpointKeysHook',
         revise_keys=[('backbone', 'img_backbone')]),
]

# ── 런타임 설정 ────────────────────────────────────────────────────────────────
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
log_level = 'INFO'
work_dir = 'work_dirs/sparseocc_eccv_r50_256x704_1f_wo_cam_mask_unified_miou'

randomness = dict(seed=0, deterministic=False)
