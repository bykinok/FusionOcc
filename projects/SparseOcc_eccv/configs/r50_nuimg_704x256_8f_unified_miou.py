"""SparseOcc_unified r50 NuScenes 704x256 8frames 설정 (mmengine 형식).

원본: projects/SparseOcc_eccv/configs/r50_nuimg_704x256_8f_ori_miou.py
변경 사항 (BEVFormer unified config 기준으로 반영):
  1. [pipeline] BEVAug 추가 (BEV X/Y flip 증강, flip_dx_ratio=0.5, flip_dy_ratio=0.5)
     - voxel_semantics, voxel_instances를 flip
     - ego2lidar를 BDA 행렬로 보정하여 카메라 투영 일관성 유지
  2. [optim_wrapper] accumulative_counts=8 추가
     → effective batch size = batch_size × num_gpus × accumulative_counts
  3. [param_scheduler] iteration 기반 cosine annealing으로 변경
     - warmup: start_factor=0.05 (1e-5/2e-4), by_epoch=False, end=500 iter
     - cosine: by_epoch=False, begin=500, end=total_epochs × num_iters_per_epoch, eta_min=1e-6
  4. [변수 추가] train_samples, num_gpus, num_iters_per_epoch 명시
  5. [미적용] train collect에서 mask_camera 제거 — 이미 미포함
  6. [미적용] randomness — 이미 존재
"""

# 플러그인 로드 (새 mmengine custom_imports 방식)
custom_imports = dict(
    imports=['projects.SparseOcc_eccv.sparseocc_eccv'],
    allow_failed_imports=False)

default_scope = 'mmdet3d'

# ── 기본 파라미터 (원본과 동일) ────────────────────────────────────────────────
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
_num_frames_ = 8
_num_queries_ = 100
_topk_training_ = [4000, 16000, 64000]
_topk_testing_ = [2000, 8000, 32000]
_topk_training_ = _topk_testing_

# ── BEV 증강 설정 (unified 변경 사항) ─────────────────────────────────────────
# X/Y 방향 flip: voxel GT와 ego2lidar 동시 보정 → 카메라 투영 일관성 유지
bda_aug_conf = dict(
    flip_dx_ratio=0.5,  # X축(전방) flip 확률
    flip_dy_ratio=0.5,  # Y축(측방) flip 확률
)

# ── LR 스케줄 계산용 변수 (unified 변경 사항) ──────────────────────────────────
# iteration 기반 cosine annealing의 end 값 계산에 사용
# --cfg-options num_gpus=N 으로 실행 시 override 가능
train_samples = 28130       # nuScenes train split 샘플 수
num_gpus = 8                # 기본 GPU 수 (batch_size=8 → 실질적으로 1 GPU당 1샘플)
_total_epochs_ = 24
_num_iters_per_epoch_ = train_samples // (num_gpus * 1)  # batch_size=1 per GPU
# BEVFormer 기준(num_gpus=2, warmup=500) 대비 epoch 비중이 동일하도록 비례 계산
# 500 × (2 / num_gpus) = 500 × (2/8) = 125 iter
_warmup_iters_ = round(500 * 2 / num_gpus)

# ── 모델 (원본과 동일) ────────────────────────────────────────────────────────
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

# ── 데이터 파이프라인 (원본과 동일) ──────────────────────────────────────────
ida_aug_conf = {
    'resize_lim': (0.38, 0.55),
    'final_dim': (256, 704),
    'bot_pct_lim': (0.0, 0.0),
    'rot_lim': (0.0, 0.0),
    'H': 900, 'W': 1600,
    'rand_flip': False,
}

# 학습 시 camera mask 미사용 (use_mask_camera=False와 일치)
# BEVAug는 LoadOccGTFromFile 이후 (voxel GT 로드 완료 후) 적용
train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=False, color_type='color'),
    dict(type='LoadMultiViewImageFromMultiSweeps', sweeps_num=_num_frames_ - 1),
    dict(type='LoadOccGTFromFile', num_classes=len(occ_class_names)),
    dict(type='BEVAug', bda_aug_conf=bda_aug_conf, is_train=True),  # unified 변경: BEV flip 증강
    dict(type='RandomTransformImage', ida_aug_conf=ida_aug_conf, training=True),
    dict(type='DefaultFormatBundle3D', class_names=det_class_names),
    dict(type='Collect3D',
         keys=['img', 'voxel_semantics', 'voxel_instances', 'instance_class_ids'],
         meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape',
                    'lidar2img', 'img_timestamp', 'ego2lidar'))
]

# 평가 시 BEVAug는 항등 변환 (is_train=False → flip 없음)
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=False, color_type='color'),
    dict(type='LoadMultiViewImageFromMultiSweeps', sweeps_num=_num_frames_ - 1, test_mode=True),
    dict(type='LoadOccGTFromFile', num_classes=len(occ_class_names)),
    dict(type='BEVAug', bda_aug_conf=bda_aug_conf, is_train=False),  # 평가 시 flip 없음
    dict(type='RandomTransformImage', ida_aug_conf=ida_aug_conf, training=False),
    dict(type='DefaultFormatBundle3D', class_names=det_class_names),
    dict(type='Collect3D',
         keys=['img', 'voxel_semantics', 'voxel_instances', 'instance_class_ids'],
         meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape',
                    'lidar2img', 'img_timestamp', 'ego2lidar', 'index'))
]

# ── DataLoader (새 mmengine 형식) ────────────────────────────────────────────
train_dataloader = dict(
    batch_size=1,
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

# ── Optimizer (unified 변경 사항: accumulative_counts=8 추가) ─────────────────
# effective batch size = batch_size(1) × num_gpus(8) × accumulative_counts(8) = 64
# → 원본 ori_miou의 batch_size=8 × num_gpus=8 = 64와 동일한 effective batch 유지
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
    accumulative_counts=8,  # unified 변경: gradient accumulation
)

# 학습 설정
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=_total_epochs_, val_interval=_total_epochs_)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# ── LR Scheduler (unified 변경: iteration 기반 cosine annealing) ─────────────
# ori_miou: epoch 기반 cosine (begin=0, end=48 epoch)
# unified:  iteration 기반 cosine (begin=500 iter, end=total_epochs × iters_per_epoch)
#   - warmup: 0→500 iter, lr: 1e-5 → 2e-4 (start_factor=0.05)
#   - cosine: 500→end iter, lr: 2e-4 → 1e-6 (eta_min=1e-6)
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.05,   # 1e-5 / 2e-4 = 0.05 (unified 기준)
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
        eta_min=1e-6,        # unified 기준 (ori: eta_min_ratio=1e-3 × 2e-4 = 2e-7)
    ),
]

# ── 체크포인트 & 로그 ──────────────────────────────────────────────────────────
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1),
    logger=dict(type='LoggerHook', interval=50),
)

# ── 사전 학습 가중치 ────────────────────────────────────────────────────────────
load_from = 'projects/SparseOcc_eccv/pretrain/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth'

# checkpoint 로드 시 'backbone' → 'img_backbone' 키 리매핑 (원본 revise_keys와 동일)
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
work_dir = 'work_dirs/sparseocc_eccv_r50_256x704_8f_unified_miou'

randomness = dict(seed=0, deterministic=False)
