# Occ3D-nuScenes GT, "unified" cross-project training recipe (same axis as
# projects/SurroundOcc/configs/surroundocc/surroundocc_occ3d_unified.py /
# projects/CONet's *_occ3d_unified.py): disable PhotoMetricDistortion,
# gradient accumulation, a shared LinearLR->CosineAnnealingLR schedule,
# 24 epochs, no periodic validation (val_interval=9999). GT/geometry
# unchanged from _ori_setting -- only the training recipe differs.
_base_ = [
    './_base_/occ3d_model.py',
    './_base_/occ3d_runtime.py',
]

data_root = 'data/nuscenes/'
occ3d_data_root = 'data/nuscenes/'
anno_root = 'projects/GaussianFormer/data/nuscenes_cam/'
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
    dict(type='LoadOccupancyOcc3D', data_root=occ3d_data_root, use_mask_camera=True),
    dict(type='ResizeCropFlipImage'),
    # "unified" recipe drops PhotoMetricDistortion (matches SurroundOcc/CONet's
    # own _unified configs, which rely on BEV-only augmentation instead).
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='NuScenesAdaptor', use_ego=True, num_cams=6),
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadOccupancyOcc3D', data_root=occ3d_data_root, use_mask_camera=True),
    dict(type='ResizeCropFlipImage'),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='NuScenesAdaptor', use_ego=True, num_cams=6),
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

return_keys = [
    'img', 'projection_mat', 'image_wh', 'occ_label', 'occ_xyz', 'occ_cam_mask',
    'ori_img', 'cam_positions', 'focal_positions', 'mask_lidar', 'index',
]
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
        return_keys=return_keys,
        phase='val'))
test_dataloader = val_dataloader

# Same canonical mIoU evaluator (mmdet3d.datasets.occ_metrics.Metric_mIoU)
# projects/STCOcc/projects/FusionOcc use directly and
# projects/SurroundOcc/projects/CONet use transitively via
# OccupancyMetricHybrid -- see gaussianformer/evaluation/occ3d_metric.py.
# `ann_file` points at the flat companion pkl (built by
# projects/GaussianFormer/tools/build_flat_ann_file.py) that
# tools/compute_metrics_from_file.py's config-introspection needs to find
# GT-by-index; unused by this evaluator directly.
val_evaluator = dict(
    type='GaussianFormerOcc3DMetric',
    use_image_mask=True,
    ann_file=anno_root + 'nuscenes_infos_val_sweeps_occ_flat.pkl',
    data_root=data_root)
test_evaluator = val_evaluator

max_epochs = 24

# LR schedule end = actual total iteration count. Matches
# projects/SurroundOcc/configs/surroundocc/surroundocc_occ3d_unified.py's own
# `train_samples`/`num_gpus`/`num_iters_per_epoch` pattern exactly --
# train_samples=28130 is GaussianFormer's own train split length too
# (confirmed: len(train_dataloader.dataset) == 28130), not a coincidence
# copied from SurroundOcc. num_gpus mirrors the same 2-GPU assumption every
# other project's unified config hardcodes here; override via
# `--cfg-options num_gpus=N` if training on a different GPU count.
train_samples = 28130
num_gpus = 2
samples_per_gpu = 1
num_iters_per_epoch = train_samples // (num_gpus * samples_per_gpu)

# lr=2e-4 (not GaussianFormer's own 4e-4, kept in _ori_setting.py) is the
# cross-project "unified" invariant -- every other project's unified config
# uses 2e-4 regardless of its own paper LR (e.g. CONet 3e-4->2e-4, FusionOcc
# 5e-5->2e-4).
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=2e-4, weight_decay=0.01),
    paramwise_cfg=dict(custom_keys={'img_backbone': dict(lr_mult=0.1)}),
    accumulative_counts=8,
    clip_grad=dict(max_norm=35, norm_type=2))
param_scheduler = [
    # start_factor=0.05 -> warmup starts at 1e-5 (0.05 * 2e-4), matching the
    # "1e-5 -> 2e-4" warmup every other project's unified config comments.
    dict(type='LinearLR', start_factor=0.05, end_factor=1.0, by_epoch=False, begin=0, end=500),
    dict(type='CosineAnnealingLR', begin=500, end=24 * num_iters_per_epoch,
         by_epoch=False, eta_min=1e-6),
]
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=9999)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
