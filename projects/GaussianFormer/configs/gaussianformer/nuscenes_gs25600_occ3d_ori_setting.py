# Occ3D-nuScenes GT, GaussianFormer's own original training recipe (same
# optimizer/schedule/epochs as configs/gaussianformer/nuscenes_gs25600.py,
# which trains against SurroundOcc GT) -- the "ori_setting" axis just swaps
# the GT source/geometry, nothing about training itself.
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
    dict(type='PhotoMetricDistortionMultiViewImage'),
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

# batch_size=4: see configs/gaussianformer/nuscenes_gs25600.py's comment --
# same model architecture/geometry-independent memory footprint (measured
# 16.67GB allocated / 17.04GB reserved at batch_size=1), sized for a 95GB
# training GPU at ~65% peak utilization. Not verified at actual 95GB scale.
train_dataloader = dict(
    batch_size=4,
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

max_epochs = 20
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=4e-4, weight_decay=0.01),
    paramwise_cfg=dict(custom_keys={'img_backbone': dict(lr_mult=0.1)}),
    clip_grad=dict(max_norm=35, norm_type=2))
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6 / 4e-4, by_epoch=False, begin=0, end=500),
    dict(type='CosineAnnealingLR', by_epoch=True, T_max=max_epochs, begin=0, end=max_epochs, eta_min=4e-5),
]
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
