# "condition_C_full": ALL occupied voxels are always supervised at train
# time regardless of camera visibility; only invisible *free* voxels are
# excluded. Eval (test_pipeline) unchanged. Same axis as
# projects/SurroundOcc's *_unified_condition_C_full.py.
_base_ = ['./nuscenes_gs25600_occ3d_unified.py']

occ3d_data_root = 'data/nuscenes/'

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadOccupancyOcc3D', data_root=occ3d_data_root, mask_mode='condition_C_full'),
    dict(type='ResizeCropFlipImage'),
    dict(type='NormalizeMultiviewImage',
         mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
    dict(type='DefaultFormatBundle'),
    dict(type='NuScenesAdaptor', use_ego=True, num_cams=6),
]
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
