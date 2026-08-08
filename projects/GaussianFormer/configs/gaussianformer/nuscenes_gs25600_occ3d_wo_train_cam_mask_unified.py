# "wo_train_cam_mask" on top of the "unified" training recipe.
_base_ = ['./nuscenes_gs25600_occ3d_unified.py']

occ3d_data_root = 'data/nuscenes/'

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadOccupancyOcc3D', data_root=occ3d_data_root, use_mask_camera=False),
    dict(type='ResizeCropFlipImage'),
    dict(type='NormalizeMultiviewImage',
         mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
    dict(type='DefaultFormatBundle'),
    dict(type='NuScenesAdaptor', use_ego=True, num_cams=6),
]
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
