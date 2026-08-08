# RayIoU evaluation instead of per-voxel mIoU: rays are cast from the LiDAR
# origin and compared against the first occupied voxel hit along each ray,
# so camera-visibility masking is irrelevant at eval time (test_pipeline
# sets use_mask_camera=False, matching projects/SurroundOcc's own
# *_rayiou configs) -- training is unaffected.
# See gaussianformer/evaluation/rayiou_metric.py for important caveats
# about the ray-origin/grid coordinate-frame assumption used here.
_base_ = ['./nuscenes_gs25600_occ3d_ori_setting.py']

occ3d_data_root = 'data/nuscenes/'

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadOccupancyOcc3D', data_root=occ3d_data_root, use_mask_camera=False),
    dict(type='ResizeCropFlipImage'),
    dict(type='NormalizeMultiviewImage',
         mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
    dict(type='DefaultFormatBundle'),
    dict(type='NuScenesAdaptor', use_ego=True, num_cams=6),
]
return_keys = [
    'img', 'projection_mat', 'image_wh', 'occ_label', 'occ_xyz', 'occ_cam_mask',
    'ori_img', 'cam_positions', 'focal_positions', 'lidar_origin',
]
val_dataloader = dict(dataset=dict(pipeline=test_pipeline, return_keys=return_keys))
test_dataloader = val_dataloader

val_evaluator = dict(type='GaussianFormerRayIoUMetric')
test_evaluator = val_evaluator
