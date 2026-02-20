# MMEngine 2.x Style Configuration
work_dir = './work_dirs/ssc_rs_base_nuscenes_cam_radar_occ3d_ori_setting'

# Custom imports for mmdet3d 2.x (mmengine)
custom_imports = dict(
    imports=['projects.LiCROcc.projects.mmdet3d_plugin'],
    allow_failed_imports=False
)

# Set default scope
default_scope = 'mmdet3d'

# Dataset configuration for occ3d
dataset_name = 'occ3d'  # 'occ3d' for occ3d GT format

_sweeps_num_ = 10
_temporal_ = []

# occ3d: CONet-style grid and range (output occupancy shape = B, 18, 200, 200, 16)
point_cloud_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]  # occ3d range
occ_size = [200, 200, 16]  # Occ3D grid [X, Y, Z] -> output (B, nbr_classes, 200, 200, 16)

# Original LiCROcc coordinate system (non-occ3d)
lims = [[-51.2, 51.2], [-51.2, 51.2], [-5, 3.0]]
sizes = [512, 512, 40]
if dataset_name == 'occ3d':
    lims = [[point_cloud_range[0], point_cloud_range[3]],   # x: [-40, 40]
            [point_cloud_range[1], point_cloud_range[4]],   # y: [-40, 40]
            [point_cloud_range[2], point_cloud_range[5]]]   # z: [-1, 5.4]
    sizes = occ_size  # [200, 200, 16]
    grid_meters = [0.4, 0.4, 0.4]  # 80m/200=0.4m, 6.4m/16=0.4m
else:
    grid_meters = [0.2, 0.2, 0.2]

# occ3d uses 18 classes (0=others, 1-16=semantic, 17=free)
nbr_classes = 18
phase = 'trainval'

use_semantic = True

# occ3d class names
class_names = ['others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle', 
               'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck', 
               'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade', 
               'vegetation', 'free']

# Class frequencies for occ3d (18 classes including 'free')
ss_class_freq = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
sc_class_freq = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]


grid_config = {
    'x': [-51.2, 51.2, 0.4],
    'y': [-51.2, 51.2, 0.4],
    'z': [-1, 5.4, 6.4],
    'depth': [1.0, 45.0, 0.5],
}
# occ3d: Match image BEV size to radar BEV by using same initial size and no upsampling
# Radar: 200x200 -> BEVUNet down3 -> [100, 50, 25]
# Image: 200x200 -> CustomResNet (stride 2,2,2) -> [100, 50, 25] -> adapter_img (scale=1) -> [100, 50, 25]
if dataset_name == 'occ3d':
    grid_config = {
        'x': [-40.0, 40.0, 0.4],   # 80/0.4 = 200 (match radar initial size)
        'y': [-40.0, 40.0, 0.4],
        'z': [-1, 5.4, 6.4],
        'depth': [1.0, 45.0, 0.5],
    }
# Keep same image branch as original cam_radar (64, 256) so pretrained load behavior is unchanged (strict=False)
numC_Trans = 64

data_config = {
    'cams': [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    'Ncams':
    6,
    'input_size': (256, 704),
    'src_size': (900, 1600),

    # BEV aug만 사용, 이미지 augmentation 비활성
    'resize': (0.0, 0.0),
    'rot': (0.0, 0.0),
    'flip': False,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}
# 👋

sync_bn = 'torch'

# For occ3d: 18 classes (update class frequencies)
if dataset_name == 'occ3d':
    ss_class_freq = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # 17 semantic classes (exclude free)
    sc_class_freq = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # 18 classes (include free)
    nbr_classes = 18

model = dict(
   type='SSC_RS',
    use_semantic=use_semantic,
    dataset_name=dataset_name,
    img_backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(2, 3),
        # frozen_stages=4,
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        with_cp=True,
        style='pytorch',
        pretrained='torchvision://resnet50',
        # frozen = True
    ),
    img_neck=dict(
        type='CustomFPN',
        in_channels=[1024, 2048],
        out_channels=256,
        num_outs=1,
        start_level=0,
        out_ids=[0],
        temporal_adapter=False),
    img_view_transformer=dict(
        type='LSSViewTransformerBEVDepth',
        grid_config=grid_config,
        input_size=data_config['input_size'],
        in_channels=256,
        out_channels=numC_Trans,
        loss_depth_weight=0.0,  # wo_DepthSV: no depth supervision
        depthnet_cfg=dict(use_dcn=False, aspp_mid_channels=96),
        downsample=16,
        temporal_adapter=False),
    img_bev_encoder_backbone=dict(
        type='CustomResNet',
        numC_input=numC_Trans,
        num_channels=[numC_Trans * 2, numC_Trans * 4, numC_Trans * 8],
        temporal_adapter=False),
    radar_voxel_encoder=dict(
        type='PcPreprocessor',
        lims=lims,
        sizes=sizes, 
        grid_meters=grid_meters, 
        init_size=sizes[-1],
        frozen=False,
        pc_dim=13
    ),
    radar_backbone=dict(
        type='SemanticBranch',
        sizes=sizes,
        nbr_class=nbr_classes-1,  # 17 semantic classes (excluding 'free')
        init_size=sizes[-1], 
        class_frequencies=ss_class_freq, 
        phase=phase,
        frozen=False,
        dataset_name=dataset_name,
        ),
    radar_middle_encoder=dict(
        type='CompletionBranch',
        init_size=sizes[-1],
        nbr_class=nbr_classes,  # 18 classes (including 'free')
        phase=phase,
        frozen=False,
        dataset_name=dataset_name,
        ),
    # occ3d: n_class=18*16=288, n_height=16, spatial=sizes[:2]=200x200 -> output (B, 18, 200, 200, 16)
    radar_bbox_head=dict(
        type='BEVUNet',
        n_class=nbr_classes*sizes[-1],
        n_height=sizes[-1], 
        use_add=False,
        use_cam=[False,True,True],
        class_frequences=sc_class_freq,
        empty_idx=17 if dataset_name == 'occ3d' else 0,  # occ3d: class 17 is 'free', original: class 0 is 'empty'
        # occ3d: same image ch as original (128,256,512); only sem/com adapter differ (scale-3: 256, 128)
        fusion_cfg=[
            dict(img_upsample_scale=1),
            dict(img_upsample_scale=1),
            dict(in_channels_sem=256, in_channels_com=128, img_upsample_scale=1),
        ] if dataset_name == 'occ3d' else None,
        ),
    train_cfg=dict(pts=dict(
        sizes=sizes,
        grid_meters=grid_meters,
        lims=lims),
        sizes=sizes,
        ),
    test_cfg=dict(pts=dict(
        sizes=sizes,
        grid_meters=grid_meters,
        lims=lims),
        sizes=sizes)
    
)


# Import custom collate function directly (no functools.partial needed)
from projects.LiCROcc.projects.mmdet3d_plugin.datasets.builder import collate

# Dataset configuration for occ3d
dataset_type = 'nuScenesDataset'
data_root = './data/nuscenes/'
# occ3d: GT path is under data_root (gts/...); dataset uses data_root as occ_root
occ_root = data_root

# occ3d specific settings
use_mask_camera = True
use_mask_camera_1_2 = False
use_mask_camera_1_4 = False
use_mask_camera_1_8 = False

train_batch_size = 1
val_batch_size = 1
test_batch_size = 1

# MMEngine 2.x: Dataloader configuration
train_dataloader = dict(
    batch_size=train_batch_size,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DistributedGroupSampler', samples_per_gpu=train_batch_size, seed=0),
    collate_fn=collate,  # Add custom collate function
    dataset=dict(
        type=dataset_type,
        split="train",
        test_mode=False,
        data_root=data_root,
        occ_root=occ_root,
        lims=lims,
        sizes=sizes,
        temporal=_temporal_,
        sweeps_num=_sweeps_num_,
        augmentation=True,
        shuffle_index=True,
        data_config=data_config,
        grid_config=grid_config,
        # occ3d specific parameters
        use_semantic=use_semantic,
        dataset_name=dataset_name,
        pc_range=point_cloud_range,
        occ_size=occ_size,
        use_ego_frame=True,  # Ego-frame for Occ3D GT
        classes=class_names,
        use_mask_camera=use_mask_camera,
        use_mask_camera_1_2=use_mask_camera_1_2,
        use_mask_camera_1_4=use_mask_camera_1_4,
        use_mask_camera_1_8=use_mask_camera_1_8,
    )
)

val_dataloader = dict(
    batch_size=val_batch_size,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DistributedSampler', shuffle=False, seed=0),
    collate_fn=collate,  # Add custom collate function
    dataset=dict(
        type=dataset_type,
        split="val",
        test_mode=True,
        data_root=data_root,
        occ_root=occ_root,
        lims=lims,
        sizes=sizes,
        temporal=_temporal_,
        sweeps_num=_sweeps_num_,
        augmentation=False,
        shuffle_index=False,
        data_config=data_config,
        grid_config=grid_config,
        # occ3d specific parameters
        use_semantic=use_semantic,
        dataset_name=dataset_name,
        pc_range=point_cloud_range,
        occ_size=occ_size,
        use_ego_frame=True,  # Ego-frame for Occ3D GT
        classes=class_names,
        use_mask_camera=use_mask_camera,
        use_mask_camera_1_2=use_mask_camera_1_2,
        use_mask_camera_1_4=use_mask_camera_1_4,
        use_mask_camera_1_8=use_mask_camera_1_8,
    )
)

test_dataloader = dict(
    batch_size=test_batch_size,
    num_workers=0,#4,
    persistent_workers=False,#True,
    drop_last=False,
    sampler=dict(type='DistributedSampler', shuffle=False, seed=0),
    collate_fn=collate,  # Add custom collate function
    dataset=dict(
        type=dataset_type,
        split="val",
        test_mode=True,
        data_root=data_root,
        occ_root=occ_root,
        lims=lims,
        sizes=sizes,
        temporal=_temporal_,
        sweeps_num=_sweeps_num_,
        augmentation=False,
        shuffle_index=False,
        data_config=data_config,
        grid_config=grid_config,
        # occ3d specific parameters
        use_semantic=use_semantic,
        dataset_name=dataset_name,
        pc_range=point_cloud_range,
        occ_size=occ_size,
        use_ego_frame=True,  # Ego-frame for Occ3D GT
        classes=class_names,
        use_mask_camera=use_mask_camera,
        use_mask_camera_1_2=use_mask_camera_1_2,
        use_mask_camera_1_4=use_mask_camera_1_4,
        use_mask_camera_1_8=use_mask_camera_1_8,
    )
)

# MMEngine 2.x: Evaluator configuration
# Use OccupancyMetricHybrid (STCOcc metric) for occ3d evaluation
val_evaluator = [
    dict(
        type='OccupancyMetricHybrid',
        dataset_name=dataset_name,
        num_classes=nbr_classes,  # 18 classes for occ3d
        use_lidar_mask=False,
        use_image_mask=True,  # Use camera mask for occ3d
        ann_file=data_root + 'nuscenes_occ_infos_val.pkl',
        data_root=data_root,
        class_names=class_names,
        eval_metric='miou',
        sort_by_timestamp=False,  # Match dataset order
    )
]
test_evaluator = [
    dict(
        type='OccupancyMetricHybrid',
        dataset_name=dataset_name,
        num_classes=nbr_classes,  # 18 classes for occ3d
        use_lidar_mask=False,
        use_image_mask=True,  # Use camera mask for occ3d
        ann_file=data_root + 'nuscenes_occ_infos_val.pkl',
        data_root=data_root,
        class_names=class_names,
        eval_metric='miou',
        sort_by_timestamp=False,  # Match dataset order
    )
]

# SurroundOcc와 동일한 optimizer 세팅 + gradient accumulation 8
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=2e-4,
        weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    clip_grad=dict(max_norm=35, norm_type=2),
    accumulative_counts=8)

# LR schedule end = 실제 총 iteration 수 (num_gpus=2 기준)
train_samples = 28130
num_gpus = 2
num_iters_per_epoch = train_samples // (num_gpus * train_batch_size)

# LR: 0-500 iter 선형 1e-5->2e-4, 이후 epoch 24까지 cosine 2e-4->1e-6
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.05,  # 1e-5 / 2e-4
        end_factor=1.0,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        type='CosineAnnealingLR',
        begin=500,
        end=24 * num_iters_per_epoch,
        by_epoch=False,
        eta_min=1e-6)
]

# MMEngine 2.x: Training configuration
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=24,
    val_interval=999999#1
)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# MMEngine 2.x: Default hooks
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, save_best='auto'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

# Environment configuration
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

# Logging
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'

# Load checkpoint
load_from = './projects/LiCROcc/pre_ckpt/bevdet-r50-4d-depth-cbgs.pth' # from bevdet: https://github.com/HuangJunJie2017/BEVDet?tab=readme-ov-file
# load_from = './projects/LiCROcc/ckpt/ori_fusion_cam_radar_epoch_24_spconv1.pth'
resume = False

# Visualization
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer'
)

# SyncBN 비활성: sync_bn은 데이터셋/모델에서 참조할 수 있으나, 모델은 norm_cfg=BN 사용. AMP 미사용(OptimWrapper).
sync_bn = 'torch'

# Reproducibility: seed=0. EMA 미사용. SyncBN/AMP 검토 완료.
randomness = dict(seed=0, deterministic=False)
