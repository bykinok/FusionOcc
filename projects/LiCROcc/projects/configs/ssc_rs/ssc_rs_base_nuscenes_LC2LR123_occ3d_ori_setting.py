# MMEngine 2.x Style Configuration - LC2LR123 (Lidar teacher, Radar student) with Occ3D GT
work_dir = './work_dirs/ssc_rs_base_nuscenes_LC2LR123_occ3d_ori_setting'

# Custom imports for mmdet3d 2.x (mmengine)
custom_imports = dict(
    imports=['projects.LiCROcc.projects.mmdet3d_plugin'],
    allow_failed_imports=False
)

default_scope = 'mmdet3d'

# Dataset configuration for occ3d
dataset_name = 'occ3d'

_sweeps_num_ = 10
_temporal_ = []

# occ3d: grid and range -> output (B, 18, 200, 200, 16)
point_cloud_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
occ_size = [200, 200, 16]

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

nbr_classes = 18
phase = 'trainval'
use_semantic = True

class_names = ['others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
               'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
               'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
               'vegetation', 'free']

ss_class_freq = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
sc_class_freq = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

grid_config = {
    'x': [-51.2, 51.2, 0.4],
    'y': [-51.2, 51.2, 0.4],
    'z': [-1, 5.4, 6.4],
    'depth': [1.0, 45.0, 0.5],
}
if dataset_name == 'occ3d':
    grid_config = {
        'x': [-40.0, 40.0, 0.4],
        'y': [-40.0, 40.0, 0.4],
        'z': [-1, 5.4, 6.4],
        'depth': [1.0, 45.0, 0.5],
    }
numC_Trans = 64

data_config = {
    'cams': [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    'Ncams': 6,
    'input_size': (256, 704),
    'src_size': (900, 1600),
    'resize': (-0.06, 0.11),
    'rot': (-0, 0),
    'flip': False,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

sync_bn = 'torch'

if dataset_name == 'occ3d':
    ss_class_freq = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    sc_class_freq = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    nbr_classes = 18

# Model: LC2LR123 (Lidar teacher -> Radar student) + Occ3D
model = dict(
    type='SSC_RS',
    use_semantic=use_semantic,
    dataset_name=dataset_name,
    Distill_1=True,
    Distill_2=True,
    Distill_3=True,
    Distill_3_mask=False,
    ratio_distill=10,
    # Distill teacher (image branch, no temporal for distill)
    img_backbone_distill=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        with_cp=True,
        style='pytorch',
        pretrained='torchvision://resnet50',
    ),
    img_neck_distill=dict(
        type='CustomFPN',
        in_channels=[1024, 2048],
        out_channels=256,
        num_outs=1,
        start_level=0,
        out_ids=[0],
        temporal_adapter=False),
    img_view_transformer_distill=dict(
        type='LSSViewTransformerBEVDepth',
        grid_config=grid_config,
        input_size=data_config['input_size'],
        in_channels=256,
        out_channels=numC_Trans,
        loss_depth_weight=1,
        depthnet_cfg=dict(use_dcn=False, aspp_mid_channels=96),
        downsample=16,
        temporal_adapter=False),
    img_bev_encoder_backbone_distill=dict(
        type='CustomResNet',
        numC_input=numC_Trans,
        num_channels=[numC_Trans * 2, numC_Trans * 4, numC_Trans * 8],
        temporal_adapter=False),
    # Main (temporal) image branch
    img_backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(2, 3),
        frozen_stages=4,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        with_cp=True,
        style='pytorch',
        pretrained='torchvision://resnet50',
    ),
    img_neck=dict(
        type='CustomFPN',
        in_channels=[1024, 2048],
        out_channels=256,
        num_outs=1,
        start_level=0,
        out_ids=[0],
        temporal_adapter=True),
    img_view_transformer=dict(
        type='LSSViewTransformerBEVDepth',
        grid_config=grid_config,
        input_size=data_config['input_size'],
        in_channels=256,
        out_channels=numC_Trans,
        loss_depth_weight=1,
        depthnet_cfg=dict(use_dcn=False, aspp_mid_channels=96),
        downsample=16,
        temporal_adapter=True),
    img_bev_encoder_backbone=dict(
        type='CustomResNet',
        numC_input=numC_Trans,
        num_channels=[numC_Trans * 2, numC_Trans * 4, numC_Trans * 8],
        temporal_adapter=True),
    # Lidar (teacher) - frozen, occ3d sizes
    pts_voxel_encoder=dict(
        type='PcPreprocessor',
        lims=lims,
        sizes=sizes,
        grid_meters=grid_meters,
        init_size=sizes[-1],
        frozen=True,
    ),
    pts_backbone=dict(
        type='SemanticBranch',
        sizes=sizes,
        nbr_class=nbr_classes - 1,
        init_size=sizes[-1],
        class_frequencies=ss_class_freq,
        phase='test',
        frozen=True,
        dataset_name=dataset_name,
    ),
    pts_middle_encoder=dict(
        type='CompletionBranch',
        init_size=sizes[-1],
        nbr_class=nbr_classes,
        phase='test',
        frozen=True,
        dataset_name=dataset_name,
    ),
    pts_bbox_head=dict(
        type='BEVUNet',
        n_class=nbr_classes * sizes[-1],
        n_height=sizes[-1],
        class_frequences=sc_class_freq,
        frozen=True,
        use_cam=[True, True, True],
        use_add=False,
        empty_idx=17 if dataset_name == 'occ3d' else 0,  # occ3d: class 17 is 'free', original: class 0 is 'empty'
        fusion_cfg=[
            dict(img_upsample_scale=1),
            dict(img_upsample_scale=1),
            dict(in_channels_sem=256, in_channels_com=128, img_upsample_scale=1),
        ] if dataset_name == 'occ3d' else None,
    ),
    # Radar (student) - trainable, occ3d sizes and fusion
    radar_voxel_encoder=dict(
        type='PcPreprocessor',
        lims=lims,
        sizes=sizes,
        grid_meters=grid_meters,
        init_size=sizes[-1],
        frozen=False,
        pc_dim=13,
    ),
    radar_backbone=dict(
        type='SemanticBranch',
        sizes=sizes,
        nbr_class=nbr_classes - 1,
        init_size=sizes[-1],
        class_frequencies=ss_class_freq,
        phase=phase,
        frozen=False,
        dataset_name=dataset_name,
    ),
    radar_middle_encoder=dict(
        type='CompletionBranch',
        init_size=sizes[-1],
        nbr_class=nbr_classes,
        phase=phase,
        frozen=False,
        dataset_name=dataset_name,
    ),
    radar_bbox_head=dict(
        type='BEVUNet',
        n_class=nbr_classes * sizes[-1],
        n_height=sizes[-1],
        class_frequences=sc_class_freq,
        use_Distill_2=True,
        use_add=False,
        use_cam=[False, True, True],
        empty_idx=17 if dataset_name == 'occ3d' else 0,  # occ3d: class 17 is 'free', original: class 0 is 'empty'
        fusion_cfg=[
            dict(img_upsample_scale=1),
            dict(img_upsample_scale=1),
            dict(in_channels_sem=256, in_channels_com=128, img_upsample_scale=1),
        ] if dataset_name == 'occ3d' else None,
    ),
    train_cfg=dict(
        pts=dict(sizes=sizes, grid_meters=grid_meters, lims=lims),
        sizes=sizes,
    ),
    test_cfg=dict(
        pts=dict(sizes=sizes, grid_meters=grid_meters, lims=lims),
        sizes=sizes,
    ),
)

from projects.LiCROcc.projects.mmdet3d_plugin.datasets.builder import collate

dataset_type = 'nuScenesDataset'
data_root = './data/nuscenes/'
occ_root = data_root

use_mask_camera = True
use_mask_camera_1_2 = False
use_mask_camera_1_4 = False
use_mask_camera_1_8 = False

train_batch_size = 1
val_batch_size = 1
test_batch_size = 1

_train_dataset = dict(
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
    use_semantic=use_semantic,
    dataset_name=dataset_name,
    pc_range=point_cloud_range,
    occ_size=occ_size,
    use_ego_frame=True,
    classes=class_names,
    use_mask_camera=use_mask_camera,
    use_mask_camera_1_2=use_mask_camera_1_2,
    use_mask_camera_1_4=use_mask_camera_1_4,
    use_mask_camera_1_8=use_mask_camera_1_8,
)

_val_test_dataset = dict(
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
    use_semantic=use_semantic,
    dataset_name=dataset_name,
    pc_range=point_cloud_range,
    occ_size=occ_size,
    use_ego_frame=True,
    classes=class_names,
    use_mask_camera=use_mask_camera,
    use_mask_camera_1_2=use_mask_camera_1_2,
    use_mask_camera_1_4=use_mask_camera_1_4,
    use_mask_camera_1_8=use_mask_camera_1_8,
)

train_dataloader = dict(
    batch_size=train_batch_size,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DistributedGroupSampler', samples_per_gpu=train_batch_size, seed=10),
    collate_fn=collate,
    dataset=_train_dataset,
)

val_dataloader = dict(
    batch_size=val_batch_size,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DistributedSampler', shuffle=False, seed=10),
    collate_fn=collate,
    dataset=_val_test_dataset,
)

test_dataloader = dict(
    batch_size=test_batch_size,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DistributedSampler', shuffle=False, seed=10),
    collate_fn=collate,
    dataset=_val_test_dataset,
)

val_evaluator = [
    dict(
        type='OccupancyMetricHybrid',
        dataset_name=dataset_name,
        num_classes=nbr_classes,
        use_lidar_mask=False,
        use_image_mask=True,
        ann_file=data_root + 'nuscenes_occ_infos_val.pkl',
        data_root=data_root,
        class_names=class_names,
        eval_metric='miou',
        sort_by_timestamp=False,
    )
]
test_evaluator = [
    dict(
        type='OccupancyMetricHybrid',
        dataset_name=dataset_name,
        num_classes=nbr_classes,
        use_lidar_mask=False,
        use_image_mask=True,
        ann_file=data_root + 'nuscenes_occ_infos_val.pkl',
        data_root=data_root,
        class_names=class_names,
        eval_metric='miou',
        sort_by_timestamp=False,
    )
]

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=2e-4, weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2),
)

param_scheduler = [
    dict(type='LinearLR', start_factor=1.0 / 3, by_epoch=False, begin=0, end=500),
    dict(type='CosineAnnealingLR', by_epoch=True, begin=0, end=24, eta_min=2e-7),
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=24, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, save_best='auto'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'

load_from = 'projects/LiCROcc/pre_ckpt/merged_model_distill_spconv1.pth'
resume = False

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer',
)

randomness = dict(seed=10, deterministic=False)
