# MMEngine 2.x Style Configuration
work_dir = './work_dirs/ssc_rs_base_nuscenes_cam_radar'

# Custom imports for mmdet3d 2.x (mmengine)
custom_imports = dict(
    imports=['projects.LiCROcc.projects.mmdet3d_plugin'],
    allow_failed_imports=False
)

# Set default scope
default_scope = 'mmdet3d'

_sweeps_num_ = 10
_temporal_ = []

lims = [[-51.2, 51.2], [-51.2, 51.2], [-5, 3.0]]
sizes = [512, 512, 40]
grid_meters = [0.2, 0.2, 0.2]
nbr_classes = 17
phase = 'trainval'

ss_class_freq = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
sc_class_freq = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]


grid_config = {
    'x': [-51.2, 51.2, 0.4],
    'y': [-51.2, 51.2, 0.4],
    'z': [-1, 5.4, 6.4],
    'depth': [1.0, 45.0, 0.5],
}
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

    # Augmentation
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}
# 👋

sync_bn = 'torch'
model = dict(
   type='SSC_RS',
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
        loss_depth_weight=1,
        depthnet_cfg=dict(use_dcn=False, aspp_mid_channels=96),
        downsample=16,
        temporal_adapter=False),
    img_bev_encoder_backbone=dict(
        type='CustomResNet',
        numC_input=numC_Trans,
        num_channels=[numC_Trans *2, numC_Trans * 4, numC_Trans * 8],
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
        nbr_class=nbr_classes-1, 
        init_size=sizes[-1], 
        class_frequencies=ss_class_freq, 
        phase=phase,
        frozen=False
        ),
    radar_middle_encoder=dict(
        type='CompletionBranch',
        init_size=sizes[-1],
        nbr_class=nbr_classes,
        phase=phase,
        frozen = False),
    radar_bbox_head=dict(
        type='BEVUNet',
        n_class=nbr_classes*sizes[-1],
        n_height=sizes[-1], 
        use_add=False,
        use_cam=[False,True,True],
        class_frequences=sc_class_freq,
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

# Dataset configuration
dataset_type = 'nuScenesDataset'
data_root = './data/nuscenes/'
occ_root = './data/nuScenes-Occupancy'

train_batch_size = 1
val_batch_size = 1
test_batch_size = 1

# MMEngine 2.x: Dataloader configuration
train_dataloader = dict(
    batch_size=train_batch_size,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DistributedGroupSampler', samples_per_gpu=train_batch_size, seed=10),
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
    )
)

val_dataloader = dict(
    batch_size=val_batch_size,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DistributedSampler', shuffle=False, seed=10),
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
    )
)

test_dataloader = dict(
    batch_size=test_batch_size,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DistributedSampler', shuffle=False, seed=10),
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
    )
)

# MMEngine 2.x: Evaluator configuration
# Use SSCEvaluator for metric calculation
val_evaluator = [
    dict(
        type='SSCEvaluator',
        # save_results=True,
        # out_file_path='val_results.pkl'
    )
]
test_evaluator = [
    dict(
        type='SSCEvaluator',
        # save_results=True,
        # out_file_path='test_results.pkl'
    )
]

# MMEngine 2.x: Optimizer wrapper
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=2e-4,
        weight_decay=0.01
    ),
    clip_grad=dict(max_norm=35, norm_type=2)
)

# MMEngine 2.x: Parameter scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0 / 3,
        by_epoch=False,
        begin=0,
        end=500
    ),
    dict(
        type='CosineAnnealingLR',
        by_epoch=True,
        begin=0,
        end=24,
        eta_min=2e-7  # lr * min_lr_ratio
    )
]

# MMEngine 2.x: Training configuration
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=24,
    val_interval=1
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

# Sync BN
sync_bn = 'torch'

randomness = dict(seed=10, deterministic=False)
