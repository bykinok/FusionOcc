_base_ = ['../../../configs/_base_/default_runtime.py']

# Enable project imports
custom_imports = dict(
    imports=['projects.STCOcc.registry_helper'],
    allow_failed_imports=False)

#
# stcocc_r50_704x256_16f_openocc_occ_only_screen_miou.py
# mIoU_openocc_all_valid eval companion for stcocc_r50_704x256_16f_openocc_occ_only_screen.py.
# NOT meant to be trained on its own -- same model/data/optim (checkpoints are
# interchangeable), only eval_metric differs (auxiliary diagnostic metric; requirement
# 6.2 says this all-voxel-valid mIoU must not be confused with Occ3D's camera-visible
# mIoU, hence the miou_scope='mIoU_openocc_all_valid' naming carried in
# research_openocc/results/summary.csv, not in the raw metric dict key itself to avoid a
# wider, riskier rename of OccupancyMetric's shared 'mIoU' key used by Occ3D tooling too).
# Use: tools/test.py this_config <ckpt from the screen training run>
#
# OpenOcc GT-generalization baseline, occupancy-only, "none" supervision policy
# (research_openocc/audit.md stage: baseline-none, per CLAUDE_CODE_OPENOCC_REQUIREMENTS_KO.md
# section 9 -- "우선 baseline-none 1개로 end-to-end training pipeline을 확인한다").
#
# What this is:
#   - Same STCOcc architecture, optimizer, augmentation, and *effective batch* (16) as the
#     already-run Occ3D screen profiles (see stcocc_r50_704x256_16f_occ3d_e12_stage2_invfree_l025.py),
#     so this is a direct GT-pipeline-transfer comparison: only the GT/adapter changes.
#   - Occupancy-only: flow_head is NOT built (omitted below, not just train_flow=False --
#     see research_openocc/audit.md section 4.1, train_flow alone does not gate flow loss/
#     inference), load_flow=False in the loader, no voxel_flows/flow key in Collect3D.
#   - lambda_inv_free=1.0 is the EXPLICIT "none" baseline policy (all valid voxels, native
#     objective, no selective weighting) -- OpenOcc has no camera visibility mask at all, so
#     this is the only policy this profile may use. A non-1.0 lambda_inv_free would now raise
#     at train time (stcocc.py::get_voxel_loss, the new T6 guard) instead of silently
#     no-op'ing, specifically to prevent this profile from ever training as if a mask existed.
#   - num_iters_per_epoch uses the SAME formula as the Occ3D screen profiles (no 4.554
#     multiplier). The upstream openocc_12e reference config has an extra "* 4.554" factor
#     whose empirical justification is not explained even in Ref/STCOcc_ori; carrying it into
#     a NEW screening profile not otherwise matching upstream's num_gpus=8/samples_per_gpu=2
#     recipe would just add an unexplained confound to what should be a controlled, directly
#     comparable (same update budget, same effective batch) GT-transfer experiment. The
#     4.554-multiplier recipe is preserved faithfully, unmodified, in
#     stcocc_r50_704x256_16f_openocc_12e.py (native reference, not run this pass).
#
# NOT run yet -- see research_openocc/commands.md for the approved smoke-test command.
#

# Dataset Config
dataset_name = 'openocc'
eval_metric = 'miou'

class_weights = [0.0682, 0.0823, 0.0671, 0.0594, 0.0732, 0.0806, 0.0680, 0.0762, 0.0675, 0.0633, 0.0521, 0.0644, 0.0557,
                 0.0551, 0.0535, 0.0533, 0.0464]  # openocc (same as stcocc_r50_704x256_16f_openocc_12e.py)

occ_class_names = ['car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle', 'motorcycle', 'pedestrian',
                   'traffic_cone', 'barrier', 'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
                   'vegetation', 'free']

train_top_k = [12000, 2400, 450]
val_top_k = [12000, 2400, 450]

# DataLoader Config
data_config = {
    'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
    'Ncams': 6,
    'input_size': (256, 704),
    'src_size': (900, 1600),
    # Augmentation
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

bda_aug_conf = dict(
    rot_lim=(0, 0),
    scale_lim=(1., 1.),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5
)

# Each nuScenes sequence is ~40 keyframes long -- split in half during training for
# step-to-step diversity, same as both Occ3D and upstream OpenOcc configs.
train_sequences_split_num = 2
test_sequences_split_num = 1

# Running Config
num_gpus = 2  # ssh mando-h100 / mando-h100_2, 2-GPU distributed (tools/dist_train.sh)
samples_per_gpu = 8  # effective batch=16, matches the already-run Occ3D screen profiles
workers_per_gpu = 4
total_epoch = 12  # 12-epoch-equivalent, matches Occ3D screen profiles for direct comparison
num_iters_per_epoch = int(28130 // (num_gpus * samples_per_gpu))  # total samples: 28130 (same info pkl as Occ3D)

# Model Config
grid_config = {
    'x': [-40, 40, 0.8],
    'y': [-40, 40, 0.8],
    'z': [-1, 5.4, 0.8],
    'depth': [1.0, 45.0, 0.5],
}

downsample_rate = 16
multi_adj_frame_id_cfg = (1, 1 + 1, 1)
forward_numC_Trans = 80

# backward params
grid_config_bevformer = grid_config
point_cloud_range = [-40, -40, -1.0, 40, 40, 5.4]
bev_h_ = 100
bev_w_ = 100
bev_z = 8
backward_num_layer = [2, 2, 2]
backward_numC_Trans = 96
_dim_ = backward_numC_Trans * 2
_pos_dim_ = backward_numC_Trans // 2
_ffn_dim_ = backward_numC_Trans * 4
_num_levels_ = 1
num_stage = 3

# others params
num_classes = len(occ_class_names)  # 17: 0-15 -> objects, 16 -> free

intermediate_pred_loss_weight = [1.0, 0.5, 0.25, 0.125]
history_frame_num = [16, 8, 4]

# "none" baseline policy (CLAUDE_CODE_OPENOCC_REQUIREMENTS_KO.md section 5): all valid
# voxels trained with the native objective, no selective weighting. Explicit 1.0, not the
# constructor default, so this is a documented choice, not an accident.
lambda_inv_free = 1.0

model = dict(
    type='STCOcc',
    use_camera_mask=False,
    lambda_inv_free=lambda_inv_free,
    num_stage=num_stage,
    bev_w=bev_h_,
    bev_h=bev_w_,
    bev_z=bev_z,
    train_top_k=train_top_k,
    val_top_k=val_top_k,
    class_weights=class_weights,
    history_frame_num=history_frame_num,
    backward_num_layer=backward_num_layer,
    empty_idx=occ_class_names.index('free'),
    intermediate_pred_loss_weight=intermediate_pred_loss_weight,
    save_results=False,
    # occupancy-only: no flow_head key at all (not flow_head=None -- simply absent, so
    # MODELS.build(flow_head) is never even attempted; see stcocc.py __init__).
    forward_projection=dict(
        type='BEVDetStereoForwardProjection',
        align_after_view_transfromation=False,
        return_intermediate=True,
        num_adj=len(range(*multi_adj_frame_id_cfg)),
        adjust_channel=backward_numC_Trans,
        img_backbone=dict(
            type='mmdet.ResNet',
            depth=50,
            num_stages=4,
            out_indices=(0, 2, 3),
            frozen_stages=-1,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=False,
            with_cp=True,
            style='pytorch'),
        img_neck=dict(
            type='CustomFPN',
            in_channels=[1024, 2048],
            out_channels=256,
            num_outs=1,
            start_level=0,
            out_ids=[0]),
        img_view_transformer=dict(
            type='LSSVStereoForwardPorjection',
            grid_config=grid_config,
            input_size=data_config['input_size'],
            in_channels=256,
            out_channels=forward_numC_Trans,
            sid=False,
            collapse_z=False,
            loss_depth_weight=0.5,
            depthnet_cfg=dict(use_dcn=False,
                              aspp_mid_channels=96,
                              stereo=True,
                              bias=5.),
            downsample=downsample_rate),
        img_bev_encoder_backbone=dict(
            type='CustomResNet3D',
            numC_input=forward_numC_Trans * (len(range(*multi_adj_frame_id_cfg)) + 1),
            num_layer=[1, 2, 4],
            with_cp=False,
            num_channels=[backward_numC_Trans, forward_numC_Trans * 2, forward_numC_Trans * 4],
            adjust_number_channel=backward_numC_Trans,
            stride=[1, 2, 2],
            backbone_output_ids=[0, 1, 2],
        ),
    ),
    backward_projection=dict(
        type='BEVFormerBackwardProjection',
        bev_h=bev_h_,
        bev_w=bev_w_,
        in_channels=backward_numC_Trans,
        out_channels=backward_numC_Trans,
        pc_range=point_cloud_range,
        transformer=dict(
            type='BEVFormer',
            use_cams_embeds=False,
            embed_dims=backward_numC_Trans,
            encoder=dict(
                type='BEVFormerEncoder',
                num_layers=2,
                use_temporal=True,
                pc_range=point_cloud_range,
                grid_config=grid_config_bevformer,
                data_config=data_config,
                return_intermediate=False,
                predictor_in_channels=backward_numC_Trans,
                predictor_out_channels=backward_numC_Trans,
                predictor_num_calsses=num_classes,
                transformerlayers=dict(
                    type='BEVFormerEncoderLayer',
                    attn_cfgs=[
                        dict(
                            type='OA_TemporalAttention',
                            num_points=bev_z,
                            embed_dims=backward_numC_Trans,
                            dropout=0.0,
                            num_levels=1),
                        dict(
                            type='OA_SpatialCrossAttention',
                            pc_range=point_cloud_range,
                            dbound=grid_config['depth'],
                            dropout=0.0,
                            deformable_attention=dict(
                                type='OA_MSDeformableAttention3D',
                                embed_dims=backward_numC_Trans,
                                num_points=bev_z,
                                num_levels=_num_levels_),
                            embed_dims=backward_numC_Trans,
                        )
                    ],
                    conv_cfgs=dict(embed_dims=backward_numC_Trans),
                    operation_order=('predictor', 'self_attn', 'norm', 'cross_attn', 'norm', 'conv')
                    )
                ),
        ),
        positional_encoding=dict(
            type='CustormLearnedPositionalEncoding',
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_,
        ),
    ),
    temporal_fusion=dict(
        type='SparseFusion',
        history_num=2,
        single_bev_num_channels=backward_numC_Trans,
        num_classes=num_classes,
        bev_w=bev_w_,
        bev_h=bev_h_,
        bev_z=bev_z,
    ),
    occupancy_head=dict(
        type='OccHead',
        in_channels=backward_numC_Trans,
        out_channels=backward_numC_Trans,
        num_classes=num_classes,
    )
)

# Data
dataset_type = 'NuScenesDatasetOccpancy'
data_root = ''
backend_args = None

train_pipeline = [
    dict(type='STCOccPrepareImageInputs', is_train=True, data_config=data_config, sequential=True),
    dict(type='STCOccLoadAnnotations'),
    dict(type='STCOccLoadOccGTFromFileOpenOcc',
         scale_1_2=True, scale_1_4=True, scale_1_8=True,
         load_ray_mask=False,  # ray_mask2 is vestigial/unused (see research_openocc/audit.md)
         load_flow=False),     # occupancy-only: do not read flow at any scale
    dict(type='STCOccBEVAug', bda_aug_conf=bda_aug_conf, classes=occ_class_names),
    dict(type='STCOccLoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=3, backend_args=backend_args),
    dict(type='STCOccPointToMultiViewDepth', downsample=1, grid_config=grid_config),
    dict(type='DefaultFormatBundle3D', class_names=occ_class_names),
    dict(type='Collect3D',
         # No voxel_flows (occupancy-only) and no voxel_mask_camera* keys -- OpenOcc has no
         # camera visibility mask; requesting those keys would either KeyError or (per the
         # old, now-fixed behavior) silently no-op the selective-lambda weighting.
         keys=['img_inputs', 'gt_depth', 'voxel_semantics', 'voxel_semantics_1_2', 'voxel_semantics_1_4',
               'voxel_semantics_1_8'])
]

test_pipeline = [
    dict(type='STCOccPrepareImageInputs', data_config=data_config, sequential=True),
    dict(type='STCOccLoadAnnotations'),
    dict(type='STCOccBEVAug', bda_aug_conf=bda_aug_conf, classes=occ_class_names, is_train=False),
    dict(type='STCOccLoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=3, backend_args=backend_args),
    dict(
        type='STCOccMultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=occ_class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points', 'img_inputs'])
        ])
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

train_dataloader = dict(
    batch_size=1,  # batch_sampler controls the real batch size below
    num_workers=workers_per_gpu,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(
        type='InfiniteGroupEachSampleInBatchSampler',
        batch_size=samples_per_gpu,
        world_size=None,
        rank=None,
        seed=None
    ),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='data/nuscenes/stcocc-nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        classes=occ_class_names,
        modality=input_modality,
        stereo=True,
        filter_empty_gt=False,
        img_info_prototype='bevdet4d',
        multi_adj_frame_id_cfg=multi_adj_frame_id_cfg,
        use_sequence_group_flag=True,
        test_mode=False,
        use_valid_flag=True,
        box_type_3d='LiDAR',
        sequences_split_num=train_sequences_split_num,
        dataset_name=dataset_name,
        eval_metric=eval_metric,
        work_dir='stcocc_r50_704x256_16f_openocc_occ_only_screen_miou',
        eval_show=True))

val_dataloader = dict(
    batch_size=1,
    num_workers=workers_per_gpu,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(
        type='InfiniteGroupEachSampleInBatchSamplerEval',
        batch_size=1,
        seed=0),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='data/nuscenes/stcocc-nuscenes_infos_val.pkl',
        pipeline=test_pipeline,
        classes=occ_class_names,
        modality=input_modality,
        stereo=True,
        filter_empty_gt=False,
        img_info_prototype='bevdet4d',
        multi_adj_frame_id_cfg=multi_adj_frame_id_cfg,
        use_sequence_group_flag=True,
        dataset_name=dataset_name,
        eval_metric=eval_metric,
        work_dir='stcocc_r50_704x256_16f_openocc_occ_only_screen_miou',
        eval_show=True,
        test_mode=True))

test_dataloader = val_dataloader

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=1e-2),
    clip_grad=dict(max_norm=5, norm_type=2))

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=200),
    dict(
        type='StepLR',
        by_epoch=False,
        step_size=total_epoch * num_iters_per_epoch,
        gamma=0.1)
]

train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=total_epoch * num_iters_per_epoch,
    val_interval=99999)

log_processor = dict(type='LogProcessor', window_size=50, by_epoch=False)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

val_evaluator = dict(
    type='OccupancyMetric',
    ann_file='data/nuscenes/stcocc-nuscenes_infos_val.pkl',
    data_root=data_root,
    dataset_name=dataset_name,
    eval_metric=eval_metric,
    num_classes=num_classes,  # 17 -> Metric_mIoU now branches to the OpenOcc class_names
    use_image_mask=False,  # OpenOcc has no camera mask; this is a dataset fact, not a choice
    compute_uncertainty_metrics=True,
    sort_by_timestamp=True)

test_evaluator = val_evaluator

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=num_iters_per_epoch),
    sampler_seed=dict(type='DistSamplerSeedHook'))

custom_hooks = [
    dict(
        type='MEGVIIEMAHook',
        init_updates=10560,
        priority='NORMAL',
        interval=2 * num_iters_per_epoch,
    )
]

# NOTE: upstream stcocc_r50_704x256_16f_openocc_12e.py points load_from at
# "projects/STCOcc/pretrained/..." (plural) which does not exist on disk -- the correct,
# existing path (singular "pretrain", same file the Occ3D configs load) is used here.
load_from = "projects/STCOcc/pretrain/forward_projection-r50-4d-stereo-pretrained.pth"

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')
