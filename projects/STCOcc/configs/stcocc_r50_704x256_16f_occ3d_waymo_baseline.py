_base_ = ['../../../configs/_base_/default_runtime.py']

custom_imports = dict(
    imports=['projects.STCOcc.registry_helper'],
    allow_failed_imports=False)

#
# stcocc_r50_704x256_16f_occ3d_waymo_baseline.py
# Occ3D-Waymo dataset-generalization baseline, occupancy-only, "none" supervision
# policy -- same architecture-neutral philosophy as the OpenOcc occ_only_screen
# profile (research_openocc/implementation_changes.md), applied to a DIFFERENT
# sensor rig (5 cameras instead of 6) and a DIFFERENT GT release (Occ3D-Waymo,
# raw free sentinel=23, remapped to 15) -- see research_waymo/ for the full audit.
#
# What this is:
#   - Same STCOcc architecture (forward projection / BEVFormer backward
#     projection / temporal fusion / occupancy head) as the Occ3D-nuScenes and
#     OpenOcc configs. NO architecture change was needed for 5 cameras: the
#     only camera-count-coupled learnable weight (BEVFormer's cams_embeds) is
#     already disabled (use_cams_embeds=False) in every STCOcc config,
#     including this one -- see research_waymo/adaptation_blockers.md.
#   - Occupancy-only: no flow_head (Waymo's voxel04 GT has no flow field at
#     all in this profile, so this is the natural default, not a removal).
#   - lambda_inv_free=1.0 is the EXPLICIT "none" baseline policy. Occ3D-Waymo
#     DOES have a usable camera-visibility-equivalent mask (`final_voxel_state`,
#     see research_waymo/dataset_schema.md) -- unlike OpenOcc -- but this first
#     profile intentionally does not wire it in (WAYMO_EXTENSION_REQUIREMENTS_KO.md
#     section 9: "우선 baseline-none 1개로 먼저"). No voxel_mask_camera key is
#     requested in Collect3D below, so the T6 guard in stcocc.py (added during
#     the OpenOcc work) correctly requires lambda_inv_free=1.0 here.
#   - point_cloud_range/voxel grid are IDENTICAL to Occ3D-nuScenes/OpenOcc
#     ([-40,-40,-1,40,40,5.4], 0.4m, [200,200,16]) -- confirmed against the
#     official Occ3D-Waymo README's coarse (voxel04) release, not assumed.
#   - class_weights below are UNIFORM (not tuned) -- no official per-class
#     frequency weighting is published for Occ3D-Waymo in the sources checked
#     this session. Flagged explicitly rather than silently reusing Occ3D-
#     nuScenes' (dataset-specific) weights.
#   - Data scope: only the LOCALLY AVAILABLE scenes are covered by the info
#     pkl this config points at -- 241/798 train scenes, 202/202 (full) val
#     scenes as of this session. See research_waymo/remaining_issues.md item
#     #8 (downloading the remaining 557 train scenes needs separate approval,
#     not done this pass).
#
# NOT fully trained -- see research_waymo/smoke_test_results.md for what was
# actually run (1-iteration backward + short run only, separate approval
# needed for a full training run).
#

dataset_name = 'occ3d_waymo'
eval_metric = 'miou'  # mIoU is the primary/default metric for Waymo, matching the reference
                      # implementation (CVT-Occ reports only mIoU for Occ3D-Waymo, not RayIoU).
                      # A RayIoU kernel (ray_metrics_waymo.py) now also exists but is a research-
                      # only approximation (non-native single-origin protocol, no official Waymo
                      # RayIoU benchmark exists) -- set eval_metric='rayiou' to opt in; see its
                      # module docstring and research_waymo/remaining_issues.md #13.

# Occ3D-Waymo class definition (research_waymo/dataset_schema.md; confirmed against
# the official Occ3D README + CVT-Occ's CLASS_NAMES/FREE_LABEL literals, cross-checked
# against real GT files this session). Raw npz free sentinel is 23, remapped to 15
# (=num_classes-1) by LoadOccGTFromFileWaymo before this ever reaches the model/loss.
occ_class_names = ['GO', 'vehicle', 'pedestrian', 'sign', 'cyclist', 'traffic_light',
                   'pole', 'construction_cone', 'bicycle', 'motorcycle', 'building',
                   'vegetation', 'tree_trunk', 'road', 'walkable', 'free']
class_weights = [1.0 / 16] * 16  # UNIFORM, not tuned -- see header note above

train_top_k = [12000, 2400, 450]  # unchanged from Occ3D/OpenOcc -- cascade candidate-count
                                  # hyperparameter, not dataset-specific; not retuned for Waymo
val_top_k = [12000, 2400, 450]

# DataLoader Config -- 5 cameras (Occ3D-Waymo), not 6 (nuScenes). Camera-pose-index <->
# image-folder mapping (including the empirically-verified index 2/3 swap) is resolved
# once at info-pkl-generation time by tools/create_data_waymo_occ.py, not here.
data_config = {
    'cams': ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_SIDE_LEFT', 'CAM_SIDE_RIGHT'],
    'Ncams': 5,
    'input_size': (256, 704),
    'src_size': (1280, 1920),  # informational only -- not actually read by the resize
                               # pipeline (it uses the real loaded image's own
                               # dimensions), and Waymo cameras are not all this size
                               # anyway (front 3 cams: 1280x1920, side 2 cams: 886x1920)
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

# Waymo captures at 10Hz vs nuScenes' 2Hz keyframes (research_waymo/camera_temporal_audit.md,
# confirmed against CVT-Occ's docs/dataset.md) -- Protocol A (frame-count matched) is used
# here: history_frame_num stays [16,8,4] like Occ3D/OpenOcc, so the 16-frame history covers
# only 1.6s of real time on Waymo vs 8.0s on nuScenes. This is a DELIBERATE, DOCUMENTED choice
# (not an oversight) -- Protocol B (duration-matched, ~80 frames) is left as a follow-up
# research decision, not auto-applied (research_waymo/camera_temporal_audit.md section 3).
train_sequences_split_num = 2
test_sequences_split_num = 1

# Running Config
num_gpus = 2  # ssh mando-h100 / mando-h100_2
samples_per_gpu = 8  # effective batch=16, matches the Occ3D/OpenOcc screen profiles
workers_per_gpu = 4
total_epoch = 12
# 47737 locally-available train samples (241/798 scenes) -- NOT the full dataset.
# No upstream multiplier (c.f. OpenOcc's audited-but-unexplained 4.554 factor) is applied.
num_iters_per_epoch = int(47737 // (num_gpus * samples_per_gpu))

# Model Config -- identical to Occ3D-nuScenes/OpenOcc; grid matches the Occ3D-Waymo
# coarse (0.4m) release exactly (research_waymo/dataset_schema.md section 2).
grid_config = {
    'x': [-40, 40, 0.8],
    'y': [-40, 40, 0.8],
    'z': [-1, 5.4, 0.8],
    'depth': [1.0, 45.0, 0.5],
}

downsample_rate = 16
multi_adj_frame_id_cfg = (1, 1 + 1, 1)
forward_numC_Trans = 80

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

num_classes = len(occ_class_names)  # 16

intermediate_pred_loss_weight = [1.0, 0.5, 0.25, 0.125]
history_frame_num = [16, 8, 4]

lambda_inv_free = 1.0  # 'none' baseline policy -- see header note. The T6 guard in
                       # stcocc.py requires this to stay 1.0 since no camera_mask key
                       # is provided to Collect3D below.

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
    # occupancy-only: no flow_head key at all, same pattern as the OpenOcc profile.
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
            use_cams_embeds=False,  # see header note -- the only camera-count-coupled
                                    # weight, already disabled for every STCOcc profile
            num_cams=5,  # 5 Waymo cameras, not the default 6 -- even with
                        # use_cams_embeds=False the cams_embeds tensor's shape must
                        # still match the actual camera count for the (zeroed)
                        # broadcast add to run at all (confirmed via smoke test:
                        # RuntimeError without this, "size of tensor a (5) must
                        # match size of tensor b (6)")
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
                            num_cams=5,  # same reason as BEVFormer's num_cams above --
                                        # this class has its own independent num_cams=6
                                        # default (confirmed via smoke test RuntimeError:
                                        # "shape '[12, 704, 96]'... size 675840" ==
                                        # 6-cams-worth of reshape attempted on 5-cams-worth
                                        # of data)
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
dataset_type = 'NuScenesDatasetOccpancy'  # reused as-is -- the Waymo info pkl mimics the
                                          # exact nuScenes info schema (see
                                          # tools/create_data_waymo_occ.py), so this
                                          # dataset class needs no Waymo-specific code
data_root = ''
backend_args = None

train_pipeline = [
    dict(type='STCOccPrepareImageInputs', is_train=True, data_config=data_config, sequential=True),
    dict(type='STCOccLoadAnnotations'),
    dict(type='STCOccLoadOccGTFromFileWaymo',
         scale_1_2=True, scale_1_4=True, scale_1_8=True,
         num_classes=num_classes, apply_fov_ignore=True,
         use_lidar_mask=False, use_camera_mask=False),
    dict(type='STCOccBEVAug', bda_aug_conf=bda_aug_conf, classes=occ_class_names),
    # load_dim=6 (not nuScenes' 5) -- Waymo's KITTI-format velodyne .bin stores 6 floats
    # per point ([x,y,z,intensity,elongation,?], confirmed this session via file size /
    # point-count divisibility check), use_dim=3 keeps only xyz, same as the nuScenes/
    # OpenOcc configs.
    dict(type='STCOccLoadPointsFromFile', coord_type='LIDAR', load_dim=6, use_dim=3, backend_args=backend_args),
    dict(type='STCOccPointToMultiViewDepth', downsample=1, grid_config=grid_config),
    dict(type='DefaultFormatBundle3D', class_names=occ_class_names),
    dict(type='Collect3D',
         # No voxel_mask_camera* keys -- 'none' policy, see header note.
         keys=['img_inputs', 'gt_depth', 'voxel_semantics', 'voxel_semantics_1_2', 'voxel_semantics_1_4',
               'voxel_semantics_1_8'])
]

test_pipeline = [
    dict(type='STCOccPrepareImageInputs', data_config=data_config, sequential=True),
    dict(type='STCOccLoadAnnotations'),
    dict(type='STCOccBEVAug', bda_aug_conf=bda_aug_conf, classes=occ_class_names, is_train=False),
    dict(type='STCOccLoadPointsFromFile', coord_type='LIDAR', load_dim=6, use_dim=3, backend_args=backend_args),
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
    batch_size=1,
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
        ann_file='data/waymo/stcocc-waymo_infos_train.pkl',
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
        work_dir='stcocc_r50_704x256_16f_occ3d_waymo_baseline',
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
        ann_file='data/waymo/stcocc-waymo_infos_val.pkl',
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
        work_dir='stcocc_r50_704x256_16f_occ3d_waymo_baseline',
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
    val_interval=99999)  # val loop not used this pass -- see header note on
                         # OccupancyMetric/RayIoU Waymo wiring being deferred;
                         # small-set mIoU for the smoke test is computed by a
                         # standalone script instead, not via this val loop.

log_processor = dict(type='LogProcessor', window_size=50, by_epoch=False)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# val_evaluator now wired for Waymo: OccupancyMetric's 3 internal GT-reload call
# sites were updated (2026-10-01) to go through a single shared helper
# (occupancy_metric_utils.load_occ_gt_npz) that knows Occ3D-Waymo's file suffix
# ('_04.npz'), key names ('voxel_label'/'final_voxel_state'), and the raw
# free=23 remap -- see research_waymo/remaining_issues.md item #12 and
# implementation_audit.md. mIoU is supported; RayIoU for Waymo is still
# NOT implemented (no ray_metrics_waymo.py kernel -- see item #13, and
# dataset_name=='occ3d_waymo' is not yet routed in _compute_rayiou, so
# eval_metric must stay 'miou' for this config until that lands).
val_evaluator = dict(
    type='OccupancyMetric',
    ann_file='data/waymo/stcocc-waymo_infos_val.pkl',
    data_root=data_root,
    dataset_name=dataset_name,
    eval_metric=eval_metric,
    num_classes=num_classes,
    use_image_mask=False,  # 'none' baseline -- see header note, same reasoning as OpenOcc
    compute_uncertainty_metrics=True,
    sort_by_timestamp=False)  # Waymo info pkl entries are not nuScenes timestamps;
                              # already in scene/frame order from the generator

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

load_from = "projects/STCOcc/pretrain/forward_projection-r50-4d-stereo-pretrained.pth"

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')
