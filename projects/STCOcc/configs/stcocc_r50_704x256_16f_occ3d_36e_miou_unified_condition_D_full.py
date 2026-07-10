_base_ = ['../../../configs/_base_/default_runtime.py']

custom_imports = dict(
    imports=['projects.STCOcc.registry_helper'],
    allow_failed_imports=False)

# -----------------------------------------------------------------------
# Ablation: condition_D_full (free-only full-range supervision)
#   - ALL free voxels at every distance: always supervised (force mask=1)
#     regardless of camera visibility.
#   - Occupied voxels: obey original mask_camera (unchanged).
#
# Fully restores free voxel supervision across all distances (0 m ~ inf).
# If RayIoU recovers fully to baseline_without_mask level, it confirms
# that free voxel masking at ALL distances contributes to the RayIoU gap.
# Compare with:
#   condition_D       → free supervised 0–20 m only
#   condition_D_prime → free supervised 0–35 m only
#   condition_D_full  → free supervised 0–∞  (this config)
# Eval: unchanged (OccupancyMetric uses mask_camera independently).
# -----------------------------------------------------------------------

dataset_name = 'occ3d'
eval_metric = 'miou'

class_weights = [0.0727, 0.0692, 0.0838, 0.0681, 0.0601, 0.0741, 0.0823, 0.0688, 0.0773, 0.0681, 0.0641, 0.0527, 0.0655, 0.0563, 0.0558, 0.0541, 0.0538, 0.0468]

occ_class_names = ['others','barrier', 'bicycle', 'bus', 'car', 'construction_vehicle','motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
                    'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade', 'vegetation','free']

train_top_k = [12500, 2500, 500]
val_top_k = [12500, 2500, 500]

data_config = {
    'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT','CAM_BACK', 'CAM_BACK_RIGHT'],
    'Ncams':6,
    'input_size': (256, 704),
    'src_size': (900, 1600),
    'resize': (0.0, 0.0),
    'rot': (0.0, 0.0),
    'flip': False,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

bda_aug_conf = dict(
    rot_lim=(0, 0),
    scale_lim=(1., 1.),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5
)

train_sequences_split_num = 2
test_sequences_split_num = 1

num_gpus = 2
samples_per_gpu = 8
workers_per_gpu = 4
total_epoch = 24
num_iters_per_epoch = int(28130 // (num_gpus * samples_per_gpu))

grid_config = {
    'x': [-40, 40, 0.8],
    'y': [-40, 40, 0.8],
    'z': [-1, 5.4, 0.8],
    'depth': [1.0, 45.0, 0.5],
}

downsample_rate = 16
multi_adj_frame_id_cfg = (1, 1+1, 1)
forward_numC_Trans = 80

grid_config_bevformer={
    'x': [-40, 40, 0.8],
    'y': [-40, 40, 0.8],
    'z': [-1, 5.4, 0.8],
    'depth': [1.0, 45.0, 0.5],
}
point_cloud_range = [-40, -40, -1.0, 40, 40, 5.4]
bev_h_ = 100
bev_w_ = 100
bev_z = 8
backward_num_layer = [2, 2, 2]
backward_numC_Trans = 96
_dim_ = backward_numC_Trans*2
_pos_dim_ = backward_numC_Trans//2
_ffn_dim_ = backward_numC_Trans * 4
_num_levels_ = 1
num_stage = 3

num_classes = len(occ_class_names)

intermediate_pred_loss_weight=[1.0, 0.5, 0.25, 0.125]
history_frame_num = [16, 8, 4]

model = dict(
    type='STCOcc',
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
            numC_input=forward_numC_Trans * (len(range(*multi_adj_frame_id_cfg))+1),
            num_layer=[1, 2, 4],
            with_cp=False,
            num_channels=[backward_numC_Trans, forward_numC_Trans*2, forward_numC_Trans*4],
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

dataset_type = 'NuScenesDatasetOccpancy'
data_root = ''
backend_args = None

# Condition D_full: ALL free voxels are always supervised regardless of distance;
# occupied voxels obey original mask_camera.
train_pipeline = [
    dict(type='STCOccPrepareImageInputs', is_train=True, data_config=data_config, sequential=True),
    dict(type='STCOccLoadAnnotations'),
    dict(
        type='STCOccLoadOccGTFromFileCVPR2023',
        scale_1_2=True, scale_1_4=True, scale_1_8=True,
        load_mask=True,
        mask_mode='condition_D_full',
        free_class_id=occ_class_names.index('free'),  # 17
        pc_range_x=80.0,
    ),
    dict(type='STCOccBEVAug', bda_aug_conf=bda_aug_conf, classes=occ_class_names),
    dict(type='STCOccLoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=3, backend_args=backend_args),
    dict(type='STCOccPointToMultiViewDepth', downsample=1, grid_config=grid_config),
    dict(type='DefaultFormatBundle3D', class_names=occ_class_names),
    dict(type='Collect3D',
         keys=['img_inputs', 'gt_depth', 'voxel_semantics', 'voxel_semantics_1_2', 'voxel_semantics_1_4', 'voxel_semantics_1_8'])
]

# Eval pipeline is unchanged (no mask applied here; OccupancyMetric handles mask independently)
test_pipeline = [
    dict(type='STCOccPrepareImageInputs', data_config=data_config, sequential=True),
    dict(type='STCOccLoadAnnotations'),
    dict(type='STCOccBEVAug',bda_aug_conf=bda_aug_conf,classes=occ_class_names,is_train=False),
    dict(type='STCOccLoadPointsFromFile',coord_type='LIDAR', load_dim=5, use_dim=3, backend_args=backend_args),
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
        seed=0
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
        work_dir='stcocc_r50_704x256_16f_occ3d_36e_condition_D_full',
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
        work_dir='stcocc_r50_704x256_16f_occ3d_36e_condition_D_full',
        eval_show=True,
        test_mode=True))

test_dataloader = val_dataloader

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
    accumulative_counts=1)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.05,
        end_factor=1.0,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        type='CosineAnnealingLR',
        begin=500,
        end=total_epoch * num_iters_per_epoch,
        by_epoch=False,
        eta_min=1e-6)
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
    num_classes=18,
    use_image_mask=True,
    sort_by_timestamp=True)

test_evaluator = val_evaluator

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=num_iters_per_epoch),
    sampler_seed=dict(type='DistSamplerSeedHook'))

custom_hooks = []

load_from = "projects/STCOcc/pretrain/forward_projection-r50-4d-stereo-pretrained.pth"

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

randomness = dict(seed=0, deterministic=False)
