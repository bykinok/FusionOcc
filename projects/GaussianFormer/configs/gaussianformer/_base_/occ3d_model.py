# Shared model/loss config for all nuscenes_gs25600_occ3d* configs.
#
# Identical to configs/gaussianformer/nuscenes_gs25600.py's `model` dict
# EXCEPT for the geometry fields that must match Occ3D-nuScenes' own grid
# convention (pc_range=[-40,-40,-1,40,40,5.4], voxel size 0.4m) instead of
# SurroundOcc GT's ([-50,-50,-5,50,50,3], 0.5m) -- both grids are 200x200x16,
# but occupy a physically different volume, so every module whose output is
# interpreted against real-world coordinates (the deformable/refine/spconv
# layers' `pc_range`, the lifter's own `pc_range`/`voxel_size` used to
# look up `pixel_gt` supervision, and the aggregator head's `cuda_kwargs`)
# must use the Occ3D range for training against Occ3D GT to be geometrically
# consistent. See projects/GaussianFormer/README.md for the full rationale,
# including why `spconv_layer.grid_size` z-component is 0.8 (not 1.0): it
# keeps the same *cell count* (8) along z as the original SurroundOcc-range
# config's spconv discretization (8m/1.0m = 8 cells), just rescaled to
# Occ3D's shorter 6.4m z-span (6.4m/0.8m = 8 cells) -- avoids a non-integer
# spatial_shape (6.4/1.0=6.4) that spconv can't represent.
embed_dims = 128
num_decoder = 4
pc_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
scale_range = [0.01, 1.8]
xyz_coordinate = 'cartesian'
phi_activation = 'sigmoid'
include_opa = True
semantics = True
semantic_dim = 17

loss = dict(
    type='MultiLoss',
    loss_cfgs=[
        dict(
            type='OccupancyLoss',
            weight=1.0,
            empty_label=17,
            num_classes=18,
            use_focal_loss=False,
            use_dice_loss=False,
            balance_cls_weight=True,
            multi_loss_weights=dict(
                loss_voxel_ce_weight=10.0,
                loss_voxel_lovasz_weight=1.0),
            use_sem_geo_scal_loss=False,
            use_lovasz_loss=True,
            lovasz_ignore=17,
            manual_class_weight=[
                1.01552756, 1.06897009, 1.30013094, 1.07253735, 0.94637502, 1.10087012,
                1.26960524, 1.06258364, 1.189019,   1.06217292, 1.00595144, 0.85706115,
                1.03923299, 0.90867526, 0.8936431,  0.85486129, 0.8527829,  0.5],
            ignore_empty=False,
            lovasz_use_softmax=False),
        dict(
            type='PixelDistributionLoss',
            weight=1.0,
            use_sigmoid=False),
    ])

loss_input_convertion = dict(
    pred_occ='pred_occ',
    sampled_xyz='sampled_xyz',
    sampled_label='sampled_label',
    occ_mask='occ_mask',
    bin_logits='bin_logits',
    density='density',
    pixel_logits='pixel_logits',
    pixel_gt='pixel_gt')

model = dict(
    type='BEVSegmentor',
    loss=loss,
    loss_input_convertion=loss_input_convertion,
    freeze_lifter=True,
    img_backbone_out_indices=[0, 1, 2, 3],
    img_backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        style='caffe',
        with_cp=True,
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True)),
    img_neck=dict(
        type='FPN',
        num_outs=4,
        start_level=1,
        out_channels=embed_dims,
        add_extra_convs='on_output',
        relu_before_extra_convs=True,
        in_channels=[256, 512, 1024, 2048]),
    lifter=dict(
        type='GaussianLifterV2',
        num_anchor=19200,
        embed_dims=embed_dims,
        anchor_grad=False,
        feat_grad=False,
        phi_activation='loop',
        semantics=semantics,
        semantic_dim=semantic_dim,
        include_opa=include_opa,
        num_samples=128,
        anchors_per_pixel=1,
        random_sampling=False,
        projection_in=None,
        # Occ3D geometry (see module docstring above) -- GaussianLifterV2
        # uses these to bounds-check/lookup `metas['occ_label']` when
        # building `pixel_gt` for PixelDistributionLoss.
        pc_range=pc_range,
        voxel_size=0.4,
        occ_resolution=[200, 200, 16],
        initializer=dict(
            type='ResNetSecondFPN',
            img_backbone_out_indices=[0, 1, 2, 3],
            img_backbone_config=dict(
                type='ResNet',
                depth=101,
                num_stages=4,
                out_indices=(0, 1, 2, 3),
                frozen_stages=1,
                norm_cfg=dict(type='BN2d', requires_grad=False),
                norm_eval=True,
                style='caffe',
                with_cp=True,
                dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
                stage_with_dcn=(False, False, True, True)),
            neck_confifg=dict(
                type='SECONDFPN',
                in_channels=[256, 512, 1024, 2048],
                out_channels=[embed_dims] * 4,
                upsample_strides=[0.5, 1, 2, 4])),
        initializer_img_downsample=None,
        pretrained_path='projects/GaussianFormer/pretrain/lifter_10.pth',
        deterministic=False,
        random_samples=6400),
    encoder=dict(
        type='GaussianOccEncoder',
        anchor_encoder=dict(
            type='SparseGaussian3DEncoder',
            embed_dims=embed_dims,
            include_opa=include_opa,
            semantics=semantics,
            semantic_dim=semantic_dim),
        norm_layer=dict(type='LN', normalized_shape=embed_dims),
        ffn=dict(
            type='AsymmetricFFN',
            in_channels=embed_dims,
            embed_dims=embed_dims,
            feedforward_channels=embed_dims * 4,
            ffn_drop=0.1,
            add_identity=False),
        deformable_model=dict(
            type='DeformableFeatureAggregation',
            embed_dims=embed_dims,
            num_groups=4,
            num_levels=4,
            num_cams=6,
            attn_drop=0.15,
            use_deformable_func=True,
            use_camera_embed=True,
            residual_mode='none',
            kps_generator=dict(
                type='SparseGaussian3DKeyPointsGenerator',
                embed_dims=embed_dims,
                phi_activation=phi_activation,
                xyz_coordinate=xyz_coordinate,
                num_learnable_pts=6,
                fix_scale=[
                    [0, 0, 0],
                    [0.45, 0, 0],
                    [-0.45, 0, 0],
                    [0, 0.45, 0],
                    [0, -0.45, 0],
                    [0, 0, 0.45],
                    [0, 0, -0.45],
                ],
                pc_range=pc_range,
                scale_range=scale_range,
                learnable_fixed_scale=6.0)),
        refine_layer=dict(
            type='SparseGaussian3DRefinementModuleV2',
            embed_dims=embed_dims,
            pc_range=pc_range,
            scale_range=scale_range,
            unit_xyz=[4.0, 4.0, 1.0],
            semantics=semantics,
            semantic_dim=semantic_dim,
            include_opa=include_opa,
            xyz_coordinate=xyz_coordinate,
            semantics_activation='identity'),
        spconv_layer=dict(
            type='SparseConv3D',
            in_channels=embed_dims,
            embed_channels=embed_dims,
            pc_range=pc_range,
            grid_size=[1.0, 1.0, 0.8],  # see module docstring: keeps 8 z-cells
            phi_activation=phi_activation,
            xyz_coordinate=xyz_coordinate,
            use_out_proj=True,
            use_multi_layer=True),
        num_decoder=num_decoder,
        operation_order=[
            'identity', 'deformable', 'add', 'norm',
            'identity', 'ffn', 'add', 'norm',
            'identity', 'spconv', 'add', 'norm',
            'identity', 'ffn', 'add', 'norm',
            'refine',
        ] * num_decoder),
    head=dict(
        type='GaussianHead',
        apply_loss_type='random_1',
        num_classes=semantic_dim + 1,
        empty_args=dict(
            mean=[0, 0, -1.0],
            scale=[100, 100, 8.0]),
        with_empty=False,
        use_localaggprob=True,
        use_localaggprob_fast=False,
        combine_geosem=True,
        cuda_kwargs=dict(
            scale_multiplier=4,
            H=200, W=200, D=16,
            pc_min=[-40.0, -40.0, -1.0],
            grid_size=0.4)))
