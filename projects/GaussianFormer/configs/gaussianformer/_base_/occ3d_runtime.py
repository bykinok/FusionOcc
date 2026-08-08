# Shared boilerplate for all nuscenes_gs25600_occ3d* configs.
default_scope = 'mmdet3d'
custom_imports = dict(
    imports=['projects.GaussianFormer.gaussianformer'],
    allow_failed_imports=False)

log_level = 'INFO'
load_from = 'projects/GaussianFormer/pretrain/r101_dcn_fcos3d_pretrain.pth'
resume = False

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, save_best='mIoU', rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

# Set in every occ3d config across every ported project in this repo
# (SurroundOcc/CONet/BEVFormer/TPVFormer/STCOcc/FusionOcc/LiCROcc/SparseOcc_eccv's
# both _ori_setting and _unified variants), not just the _unified axis --
# was missing here.
randomness = dict(seed=0, deterministic=False)
