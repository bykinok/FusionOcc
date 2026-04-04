# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
#  Migration: mmcv.runner → mmengine.runner (새 OpenMMLab API)
# ---------------------------------------------
"""
이 파일은 구버전 mmcv.runner 기반 학습 루프를 새 mmengine.runner 로 이식합니다.
원본 로직 구조는 최대한 유지합니다.
"""
import random
import warnings
import time
import os.path as osp

import numpy as np
import torch
import torch.distributed as dist

try:
    from mmengine.runner import Runner
    from mmengine.optim import build_optim_wrapper
    from mmengine.dist import get_dist_info
    _HAS_MMENGINE = True
except ImportError:
    _HAS_MMENGINE = False
    try:
        from mmcv.runner import get_dist_info
    except ImportError:
        def get_dist_info():
            return 0, 1

try:
    from mmengine import MMLogger
    def get_root_logger(log_level='INFO'):
        return MMLogger.get_current_instance()
except ImportError:
    try:
        from mmdet.utils import get_root_logger
    except ImportError:
        import logging
        def get_root_logger(log_level='INFO'):
            return logging.getLogger('sparseocc')

from ..datasets.builder import build_dataloader
from ..core.evaluation.eval_hooks import OccDistEvalHook
from ..datasets import custom_build_dataset


def custom_train_detector(model,
                          dataset,
                          cfg,
                          distributed=False,
                          validate=False,
                          timestamp=None,
                          eval_model=None,
                          meta=None):
    """새 mmengine Runner 기반 학습 함수.

    구버전 mmcv.runner EpochBasedRunner 로직을 mmengine.runner.Runner 로 이식합니다.
    """
    logger = get_root_logger(cfg.get('log_level', 'INFO'))
    rank, world_size = get_dist_info()

    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]

    # samples_per_gpu 호환
    if hasattr(cfg, 'data') and 'imgs_per_gpu' in cfg.data:
        cfg.data.samples_per_gpu = cfg.data.imgs_per_gpu

    # DataLoader 구성
    train_loader_cfg = dict(
        samples_per_gpu=cfg.data.get('samples_per_gpu', 1),
        workers_per_gpu=cfg.data.get('workers_per_gpu', 4),
        num_gpus=len(cfg.gpu_ids) if hasattr(cfg, 'gpu_ids') else world_size,
        dist=distributed,
        seed=cfg.get('seed', None),
        shuffler_sampler=cfg.data.get('shuffler_sampler', dict(type='DistributedGroupSampler')),
        nonshuffler_sampler=cfg.data.get('nonshuffler_sampler', dict(type='DistributedSampler')),
    )
    data_loaders = [build_dataloader(ds, **train_loader_cfg) for ds in dataset]

    # 모델을 GPU에 올리기
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        try:
            from torch.nn.parallel import DistributedDataParallel as DDP
            model = DDP(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters)
        except Exception:
            from mmcv.parallel import MMDistributedDataParallel
            model = MMDistributedDataParallel(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters)
    else:
        gpu_ids = getattr(cfg, 'gpu_ids', [0])
        try:
            from mmcv.parallel import MMDataParallel
            model = MMDataParallel(model.cuda(gpu_ids[0]), device_ids=gpu_ids)
        except ImportError:
            model = model.cuda(gpu_ids[0])

    # Optimizer
    try:
        from mmengine.optim import build_optim_wrapper
        optimizer = build_optim_wrapper(model, cfg.optimizer)
    except Exception:
        try:
            from mmcv.runner import build_optimizer
            optimizer = build_optimizer(model, cfg.optimizer)
        except Exception:
            import torch.optim as optim
            optimizer = optim.AdamW(model.parameters(), lr=cfg.optimizer.get('lr', 1e-4))

    # Runner 구성 (mmengine 우선, 폴백으로 mmcv.runner)
    if _HAS_MMENGINE:
        _run_with_mmengine(
            model, optimizer, data_loaders, cfg, validate, timestamp, meta,
            distributed, eval_model, logger)
    else:
        _run_with_mmcv(
            model, optimizer, data_loaders, cfg, validate, timestamp, meta,
            distributed, eval_model, logger)


def _run_with_mmengine(model, optimizer, data_loaders, cfg, validate,
                       timestamp, meta, distributed, eval_model, logger):
    """mmengine.runner.Runner 를 이용한 학습."""
    from mmengine.runner import Runner

    runner_cfg = dict(
        model=model,
        work_dir=cfg.work_dir,
        train_dataloader=data_loaders[0],
        optim_wrapper=optimizer if hasattr(optimizer, 'update_params') else dict(optimizer=optimizer),
        train_cfg=dict(
            type='EpochBasedTrainLoop',
            max_epochs=cfg.runner.get('max_epochs', 24) if hasattr(cfg, 'runner') else 24,
            val_interval=cfg.evaluation.get('interval', 1) if hasattr(cfg, 'evaluation') else 1,
        ),
        default_hooks=dict(
            checkpoint=cfg.get('checkpoint_config', dict(interval=1)),
            logger=dict(type='LoggerHook', interval=50),
        ),
        resume=bool(cfg.get('resume_from')),
        load_from=cfg.get('load_from'),
    )

    if validate:
        val_dataset = custom_build_dataset(cfg.data.val, dict(test_mode=True))
        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.get('workers_per_gpu', 4),
            dist=distributed,
            shuffle=False,
        )
        runner_cfg['val_dataloader'] = val_dataloader
        runner_cfg['val_cfg'] = dict(type='ValLoop')

    runner = Runner.from_cfg(runner_cfg)
    runner.train()


def _run_with_mmcv(model, optimizer, data_loaders, cfg, validate,
                   timestamp, meta, distributed, eval_model, logger):
    """폴백: 구버전 mmcv.runner 기반 학습."""
    from mmcv.runner import (HOOKS, DistSamplerSeedHook, EpochBasedRunner,
                             Fp16OptimizerHook, OptimizerHook, build_runner)
    from mmcv.utils import build_from_cfg

    try:
        from mmdet.core import EvalHook
    except ImportError:
        EvalHook = None

    if 'runner' not in cfg:
        cfg.runner = dict(type='EpochBasedRunner', max_epochs=cfg.get('total_epochs', 24))

    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))

    runner.timestamp = timestamp

    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    runner.register_training_hooks(
        cfg.lr_config, optimizer_config,
        cfg.checkpoint_config, cfg.log_config,
        cfg.get('momentum_config', None))

    if distributed and isinstance(runner, EpochBasedRunner):
        runner.register_hook(DistSamplerSeedHook())

    if validate:
        val_dataset = custom_build_dataset(cfg.data.val, dict(test_mode=True))
        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False,
            shuffler_sampler=cfg.data.shuffler_sampler,
            nonshuffler_sampler=cfg.data.nonshuffler_sampler,
        )
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_cfg['jsonfile_prefix'] = osp.join(
            'val', cfg.work_dir, time.ctime().replace(' ', '_').replace(':', '_'))
        eval_hook = OccDistEvalHook if distributed else (EvalHook or OccDistEvalHook)
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg))

    if cfg.get('resume_from'):
        runner.resume(cfg.resume_from)
    elif cfg.get('load_from'):
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow)
