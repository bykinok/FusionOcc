
# Copyright (c) OpenMMLab. All rights reserved.
import copy
import platform
import random
from functools import partial

import numpy as np
import torch
try:
    from mmcv.parallel import collate
except ImportError:
    # MMEngine 2.x: pseudo_collate has different behavior
    # We need a custom collate function to maintain list[dict] format
    def collate(batch):
        """
        Custom collate function to maintain MMDetection 0.x format.
        Returns list[dict] instead of dict[list].
        """

        if not isinstance(batch, list):
            raise TypeError(f'batch must be a list, but got {type(batch)}')
        
        # For each sample in batch, convert to the expected format
        # batch is a list of dicts from __getitem__
        img_metas_list = []
        points_list = []
        radar_pc_list = []
        target_list = []
        img_inputs_dict = {
            'imgs': [],
            'intrins': [],
            'sensor2egos': [],
            'post_rots': [],
            'post_trans': [],
            'flip_type': [],
            'gt_depth': []
        }
        
        for sample in batch:
            # img_metas should be a list of dicts
            img_metas_list.append(sample['img_metas'])
            points_list.append(sample['points'])
            radar_pc_list.append(sample['radar_pc'])
            # Convert target to tensor if it's numpy array
            if isinstance(sample['target'], np.ndarray):
                target_list.append(torch.from_numpy(sample['target']))
            else:
                target_list.append(sample['target'])
            
            # Stack img_inputs
            for key in img_inputs_dict.keys():
                if key in sample['img_inputs']:
                    img_inputs_dict[key].append(sample['img_inputs'][key])
        
        # Stack tensors in img_inputs
        for key in img_inputs_dict.keys():
            if len(img_inputs_dict[key]) > 0:
                if isinstance(img_inputs_dict[key][0], torch.Tensor):
                    img_inputs_dict[key] = torch.stack(img_inputs_dict[key], dim=0)
                elif isinstance(img_inputs_dict[key][0], (list, tuple)):
                    img_inputs_dict[key] = img_inputs_dict[key]
        
        # Stack target if they are tensors
        if len(target_list) > 0 and isinstance(target_list[0], torch.Tensor):
            target_stacked = torch.stack(target_list, dim=0)
        else:
            target_stacked = target_list
        
        return dict(
            img_metas=img_metas_list,
            points=points_list,
            radar_pc=radar_pc_list,
            target=target_stacked,
            img_inputs=img_inputs_dict
        )

try:
    from mmcv.runner import get_dist_info
except ImportError:
    def get_dist_info():
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()
        return 0, 1

try:
    from mmcv.utils import Registry, build_from_cfg
except ImportError:
    from mmengine.registry import Registry, build_from_cfg
from torch.utils.data import DataLoader

try:
    from mmdet.datasets.samplers import GroupSampler
except ImportError:
    # MMEngine 2.x: GroupSampler is removed
    GroupSampler = None
from .samplers.group_sampler import DistributedGroupSampler
from .samplers.distributed_sampler import DistributedSampler
from .samplers.sampler import build_sampler

def build_dataloader(dataset,
                     samples_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     dist=True,
                     shuffle=True,
                     seed=None,
                     shuffler_sampler=None,
                     nonshuffler_sampler=None,
                     **kwargs):
    """Build PyTorch DataLoader.
    In distributed training, each GPU/process has a dataloader.
    In non-distributed training, there is only one dataloader for all GPUs.
    Args:
        dataset (Dataset): A PyTorch dataset.
        samples_per_gpu (int): Number of training samples on each GPU, i.e.,
            batch size of each GPU.
        workers_per_gpu (int): How many subprocesses to use for data loading
            for each GPU.
        num_gpus (int): Number of GPUs. Only used in non-distributed training.
        dist (bool): Distributed training/test or not. Default: True.
        shuffle (bool): Whether to shuffle the data at every epoch.
            Default: True.
        kwargs: any keyword argument to be used to initialize DataLoader
    Returns:
        DataLoader: A PyTorch dataloader.
    """
    rank, world_size = get_dist_info()
    if dist:
        # DistributedGroupSampler will definitely shuffle the data to satisfy
        # that images on each GPU are in the same group
        if shuffle:
            sampler = build_sampler(shuffler_sampler if shuffler_sampler is not None else dict(type='DistributedGroupSampler'),
                                     dict(
                                         dataset=dataset,
                                         samples_per_gpu=samples_per_gpu,
                                         num_replicas=world_size,
                                         rank=rank,
                                         seed=seed)
                                     )

        else:
            sampler = build_sampler(nonshuffler_sampler if nonshuffler_sampler is not None else dict(type='DistributedSampler'),
                                     dict(
                                         dataset=dataset,
                                         num_replicas=world_size,
                                         rank=rank,
                                         shuffle=shuffle,
                                         seed=seed)
                                     )

        batch_size = samples_per_gpu if shuffle else 1
        num_workers = workers_per_gpu
    else:
        # assert False, 'not support in bevformer'
        print('WARNING!!!!, Only can be used for obtain inference speed!!!!')
        sampler = GroupSampler(dataset, samples_per_gpu) if shuffle else None
        batch_size = num_gpus * samples_per_gpu
        num_workers = num_gpus * workers_per_gpu

    init_fn = partial(
        worker_init_fn, num_workers=num_workers, rank=rank,
        seed=seed) if seed is not None else None

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
        pin_memory=False,
        worker_init_fn=init_fn,
        **kwargs)

    return data_loader


def worker_init_fn(worker_id, num_workers, rank, seed):
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
