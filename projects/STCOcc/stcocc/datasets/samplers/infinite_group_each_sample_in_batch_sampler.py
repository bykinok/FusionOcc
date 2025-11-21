# Copyright (c) OpenMMLab. All rights reserved.
import copy
import itertools
import torch
import numpy as np
from torch.utils.data import Sampler
from mmengine.dist import get_dist_info, sync_random_seed
from mmdet3d.registry import DATA_SAMPLERS  # SAMPLERS 대신 DATA_SAMPLERS 사용

@DATA_SAMPLERS.register_module()  # SAMPLERS 대신 DATA_SAMPLERS 사용
class InfiniteGroupEachSampleInBatchSampler(Sampler):
    """
    Pardon this horrendous name. Basically, we want every sample to be from its own group.
    If batch size is 4 and # of GPUs is 8, each sample of these 32 should be operating on
    its own group.
    Shuffling is only done for group order, not done within groups.
    """

    def __init__(self, 
                 dataset=None,       # dataset은 선택적 인자로 변경
                 batch_size=1,
                 world_size=None,
                 rank=None,
                 seed=0,
                 sampler=None,  # MMEngine이 전달하는 인자 (사용하지 않음)
                 **kwargs):     # 기타 예상치 못한 인자들 무시

        # MMEngine이 sampler를 전달하면, 여기서 dataset 추출
        if dataset is None and sampler is not None:
            if hasattr(sampler, 'dataset'):
                dataset = sampler.dataset
            else:
                raise ValueError("dataset must be provided either directly or through sampler")
        
        if dataset is None:
            raise ValueError("dataset is required")

        _rank, _world_size = get_dist_info()
        if world_size is None:
            world_size = _world_size
        if rank is None:
            rank = _rank

        self.dataset = dataset
        self.batch_size = batch_size
        self.world_size = world_size
        self.rank = rank
        self.seed = sync_random_seed(seed)
        self.sampler = sampler  # MMEngine의 DistSamplerSeedHook이 접근하는 속성

        self.size = len(self.dataset)

        assert hasattr(self.dataset, 'flag')
        self.flag = self.dataset.flag
        self.group_sizes = np.bincount(self.flag)
        self.groups_num = len(self.group_sizes)
        self.global_batch_size = batch_size * world_size

        assert self.groups_num >= self.global_batch_size

        # Now, for efficiency, make a dict group_idx: List[dataset sample_idxs]
        self.group_idx_to_sample_idxs = {
            group_idx: np.where(self.flag == group_idx)[0].tolist()
            for group_idx in range(self.groups_num)}        

        # Get a generator per sample idx. Considering samples over all
        # GPUs, each sample position has its own generator 
        self.group_indices_per_global_sample_idx = [
            self._group_indices_per_global_sample_idx(self.rank * self.batch_size + local_sample_idx) 
            for local_sample_idx in range(self.batch_size)]
        
        # Keep track of a buffer of dataset sample idxs for each local sample idx
        self.buffer_per_local_sample = [[] for _ in range(self.batch_size)]

    def _infinite_group_indices(self):
        # 랜덤성 제거: 순차적 그룹 순서 사용
        while True:
            yield from list(range(self.groups_num))
        # g = torch.Generator()
        # g.manual_seed(self.seed)
        # while True:
        #     yield from torch.randperm(self.groups_num, generator=g).tolist()

    def _group_indices_per_global_sample_idx(self, global_sample_idx):
        yield from itertools.islice(self._infinite_group_indices(), 
                                    global_sample_idx, 
                                    None,
                                    self.global_batch_size)

    def __iter__(self):
        while True:
            curr_batch = []
            for local_sample_idx in range(self.batch_size):
                if len(self.buffer_per_local_sample[local_sample_idx]) == 0:
                    # Finished current group, refill with next group
                    new_group_idx = next(self.group_indices_per_global_sample_idx[local_sample_idx])
                    self.buffer_per_local_sample[local_sample_idx] = \
                        copy.deepcopy(
                            self.group_idx_to_sample_idxs[new_group_idx])

                curr_batch.append(self.buffer_per_local_sample[local_sample_idx].pop(0))
            
            yield curr_batch

    def __len__(self):
        """Length of base dataset."""
        return self.size
        
    def set_epoch(self, epoch):
        self.epoch = epoch