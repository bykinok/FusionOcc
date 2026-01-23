# Copyright (c) OpenMMLab. All rights reserved.
"""
SyncBN Hook for MMEngine

This hook converts BatchNorm layers to SyncBatchNorm at a specified epoch
for better distributed training performance.

Adapted from mmdet3d.core.hook.syncbncontrol for MMEngine compatibility.
"""

from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from torch.nn import SyncBatchNorm


def is_parallel(model):
    """Check if model is wrapped by DistributedDataParallel or DataParallel."""
    from torch.nn.parallel import DataParallel, DistributedDataParallel
    return isinstance(model, (DataParallel, DistributedDataParallel))


@HOOKS.register_module()
class SyncBNHook(Hook):
    """Hook to convert BatchNorm to SyncBatchNorm during training.
    
    This hook converts all BatchNorm layers in the model to SyncBatchNorm
    layers at the specified epoch. This is useful for distributed training
    to synchronize batch statistics across GPUs.
    
    Args:
        syncbn_start_epoch (int): Epoch to start using SyncBatchNorm.
            Default: 0 (convert from the beginning).
        priority (int or str): Priority of the hook. Default: 'NORMAL'.
    
    Example:
        >>> custom_hooks = [
        ...     dict(type='SyncBNHook', syncbn_start_epoch=0)
        ... ]
    """
    
    priority = 'NORMAL'
    
    def __init__(self, syncbn_start_epoch: int = 0):
        super().__init__()
        self.syncbn_start_epoch = syncbn_start_epoch
        self.is_syncbn = False
    
    def _convert_syncbn(self, runner):
        """Convert BatchNorm to SyncBatchNorm."""
        model = runner.model
        
        # Handle different model wrapping scenarios
        if is_parallel(model):
            # model is wrapped by DDP/DP
            if hasattr(model, 'module'):
                # Check if module is further wrapped
                if is_parallel(model.module):
                    # model.module.module is the actual model
                    model.module.module = SyncBatchNorm.convert_sync_batchnorm(
                        model.module.module, process_group=None)
                else:
                    # model.module is the actual model
                    model.module = SyncBatchNorm.convert_sync_batchnorm(
                        model.module, process_group=None)
        else:
            # model is not wrapped
            runner.model = SyncBatchNorm.convert_sync_batchnorm(
                runner.model, process_group=None)
        
        runner.logger.info('✅ Converted BatchNorm to SyncBatchNorm')
    
    def before_train_epoch(self, runner):
        """Convert to SyncBatchNorm at the specified epoch.
        
        Args:
            runner (Runner): The runner of the training process.
        """
        if runner.epoch >= self.syncbn_start_epoch and not self.is_syncbn:
            runner.logger.info(
                f'Converting BatchNorm to SyncBatchNorm at epoch {runner.epoch}...')
            self._convert_syncbn(runner)
            self.is_syncbn = True


# Alias for backward compatibility
SyncbnControlHook = SyncBNHook

__all__ = ['SyncBNHook', 'SyncbnControlHook']
