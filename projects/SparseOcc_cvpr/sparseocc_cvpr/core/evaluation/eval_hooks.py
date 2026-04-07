# Note: Considering that MMCV's EvalHook updated its interface in V1.3.16,
# in order to avoid strong version dependency, we did not directly
# inherit EvalHook but BaseDistEvalHook.
#
# Migration: mmcv.runner EvalHook → mmengine Hook 기반으로 이식
# 원본 로직은 최대한 유지합니다.

import bisect
import os.path as osp

import torch.distributed as dist
from torch.nn.modules.batchnorm import _BatchNorm


# --- 호환성 임포트 ---
try:
    from mmengine.utils import is_list_of
except ImportError:
    try:
        import mmcv
        is_list_of = mmcv.is_list_of
    except ImportError:
        def is_list_of(obj, expected_type):
            return isinstance(obj, list) and all(isinstance(x, expected_type) for x in obj)

try:
    from mmengine.hooks import Hook as BaseEvalHook
    from mmengine.hooks import Hook as BaseDistEvalHook
    _HAS_MMENGINE_HOOK = True
except ImportError:
    _HAS_MMENGINE_HOOK = False
    try:
        from mmcv.runner import DistEvalHook as BaseDistEvalHook
        from mmcv.runner import EvalHook as BaseEvalHook
    except ImportError:
        class BaseEvalHook:
            pass
        BaseDistEvalHook = BaseEvalHook


def _calc_dynamic_intervals(start_interval, dynamic_interval_list):
    assert is_list_of(dynamic_interval_list, tuple)
    dynamic_milestones = [0]
    dynamic_milestones.extend([di[0] for di in dynamic_interval_list])
    dynamic_intervals = [start_interval]
    dynamic_intervals.extend([di[1] for di in dynamic_interval_list])
    return dynamic_milestones, dynamic_intervals


class OccDistEvalHook(BaseDistEvalHook):
    def __init__(self, *args, dynamic_intervals=None, **kwargs):
        super(OccDistEvalHook, self).__init__(*args, **kwargs)
        self.use_dynamic_intervals = dynamic_intervals is not None
        if self.use_dynamic_intervals:
            self.dynamic_milestones, self.dynamic_intervals = \
                _calc_dynamic_intervals(self.interval, dynamic_intervals)

    def _decide_interval(self, runner):
        if self.use_dynamic_intervals:
            progress = runner.epoch if self.by_epoch else runner.iter
            step = bisect.bisect(self.dynamic_milestones, (progress + 1))
            self.interval = self.dynamic_intervals[step - 1]

    def before_train_epoch(self, runner):
        self._decide_interval(runner)
        super().before_train_epoch(runner)

    def before_train_iter(self, runner, **kwargs):
        self._decide_interval(runner)
        super().before_train_iter(runner, **kwargs)

    def _do_evaluate(self, runner):
        """평가 수행 및 체크포인트 저장."""
        if self.broadcast_bn_buffer:
            model = runner.model
            for name, module in model.named_modules():
                if isinstance(module, _BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)

        if not self._should_evaluate(runner):
            return

        tmpdir = self.tmpdir
        if tmpdir is None:
            tmpdir = osp.join(runner.work_dir, '.eval_hook')

        from ..apis.test import custom_multi_gpu_test

        results = custom_multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=tmpdir,
            gpu_collect=self.gpu_collect)

        if runner.rank == 0:
            print('\n')
            key_score = self.evaluate(runner, results)
            if self.save_best:
                self._save_ckpt(runner, key_score)
