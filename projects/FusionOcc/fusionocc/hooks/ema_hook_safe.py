# Copyright (c) OpenMMLab. All rights reserved.
"""EMAHook that is safe for test: no-op when ema_model was never created (test-only run)."""
from mmengine.registry import HOOKS
from mmengine.hooks import EMAHook


@HOOKS.register_module()
class EMAHookSafeForTest(EMAHook):
    """EMAHook that does not crash in test mode when before_run was never called.

    Training: before_run() is called -> ema_model is created -> behavior same as EMAHook.
    Test: before_run() is not called -> ema_model is missing -> before_test_epoch/after_test_epoch no-op.
    """

    def before_test_epoch(self, runner) -> None:
        if getattr(self, 'ema_model', None) is not None:
            self._swap_ema_parameters()

    def after_test_epoch(self, runner, metrics=None) -> None:
        if getattr(self, 'ema_model', None) is not None:
            self._swap_ema_parameters()
