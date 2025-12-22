# Converted to mmengine API
from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from ...models.utils import run_time


@HOOKS.register_module()
class TransferWeight(Hook):
    """Hook to transfer weights from training model to eval model."""
    
    def __init__(self, every_n_iters=1):
        super().__init__()
        self.every_n_iters = every_n_iters

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        """Transfer weights after training iteration."""
        if self.every_n_iters > 0 and (runner.iter + 1) % self.every_n_iters == 0:
            if hasattr(runner, 'eval_model') and runner.eval_model is not None:
                runner.eval_model.load_state_dict(runner.model.state_dict())

