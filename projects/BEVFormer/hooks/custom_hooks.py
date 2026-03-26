try:
    from mmcv.runner.hooks.hook import HOOKS, Hook
except ImportError:
    try:
        from mmengine.hooks import Hook, HOOKS
    except ImportError:
        from mmengine.hooks import Hook
        from mmdet3d.registry import HOOKS
from projects.BEVFormer.utils import run_time


@HOOKS.register_module()
class TransferWeight(Hook):
    
    def __init__(self, every_n_inters=1):
        self.every_n_inters=every_n_inters

    def after_train_iter(self, runner):
        if self.every_n_inner_iters(runner, self.every_n_inters):
            runner.eval_model.load_state_dict(runner.model.state_dict())


@HOOKS.register_module()
class DepthLossAnnealingHook(Hook):
    """Anneals the depth supervision loss_weight epoch-by-epoch.

    The hook inspects ``runner.model.depth_head.loss_weight`` (or
    ``runner.model.module.depth_head.loss_weight`` for DDP models) and sets
    it to the value defined in ``annealing_schedule`` for the current epoch.

    Args:
        annealing_schedule (list[tuple[int, float]]): Each entry is
            ``(epoch_start, weight)``, 1-indexed.  The last entry whose
            ``epoch_start <= current_epoch`` is used.  Entries are sorted
            automatically so order in the config does not matter.

    Example config::

        custom_hooks = [
            dict(
                type='DepthLossAnnealingHook',
                annealing_schedule=[
                    (1,  4.0),   # epoch  1-8 : strong geometric supervision
                    (9,  0.5),   # epoch  9-16: gentle regularisation
                    (17, 0.1),   # epoch 17-24: near-zero, occupancy focus
                ],
            )
        ]
    """

    def __init__(self, annealing_schedule):
        super().__init__()
        if not annealing_schedule:
            raise ValueError('annealing_schedule must not be empty.')
        self.annealing_schedule = sorted(annealing_schedule, key=lambda x: x[0])

    def _get_depth_head(self, runner):
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module
        return getattr(model, 'depth_head', None)

    def _resolve_weight(self, current_epoch):
        weight = self.annealing_schedule[0][1]
        for epoch_start, w in self.annealing_schedule:
            if current_epoch >= epoch_start:
                weight = w
        return weight

    def before_train(self, runner):
        self._apply(runner, epoch=1)

    def before_train_epoch(self, runner):
        current_epoch = runner.epoch + 1  # runner.epoch is 0-indexed
        self._apply(runner, epoch=current_epoch)

    def _apply(self, runner, epoch):
        depth_head = self._get_depth_head(runner)
        if depth_head is None:
            return
        weight = self._resolve_weight(epoch)
        depth_head.loss_weight = weight
        runner.logger.info(
            f'[DepthLossAnnealingHook] epoch {epoch:>2d} → '
            f'depth loss_weight = {weight}'
        )

