"""Unit test for DepthLossAnnealingHook.

Run from the workspace root:
    python projects/TPVFormer/tests/test_depth_loss_annealing_hook.py
"""

import sys
import os
import logging

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Register TPVFormer modules (needed so HOOKS registry is populated)
from mmengine.utils import import_modules_from_strings
import_modules_from_strings('projects.TPVFormer.tpvformer')

from projects.TPVFormer.tpvformer.hooks import DepthLossAnnealingHook  # noqa: E402

# ---------------------------------------------------------------------------
# Minimal stubs — lightweight runner/model mocks
# ---------------------------------------------------------------------------

class _FakeDepthHead:
    def __init__(self, initial_weight=4.0):
        self.loss_weight = initial_weight


class _FakeModel:
    """Plain model (no DDP wrapping)."""
    def __init__(self):
        self.depth_head = _FakeDepthHead(initial_weight=4.0)


class _FakeDDPModel:
    """DDP-wrapped model (has .module attribute)."""
    def __init__(self):
        self.module = _FakeModel()


class _FakeRunner:
    def __init__(self, model):
        self.model = model
        self.epoch = 0  # 0-indexed, mirrors mmengine Runner
        self.logger = logging.getLogger('test_hook')


# ---------------------------------------------------------------------------
# Schedule & expected mapping (matches the config)
# ---------------------------------------------------------------------------

SCHEDULE = [
    (1,  4.0),
    (9,  0.5),
    (17, 0.1),
]

EXPECTED = {e: w for (start, w), end in zip(
    SCHEDULE,
    [s[0] for s in SCHEDULE[1:]] + [25]
) for e in range(start, end)}


# ---------------------------------------------------------------------------
# Simulation helper
# ---------------------------------------------------------------------------

def _run_simulation(model):
    """Simulate 24 epochs; return {epoch (1-indexed) → loss_weight}."""
    hook = DepthLossAnnealingHook(annealing_schedule=SCHEDULE)
    runner = _FakeRunner(model)

    # Resolve depth_head through possible DDP wrapping
    m = model.module if hasattr(model, 'module') else model
    depth_head = m.depth_head

    results = {}

    # before_train fires once, before epoch 1
    hook.before_train(runner)
    results[1] = depth_head.loss_weight

    for epoch_idx in range(24):      # 0-indexed → epochs 1..24
        runner.epoch = epoch_idx
        hook.before_train_epoch(runner)
        results[epoch_idx + 1] = depth_head.loss_weight

    return results


# ---------------------------------------------------------------------------
# Pretty-print helper
# ---------------------------------------------------------------------------

def _check(results, label):
    print(f'\n  Epoch | Expected | Actual  | Status')
    print(f'  ------|----------|---------|-------')
    all_pass = True
    prev = None
    for epoch in range(1, 25):
        exp = EXPECTED[epoch]
        act = results[epoch]
        ok = abs(act - exp) < 1e-9
        tag = ' ← phase change' if (prev is not None and act != prev) else ''
        status = 'PASS' if ok else f'FAIL (expected {exp})'
        print(f'  {epoch:>5} | {exp:>8.1f} | {act:>7.1f} | {status}{tag}')
        if not ok:
            all_pass = False
        prev = act
    verdict = f'ALL {len(results)} epochs PASSED ✓' if all_pass else 'SOME epochs FAILED ✗'
    print(f'\n  [{label}] {verdict}')
    return all_pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format='%(message)s')


def test_plain_model():
    print('\n=== Test 1: Plain model (no DDP) ===')
    _check(_run_simulation(_FakeModel()), 'plain')


def test_ddp_model():
    print('\n=== Test 2: DDP-wrapped model (.module) ===')
    _check(_run_simulation(_FakeDDPModel()), 'DDP')


def test_no_depth_head():
    print('\n=== Test 3: Model without depth_head (should not raise) ===')
    class _NoDepthModel:
        pass
    hook = DepthLossAnnealingHook(annealing_schedule=SCHEDULE)
    runner = _FakeRunner(_NoDepthModel())
    try:
        hook.before_train(runner)
        hook.before_train_epoch(runner)
        print('  PASS — no exception when depth_head is absent')
    except Exception as e:
        print(f'  FAIL — unexpected exception: {e}')


def test_empty_schedule_raises():
    print('\n=== Test 4: Empty schedule should raise ValueError ===')
    try:
        DepthLossAnnealingHook(annealing_schedule=[])
        print('  FAIL — ValueError was not raised')
    except ValueError as e:
        print(f'  PASS — ValueError raised: {e}')


def test_unsorted_schedule():
    print('\n=== Test 5: Out-of-order entries are sorted automatically ===')
    hook = DepthLossAnnealingHook(
        annealing_schedule=[(17, 0.1), (1, 4.0), (9, 0.5)])
    assert hook.annealing_schedule[0] == (1, 4.0)
    assert hook.annealing_schedule[-1] == (17, 0.1)
    print('  PASS — schedule sorted correctly')


if __name__ == '__main__':
    test_plain_model()
    test_ddp_model()
    test_no_depth_head()
    test_empty_schedule_raises()
    test_unsorted_schedule()
    print('\n=== All tests complete ===')
