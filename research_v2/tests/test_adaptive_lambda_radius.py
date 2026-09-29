"""E3: unit test for STCOcc.build_inv_free_voxel_weight's new radius_piecewise dict mode.

CPU-only (no CUDA needed -- .cuda() calls in the real model's __init__ are bypassed by not
instantiating the full model; we test build_inv_free_voxel_weight/_get_voxel_radius_map as a
bound method on a bare object with just the attributes they need).
"""
import sys
import os
import types
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from projects.STCOcc.stcocc.detectors.stcocc import STCOcc

EMPTY_IDX = 17


def make_fake_model(lambda_inv_free):
    m = types.SimpleNamespace()
    m.lambda_inv_free = lambda_inv_free
    m.empty_idx = EMPTY_IDX
    m._inv_free_sanity_logged = True  # suppress logging path
    m._radius_map_cache = {}
    m._get_voxel_radius_map = STCOcc._get_voxel_radius_map.__get__(m)
    m._stack_voxel_field = staticmethod(STCOcc._stack_voxel_field).__get__(m)
    m.build_inv_free_voxel_weight = STCOcc.build_inv_free_voxel_weight.__get__(m)
    return m


def test_scalar_mode_unchanged():
    m = make_fake_model(0.25)
    B, X, Y, Z = 2, 20, 20, 4
    target = torch.full((B, X, Y, Z), EMPTY_IDX, dtype=torch.long)
    target[:, :10] = 0  # half occupied
    mask = torch.zeros((B, X, Y, Z), dtype=torch.bool)  # all invisible
    w = m.build_inv_free_voxel_weight(target, mask, device='cpu')
    inv_free = (target == EMPTY_IDX) & (~mask)
    assert torch.allclose(w[inv_free], torch.full_like(w[inv_free], 0.25)), 'scalar mode broken'
    assert torch.allclose(w[~inv_free], torch.ones_like(w[~inv_free])), 'non-inv-free voxels must stay 1.0'
    print('PASS: scalar mode unchanged')


def test_radius_piecewise_mode():
    m = make_fake_model({'mode': 'radius_piecewise', 'boundary_m': 20.0, 'near': 0.10, 'far': 0.50})
    B, X, Y, Z = 1, 200, 200, 16
    target = torch.full((B, X, Y, Z), EMPTY_IDX, dtype=torch.long)
    mask = torch.zeros((B, X, Y, Z), dtype=torch.bool)  # all invisible, all free -> all inv_free
    w = m.build_inv_free_voxel_weight(target, mask, device='cpu')

    radius = m._get_voxel_radius_map((X, Y, Z), 'cpu')
    near_mask = radius < 20.0
    far_mask = ~near_mask

    assert torch.allclose(w[0][near_mask], torch.full_like(w[0][near_mask], 0.10), atol=1e-5), \
        'near-region weight should be 0.10'
    assert torch.allclose(w[0][far_mask], torch.full_like(w[0][far_mask], 0.50), atol=1e-5), \
        'far-region weight should be 0.50'
    print(f'PASS: radius_piecewise mode -- near voxels: {int(near_mask.sum())}, '
          f'far voxels: {int(far_mask.sum())}')


def test_radius_piecewise_multiscale_shapes():
    """Confirm the radius map correctly re-derives per scale (e.g. 1/2, 1/4 resolution)."""
    m = make_fake_model({'mode': 'radius_piecewise', 'boundary_m': 20.0, 'near': 0.1, 'far': 0.5})
    for X, Y, Z in [(200, 200, 16), (100, 100, 8), (50, 50, 4), (25, 25, 2)]:
        target = torch.full((1, X, Y, Z), EMPTY_IDX, dtype=torch.long)
        mask = torch.zeros((1, X, Y, Z), dtype=torch.bool)
        w = m.build_inv_free_voxel_weight(target, mask, device='cpu')
        assert w.shape == (1, X, Y, Z)
        uniq = torch.unique(w)
        assert all(min(abs(v - 0.1), abs(v - 0.5)) < 1e-5 for v in uniq.tolist()), \
            f'unexpected weight values at scale {(X,Y,Z)}: {uniq}'
    print('PASS: radius_piecewise works correctly across all 4 model scales')


def test_lambda1_equivalence_still_untouched():
    """lambda_inv_free=1.0 scalar path is unaffected by the dict-mode addition (regression)."""
    m = make_fake_model(1.0)
    B, X, Y, Z = 1, 10, 10, 2
    target = torch.randint(0, EMPTY_IDX + 1, (B, X, Y, Z))
    mask = torch.rand(B, X, Y, Z) > 0.5
    w = m.build_inv_free_voxel_weight(target, mask, device='cpu')
    valid = target != 255
    assert torch.allclose(w[valid], torch.ones_like(w[valid])), 'lambda=1.0 must remain a no-op weight map'
    print('PASS: lambda=1.0 scalar still a no-op (unaffected by dict-mode addition)')


if __name__ == '__main__':
    test_scalar_mode_unchanged()
    test_radius_piecewise_mode()
    test_radius_piecewise_multiscale_shapes()
    test_lambda1_equivalence_still_untouched()
    print('ALL PASS')
