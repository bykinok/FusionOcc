"""P0 CPU-only unit tests for invisible-free loss weighting (spec section 4.2).

Runs entirely on CPU (no CUDA), by design: no GPU has been designated for this
session (spec requires GPU ids + budget approval before any GPU is touched).
Tests the shared reduction primitives directly (bypassing CustomFocalLoss's
CUDA-only __init__, and bypassing mmcv's CUDA sigmoid_focal_loss op) rather
than the full CustomFocalLoss module -- see loss_inventory.json
"known_gaps_for_new_spec_requirements" for what remains untested as a result.

Writes one row per (loss, test) to research/audits/gradient_coverage.csv.
Exit code 0 iff every test PASSED (edge-case tests) / gradients matched
within tolerance (equivalence tests). Never silently skips a failure.
"""
import csv
import os
import sys

import torch
import torch.nn.functional as F

REPO_ROOT = "/workspace/FusionOcc"
sys.path.insert(0, os.path.join(REPO_ROOT, "projects", "STCOcc", "stcocc"))

from losses.focal_loss import py_sigmoid_focal_loss, _reduce_per_voxel_loss  # noqa: E402
from losses.semkitti import sem_scal_loss, geo_scal_loss  # noqa: E402

torch.manual_seed(0)

rows = []


def record(loss_name, test_name, status, detail):
    rows.append({"loss_name": loss_name, "test_name": test_name, "status": status, "detail": detail})
    print(f"[{status}] {loss_name} :: {test_name} -- {detail}")


NUM_CLASSES = 18
FREE_IDX = 17
IGNORE_IDX = 255
SHAPE = (2, 6, 6, 4)  # (B, X, Y, Z) small synthetic grid


def make_synthetic(seed=0, all_ignore=False, no_free=False, no_occupied=False,
                    all_visible=False, all_invisible=False, missing_class=None):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(SHAPE + (NUM_CLASSES,), generator=g, requires_grad=True)
    if all_ignore:
        target = torch.full(SHAPE, IGNORE_IDX, dtype=torch.long)
    else:
        target = torch.randint(0, NUM_CLASSES, SHAPE, generator=g)
        if no_free:
            target = torch.where(target == FREE_IDX, torch.tensor(0), target)
        if no_occupied:
            target = torch.full_like(target, FREE_IDX)
        if missing_class is not None:
            target = torch.where(target == missing_class, torch.tensor((missing_class + 1) % NUM_CLASSES), target)
        # sprinkle some ignore voxels too (realistic mixed case), unless caller wants none
        ignore_mask = torch.rand(SHAPE, generator=g) < 0.1
        target = torch.where(ignore_mask, torch.tensor(IGNORE_IDX), target)

    if all_visible:
        camera_mask = torch.ones(SHAPE, dtype=torch.bool)
    elif all_invisible:
        camera_mask = torch.zeros(SHAPE, dtype=torch.bool)
    else:
        camera_mask = torch.rand(SHAPE, generator=g) > 0.6  # ~40% visible
    return logits, target, camera_mask


def ce_focal_loss(logits_5d, target, voxel_weight):
    """Mirrors CustomFocalLoss.forward's own preprocessing (visible_mask select +
    one-hot), then calls the CPU-fallback py_sigmoid_focal_loss primitive directly
    -- bypassing CustomFocalLoss itself (its __init__ hard-requires CUDA for the
    BEV-radial `self.c` term, see loss_inventory.json known_gaps).

    logits_5d: (B, C, X, Y, Z), requires_grad leaf.
    target:    (B, X, Y, Z) raw class ids, IGNORE_IDX = ignore.
    voxel_weight: (B, X, Y, Z) float or None.
    Returns the scalar loss (grad flows back into logits_5d).
    """
    num_classes = logits_5d.shape[1]
    pred_flat = logits_5d.permute(0, 2, 3, 4, 1).reshape(-1, num_classes)
    target_flat = target.reshape(-1)
    visible_mask = (target_flat != IGNORE_IDX).nonzero().squeeze(-1)
    pred_sel = pred_flat[visible_mask]
    target_sel = target_flat[visible_mask].clamp(min=0, max=num_classes - 1)
    target_oh = F.one_hot(target_sel, num_classes=num_classes + 1)[:, :num_classes].float()
    vw_sel = voxel_weight.reshape(-1)[visible_mask] if voxel_weight is not None else None
    return py_sigmoid_focal_loss(pred_sel, target_oh, weight=None, voxel_weight=vw_sel)


def build_weight(target, camera_mask, lam):
    valid = target != IGNORE_IDX
    is_free = target == FREE_IDX
    invisible = ~camera_mask
    inv_free = invisible & is_free & valid
    w = torch.ones_like(target, dtype=torch.float32)
    w[inv_free] = lam
    w[~valid] = 0.0
    return w, inv_free


# ---------------------------------------------------------------------------
# 1. lambda=1 equivalence: voxel_weight=None must equal voxel_weight=ones_like
# ---------------------------------------------------------------------------
def test_lambda1_equivalence():
    logits, target, camera_mask = make_synthetic(seed=1)
    ones = torch.ones_like(target, dtype=torch.float32)
    ones[target == IGNORE_IDX] = 0.0  # matches build_weight's ~valid handling at lambda=1

    # --- CE / focal path (via the CPU-fallback reduction primitive) ---
    pred = logits.permute(0, 4, 1, 2, 3).clone().detach().requires_grad_(True)
    per_voxel = ce_focal_loss(pred, target, voxel_weight=None)
    per_voxel.backward()
    grad_none = pred.grad.clone()

    pred2 = logits.permute(0, 4, 1, 2, 3).clone().detach().requires_grad_(True)
    per_voxel2 = ce_focal_loss(pred2, target, voxel_weight=ones)
    per_voxel2.backward()
    grad_ones = pred2.grad.clone()

    val_close = torch.allclose(per_voxel.detach(), per_voxel2.detach(), atol=1e-6)
    grad_close = torch.allclose(grad_none, grad_ones, atol=1e-6)
    record("CE_focal_primitive", "lambda1_equivalence",
           "PASS" if (val_close and grad_close) else "FAIL",
           f"value_match={val_close} grad_match={grad_close} "
           f"max_grad_diff={(grad_none - grad_ones).abs().max().item():.3e}")

    # --- sem_scal / geo_scal path ---
    for name, fn, kwargs in [
        ("sem_scal", sem_scal_loss, {}),
        ("geo_scal", geo_scal_loss, {"empty_idx": FREE_IDX}),
    ]:
        p1 = logits.permute(0, 4, 1, 2, 3).clone().detach().requires_grad_(True)
        l1 = fn(p1, target, ignore_index=IGNORE_IDX, voxel_weight=None, **kwargs)
        l1.backward()
        g1 = p1.grad.clone()

        p2 = logits.permute(0, 4, 1, 2, 3).clone().detach().requires_grad_(True)
        l2 = fn(p2, target, ignore_index=IGNORE_IDX, voxel_weight=ones, **kwargs)
        l2.backward()
        g2 = p2.grad.clone()

        val_close = torch.allclose(l1.detach(), l2.detach(), atol=1e-6)
        grad_close = torch.allclose(g1, g2, atol=1e-6)
        record(name, "lambda1_equivalence",
               "PASS" if (val_close and grad_close) else "FAIL",
               f"value_match={val_close} ({l1.item():.6f} vs {l2.item():.6f}) grad_match={grad_close} "
               f"max_grad_diff={(g1 - g2).abs().max().item():.3e}")


# ---------------------------------------------------------------------------
# 2. lambda=0 partial gradient: invisible-free logit gradient must be exactly 0
# ---------------------------------------------------------------------------
def test_lambda0_zero_gradient():
    logits, target, camera_mask = make_synthetic(seed=2)
    w, inv_free = build_weight(target, camera_mask, lam=0.0)
    if inv_free.sum() == 0:
        record("ALL", "lambda0_zero_gradient", "SKIP", "synthetic batch has 0 invisible-free voxels")
        return

    # sem_scal / geo_scal
    for name, fn, kwargs in [
        ("sem_scal", sem_scal_loss, {}),
        ("geo_scal", geo_scal_loss, {"empty_idx": FREE_IDX}),
    ]:
        p = logits.permute(0, 4, 1, 2, 3).clone().detach().requires_grad_(True)
        loss = fn(p, target, ignore_index=IGNORE_IDX, voxel_weight=w, **kwargs)
        loss.backward()
        grad = p.grad  # (B, C, X, Y, Z)
        grad_at_invfree = grad.permute(0, 2, 3, 4, 1)[inv_free]  # (Ninvfree, C)
        max_abs = grad_at_invfree.abs().max().item()
        record(name, "lambda0_zero_gradient",
               "PASS" if max_abs == 0.0 else "FAIL",
               f"n_invfree_voxels={int(inv_free.sum())} max_abs_grad_at_invfree={max_abs:.3e}")

    # CE / focal primitive
    pred = logits.permute(0, 4, 1, 2, 3).clone().detach().requires_grad_(True)
    loss = ce_focal_loss(pred, target, voxel_weight=w)
    loss.backward()
    grad = pred.grad
    grad_at_invfree = grad.permute(0, 2, 3, 4, 1)[inv_free]
    max_abs = grad_at_invfree.abs().max().item()
    record("CE_focal_primitive", "lambda0_zero_gradient",
           "PASS" if max_abs == 0.0 else "FAIL",
           f"n_invfree_voxels={int(inv_free.sum())} max_abs_grad_at_invfree={max_abs:.3e}")


# ---------------------------------------------------------------------------
# 3. Edge cases: no NaN/Inf
# ---------------------------------------------------------------------------
def test_edge_cases():
    cases = {
        "all_ignore": dict(all_ignore=True),
        "no_free": dict(no_free=True),
        "no_occupied": dict(no_occupied=True),
        "all_visible": dict(all_visible=True),
        "all_invisible": dict(all_invisible=True),
        "missing_class_5": dict(missing_class=5),
    }
    for case_name, kw in cases.items():
        logits, target, camera_mask = make_synthetic(seed=3, **kw)
        w, _ = build_weight(target, camera_mask, lam=0.25)
        pred = logits.permute(0, 4, 1, 2, 3).clone().detach().requires_grad_(True)

        any_fail = False
        details = []
        for name, fn, kwargs in [
            ("sem_scal", sem_scal_loss, {}),
            ("geo_scal", geo_scal_loss, {"empty_idx": FREE_IDX}),
        ]:
            p = pred.clone().detach().requires_grad_(True)
            try:
                loss = fn(p, target, ignore_index=IGNORE_IDX, voxel_weight=w, **kwargs)
                bad = bool(torch.isnan(loss) or torch.isinf(loss))
                if not bad:
                    loss.backward()
                    bad = bool(torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
                details.append(f"{name}={'NaN/Inf' if bad else 'finite'}")
                any_fail = any_fail or bad
            except Exception as e:
                details.append(f"{name}=EXCEPTION({e})")
                any_fail = True

        try:
            p = pred.clone().detach().requires_grad_(True)
            loss = ce_focal_loss(p, target, voxel_weight=w)
            bad = bool(torch.isnan(loss) or torch.isinf(loss))
            if not bad:
                loss.backward()
                bad = bool(torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
            details.append(f"CE_focal={'NaN/Inf' if bad else 'finite'}")
            any_fail = any_fail or bad
        except Exception as e:
            details.append(f"CE_focal=EXCEPTION({e})")
            any_fail = True

        record("ALL", f"edge_case_{case_name}", "FAIL" if any_fail else "PASS", "; ".join(details))


if __name__ == "__main__":
    test_lambda1_equivalence()
    test_lambda0_zero_gradient()
    test_edge_cases()

    out_path = "/workspace/FusionOcc/research/audits/gradient_coverage.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["loss_name", "test_name", "status", "detail"])
        writer.writeheader()
        writer.writerows(rows)

    n_fail = sum(1 for r in rows if r["status"] == "FAIL")
    n_pass = sum(1 for r in rows if r["status"] == "PASS")
    n_skip = sum(1 for r in rows if r["status"] == "SKIP")
    print(f"\n{'='*60}\nSummary: {n_pass} PASS, {n_fail} FAIL, {n_skip} SKIP -> {out_path}\n{'='*60}")
    sys.exit(1 if n_fail else 0)
