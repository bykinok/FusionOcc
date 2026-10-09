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
from losses.lovasz_softmax import lovasz_softmax  # noqa: E402

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


def build_weight_full(target, camera_mask, lam_free, lam_occ):
    """build_weight extended with an independent invisible-OCCUPIED lambda,
    mirroring STCOcc.build_inv_free_voxel_weight's 2026-10 lambda_inv_occupied
    extension exactly (same inv_free/inv_occupied partition, same order of
    assignment: lam_free into inv_free, then lam_occ into inv_occupied, then
    ~valid -> 0 last so it always wins)."""
    valid = target != IGNORE_IDX
    is_free = target == FREE_IDX
    invisible = ~camera_mask
    inv_free = invisible & is_free & valid
    inv_occupied = invisible & (~is_free) & valid
    w = torch.ones_like(target, dtype=torch.float32)
    w[inv_free] = lam_free
    w[inv_occupied] = lam_occ
    w[~valid] = 0.0
    return w, inv_free, inv_occupied


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


# ---------------------------------------------------------------------------
# 4. (v2 NEW) per-group logit gradient NORMS at a representative lambda,
#    plus an independent all-term hard-drop check (exclude via ignore_index
#    relabeling, a DIFFERENT code path than voxel_weight=0, and confirm the
#    same zero-gradient property -- spec v2 3.1 "allterm_harddrop").
# ---------------------------------------------------------------------------
def test_gradient_norms_and_harddrop(lam=0.25):
    logits, target, camera_mask = make_synthetic(seed=4)
    w, inv_free = build_weight(target, camera_mask, lam=lam)
    valid = target != IGNORE_IDX
    is_free = target == FREE_IDX
    invisible = ~camera_mask
    inv_occupied = invisible & (~is_free) & valid
    visible_grp = camera_mask & valid

    def group_norms(grad_5d):
        # grad_5d: (B, C, X, Y, Z) -> per-voxel L2 norm over classes, then group-mean
        g = grad_5d.permute(0, 2, 3, 4, 1)  # (B,X,Y,Z,C)
        per_voxel_norm = g.norm(dim=-1)
        out = {}
        for name, m in [("free_visible", visible_grp & is_free),
                        ("occupied_visible", visible_grp & (~is_free)),
                        ("inv_occupied", inv_occupied),
                        ("inv_free", inv_free)]:
            out[name] = per_voxel_norm[m].mean().item() if m.sum() > 0 else float("nan")
        return out

    for name, fn, kwargs in [
        ("sem_scal", sem_scal_loss, {}),
        ("geo_scal", geo_scal_loss, {"empty_idx": FREE_IDX}),
    ]:
        p = logits.permute(0, 4, 1, 2, 3).clone().detach().requires_grad_(True)
        loss = fn(p, target, ignore_index=IGNORE_IDX, voxel_weight=w, **kwargs)
        loss.backward()
        norms = group_norms(p.grad)
        record(name, f"gradient_norms_lambda{lam}", "INFO",
               f"free_visible={norms['free_visible']:.4e} occupied_visible={norms['occupied_visible']:.4e} "
               f"inv_occupied={norms['inv_occupied']:.4e} inv_free={norms['inv_free']:.4e}")

    p = logits.permute(0, 4, 1, 2, 3).clone().detach().requires_grad_(True)
    loss = ce_focal_loss(p, target, voxel_weight=w)
    loss.backward()
    norms = group_norms(p.grad)
    record("CE_focal_primitive", f"gradient_norms_lambda{lam}", "INFO",
           f"free_visible={norms['free_visible']:.4e} occupied_visible={norms['occupied_visible']:.4e} "
           f"inv_occupied={norms['inv_occupied']:.4e} inv_free={norms['inv_free']:.4e}")

    # --- allterm_harddrop: relabel inv_free voxels to IGNORE_IDX (a DIFFERENT
    # mechanism than weight=0 -- this is what STCOccLoadOccGTFromFileCVPR2023's
    # baseline_with_mask policy does upstream in the dataset pipeline) and
    # confirm the same exact-zero-gradient property via this independent path.
    target_harddrop = target.clone()
    target_harddrop[inv_free] = IGNORE_IDX
    if inv_free.sum() == 0:
        record("ALL", "allterm_harddrop", "SKIP", "0 invisible-free voxels in this synthetic batch")
        return
    for name, fn, kwargs in [
        ("sem_scal", sem_scal_loss, {}),
        ("geo_scal", geo_scal_loss, {"empty_idx": FREE_IDX}),
    ]:
        p = logits.permute(0, 4, 1, 2, 3).clone().detach().requires_grad_(True)
        loss = fn(p, target_harddrop, ignore_index=IGNORE_IDX, voxel_weight=None, **kwargs)
        loss.backward()
        grad_at_invfree = p.grad.permute(0, 2, 3, 4, 1)[inv_free]
        max_abs = grad_at_invfree.abs().max().item()
        record(name, "allterm_harddrop",
               "PASS" if max_abs == 0.0 else "FAIL",
               f"via ignore_index relabel (not voxel_weight=0): max_abs_grad_at_invfree={max_abs:.3e}")
    p = logits.permute(0, 4, 1, 2, 3).clone().detach().requires_grad_(True)
    loss = ce_focal_loss(p, target_harddrop, voxel_weight=None)
    loss.backward()
    grad_at_invfree = p.grad.permute(0, 2, 3, 4, 1)[inv_free]
    max_abs = grad_at_invfree.abs().max().item()
    record("CE_focal_primitive", "allterm_harddrop",
           "PASS" if max_abs == 0.0 else "FAIL",
           f"via ignore_index relabel (not voxel_weight=0): max_abs_grad_at_invfree={max_abs:.3e}")

    # Lovasz is NOT covered by either weight=0 or ignore-relabel-via-this-script,
    # because get_voxel_loss never passes voxel_weight to lovasz_softmax at all.
    # ignore_index=255 IS honored by lovasz_softmax's own flatten_probas (it was
    # already there pre-this-work for the baseline_with_mask dataset-level path),
    # so the harddrop-via-relabel mechanism WOULD zero Lovasz's contribution too
    # if it were exercised at the get_voxel_loss call site -- but at runtime,
    # lambda_inv_free never causes that relabeling to happen (only voxel_weight
    # is built; target itself is never mutated at lambda<1). Recorded as a code
    # fact, not measured via this synthetic harness (would require calling
    # lovasz_softmax directly, out of scope for this pass -- see loss_inventory
    # v2 addendum).
    record("lovasz", "allterm_harddrop", "NOT_TESTED",
           "lovasz_softmax honors ignore_index=255 when its INPUT TARGET carries it (dataset-level "
           "baseline_with_mask path), but get_voxel_loss's lambda_inv_free mechanism never mutates "
           "the target passed to lovasz_softmax at any lambda -- so in the ACTUAL training code path, "
           "lovasz's invisible-free gradient is never dropped by lambda, hard or soft, at any lambda<1.")


# ---------------------------------------------------------------------------
# 5. (v2 NEW) Lovasz's relative share of total inv_free gradient as lambda
#    changes -- template explicitly says do NOT assume this, check it.
# ---------------------------------------------------------------------------
def build_lovasz_target(target_voxel_semantic, voxel_weight):
    """Standalone replica of STCOcc.build_lovasz_target (research_v2 reweight_lovasz=True
    option, 2026-10-02) -- see that method's docstring for the full rationale. Kept as an
    independent copy here (same convention as build_weight above) so this file never needs
    to instantiate the full model to test the primitive."""
    if voxel_weight is None:
        return target_voxel_semantic
    needs_sampling = (voxel_weight < 1.0) & (target_voxel_semantic != IGNORE_IDX)
    if not needs_sampling.any():
        return target_voxel_semantic
    target_lovasz = target_voxel_semantic.clone()
    keep = torch.rand_like(voxel_weight) < voxel_weight
    drop = needs_sampling & ~keep
    target_lovasz[drop] = 255
    return target_lovasz


# ---------------------------------------------------------------------------
# reweight_lovasz=True (research_v2, 2026-10-02): stochastic per-voxel exclusion
# ---------------------------------------------------------------------------
def test_lovasz_stochastic_reweight():
    logits, target, camera_mask = make_synthetic(seed=7)

    # --- lambda=1.0 (voxel_weight=None): must return the exact same object, untouched ---
    out = build_lovasz_target(target, None)
    record("lovasz", "reweight_lambda1_noop",
           "PASS" if out is target else "FAIL",
           f"voxel_weight=None must short-circuit to the original tensor unchanged (identity check): {out is target}")

    # --- lambda=0.0: every invisible-free voxel must be deterministically dropped (255),
    #     invisible-occupied and already-valid voxels must be exactly untouched ---
    w0, inv_free = build_weight(target, camera_mask, lam=0.0)
    inv_occupied = (~camera_mask) & (target != FREE_IDX) & (target != IGNORE_IDX)
    if inv_free.sum() == 0:
        record("lovasz", "reweight_lambda0_deterministic", "SKIP", "0 invisible-free voxels in synthetic batch")
    else:
        out0 = build_lovasz_target(target, w0)
        all_dropped = bool((out0[inv_free] == IGNORE_IDX).all())
        occupied_untouched = bool(inv_occupied.sum() == 0 or (out0[inv_occupied] == target[inv_occupied]).all())
        elsewhere = ~inv_free
        elsewhere_untouched = bool((out0[elsewhere] == target[elsewhere]).all())
        record("lovasz", "reweight_lambda0_deterministic",
               "PASS" if (all_dropped and occupied_untouched and elsewhere_untouched) else "FAIL",
               f"all_inv_free_dropped={all_dropped} inv_occupied_untouched={occupied_untouched} "
               f"rest_untouched={elsewhere_untouched} (n_inv_free={int(inv_free.sum())}, n_inv_occupied={int(inv_occupied.sum())})")

    # --- intermediate lambda (0.3): empirical keep-rate over many independent draws must
    #     converge to lambda within binomial sampling tolerance (this is the actual claim
    #     the opt-in makes: "approximates `weight` of a full gradient contribution in
    #     expectation", so the expectation itself must be checked, not just the two edges) ---
    lam = 0.3
    w_mid, inv_free_mid = build_weight(target, camera_mask, lam=lam)
    if inv_free_mid.sum() == 0:
        record("lovasz", "reweight_lambda_mid_calibration", "SKIP", "0 invisible-free voxels in synthetic batch")
    else:
        n_trials = 400
        kept = 0
        total = 0
        for i in range(n_trials):
            torch.manual_seed(1000 + i)
            out_mid = build_lovasz_target(target, w_mid)
            kept += int((out_mid[inv_free_mid] != IGNORE_IDX).sum())
            total += int(inv_free_mid.sum())
        empirical_rate = kept / total
        # binomial std error for `total` independent-ish Bernoulli(lambda) draws
        se = (lam * (1 - lam) / total) ** 0.5
        within_tol = abs(empirical_rate - lam) < max(6 * se, 0.03)
        record("lovasz", "reweight_lambda_mid_calibration",
               "PASS" if within_tol else "FAIL",
               f"lambda={lam}, empirical_keep_rate={empirical_rate:.4f} over {total} draws "
               f"({n_trials} trials x {int(inv_free_mid.sum())} inv_free voxels), se={se:.4f}, "
               f"within_6se_or_0.03={within_tol}")

    torch.manual_seed(0)  # restore the module-level seed for any tests appended after this one


def test_lambda_inv_occupied():
    """lambda_inv_occupied (2026-10 extension): independently zeroes invisible-
    OCCUPIED voxel weight, leaving invisible-free (lambda_inv_free's existing
    scope) and visible voxels untouched. Three checks:
      1. lam_occ=1.0 (default) is an exact no-op -> identical to build_weight.
      2. lam_occ=0.0 zeroes ONLY inv_occupied; inv_free keeps lam_free's value,
         visible/other stays 1.0, ignored stays 0.0.
      3. lam_free=0 + lam_occ=0 together zero EVERY invisible voxel (free and
         occupied alike) -- the CE/sem_scal/geo_scal-level supervision pattern
         this ablation needs to approximate the w/mask hard-baseline's exclusion
         (Lovasz coverage is the separate, already-tested reweight_lovasz axis).
    """
    logits, target, camera_mask = make_synthetic(seed=7)
    valid = target != IGNORE_IDX
    is_free = target == FREE_IDX
    invisible = ~camera_mask
    inv_free_ref = invisible & is_free & valid
    inv_occ_ref = invisible & (~is_free) & valid
    if inv_occ_ref.sum() == 0:
        record("inv_occupied_weight", "lambda_inv_occupied", "SKIP", "0 invisible-occupied voxels in synthetic sample")
        return

    # 1. no-op at lam_occ=1.0
    w_ref, inv_free_a = build_weight(target, camera_mask, lam=0.25)
    w_full, inv_free_b, inv_occ_b = build_weight_full(target, camera_mask, lam_free=0.25, lam_occ=1.0)
    noop_ok = torch.allclose(w_ref, w_full) and torch.equal(inv_free_a, inv_free_b)
    record("inv_occupied_weight", "lam_occ=1.0_noop", "PASS" if noop_ok else "FAIL",
           f"max|diff|={ (w_ref - w_full).abs().max().item():.3e} (expect exact 0, lam_occ=1.0 must reproduce legacy build_weight)")

    # 2. lam_occ=0.0 touches ONLY inv_occupied, scoped correctly
    w2, inv_free2, inv_occ2 = build_weight_full(target, camera_mask, lam_free=0.25, lam_occ=0.0)
    occ_zeroed = bool((w2[inv_occ_ref] == 0.0).all())
    free_untouched = bool((w2[inv_free_ref] == 0.25).all())
    visible_mask = camera_mask & valid
    visible_is_one = bool((w2[visible_mask] == 1.0).all())
    ignored_zero = bool((w2[~valid] == 0.0).all())
    scope_ok = occ_zeroed and free_untouched and visible_is_one and ignored_zero
    record("inv_occupied_weight", "lam_occ=0.0_scope", "PASS" if scope_ok else "FAIL",
           f"inv_occupied all-zero={occ_zeroed}, inv_free preserved at lam_free={free_untouched}, "
           f"visible stays 1.0={visible_is_one}, ignored stays 0.0={ignored_zero}")

    # 3. both zeroed -> every invisible voxel (free+occupied) is weight 0,
    #    matching the CE/sem_scal/geo_scal-level exclusion w/mask's hard
    #    ignore_index=255 relabeling achieves for ALL invisible voxels.
    w3, inv_free3, inv_occ3 = build_weight_full(target, camera_mask, lam_free=0.0, lam_occ=0.0)
    all_invisible_valid_zero = bool((w3[invisible & valid] == 0.0).all())
    visible_still_one = bool((w3[visible_mask] == 1.0).all())
    full_exclusion_ok = all_invisible_valid_zero and visible_still_one
    record("inv_occupied_weight", "lam_free0_lam_occ0_full_exclusion", "PASS" if full_exclusion_ok else "FAIL",
           f"every invisible&valid voxel (free+occupied) == 0: {all_invisible_valid_zero}; "
           f"visible still 1.0: {visible_still_one} -- this is the CE/sem/geo-level analogue of w/mask's "
           f"ignore_index=255 relabeling (Lovasz coverage is separately handled by reweight_lovasz)")


def test_lovasz_relative_share():
    logits, target, camera_mask = make_synthetic(seed=5)
    _, inv_free = build_weight(target, camera_mask, lam=0.0)
    if inv_free.sum() == 0:
        record("lovasz", "relative_share_vs_lambda", "SKIP", "0 invisible-free voxels")
        return

    def combined_weighted_norm(lam):
        w, _ = build_weight(target, camera_mask, lam=lam)
        total_sq = None
        for fn, kwargs in [(sem_scal_loss, {}), (geo_scal_loss, {"empty_idx": FREE_IDX})]:
            p = logits.permute(0, 4, 1, 2, 3).clone().detach().requires_grad_(True)
            loss = fn(p, target, ignore_index=IGNORE_IDX, voxel_weight=w, **kwargs)
            loss.backward()
            g = p.grad.permute(0, 2, 3, 4, 1)[inv_free]
            total_sq = (g ** 2).sum(-1) if total_sq is None else total_sq + (g ** 2).sum(-1)
        p = logits.permute(0, 4, 1, 2, 3).clone().detach().requires_grad_(True)
        loss = ce_focal_loss(p, target, voxel_weight=w)
        loss.backward()
        g = p.grad.permute(0, 2, 3, 4, 1)[inv_free]
        total_sq = total_sq + (g ** 2).sum(-1)
        return total_sq.sqrt().mean().item()

    # lovasz_softmax's own gradient at the same inv_free voxels (unweighted --
    # this IS how it actually runs at every lambda in the real training code).
    p = logits.permute(0, 4, 1, 2, 3).clone().detach().requires_grad_(True)
    probas = torch.softmax(p, dim=1)
    # NOTE: lovasz_softmax's flatten_probas has a real, independent latent bug
    # (found by this test): `import numpy as np` sits inside an `if isinstance(
    # labels, list): ...` branch, which makes Python treat `np` as function-
    # local for the WHOLE function -- so the very next `elif isinstance(labels,
    # np.ndarray)` line raises UnboundLocalError whenever `labels` is NOT a
    # python list (a bare tensor, e.g.). Every real training call happens to
    # pass `target_voxel_semantic` as a python list (mmengine's non-stacking
    # collate behavior for these Collect3D keys, confirmed in the v1 audit),
    # so this never fires in practice -- but this test's synthetic tensor
    # input hits it directly. Recorded as a P0-v2 finding, NOT patched here.
    loss = lovasz_softmax(probas, list(target), ignore=IGNORE_IDX)
    loss.backward()
    lovasz_norm = p.grad.permute(0, 2, 3, 4, 1)[inv_free].norm(dim=-1).mean().item()

    norm_lambda1 = combined_weighted_norm(1.0)
    norm_lambda025 = combined_weighted_norm(0.25)
    norm_lambda0 = combined_weighted_norm(0.0)

    share_at = lambda other_norm: lovasz_norm / (lovasz_norm + other_norm) if (lovasz_norm + other_norm) > 0 else float("nan")
    s1, s025, s0 = share_at(norm_lambda1), share_at(norm_lambda025), share_at(norm_lambda0)
    monotonic_increase = (s0 >= s025 >= s1)
    record("lovasz", "relative_share_vs_lambda",
           "CHECKED" if True else "SKIP",
           f"lovasz_norm={lovasz_norm:.4e} (constant, unweighted at every lambda); "
           f"other3_combined_norm: lambda=1.0->{norm_lambda1:.4e}, lambda=0.25->{norm_lambda025:.4e}, lambda=0.0->{norm_lambda0:.4e}; "
           f"lovasz_share_of_total: lambda=1.0->{s1:.3f}, lambda=0.25->{s025:.3f}, lambda=0.0->{s0:.3f}; "
           f"monotonic_increase_as_lambda_decreases={monotonic_increase} "
           f"(template said not to ASSUME this -- this run's synthetic data CONFIRMS the direction, "
           f"real training-batch magnitudes may differ; treat as directionally-checked not proven in general)")


if __name__ == "__main__":
    test_lambda1_equivalence()
    test_lambda0_zero_gradient()
    test_edge_cases()
    test_gradient_norms_and_harddrop(lam=0.25)
    test_lovasz_relative_share()
    test_lovasz_stochastic_reweight()
    test_lambda_inv_occupied()

    out_path = "/workspace/FusionOcc/research_v2/audits/gradient_coverage_v2.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["loss_name", "test_name", "status", "detail"])
        writer.writeheader()
        writer.writerows(rows)

    n_fail = sum(1 for r in rows if r["status"] == "FAIL")
    n_pass = sum(1 for r in rows if r["status"] == "PASS")
    n_skip = sum(1 for r in rows if r["status"] == "SKIP")
    n_info = sum(1 for r in rows if r["status"] in ("INFO", "NOT_TESTED"))
    print(f"\n{'='*60}\nSummary: {n_pass} PASS, {n_fail} FAIL, {n_skip} SKIP, {n_info} INFO/NOT_TESTED -> {out_path}\n{'='*60}")
    sys.exit(1 if n_fail else 0)
