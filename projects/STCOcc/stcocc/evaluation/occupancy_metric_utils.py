# Copyright (c) OpenMMLab. All rights reserved.
"""Expected Calibration Error (ECE), NLL, and streaming AUROC/FPR95 for occupancy metrics."""
import numpy as np
from typing import List, Tuple

# Fixed number of score bins for streaming AUROC/FPR95 (avoids unbounded memory)
AUROC_HIST_BINS = 256


def compute_ece(conf: np.ndarray, acc: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE).

    Bins samples by confidence (equal-width bins), then ECE = sum_m (|B_m|/n) * |acc(B_m) - conf(B_m)|.

    Args:
        conf: 1D array of confidence scores (e.g. max prob or 1 - uncertainty) in [0, 1].
        acc: 1D array of 0/1 accuracy (1 if correct).
        n_bins: Number of bins. Default 10.

    Returns:
        ECE in [0, 1], or nan if inputs invalid/empty.
    """
    conf = np.asarray(conf, dtype=np.float64).reshape(-1)
    acc = np.asarray(acc, dtype=np.float64).reshape(-1)
    if conf.size != acc.size or conf.size == 0:
        return float('nan')
    n = conf.size
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        count = mask.sum()
        if count == 0:
            continue
        avg_conf = np.mean(conf[mask])
        avg_acc = np.mean(acc[mask])
        ece += (count / n) * np.abs(avg_conf - avg_acc)
    return float(ece)


def compute_nll(probs: np.ndarray, gt: np.ndarray, eps: float = 1e-10) -> float:
    """Compute mean Negative Log-Likelihood (NLL) for classification.

    NLL = -mean(log(p_true)), where p_true is the probability assigned to the true class.
    Clips probabilities to [eps, 1] to avoid log(0).

    Args:
        probs: (N, num_classes) array of class probabilities (e.g. softmax).
        gt: (N,) integer array of ground-truth class indices in [0, num_classes-1].

    Returns:
        Mean NLL (scalar), or nan if inputs invalid/empty.
    """
    probs = np.asarray(probs, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.int64).reshape(-1)
    if probs.ndim == 1:
        # single-class / binary: treat as (N,) prob of positive
        if gt.size != probs.size:
            return float('nan')
        p_true = np.where(gt > 0, probs, 1.0 - probs)
    else:
        N = probs.shape[0]
        if gt.size != N or probs.shape[1] == 0:
            return float('nan')
        # p_true[i] = probs[i, gt[i]]
        p_true = probs[np.arange(N), np.clip(gt, 0, probs.shape[1] - 1)]
    p_true = np.clip(p_true, eps, 1.0)
    nll = -np.mean(np.log(p_true))
    return float(nll)


def nll_neglog_sum_count(probs: np.ndarray, gt: np.ndarray, eps: float = 1e-10) -> Tuple[float, int]:
    """Return (sum(-log(p_true)), count) for incremental NLL. NLL = neglog_sum / count."""
    probs = np.asarray(probs, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.int64).reshape(-1)
    if probs.ndim == 1:
        if gt.size != probs.size:
            return 0.0, 0
        p_true = np.where(gt > 0, probs, 1.0 - probs)
    else:
        N = probs.shape[0]
        if gt.size != N or probs.shape[1] == 0:
            return 0.0, 0
        p_true = probs[np.arange(N), np.clip(gt, 0, probs.shape[1] - 1)]
    p_true = np.clip(p_true, eps, 1.0)
    neglog = -np.log(p_true)
    return float(np.sum(neglog)), int(p_true.size)


def ece_bin_stats_update(bin_stats: List[Tuple[float, float, int]], conf: np.ndarray, acc: np.ndarray, n_bins: int = 10) -> None:
    """Update bin_stats in place. bin_stats[i] = (sum_conf, sum_acc, count) for bin i."""
    conf = np.asarray(conf, dtype=np.float64).reshape(-1)
    acc = np.asarray(acc, dtype=np.float64).reshape(-1)
    if conf.size != acc.size or conf.size == 0:
        return
    while len(bin_stats) < n_bins:
        bin_stats.append((0.0, 0.0, 0))
    bin_edges = np.linspace(0, 1, n_bins + 1)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        c = int(mask.sum())
        if c == 0:
            continue
        sc_old, sa_old, cnt_old = bin_stats[i]
        bin_stats[i] = (
            sc_old + float(np.sum(conf[mask])),
            sa_old + float(np.sum(acc[mask])),
            cnt_old + c,
        )


def ece_from_bin_stats(bin_stats: List[Tuple[float, float, int]], n_bins: int = 10) -> float:
    """Compute ECE from pre-aggregated bin stats. bin_stats[i] = (sum_conf, sum_acc, count)."""
    n = sum(b[2] for b in bin_stats)
    if n == 0:
        return float('nan')
    ece = 0.0
    for i in range(n_bins):
        if i >= len(bin_stats) or bin_stats[i][2] == 0:
            continue
        sc, sa, cnt = bin_stats[i]
        avg_conf = sc / cnt
        avg_acc = sa / cnt
        ece += (cnt / n) * np.abs(avg_conf - avg_acc)
    return float(ece)


def auroc_histogram_update(
    hist: np.ndarray,
    y_binary: np.ndarray,
    scores: np.ndarray,
    n_bins: int = AUROC_HIST_BINS,
) -> None:
    """Update AUROC histogram in place. hist[b, 0] = count correct (y=0), hist[b, 1] = count incorrect (y=1).
    Scores in [0, 1]; higher score = more uncertain. Clips scores to [0, 1] and bins by floor(s * n_bins).
    """
    y_binary = np.asarray(y_binary, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if y_binary.size != scores.size or y_binary.size == 0 or hist.shape[0] < n_bins or hist.shape[1] < 2:
        return
    scores = np.clip(scores, 0.0, 1.0)
    bin_idx = np.clip((scores * n_bins).astype(np.int64), 0, n_bins - 1)
    for b in range(n_bins):
        mask = (bin_idx == b)
        if not mask.any():
            continue
        n0 = int((y_binary[mask] == 0).sum())
        n1 = int((y_binary[mask] == 1).sum())
        hist[b, 0] += n0
        hist[b, 1] += n1


def compute_auroc_fpr95_from_histogram(
    hist: np.ndarray,
    n_bins: int = AUROC_HIST_BINS,
) -> Tuple[float, float]:
    """Compute AUROC and FPR95 from a binned (n_bins, 2) histogram.
    hist[b, 0] = count correct (neg), hist[b, 1] = count incorrect (pos).
    Higher bin index = higher score = more uncertain.
    Returns (auroc, fpr95) or (nan, nan) if insufficient data.
    """
    hist = np.asarray(hist, dtype=np.float64)
    if hist.shape[0] < n_bins or hist.shape[1] < 2:
        return float('nan'), float('nan')
    hist = hist[:n_bins, :2].copy()
    n_neg = float(hist[:, 0].sum())
    n_pos = float(hist[:, 1].sum())
    if n_neg <= 0 or n_pos <= 0:
        return float('nan'), float('nan')
    # AUROC = P(score_pos > score_neg) + 0.5 * P(tie)
    # = (1/(n_neg*n_pos)) * ( sum_{b_pos > b_neg} hist[b_pos,1]*hist[b_neg,0] + 0.5 * sum_b hist[b,0]*hist[b,1] )
    auroc_sum = 0.0
    for b_pos in range(n_bins):
        for b_neg in range(n_bins):
            if b_pos > b_neg:
                auroc_sum += hist[b_pos, 1] * hist[b_neg, 0]
            elif b_pos == b_neg:
                auroc_sum += 0.5 * hist[b_pos, 1] * hist[b_neg, 0]
    auroc = auroc_sum / (n_neg * n_pos)
    # FPR95: threshold from high score (high bin) down; TPR = fraction of positives with score >= t
    cum_pos_high = 0.0
    cum_neg_high = 0.0
    target_tpr = 0.95
    fpr95 = 0.0
    for b in range(n_bins - 1, -1, -1):
        cum_pos_high += hist[b, 1]
        cum_neg_high += hist[b, 0]
        tpr = cum_pos_high / n_pos if n_pos > 0 else 0.0
        if tpr >= target_tpr:
            fpr95 = cum_neg_high / n_neg if n_neg > 0 else 0.0
            break
    return float(auroc), float(fpr95)
