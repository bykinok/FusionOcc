#!/usr/bin/env python3
"""
SparseOcc 전용 불확실성 보정(calibration) 평가 스크립트.

핵심 차이점 (vs compute_metrics_from_file_v2.py):
  - 평가를 **sparse 위치(모델이 occupied로 예측한 ~32K voxel)에서만** 수행
  - dense 채움(non-predicted → free class confidence=1) 시 발생하는
    False Negative 지배 문제(AUROC < 50%) 해결
  - GT=free인 sparse 위치(FP)는 선택적으로 포함 가능

두 가지 입력 포맷 지원:
  Mode A (concatenated npz):
    export_occ_logits.py 기본 출력물 사용
    Keys: logits [N_total, C], gt [N_total], indices [N_total, 3]
    Usage:
      python tools/compute_calibration_sparse.py \
        --npz work_dirs/.../sparseocc_occ_logits_val_calib_train.npz

  Mode B (per-sample npz directory):
    export_occ_logits.py --per-sample-dir 출력물 사용
    Keys: probs [N_sparse, C], sparse_indices [N_sparse, 3],
          gt_full [H, W, Z], mask_camera [H, W, Z]
    Usage:
      python tools/compute_calibration_sparse.py \
        --per-sample-dir work_dirs/.../sparseocc_per_sample/

두 모드 모두 동일한 메트릭을 계산:
  - AUROC / FPR95 (uncertainty_msp, uncertainty_entropy)
  - ECE (15-bin)
  - NLL
  - 거리별 분석 (indices 있을 때)
  - 높이별 분석 (indices 있을 때)
  - 클래스별 AUROC/FPR95 (indices 있을 때)
"""
import argparse
import glob
import os
import sys

import numpy as np

# ── AUROC / FPR95 ────────────────────────────────────────────────────
def _compute_auroc_fpr95(y_incorrect: np.ndarray, score: np.ndarray):
    """AUROC 및 FPR95 계산.

    Args:
        y_incorrect: [N] int, 1=오류, 0=정답
        score: [N] float, 불확실성 점수 (높을수록 오류일 것으로 예상)

    Returns:
        auroc (float), fpr95 (float)
    """
    try:
        from sklearn.metrics import roc_auc_score, roc_curve
    except ImportError:
        print('[ERROR] sklearn이 필요합니다: pip install scikit-learn')
        return float('nan'), float('nan')

    if len(np.unique(y_incorrect)) < 2:
        return float('nan'), float('nan')

    auroc = float(roc_auc_score(y_incorrect, score))
    fpr, tpr, _ = roc_curve(y_incorrect, score)
    idx = np.searchsorted(tpr, 0.95)
    fpr95 = float(fpr[min(idx, len(fpr) - 1)])
    return auroc, fpr95


# ── ECE ─────────────────────────────────────────────────────────────
def _compute_ece(probs: np.ndarray, gt: np.ndarray, n_bins: int = 15) -> float:
    """ECE (Expected Calibration Error).

    Args:
        probs: [N, C] softmax 확률
        gt: [N] GT 레이블
        n_bins: 빈 수

    Returns:
        ECE (0~1 float)
    """
    confidence = probs.max(axis=-1)
    pred_class = probs.argmax(axis=-1)
    correct = (pred_class == gt).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(gt)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidence >= lo) & (confidence < hi)
        if mask.sum() == 0:
            continue
        avg_conf = confidence[mask].mean()
        avg_acc  = correct[mask].mean()
        ece += (mask.sum() / n) * abs(avg_conf - avg_acc)
    return float(ece)


# ── NLL ─────────────────────────────────────────────────────────────
def _compute_nll(probs: np.ndarray, gt: np.ndarray, eps: float = 1e-9) -> float:
    """평균 NLL.

    Args:
        probs: [N, C]
        gt: [N] int, 유효한 클래스 인덱스

    Returns:
        평균 NLL (float)
    """
    n = len(gt)
    if n == 0:
        return float('nan')
    p_gt = probs[np.arange(n), gt.astype(int)].clip(eps, 1.0)
    return float(-np.log(p_gt).mean())


# ── 거리/높이 계산 (point cloud range 기준) ──────────────────────────
def _voxel_to_radius(indices: np.ndarray,
                     occ_size=(200, 200, 16),
                     pc_range=(-40, -40, -1.0, 40, 40, 5.4)) -> np.ndarray:
    """sparse_indices [N, 3] → 각 voxel의 XY 거리(m) [N]."""
    H, W, Z = occ_size
    x_min, y_min = pc_range[0], pc_range[1]
    x_max, y_max = pc_range[3], pc_range[4]
    vox_x = (x_max - x_min) / H
    vox_y = (y_max - y_min) / W

    cx = x_min + (indices[:, 0] + 0.5) * vox_x
    cy = y_min + (indices[:, 1] + 0.5) * vox_y
    return np.sqrt(cx ** 2 + cy ** 2)


def _voxel_to_height(indices: np.ndarray,
                     occ_size=(200, 200, 16),
                     pc_range=(-40, -40, -1.0, 40, 40, 5.4)) -> np.ndarray:
    """sparse_indices [N, 3] → 각 voxel의 실제 z 좌표(m) [N]."""
    Z = occ_size[2]
    z_min, z_max = pc_range[2], pc_range[5]
    vox_z = (z_max - z_min) / Z
    return z_min + (indices[:, 2] + 0.5) * vox_z


# ── softmax with optional temperature ────────────────────────────────
def _logits_to_probs(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """log(semseg) → softmax 확률. temperature > 1이면 분포를 부드럽게 만듦.

    temperature scaling: softmax(logits / T)
      T > 1 → 분포 softening (overconfident 모델 보정)
      T < 1 → 분포 sharpening
      T = 1 → 원본 유지
    """
    scaled = logits / max(temperature, 1e-6)
    v = scaled - scaled.max(axis=-1, keepdims=True)
    exp_v = np.exp(v)
    return (exp_v / (exp_v.sum(axis=-1, keepdims=True) + 1e-9)).astype(np.float32)


# ── Mode A: concatenated npz 로드 ────────────────────────────────────
def _load_concatenated(npz_path: str, temperature: float = 1.0):
    """export_occ_logits.py 기본 출력 로드."""
    data = np.load(npz_path, allow_pickle=True)
    logits  = data['logits'].astype(np.float32)  # [N, C]
    gt      = data['gt'].astype(np.int64)         # [N]
    indices = data.get('indices')                 # [N, 3] or None

    probs = _logits_to_probs(logits, temperature=temperature)

    if indices is not None:
        indices = indices.astype(np.int32)
    return probs, gt, indices


# ── Mode B: per-sample npz 디렉토리 로드 ────────────────────────────
def _load_per_sample(per_sample_dir: str, use_mask: bool = True, temperature: float = 1.0):
    """per-sample npz 디렉토리 → 누적 probs, gt, indices."""
    files = sorted(glob.glob(os.path.join(per_sample_dir, 'sample_*.npz')))
    if not files:
        raise FileNotFoundError(f'sample_*.npz 파일 없음: {per_sample_dir}')
    print(f'[per-sample 모드] {len(files)}개 파일 로드 중...')

    all_probs, all_gt, all_indices = [], [], []
    for fp in files:
        d = np.load(fp, allow_pickle=True)
        probs_s   = d['probs'].astype(np.float32)      # [N_sparse, C] 이미 정규화됨
        spi       = d['sparse_indices'].astype(np.int32)  # [N_sparse, 3]
        gt_full   = d['gt_full']                         # [H, W, Z]
        mask_cam  = d.get('mask_camera')                 # [H, W, Z] or None

        xi, yi, zi = spi[:, 0].astype(int), spi[:, 1].astype(int), spi[:, 2].astype(int)

        # gt_full 범위 검증
        H, W, Z = gt_full.shape
        coord_ok = ((xi >= 0) & (xi < H) & (yi >= 0) & (yi < W) & (zi >= 0) & (zi < Z))
        if not coord_ok.all():
            n_bad = int((~coord_ok).sum())
            print(f'  [WARNING] {os.path.basename(fp)}: {n_bad}개 out-of-bounds, 제외')
            probs_s = probs_s[coord_ok]
            spi     = spi[coord_ok]
            xi, yi, zi = xi[coord_ok], yi[coord_ok], zi[coord_ok]

        gt_sp = gt_full[xi, yi, zi]
        if use_mask and mask_cam is not None:
            mask_sp = mask_cam.astype(bool)[xi, yi, zi]
        else:
            mask_sp = np.ones(len(gt_sp), dtype=bool)

        if mask_sp.sum() == 0:
            continue
        p = probs_s[mask_sp]
        # per-sample probs는 이미 정규화됨. temperature scaling 적용 (T≠1일 때)
        if abs(temperature - 1.0) > 1e-6:
            # probs → log(probs)로 역변환 후 T 적용 (geometric scaling과 동일)
            log_p = np.log(p.clip(1e-9, 1.0))
            p = _logits_to_probs(log_p, temperature=temperature)
        all_probs.append(p)
        all_gt.append(gt_sp[mask_sp])
        all_indices.append(spi[mask_sp])

    if not all_probs:
        raise RuntimeError('유효한 데이터 없음')

    probs   = np.concatenate(all_probs, axis=0)
    gt      = np.concatenate(all_gt,    axis=0).astype(np.int64)
    indices = np.concatenate(all_indices, axis=0).astype(np.int32)
    return probs, gt, indices


# ── 메인 평가 로직 ────────────────────────────────────────────────────
def _evaluate(probs: np.ndarray, gt: np.ndarray, indices,
              free_class: int = 17,
              include_free_gt: bool = False,
              pc_range=(-40, -40, -1.0, 40, 40, 5.4),
              occ_size=(200, 200, 16)):
    """sparse 위치에서 불확실성 메트릭 계산.

    Args:
        probs:  [N, C] 정규화 확률 (semseg_norm)
        gt:     [N] GT 레이블
        indices:[N, 3] or None  voxel 좌표
        free_class: free class 인덱스
        include_free_gt: True이면 GT=free인 sparse 위치(FP)도 포함
    """
    print(f'\n  총 sparse voxel 수 : {len(probs):,}')

    # ── 필터링 ──────────────────────────────────────────────────────
    if include_free_gt:
        mask = np.ones(len(gt), dtype=bool)
    else:
        mask = (gt != free_class)
        print(f'  GT=free 제외 후    : {mask.sum():,}  (GT=free: {(~mask).sum():,}개 제외)')

    probs_f   = probs[mask]    # [M, C]
    gt_f      = gt[mask]       # [M]
    indices_f = indices[mask] if indices is not None else None

    if len(gt_f) == 0:
        print('  [ERROR] 유효한 voxel 없음 (GT 필터링 후 0개)')
        return {}

    # ── 기본 통계 ────────────────────────────────────────────────────
    pred_cls  = probs_f.argmax(axis=-1)
    correct   = (pred_cls == gt_f)
    accuracy  = correct.mean()
    print(f'  정확도 (occupied GT): {accuracy*100:.2f}%')

    # ── 불확실성 계산 ────────────────────────────────────────────────
    msp     = probs_f.max(axis=-1)
    unc_msp = 1.0 - msp
    eps     = 1e-9
    unc_ent = -(probs_f * np.log(probs_f + eps)).sum(axis=-1)

    y_incorrect = (~correct).astype(int)

    print(f'\n  uncertainty_msp:  mean={unc_msp.mean():.4f}  std={unc_msp.std():.4f}')
    print(f'  uncertainty_ent:  mean={unc_ent.mean():.4f}  std={unc_ent.std():.4f}')
    print(f'  오류율 (y_incorrect=1): {y_incorrect.mean()*100:.2f}%')

    # ── 전체 AUROC/FPR95 ────────────────────────────────────────────
    auroc_msp,  fpr95_msp  = _compute_auroc_fpr95(y_incorrect, unc_msp)
    auroc_ent,  fpr95_ent  = _compute_auroc_fpr95(y_incorrect, unc_ent)
    ece   = _compute_ece(probs_f, gt_f)
    nll   = _compute_nll(probs_f, gt_f)

    metrics = {
        'accuracy':    accuracy,
        'AUROC_msp':   auroc_msp,
        'FPR95_msp':   fpr95_msp,
        'AUROC_ent':   auroc_ent,
        'FPR95_ent':   fpr95_ent,
        'ECE':         ece,
        'NLL':         nll,
    }

    # ── 거리별 분석 ─────────────────────────────────────────────────
    if indices_f is not None:
        radius = _voxel_to_radius(indices_f, occ_size=occ_size, pc_range=pc_range)
        r_bins = [(0, 20), (20, 35), (35, 50)]
        r_results = {}
        for lo, hi in r_bins:
            r_mask = (radius >= lo) & (radius < hi)
            if r_mask.sum() < 10:
                continue
            ra, rf = _compute_auroc_fpr95(y_incorrect[r_mask], unc_msp[r_mask])
            ece_r  = _compute_ece(probs_f[r_mask], gt_f[r_mask])
            nll_r  = _compute_nll(probs_f[r_mask], gt_f[r_mask])
            r_results[f'{lo}-{hi}m'] = dict(
                n=int(r_mask.sum()),
                AUROC_msp=ra, FPR95_msp=rf,
                ECE=ece_r, NLL=nll_r,
            )
        metrics['radius'] = r_results

    # ── 높이별 분석 ─────────────────────────────────────────────────
    if indices_f is not None:
        z_min, z_max = pc_range[2], pc_range[5]
        height = _voxel_to_height(indices_f, occ_size=occ_size, pc_range=pc_range)
        # compute_metrics_from_file_v2.py 와 동일한 높이 구간
        h_bins = [(z_min, z_min + 2), (z_min + 2, z_min + 4), (z_min + 4, z_max + 1)]
        h_labels = ['0-2m', '2-4m', '4-6m+']
        h_results = {}
        for (lo, hi), label in zip(h_bins, h_labels):
            h_mask = (height >= lo) & (height < hi)
            if h_mask.sum() < 10:
                continue
            ha, hf = _compute_auroc_fpr95(y_incorrect[h_mask], unc_msp[h_mask])
            ece_h  = _compute_ece(probs_f[h_mask], gt_f[h_mask])
            nll_h  = _compute_nll(probs_f[h_mask], gt_f[h_mask])
            h_results[label] = dict(
                n=int(h_mask.sum()),
                z_range=f'{lo:.1f}~{min(hi, z_max):.1f}m',
                AUROC_msp=ha, FPR95_msp=hf,
                ECE=ece_h, NLL=nll_h,
            )
        metrics['height'] = h_results

    # ── 클래스별 AUROC/FPR95/ECE ────────────────────────────────────
    cls_results = {}
    n_classes = probs_f.shape[1]
    for cls_id in range(n_classes):
        cls_mask = (gt_f == cls_id)
        if cls_mask.sum() < 10:
            continue
        # 해당 클래스 GT에서 오답(다른 클래스로 예측) 여부
        y_inc_cls = y_incorrect[cls_mask]
        ece_cls = _compute_ece(probs_f[cls_mask], gt_f[cls_mask])
        if len(np.unique(y_inc_cls)) < 2:
            cls_results[cls_id] = dict(
                n=int(cls_mask.sum()),
                AUROC_msp=float('nan'), FPR95_msp=float('nan'),
                ECE=ece_cls,
            )
            continue
        ca, cf = _compute_auroc_fpr95(y_inc_cls, unc_msp[cls_mask])
        cls_results[cls_id] = dict(
            n=int(cls_mask.sum()),
            AUROC_msp=ca, FPR95_msp=cf,
            ECE=ece_cls,
        )
    metrics['per_class'] = cls_results

    return metrics


_OCC_CLASS_NAMES = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
    'vegetation',
]


# ── 출력 포맷 ─────────────────────────────────────────────────────────
def _print_results(metrics: dict, tag: str = ''):
    title = f'[Sparse-Only Calibration{" " + tag if tag else ""}]'
    print(f'\n{"="*60}')
    print(f'  {title}')
    print(f'{"="*60}')
    if not metrics:
        print('  계산 결과 없음')
        return

    print(f"  정확도      : {metrics.get('accuracy', float('nan'))*100:.2f}%")
    print()

    # ── 전체 AUROC/FPR95 ────────────────────────────────────────────
    hdr = f"  {'Measure':>22s} | {'AUROC %':>8s} | {'FPR95 %':>8s}"
    sep = '  ' + '-' * 44
    print(hdr); print(sep)
    for key, label in [('msp', 'uncertainty_msp'), ('ent', 'uncertainty_entropy')]:
        a = metrics.get(f'AUROC_{key}', float('nan'))
        f = metrics.get(f'FPR95_{key}', float('nan'))
        print(f"  {label:>22s} | {a*100:>7.2f} | {f*100:>7.2f}")
    print()
    print(f"  ECE  : {metrics.get('ECE', float('nan'))*100:.2f}%")
    print(f"  NLL  : {metrics.get('NLL', float('nan')):.4f}")

    # ── 거리별 ──────────────────────────────────────────────────────
    if 'radius' in metrics and metrics['radius']:
        print()
        print(f"===> 거리별 AUROC/FPR95/ECE/NLL Summary:")
        print(f"  {'Range':>10s} | {'N':>10s} | {'AUROC_msp':>9s} | {'FPR95_msp':>9s} | {'ECE %':>7s} | {'NLL':>7s}")
        print('  ' + '-' * 67)
        for rng, rv in metrics['radius'].items():
            print(f"  {rng:>10s} | {rv['n']:>10,d} | "
                  f"{rv['AUROC_msp']*100:>8.2f} | "
                  f"{rv['FPR95_msp']*100:>8.2f} | "
                  f"{rv['ECE']*100:>6.2f} | "
                  f"{rv['NLL']:>6.4f}")

    # ── 높이별 ──────────────────────────────────────────────────────
    if 'height' in metrics and metrics['height']:
        print()
        print(f"===> 높이별 AUROC/FPR95/ECE/NLL Summary:")
        print(f"  {'Height':>8s} | {'Actual Z':>12s} | {'N':>10s} | {'AUROC_msp':>9s} | {'FPR95_msp':>9s} | {'ECE %':>7s} | {'NLL':>7s}")
        print('  ' + '-' * 80)
        for hlabel, hv in metrics['height'].items():
            print(f"  {hlabel:>8s} | {hv['z_range']:>12s} | {hv['n']:>10,d} | "
                  f"{hv['AUROC_msp']*100:>8.2f} | "
                  f"{hv['FPR95_msp']*100:>8.2f} | "
                  f"{hv['ECE']*100:>6.2f} | "
                  f"{hv['NLL']:>6.4f}")

    # ── 클래스별 AUROC/FPR95/ECE ────────────────────────────────────
    if 'per_class' in metrics and metrics['per_class']:
        cls_res = metrics['per_class']
        print()
        print(f"===> 클래스별 AUROC/FPR95/ECE (uncertainty_msp):")
        print(f"  {'Class':>22s} | {'N':>10s} | {'AUROC %':>8s} | {'FPR95 %':>8s} | {'ECE %':>7s}")
        print('  ' + '-' * 68)
        auroc_vals, ece_vals = [], []
        for cls_id, cv in sorted(cls_res.items()):
            name = _OCC_CLASS_NAMES[cls_id] if cls_id < len(_OCC_CLASS_NAMES) else f'class_{cls_id}'
            auroc_str = f"{cv['AUROC_msp']*100:>7.2f}" if not np.isnan(cv['AUROC_msp']) else f"{'N/A':>7s}"
            fpr95_str = f"{cv['FPR95_msp']*100:>7.2f}" if not np.isnan(cv['FPR95_msp']) else f"{'N/A':>7s}"
            print(f"  {name:>22s} | {cv['n']:>10,d} | "
                  f"{auroc_str} | "
                  f"{fpr95_str} | "
                  f"{cv['ECE']*100:>6.2f}")
            if not np.isnan(cv['AUROC_msp']):
                auroc_vals.append(cv['AUROC_msp'])
            ece_vals.append(cv['ECE'])
        print('  ' + '-' * 68)
        mean_auroc = float(np.mean(auroc_vals)) if auroc_vals else float('nan')
        mean_ece   = float(np.mean(ece_vals))   if ece_vals   else float('nan')
        print(f"  {'mean':>22s} | {'':>10s} | {mean_auroc*100:>7.2f} | {'':>7s} | {mean_ece*100:>6.2f}")

    print(f'\n{"="*60}')


# ── main ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='SparseOcc sparse-only 불확실성 보정 평가'
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument('--npz', type=str,
                     help='[Mode A] export_occ_logits.py concatenated .npz 경로')
    src.add_argument('--per-sample-dir', type=str,
                     help='[Mode B] export_occ_logits.py --per-sample-dir 디렉토리')

    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Temperature scaling 값 (기본: 1.0 = uncalibrated). '
                             'train_temperature.py 학습 결과 T 값을 입력하면 calibrated 평가.')
    parser.add_argument('--free-class', type=int, default=17,
                        help='Free class 인덱스 (기본: 17)')
    parser.add_argument('--include-free-gt', action='store_true',
                        help='GT=free인 sparse 위치(False Positive)도 포함')
    parser.add_argument('--no-mask', action='store_true',
                        help='[Mode B] camera mask 미적용 (기본: 적용)')
    parser.add_argument('--pc-range', nargs=6, type=float,
                        default=[-40, -40, -1.0, 40, 40, 5.4],
                        help='Point cloud range (기본: -40 -40 -1 40 40 5.4)')
    parser.add_argument('--occ-size', nargs=3, type=int, default=[200, 200, 16],
                        metavar=('H', 'W', 'Z'),
                        help='격자 크기 (기본: 200 200 16)')
    parser.add_argument('--save', type=str, default=None,
                        help='결과를 .txt 파일로 저장')
    args = parser.parse_args()

    # ── 데이터 로드 ──────────────────────────────────────────────────
    T = args.temperature
    if abs(T - 1.0) > 1e-6:
        print(f'[Temperature Scaling] T={T:.4f} 적용')
    else:
        print(f'[Uncalibrated] T=1.0 (temperature scaling 미적용)')

    if args.npz:
        print(f'[Mode A] 로드: {args.npz}')
        probs, gt, indices = _load_concatenated(args.npz, temperature=T)
    else:
        print(f'[Mode B] 로드: {args.per_sample_dir}')
        probs, gt, indices = _load_per_sample(
            args.per_sample_dir, use_mask=not args.no_mask, temperature=T
        )

    print(f'probs  shape: {probs.shape}')
    print(f'gt     shape: {gt.shape}')
    print(f'indices: {indices.shape if indices is not None else "없음"}')

    # ── 평가 ────────────────────────────────────────────────────────
    calib_tag = f'T={T:.4f}' if abs(T - 1.0) > 1e-6 else 'uncalibrated'
    tag = f'GT=free {"포함" if args.include_free_gt else "제외"}, {calib_tag}'
    metrics = _evaluate(
        probs, gt, indices,
        free_class=args.free_class,
        include_free_gt=args.include_free_gt,
        pc_range=tuple(args.pc_range),
        occ_size=tuple(args.occ_size),
    )

    _print_results(metrics, tag=tag)

    # ── 저장 ────────────────────────────────────────────────────────
    if args.save:
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            # 로드 요약 포함
            print(f'probs  shape: {probs.shape}')
            print(f'gt     shape: {gt.shape}')
            print(f'indices: {indices.shape if indices is not None else "없음"}')
            _print_results(metrics, tag=tag)
        out_str = buf.getvalue()
        with open(args.save, 'w') as fh:
            fh.write(out_str)
        print(f'\n결과 저장: {args.save}')


if __name__ == '__main__':
    main()
