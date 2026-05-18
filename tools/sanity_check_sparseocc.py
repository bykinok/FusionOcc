#!/usr/bin/env python3
"""
SparseOcc 데이터 진단 스크립트.

export_occ_logits.py 로 추출된 .npz 파일을 읽어 다음을 검증한다:
  1. semseg(logits/probs) 값 통계
  2. sparse_indices 좌표 범위 · 중복 · bounds 검증
  3. GT 레이블 분포 (sparse 위치에서의 free vs occupied 비율)
  4. 예측 vs GT 교차 검증 (정확도, 오류 패턴)

Usage:
  python tools/sanity_check_sparseocc.py \
      work_dirs/.../sparseocc_occ_logits_val_calib_train.npz \
      [--occ-size 200 200 16] [--free-class 17]
"""
import argparse
import sys
import os
import numpy as np


def _print_banner(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def _hist_classes(arr: np.ndarray, free_class: int, label: str):
    unique, counts = np.unique(arr, return_counts=True)
    total = len(arr)
    print(f"\n  {label} 클래스 분포 (총 {total:,}개):")
    for cls, cnt in sorted(zip(unique, counts), key=lambda x: -x[1]):
        tag = "  ← FREE" if int(cls) == free_class else ""
        print(f"    class {int(cls):3d}: {cnt:10,d}  ({100.*cnt/total:6.2f}%){tag}")


def main():
    parser = argparse.ArgumentParser(
        description="SparseOcc exported .npz 진단 스크립트"
    )
    parser.add_argument("npz_path", help="export_occ_logits.py 출력 .npz 경로")
    parser.add_argument(
        "--occ-size", nargs=3, type=int, default=[200, 200, 16],
        metavar=("H", "W", "Z"),
        help="점유 격자 크기 (기본: 200 200 16)",
    )
    parser.add_argument(
        "--free-class", type=int, default=17,
        help="Free-space 클래스 인덱스 (기본: 17)",
    )
    parser.add_argument(
        "--n-classes", type=int, default=None,
        help="총 클래스 수 (기본: 자동 감지)",
    )
    args = parser.parse_args()

    H, W, Z = args.occ_size
    free_class = args.free_class

    if not os.path.isfile(args.npz_path):
        print(f"[ERROR] 파일 없음: {args.npz_path}")
        sys.exit(1)

    print(f"Loading: {args.npz_path}")
    data = np.load(args.npz_path, allow_pickle=True)
    keys = list(data.keys())
    print(f"Keys   : {keys}")

    # ---------------------------------------------------------------
    # 1. Logits / Probs 통계
    # ---------------------------------------------------------------
    logit_key = None
    if "logits" in data:
        logit_key = "logits"
    elif "probs" in data:
        logit_key = "probs"

    if logit_key:
        vals = data[logit_key].astype(np.float32)   # [N, C]
        _print_banner(f"[1] {logit_key} 통계")
        print(f"  shape  : {vals.shape}")
        print(f"  dtype  : {vals.dtype}")
        print(f"  min    : {vals.min():.6f}")
        print(f"  max    : {vals.max():.6f}")
        print(f"  mean   : {vals.mean():.6f}")
        print(f"  std    : {vals.std():.6f}")

        nan_cnt = int(np.isnan(vals).sum())
        inf_cnt = int(np.isinf(vals).sum())
        neg30_cnt = int((vals <= -29.9).sum())
        print(f"  NaN    : {nan_cnt}")
        print(f"  Inf    : {inf_cnt}")
        print(f"  ≤ -30  : {neg30_cnt}  (dummy free-class 채널이 있을 경우 예상)")

        row_sums = vals.sum(axis=-1)
        print(f"\n  row_sum (모든 클래스 합): mean={row_sums.mean():.4f}, std={row_sums.std():.4f}")
        if abs(row_sums.mean() - 1.0) < 0.05:
            print("  ✓ row_sum ≈ 1.0  →  정규화된 확률(probs)")
        elif logit_key == "logits":
            print("  ✓ row_sum ≠ 1.0  →  log(semseg) 형태(logits); softmax 후 probs 도출 가능")
        else:
            print("  ⚠ row_sum이 1.0과 크게 다름  →  정규화 필요")

        # argmax 기반 예측 클래스 분포
        pred_classes = np.argmax(vals, axis=-1)
        n_classes = vals.shape[1]
        _hist_classes(pred_classes, free_class, f"argmax({logit_key}) 예측")

        # softmax 후 MSP 분포
        # softmax(log(semseg)) == semseg / sum(semseg) ← 수학적 등가
        if logit_key == "logits":
            # safe softmax
            v_shift = vals - vals.max(axis=-1, keepdims=True)
            exp_v = np.exp(v_shift)
            probs = exp_v / (exp_v.sum(axis=-1, keepdims=True) + 1e-9)
        else:
            probs = vals / (vals.sum(axis=-1, keepdims=True) + 1e-9)

        msp = probs.max(axis=-1)
        uncertainty_msp = 1.0 - msp
        print(f"\n  MSP (softmax 후 최대 확률) 분포:")
        for thr in [0.3, 0.5, 0.7, 0.9]:
            pct = 100.0 * (msp >= thr).mean()
            print(f"    MSP >= {thr:.1f}: {pct:6.2f}%")
        print(f"  uncertainty_msp:  mean={uncertainty_msp.mean():.4f},  "
              f"std={uncertainty_msp.std():.4f},  "
              f"median={np.median(uncertainty_msp):.4f}")
    else:
        logit_key = None
        probs = None
        pred_classes = None
        print("\n[INFO] logits/probs 키 없음: 불확실성 분석 생략")

    # ---------------------------------------------------------------
    # 2. sparse_indices 통계
    # ---------------------------------------------------------------
    indices = data.get("indices")
    if indices is not None:
        _print_banner("[2] sparse_indices 통계")
        indices = np.asarray(indices)
        print(f"  shape  : {indices.shape}")
        print(f"  dtype  : {indices.dtype}")

        dim_names = ["x (H-dim)", "y (W-dim)", "z (Z-dim)"]
        bounds = [H, W, Z]
        all_ok = True
        for d, (name, bound) in enumerate(zip(dim_names, bounds)):
            vmin, vmax = int(indices[:, d].min()), int(indices[:, d].max())
            ok = (vmin >= 0) and (vmax < bound)
            mark = "✓" if ok else "✗ OUT-OF-BOUNDS"
            print(f"  dim[{d}] {name:12s}: min={vmin:4d}  max={vmax:4d}  "
                  f"(기대 0~{bound-1})  {mark}")
            all_ok = all_ok and ok

        # 범위 검증 assertion
        try:
            assert indices[:, 0].max() < H, f"x overflow: {indices[:,0].max()} >= {H}"
            assert indices[:, 1].max() < W, f"y overflow: {indices[:,1].max()} >= {W}"
            assert indices[:, 2].max() < Z, f"z overflow: {indices[:,2].max()} >= {Z}"
            assert indices.min() >= 0,      f"음수 좌표 발견: min={indices.min()}"
            print("  ✓ 모든 좌표 bounds assertion 통과")
        except AssertionError as e:
            print(f"  ✗ Assertion FAILED: {e}")

        # 중복 좌표 검사 (최대 50,000개 샘플)
        check_n = min(50_000, len(indices))
        tuples = [tuple(r) for r in indices[:check_n]]
        n_unique = len(set(tuples))
        dup = check_n - n_unique
        if dup == 0:
            print(f"  ✓ 중복 없음 (검사 {check_n:,}개)")
        else:
            print(f"  ⚠ 중복 {dup:,}개 발견 (검사 {check_n:,}개)")

        # 격자당 인덱스 밀도
        if len(indices) > 0:
            total_voxels = H * W * Z
            sparse_ratio = len(indices) / total_voxels
            print(f"\n  sparse 예측 밀도: {len(indices):,} / {total_voxels:,} "
                  f"= {sparse_ratio*100:.2f}%")
            if sparse_ratio < 0.1:
                print(f"  ✓ 희소 예측 (SparseOcc 정상 동작: ~5% 예상)")
            else:
                print(f"  ⚠ 밀도가 예상보다 높음 (SparseOcc는 Top-K occupied만 예측)")
    else:
        indices = None
        print("\n[INFO] indices 키 없음: 좌표 검증 생략 (dense 모델)")

    # ---------------------------------------------------------------
    # 3. GT 분포
    # ---------------------------------------------------------------
    gt = data.get("gt")
    if gt is not None:
        _print_banner("[3] GT 레이블 통계 (sparse 위치에서 추출)")
        gt = np.asarray(gt).ravel()
        print(f"  shape   : {gt.shape}")
        _hist_classes(gt, free_class, "GT")

        free_ratio = float((gt == free_class).mean())
        occ_ratio  = 1.0 - free_ratio
        print(f"\n  Occupied GT 비율 (sparse 위치): {occ_ratio*100:.2f}%")
        if occ_ratio >= 0.5:
            print("  ✓ Sparse 위치의 절반 이상이 occupied GT → 정상")
            print("    (SparseOcc는 occupied voxel을 예측하므로 GT도 주로 occupied여야 함)")
        else:
            print("  ⚠ Sparse 위치에서 GT-free 비율이 높음 "
                  "→ 좌표 매핑 오류 가능성 검토 필요")
    else:
        gt = None
        print("\n[INFO] gt 키 없음: GT 분포 분석 생략")

    # ---------------------------------------------------------------
    # 4. 교차 검증
    # ---------------------------------------------------------------
    if logit_key and gt is not None and probs is not None and pred_classes is not None:
        _print_banner("[4] 예측 vs GT 교차 검증")

        # 길이 일치 확인
        N = min(len(probs), len(gt))
        if len(probs) == len(gt):
            print(f"  ✓ 배열 길이 일치: {N:,}")
        else:
            print(f"  ⚠ 길이 불일치: probs={len(probs)}, gt={len(gt)} → min({N})으로 자름")
        probs_   = probs[:N]
        gt_      = gt[:N]
        pred_    = pred_classes[:N]

        # 전체 정확도 (sparse 위치)
        acc_all = float((pred_ == gt_).mean())
        print(f"  정확도 (전체, sparse 위치): {acc_all*100:.2f}%")

        # free GT 제외 정확도
        non_free = gt_ != free_class
        if non_free.sum() > 0:
            acc_occ = float((pred_[non_free] == gt_[non_free]).mean())
            print(f"  정확도 (occupied GT만):     {acc_occ*100:.2f}%")
        else:
            print("  ⚠ occupied GT가 없음 → 좌표 매핑 오류 의심")

        # free GT → 모델 예측 분포 (False Positive 분석)
        free_gt_mask = gt_ == free_class
        fp_count = int(free_gt_mask.sum())
        if fp_count > 0:
            fp_preds = pred_[free_gt_mask]
            print(f"\n  GT=FREE인데 sparse로 예측된 voxel (False Positive): {fp_count:,}개")
            unique_fp, cnt_fp = np.unique(fp_preds, return_counts=True)
            for cls, cnt in sorted(zip(unique_fp, cnt_fp), key=lambda x: -x[1])[:5]:
                print(f"    FP → class {int(cls):3d}: {cnt:,}개")

        # uncertainty_msp vs correct/incorrect
        msp_ = probs_.max(axis=-1)
        unc_ = 1.0 - msp_
        correct_ = (pred_ == gt_)
        print(f"\n  uncertainty_msp 통계:")
        print(f"    correct  voxels: unc_mean={unc_[correct_].mean():.4f}  "
              f"n={correct_.sum():,}")
        if (~correct_).sum() > 0:
            print(f"    incorrect voxels: unc_mean={unc_[~correct_].mean():.4f}  "
                  f"n={(~correct_).sum():,}")

        # occupied GT만 AUROC 방향 예비 분석
        try:
            from sklearn.metrics import roc_auc_score
            y_inc = (~correct_[non_free]).astype(int)
            u_msp = unc_[non_free]
            if len(np.unique(y_inc)) == 2:
                auroc = roc_auc_score(y_inc, u_msp)
                print(f"\n  [예비 AUROC at sparse positions, occupied GT]")
                print(f"    AUROC(uncertainty_msp): {auroc*100:.2f}%")
                if auroc > 0.5:
                    print(f"    ✓ > 50%: 불확실성이 오류 방향과 일치 (올바른 방향)")
                else:
                    print(f"    ✗ < 50%: 불확실성 방향 역전 → semseg 정규화 방식 확인 필요")
            else:
                print("\n  [예비 AUROC] y_incorrect에 클래스가 1개뿐 → 계산 불가")
        except ImportError:
            print("\n  [INFO] sklearn 없음: AUROC 예비 계산 생략")

    # ---------------------------------------------------------------
    # 5. 인덱스 → GT 매핑 검증 (gt_full 키가 있는 경우)
    # ---------------------------------------------------------------
    gt_full = data.get("gt_full")
    if gt_full is not None and indices is not None and gt is not None:
        _print_banner("[5] sparse_indices → gt_full 매핑 검증")
        gt_full = np.asarray(gt_full)   # [H, W, Z]
        print(f"  gt_full shape: {gt_full.shape}  (기대: {H}×{W}×{Z})")

        xi = indices[:, 0].astype(int)
        yi = indices[:, 1].astype(int)
        zi = indices[:, 2].astype(int)

        valid_mask = (
            (xi >= 0) & (xi < H) &
            (yi >= 0) & (yi < W) &
            (zi >= 0) & (zi < Z)
        )
        if valid_mask.all():
            gt_at_sparse = gt_full[xi, yi, zi]
            match = (gt_at_sparse == gt).mean()
            print(f"  gt_full[xi,yi,zi] == gt(sparse) 일치율: {match*100:.2f}%")
            if match > 0.99:
                print("  ✓ 좌표 순서 (x, y, z) → (H, W, Z) 매핑 올바름")
            else:
                print("  ✗ 불일치! 좌표 순서 또는 GT 추출 방식 확인 필요")
                # 대안 좌표 순서 시도
                for perm, pname in [
                    ((1, 0, 2), "y,x,z"),
                    ((2, 1, 0), "z,y,x"),
                    ((0, 2, 1), "x,z,y"),
                ]:
                    a, b, c = indices[:, perm[0]], indices[:, perm[1]], indices[:, perm[2]]
                    vm = (a >= 0) & (a < gt_full.shape[0]) & \
                         (b >= 0) & (b < gt_full.shape[1]) & \
                         (c >= 0) & (c < gt_full.shape[2])
                    if vm.all():
                        alt_gt = gt_full[a.astype(int), b.astype(int), c.astype(int)]
                        alt_match = (alt_gt == gt).mean()
                        print(f"    대안 순서 ({pname}): 일치율={alt_match*100:.2f}%")
        else:
            bad_n = int((~valid_mask).sum())
            print(f"  ✗ 유효하지 않은 좌표 {bad_n:,}개 → gt_full 검증 불가")

    _print_banner("진단 완료")
    print()


if __name__ == "__main__":
    main()
