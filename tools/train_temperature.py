# Copyright (c) OpenMMLab. All rights reserved.
"""
Train a single scalar temperature T for temperature scaling (calibration).

Minimizes NLL on validation logits:  p = softmax(logits / T),  NLL = -mean(log p(y_true)).
Model-agnostic: expects a pre-exported .npz (or directory of .npz) with keys:
  - logits: (N, num_classes) float  — 또는 (H,W,Z,C) / (B,H,W,Z,C) (자동 flatten)
  - gt: (N,) int64
  - mask: (N,) bool, optional (없으면 all-True)

지원 모델 및 권장 옵션:
  BEVFormer / STCOcc / FusionOcc / SurroundOcc / TPVFormer / CONet
      dense 예측, 18-class → --free-class 17 권장 (free 제외)
  LiCROcc
      dense 예측, 클래스 수 가변 → --free-class <free_id> 권장
  SparseOcc_eccv
      sparse 예측, 17-class (non-free), GT 0–17 포함
      → --free-class 17 필수 (gt=17이 num_classes=17을 초과하므로)

Usage:
  # 1) Export logits
  python tools/export_occ_logits.py config.py checkpoint.pth --output work_dirs/occ_logits_val.npz

  # 2) Train T by minimizing NLL (SparseOcc 예시)
  python tools/train_temperature.py \\
      --logits-file work_dirs/occ_logits_val.npz \\
      --free-class 17 \\
      --output work_dirs/temperature.pt

  # 3) Use T at inference: config에 model.temperature=<T> 설정 또는
  #    state = torch.load('work_dirs/temperature.pt'); model.temperature = state['temperature']

주의:
  - --logits-dir 로 여러 .npz를 합칠 때 모든 파일의 num_classes가 같아야 함
  - Dense 모델 val 전체(수억 voxel)는 RAM이 충분한지 확인 (auto-device=cpu+batch)
"""

import argparse
import os
import numpy as np
import torch


def _filter_valid(
    logits: torch.Tensor, gt: torch.Tensor, free_class: int, num_classes: int
):
    """NLL 계산에 사용할 유효 voxel만 선택한다.

    제외 조건 (인덱싱 전 필터링):
      1. gt == free_class  (--free-class 지정 시)
      2. gt < 0            (음수 레이블 — ignore/padding 값)
      3. gt >= num_classes (모델 출력 클래스 수 초과 — SparseOcc 17-class vs GT 0-17 등)
    """
    if free_class >= 0:
        mask = gt != free_class
        logits = logits[mask]
        gt = gt[mask]
    valid = (gt >= 0) & (gt < num_classes)
    if not valid.all():
        logits = logits[valid]
        gt = gt[valid]
    return logits, gt


def nll_temperature(
    logits: torch.Tensor, gt: torch.Tensor, temperature: torch.Tensor,
    free_class: int = -1,
) -> torch.Tensor:
    """NLL with temperature scaling: p = softmax(logits / T), NLL = -mean(log p(y_true)).

    free_class >= 0 이면 gt == free_class인 voxel을 NLL에서 제외한다.
    gt < 0 또는 gt >= num_classes인 voxel도 자동 제외한다.

    NOTE: 필터링은 반드시 log_softmax 인덱싱 이전에 수행한다.
    (SparseOcc: 17-class logits, GT 0-17 → gt=17을 먼저 걸러야 IndexError 방지)
    """
    t = temperature.clamp(min=1e-3, max=1e3)
    logits, gt = _filter_valid(logits, gt, free_class, logits.shape[-1])
    if gt.numel() == 0:
        return torch.tensor(0.0, device=gt.device, requires_grad=True)
    log_probs = torch.log_softmax(logits / t, dim=-1)
    p_true = log_probs[torch.arange(gt.numel(), device=gt.device), gt]
    return -p_true.mean()


def nll_temperature_minibatch(
    logits: torch.Tensor, gt: torch.Tensor, temperature: torch.Tensor,
    batch_size: int, device: torch.device,
    free_class: int = -1,
) -> torch.Tensor:
    """NLL in minibatches (logits/gt on CPU) to avoid GPU OOM. Returns scalar for backward.

    free_class >= 0 이면 occupied voxel(gt != free_class)만 사용한다.
    gt < 0 또는 gt >= num_classes인 voxel도 자동 제외한다.
    """
    n_total = logits.shape[0]
    t = temperature.clamp(min=1e-3, max=1e3)
    total_nll = 0.0
    n_counted = 0
    for start in range(0, n_total, batch_size):
        end = min(start + batch_size, n_total)
        l_b = logits[start:end].to(device, non_blocking=True)
        g_b = gt[start:end].to(device, non_blocking=True)
        l_b, g_b = _filter_valid(l_b, g_b, free_class, l_b.shape[-1])
        n_b = g_b.numel()
        if n_b == 0:
            continue
        log_probs = torch.log_softmax(l_b / t, dim=-1)
        p_true = log_probs[torch.arange(n_b, device=device), g_b]
        nll_b = -p_true.mean()
        total_nll = total_nll + nll_b * n_b
        n_counted += n_b
    if n_counted == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    return total_nll / n_counted


def load_logits_gt_from_npz(path: str):
    """Load logits, gt, and optional mask from a single .npz file."""
    data = np.load(path, allow_pickle=True)
    logits = data['logits'].astype(np.float32)   # (N, C) or (H,W,Z,C) -> flatten to (N,C)
    gt = data['gt'].astype(np.int64).ravel()
    if 'mask' in data:
        mask = np.asarray(data['mask']).ravel().astype(bool)
    else:
        mask = np.ones(gt.shape, dtype=bool)
    if logits.ndim == 4:
        # (H, W, Z, C) -> (N, C)
        logits = logits.reshape(-1, logits.shape[-1])
    elif logits.ndim == 5:
        # (B, H, W, Z, C) -> (N, C)
        logits = logits.reshape(-1, logits.shape[-1])
    if gt.ndim > 1:
        gt = gt.ravel()
    n = min(logits.shape[0], gt.shape[0], mask.shape[0])
    logits = logits[:n]
    gt = gt[:n]
    mask = mask[:n]
    return logits, gt, mask


def main():
    parser = argparse.ArgumentParser(description='Train temperature T by minimizing NLL on validation logits')
    parser.add_argument('--logits-file', type=str, default=None, help='Single .npz with keys logits, gt, [mask]')
    parser.add_argument('--logits-dir', type=str, default=None, help='Directory of .npz files (each with logits, gt, [mask])')
    parser.add_argument('--output', type=str, default='temperature.pt', help='Output path for T (e.g. temperature.pt or temperature.json)')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate for T')
    parser.add_argument('--epochs', type=int, default=50, help='Number of optimization steps')
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'LBFGS'], help='Optimizer for T')
    parser.add_argument('--init-temp', type=float, default=1.5, help='Initial temperature value')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'],
                        help='Device: auto (use CPU when data is large to avoid OOM), cpu, or cuda')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Minibatch size for GPU (default: full on CPU, 2^20 on GPU if OOM). Reduce if OOM.')
    parser.add_argument('--free-class', type=int, default=-1,
                        help='Free-space class index to exclude from NLL (default: -1 = include all). '
                             'Set to 17 for Occ3D-nuScenes to train T on occupied voxels only.')
    args = parser.parse_args()

    if (args.logits_file is None) == (args.logits_dir is None):
        raise ValueError('Exactly one of --logits-file or --logits-dir must be set.')

    if args.logits_file:
        logits, gt, mask = load_logits_gt_from_npz(args.logits_file)
        logits_list, gt_list, mask_list = [logits], [gt], [mask]
    else:
        logits_list, gt_list, mask_list = [], [], []
        for f in sorted(os.listdir(args.logits_dir)):
            if not f.endswith('.npz'):
                continue
            l, g, m = load_logits_gt_from_npz(os.path.join(args.logits_dir, f))
            logits_list.append(l)
            gt_list.append(g)
            mask_list.append(m)
        if not logits_list:
            raise FileNotFoundError(f'No .npz files in {args.logits_dir}')
        # 클래스 수 불일치 사전 검증 (concatenate 전에 명시적 오류 발생)
        class_counts = [l.shape[1] for l in logits_list]
        if len(set(class_counts)) > 1:
            raise ValueError(
                f'여러 .npz 파일의 클래스 수(num_classes)가 일치하지 않습니다: {class_counts}. '
                f'동일 모델·설정으로 export한 파일만 함께 사용해주세요.'
            )

    # Concatenate and apply mask
    logits_all = np.concatenate([l[m] for l, m in zip(logits_list, mask_list)], axis=0)
    gt_all = np.concatenate([g[m] for g, m in zip(gt_list, mask_list)], axis=0)
    n_total = logits_all.shape[0]
    num_classes = logits_all.shape[1]
    free_class = args.free_class

    # 실제 NLL에 사용될 유효 voxel 수 계산 (free_class + 범위 초과 + 음수 제외)
    gt_np = gt_all  # numpy array
    valid_mask = (gt_np >= 0) & (gt_np < num_classes)
    if free_class >= 0:
        valid_mask = valid_mask & (gt_np != free_class)
    n_valid = int(valid_mask.sum())
    n_excluded = n_total - n_valid

    if free_class >= 0:
        print(f'Loaded {n_total:,} voxels, {num_classes} classes. '
              f'NLL에 사용: {n_valid:,} voxels '
              f'(제외: {n_excluded:,} — free_class={free_class} / gt<0 / gt≥{num_classes}).')
    else:
        excluded_str = f', {n_excluded:,} 제외(gt<0 또는 gt≥{num_classes})' if n_excluded > 0 else ''
        print(f'Loaded {n_total:,} voxels, {num_classes} classes{excluded_str}.')

    batch_size = args.batch_size
    # Device: avoid GPU OOM for large data (e.g. 258M voxels * 18 * 4 bytes ~ 17GB+)
    if args.device == 'auto':
        if n_total > 5e6:
            device = torch.device('cpu')
            if batch_size is None:
                batch_size = 5_000_000  # chunk to limit RAM on CPU
            print(f'Using CPU (data large: {n_total} voxels). Processing in batches of {batch_size}.')
        elif torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    # Minibatch size for GPU (and optionally CPU when data is huge)
    if batch_size is None and device.type == 'cuda':
        batch_size = min(n_total, max(50000, int(2e9 // (num_classes * 4))))
        print(f'Using GPU with batch_size={batch_size} to avoid OOM.')

    if batch_size is not None and batch_size >= n_total:
        batch_size = None

    logits_t = torch.from_numpy(logits_all).float()
    gt_t = torch.from_numpy(gt_all).long()
    if batch_size is None:
        logits_t = logits_t.to(device)
        gt_t = gt_t.to(device)

    # Learnable temperature (single scalar). Keep T > 0 via clamp in loss
    temperature = torch.nn.Parameter(torch.tensor(float(args.init_temp), device=device))

    def _compute_nll(temp):
        """현재 temperature로 NLL 계산 (full-batch 또는 minibatch)."""
        if batch_size is None:
            return nll_temperature(logits_t, gt_t, temp, free_class=free_class)
        return nll_temperature_minibatch(logits_t, gt_t, temp, batch_size, device,
                                         free_class=free_class)

    if args.optimizer == 'LBFGS':
        # LBFGS: optimizer.step(closure) 한 번 호출로 내부에서 max_iter회 반복
        # --epochs는 LBFGS 내부 최대 함수 평가 횟수(max_iter)를 의미함 (Adam의 epoch와 다름)
        print(f'Optimizer: LBFGS (max_iter={args.epochs}, lr={args.lr})')
        optimizer = torch.optim.LBFGS([temperature], lr=args.lr, max_iter=args.epochs)

        def closure():
            optimizer.zero_grad()
            nll = _compute_nll(temperature)
            nll.backward()
            return nll

        optimizer.step(closure)
        with torch.no_grad():
            nll_val = _compute_nll(temperature).item()
            print(f'LBFGS done: T={temperature.clamp(1e-3, 1e3).item():.4f}, NLL={nll_val:.4f}')
    else:
        print(f'Optimizer: Adam (epochs={args.epochs}, lr={args.lr})')
        optimizer = torch.optim.Adam([temperature], lr=args.lr)
        log_interval = max(1, args.epochs // 10)
        for epoch in range(args.epochs):
            optimizer.zero_grad()
            nll = _compute_nll(temperature)
            nll.backward()
            optimizer.step()
            if (epoch + 1) % log_interval == 0 or epoch == args.epochs - 1:
                with torch.no_grad():
                    t_cur = temperature.clamp(1e-3, 1e3).item()
                print(f'  Epoch {epoch+1:4d}/{args.epochs}: NLL={nll.item():.4f}, T={t_cur:.4f}')

    with torch.no_grad():
        t_final = temperature.clamp(min=1e-3, max=1e3).item()
        nll_final = _compute_nll(temperature).item()

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    result = {'temperature': t_final, 'nll': nll_final, 'num_classes': num_classes,
              'free_class': free_class, 'n_valid_voxels': n_valid}
    if args.output.endswith('.pt') or args.output.endswith('.pth'):
        torch.save(result, args.output)
    else:
        import json
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
    print(f'Saved: T={t_final:.4f}, NLL={nll_final:.4f}, '
          f'num_classes={num_classes}, n_valid={n_valid:,} → {args.output}')


if __name__ == '__main__':
    main()
