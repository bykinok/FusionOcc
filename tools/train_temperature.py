# Copyright (c) OpenMMLab. All rights reserved.
"""
Train a single scalar temperature T for temperature scaling (calibration).

Minimizes NLL on validation logits:  p = softmax(logits / T),  NLL = -mean(log p(y_true)).
Model-agnostic: expects a pre-exported .npz (or directory of .npz) with keys:
  - logits: (N, num_classes) float
  - gt: (N,) int64
  - mask: (N,) bool, optional (if absent, all positions are valid)

Usage:
  # 1) Export logits (BEVFormer example)
  python tools/export_occ_logits.py config.py checkpoint.pth --output work_dirs/occ_logits_val.npz

  # 2) Train T by minimizing NLL
  python tools/train_temperature.py --logits-file work_dirs/occ_logits_val.npz --output work_dirs/temperature.pt

  # 3) Use T at inference: in config set pts_bbox_head.temperature=<T>, or after load_checkpoint:
  #    state = torch.load('work_dirs/temperature.pt'); model.pts_bbox_head.temperature = state['temperature']
"""

import argparse
import os
import numpy as np
import torch


def nll_temperature(logits: torch.Tensor, gt: torch.Tensor, temperature: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """NLL with temperature scaling: p = softmax(logits / T), NLL = -mean(log p(y_true))."""
    t = temperature.clamp(min=1e-3, max=1e3)
    logits_scaled = logits / t
    log_probs = torch.log_softmax(logits_scaled, dim=-1)
    n = gt.numel()
    p_true = log_probs[torch.arange(n, device=gt.device), gt]
    return -p_true.mean()


def nll_temperature_minibatch(
    logits: torch.Tensor, gt: torch.Tensor, temperature: torch.Tensor,
    batch_size: int, device: torch.device, eps: float = 1e-10
) -> torch.Tensor:
    """NLL in minibatches (logits/gt on CPU) to avoid GPU OOM. Returns scalar for backward."""
    n_total = logits.shape[0]
    t = temperature.clamp(min=1e-3, max=1e3)
    total_nll = 0.0
    for start in range(0, n_total, batch_size):
        end = min(start + batch_size, n_total)
        l_b = logits[start:end].to(device, non_blocking=True)
        g_b = gt[start:end].to(device, non_blocking=True)
        logits_scaled = l_b / t
        log_probs = torch.log_softmax(logits_scaled, dim=-1)
        n_b = g_b.numel()
        p_true = log_probs[torch.arange(n_b, device=device), g_b]
        nll_b = -p_true.mean()
        total_nll = total_nll + nll_b * (end - start)
    return total_nll / n_total


def load_logits_gt_from_npz(path: str):
    """Load logits, gt, and optional mask from a single .npz file."""
    data = np.load(path, allow_pickle=True)
    logits = data['logits'].astype(np.float64)   # (N, C) or (H,W,Z,C) -> flatten to (N,C)
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
    parser.add_argument('--eps', type=float, default=1e-10, help='Clip for log probs')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'],
                        help='Device: auto (use CPU when data is large to avoid OOM), cpu, or cuda')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Minibatch size for GPU (default: full on CPU, 2^20 on GPU if OOM). Reduce if OOM.')
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

    # Concatenate and apply mask
    logits_all = np.concatenate([l[m] for l, m in zip(logits_list, mask_list)], axis=0)
    gt_all = np.concatenate([g[m] for g, m in zip(gt_list, mask_list)], axis=0)
    n_total = logits_all.shape[0]
    num_classes = logits_all.shape[1]
    print(f'Loaded {n_total} voxels, {num_classes} classes.')

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

    if args.optimizer == 'LBFGS':
        optimizer = torch.optim.LBFGS([temperature], lr=args.lr, max_iter=args.epochs)

        def closure():
            optimizer.zero_grad()
            if batch_size is None:
                nll = nll_temperature(logits_t, gt_t, temperature, eps=args.eps)
            else:
                nll = nll_temperature_minibatch(logits_t, gt_t, temperature, batch_size, device, eps=args.eps)
            nll.backward()
            return nll

        optimizer.step(closure)
    else:
        optimizer = torch.optim.Adam([temperature], lr=args.lr)
        for _ in range(args.epochs):
            optimizer.zero_grad()
            if batch_size is None:
                nll = nll_temperature(logits_t, gt_t, temperature, eps=args.eps)
            else:
                nll = nll_temperature_minibatch(logits_t, gt_t, temperature, batch_size, device, eps=args.eps)
            nll.backward()
            optimizer.step()
    with torch.no_grad():
        t_final = temperature.clamp(min=1e-3, max=1e3).item()
        if batch_size is None:
            nll_final = nll_temperature(logits_t, gt_t, temperature, eps=args.eps).item()
        else:
            nll_final = nll_temperature_minibatch(logits_t, gt_t, temperature, batch_size, device, eps=args.eps).item()

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    if args.output.endswith('.pt') or args.output.endswith('.pth'):
        torch.save({'temperature': t_final, 'nll': nll_final}, args.output)
        print(f'Saved temperature T = {t_final:.4f} (NLL = {nll_final:.4f}) to {args.output}')
    else:
        import json
        with open(args.output, 'w') as f:
            json.dump({'temperature': t_final, 'nll': nll_final}, f, indent=2)
        print(f'Saved temperature T = {t_final:.4f} (NLL = {nll_final:.4f}) to {args.output}')


if __name__ == '__main__':
    main()
