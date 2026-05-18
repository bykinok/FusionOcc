# Copyright (c) OpenMMLab. All rights reserved.
"""
Export occupancy logits and ground truth from a trained model for temperature scaling.

Runs the model on the test/val set with export_occ_logits=True and saves (logits, gt, mask,
optionally indices) to a single .npz file that tools/train_temperature.py can consume.

Supports two modes (auto-detected per sample):
  Dense mode  : model outputs dense logits (H,W,Z,C)
                Supported models: BEVFormer, STCOcc, FusionOcc, SurroundOcc, TPVFormer,
                                  CONet, LiCROcc
  Sparse mode : model outputs sparse logits (N_sparse,C) + sparse_indices (N_sparse,3)
                Supported models: SparseOcc_eccv

Return format per model:
  - BEVFormer  : custom test_step → list[dict]  (occ_logits, voxel_semantics, mask_camera)
  - STCOcc     : simple_test → list[dict]       (occ_logits, voxel_semantics, mask_camera)
  - FusionOcc  : predict() → list[dict]         (occ_logits, voxel_semantics, mask_camera)
  - SurroundOcc: predict() → list[dict]         (occ_logits, voxel_semantics, mask_camera)
  - TPVFormer  : predict() → list[dict]         (occ_logits, voxel_semantics, mask_camera)
  - CONet      : predict() → list[dict]         (occ_logits, voxel_semantics, mask_camera)
  - LiCROcc    : forward()  → list[dict]        (occ_logits, voxel_semantics*, mask_camera*)
  - SparseOcc_eccv: predict() → list[Det3DDataSample]  (occ_logits, sparse_indices, ...)

  * optional: depends on dataset loading config

Usage (SparseOcc example):
  python tools/export_occ_logits.py \
    projects/SparseOcc_eccv/configs/r50_nuimg_704x256_8f_wo_cam_mask_unified_calib_train_miou.py \
    work_dirs/.../epoch_24.pth \
    --output work_dirs/occ_logits_val.npz

For new models: implement export_occ_logits logic in the detector so that when
model.export_occ_logits=True, predict()/test_step() returns a list of dict or
Det3DDataSample with keys 'occ_logits' (H,W,Z,C float32), 'voxel_semantics' (H,W,Z int),
and optionally 'mask_camera' (H,W,Z bool).  Sparse models additionally return
'sparse_indices' (N,3 int) alongside 'occ_logits' (N,C).
"""

import argparse
import os
import numpy as np
import torch
from tqdm import tqdm
from scipy.special import softmax as scipy_softmax

from mmengine.config import Config, DictAction
from mmengine.registry import DefaultScope
from mmengine.runner import Runner

try:
    from mmdet3d.utils import replace_ceph_backend
except ImportError:
    replace_ceph_backend = None
import mmdet3d  # noqa: F401 - register mmdet3d modules


def parse_args():
    parser = argparse.ArgumentParser(description='Export occ logits and GT for temperature scaling')
    parser.add_argument('config', help='Test config file path')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--output', type=str, default='occ_logits_val.npz',
                        help='Output .npz path (default: occ_logits_val.npz)')
    parser.add_argument('--per-sample-dir', type=str, default=None,
                        help='디렉토리 지정 시 샘플별 .npz 저장 모드 활성화. '
                             '각 파일에 probs(normalized semseg), sparse_indices, '
                             'gt_full [H,W,Z], mask_camera [H,W,Z]를 저장한다. '
                             'compute_calibration_sparse.py와 함께 사용.')
    parser.add_argument('--occ-size', nargs=3, type=int, default=[200, 200, 16],
                        metavar=('H', 'W', 'Z'),
                        help='점유 격자 크기 (기본: 200 200 16). '
                             'per-sample 모드에서 좌표 bounds 검증에 사용.')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Max samples to export (default: all)')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction, help='Override config options')
    args = parser.parse_args()
    return args


def _get_field(sample, name, default=None):
    """Det3DDataSample(속성 접근) 또는 dict(키 접근) 모두 지원."""
    if isinstance(sample, dict):
        return sample.get(name, default)
    return getattr(sample, name, default)


def _has_field(sample, name):
    """Det3DDataSample 또는 dict에서 필드 존재 여부 및 비-None 값 확인."""
    if isinstance(sample, dict):
        return name in sample and sample[name] is not None
    return hasattr(sample, name) and getattr(sample, name, None) is not None


def _unwrap_output(out):
    """test_step/predict()의 다양한 반환 형식을 평탄화된 샘플 리스트로 변환.

    처리 케이스:
      - None                   → []
      - single sample (non-list) → [sample]
      - list[sample]           → list[sample]   (정상 케이스)
      - list[list[sample]]     → list[sample]   (중첩 래핑 케이스)
    """
    if out is None:
        return []
    if not isinstance(out, list):
        return [out]
    # 중첩 리스트 평탄화: [[s1, s2], [s3]] → [s1, s2, s3]
    if len(out) > 0 and isinstance(out[0], list):
        flat = []
        for item in out:
            flat.extend(item) if isinstance(item, list) else flat.append(item)
        return flat
    return out


def _validate_sparse_indices(sparse_indices: np.ndarray, occ_size: list, sample_id: int = 0):
    """sparse_indices 좌표 범위 검증. 이상 시 경고 및 유효 마스크 반환."""
    H, W, Z = occ_size
    xi, yi, zi = sparse_indices[:, 0], sparse_indices[:, 1], sparse_indices[:, 2]
    valid_mask = (
        (xi >= 0) & (xi < H) &
        (yi >= 0) & (yi < W) &
        (zi >= 0) & (zi < Z)
    )
    n_bad = int((~valid_mask).sum())
    if n_bad > 0:
        print(f'[WARNING] 샘플 {sample_id}: {n_bad}개 out-of-bounds 좌표 발견 '
              f'(x max={int(xi.max())}<{H}, y max={int(yi.max())}<{W}, z max={int(zi.max())}<{Z}). '
              f'해당 voxel 제외.')
    return valid_mask


def _process_sparse_sample(sample, all_logits, all_gt, all_indices, warn_once,
                            per_sample_dir=None, sample_id=0, occ_size=None):
    """sparse_indices 키가 있는 SparseOcc 계열 샘플 처리.

    per_sample_dir 지정 시:
      - probs [N_sparse, C]  (log(semseg) → softmax → semseg_norm)
      - sparse_indices [N_sparse, 3]
      - gt_full [H, W, Z]
      - mask_camera [H, W, Z]
      를 개별 .npz 파일로 저장한다 (compute_calibration_sparse.py 용).

    기본 모드(concatenated):
      - logits [N_sparse, C], gt [N_sparse], indices [N_sparse, 3] 를 누적한다.
    """
    if occ_size is None:
        occ_size = [200, 200, 16]

    logits = np.asarray(_get_field(sample, 'occ_logits'), dtype=np.float32)   # [N_sparse, C]
    sparse_indices = np.asarray(_get_field(sample, 'sparse_indices'))          # [N_sparse, 3] int32
    gt_full_raw = _get_field(sample, 'voxel_semantics')
    if gt_full_raw is None:
        if warn_once['sparse_gt']:
            print('[WARNING] voxel_semantics(GT)가 없는 샘플을 건너뜁니다. '
                  'calib_train 설정에서 LoadOccGTFromFile을 활성화해 주세요.')
            warn_once['sparse_gt'] = False
        return False
    gt_full = np.asarray(gt_full_raw)                               # [H, W, Z]

    # ── 좌표 bounds 검증 ────────────────────────────────────────────
    coord_valid = _validate_sparse_indices(sparse_indices, occ_size, sample_id)
    if coord_valid.sum() == 0:
        print(f'[WARNING] 샘플 {sample_id}: 유효한 좌표가 없어 건너뜁니다.')
        return False
    if not coord_valid.all():
        sparse_indices = sparse_indices[coord_valid]
        logits = logits[coord_valid]

    xi, yi, zi = (sparse_indices[:, 0].astype(int),
                  sparse_indices[:, 1].astype(int),
                  sparse_indices[:, 2].astype(int))

    # ── per-sample 저장 모드 ────────────────────────────────────────
    if per_sample_dir is not None:
        mask_full_raw = _get_field(sample, 'mask_camera')
        mask_full = (np.asarray(mask_full_raw).astype(bool)
                     if mask_full_raw is not None
                     else np.ones(gt_full.shape, dtype=bool))

        # log(semseg) → softmax → semseg_norm (수학적으로 동일)
        # scipy_softmax는 수치 안정성을 위해 내부적으로 max 빼기 적용
        probs = scipy_softmax(logits, axis=-1).astype(np.float32)   # [N_sparse, C]

        out_path = os.path.join(per_sample_dir, f'sample_{sample_id:06d}.npz')
        np.savez_compressed(
            out_path,
            probs=probs,                    # [N_sparse, C] 정규화 확률
            sparse_indices=sparse_indices,  # [N_sparse, 3] int32
            gt_full=gt_full,                # [H, W, Z] 전체 GT
            mask_camera=mask_full,          # [H, W, Z] bool
        )
        # concatenated 버퍼에도 logits/gt/indices 추가 (호환성)
        gt_sparse = gt_full[xi, yi, zi]
        mask_sparse = mask_full[xi, yi, zi]
        valid = mask_sparse
        if valid.sum() > 0:
            all_logits.append(logits[valid])
            all_gt.append(gt_sparse[valid])
            all_indices.append(sparse_indices[valid])
        return True

    # ── 기본 concatenated 모드 ──────────────────────────────────────
    # GT: sparse 위치의 voxel_semantics 값만 추출
    gt_sparse = gt_full[xi, yi, zi]                                 # [N_sparse]

    # mask_camera: sparse 위치에서만 추출
    mask_full = _get_field(sample, 'mask_camera')
    if mask_full is not None:
        mask_sparse = np.asarray(mask_full).astype(bool)[xi, yi, zi]
    else:
        mask_sparse = np.ones(len(gt_sparse), dtype=bool)

    valid = mask_sparse
    if valid.sum() == 0:
        return False

    all_logits.append(logits[valid])
    all_gt.append(gt_sparse[valid])
    all_indices.append(sparse_indices[valid])
    return True


def _process_dense_sample(sample, all_logits, all_gt, all_indices, warn_once):
    """dense logits (H,W,Z,C) 모델 (BEVFormer 등) 샘플 처리 — 하위 호환."""
    logits = np.asarray(_get_field(sample, 'occ_logits'))
    gt = _get_field(sample, 'voxel_semantics')
    if gt is None:
        if warn_once['dense_gt']:
            print('[WARNING] voxel_semantics(GT)가 없는 샘플을 건너뜁니다. '
                  'calib_train 설정에서 GT 로딩이 활성화되어 있는지 확인해 주세요.')
            warn_once['dense_gt'] = False
        return False
    gt = np.asarray(gt)

    mask = _get_field(sample, 'mask_camera')
    if mask is None:
        mask = np.ones(gt.shape, dtype=bool)
    else:
        mask = np.asarray(mask).astype(bool)

    # Flatten: (H,W,Z,C) -> (N,C),  (H,W,Z) -> (N,)
    logits_flat = logits.reshape(-1, logits.shape[-1])
    gt_flat = gt.ravel()
    mask_flat = mask.ravel()
    n = min(logits_flat.shape[0], gt_flat.shape[0], mask_flat.shape[0])
    logits_flat = logits_flat[:n]
    gt_flat     = gt_flat[:n]
    mask_flat   = mask_flat[:n]

    valid = mask_flat.astype(bool)
    if valid.sum() == 0:
        return False

    all_logits.append(logits_flat[valid])
    all_gt.append(gt_flat[valid])
    all_indices.append(None)   # dense 모드는 spatial 인덱스 없음
    return True


def main():
    args = parse_args()
    DefaultScope.get_instance('mmdet3d', scope_name='mmdet3d')
    cfg = Config.fromfile(args.config)

    if hasattr(cfg, 'custom_imports') and cfg.custom_imports:
        import importlib
        for name in cfg.custom_imports.get('imports', []):
            importlib.import_module(name)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    if replace_ceph_backend is not None and getattr(args, 'ceph', False):
        cfg = replace_ceph_backend(cfg)

    if cfg.get('work_dir', None) is None:
        cfg.work_dir = os.path.join('./work_dirs', os.path.splitext(os.path.basename(args.config))[0])
    cfg.load_from = args.checkpoint

    # DistributedSampler는 단일 프로세스에서 init_process_group 없이 동작하지 않으므로
    # DefaultSampler(shuffle=False)로 교체한다.
    _DIST_SAMPLER_TYPES = {'DistributedSampler', 'DistributedGroupSampler', 'InfiniteSampler'}

    def _replace_sampler(dl_cfg):
        if not isinstance(dl_cfg, dict):
            return
        sampler = dl_cfg.get('sampler', {})
        if isinstance(sampler, dict) and sampler.get('type') in _DIST_SAMPLER_TYPES:
            dl_cfg['sampler'] = dict(type='DefaultSampler', shuffle=False)

    for key in ('test_dataloader', 'val_dataloader'):
        if hasattr(cfg, key):
            _replace_sampler(getattr(cfg, key))

    runner = Runner.from_cfg(cfg)
    from mmengine.runner import load_checkpoint
    load_checkpoint(runner.model, args.checkpoint, map_location='cpu', strict=False)

    # export_occ_logits 플래그를 모델에 설정 (DDP 래핑 시 내부 모델에 설정)
    model = getattr(runner.model, 'module', runner.model)
    model.export_occ_logits = True

    runner.model.eval()
    test_loop = runner.build_test_loop(runner._test_loop)
    dataloader = test_loop.dataloader

    # per-sample 디렉토리 준비
    per_sample_dir = getattr(args, 'per_sample_dir', None)
    occ_size = getattr(args, 'occ_size', [200, 200, 16])
    if per_sample_dir is not None:
        os.makedirs(per_sample_dir, exist_ok=True)
        print(f'[per-sample 모드] 샘플별 .npz 저장: {per_sample_dir}')
        print(f'  probs(semseg_norm), sparse_indices, gt_full, mask_camera 포함')

    all_logits, all_gt, all_indices = [], [], []
    n_export = 0
    n_skip_no_logits = 0
    max_samples = args.max_samples or float('inf')
    has_sparse = None  # None = 아직 미결정, True/False = 첫 샘플에서 결정
    # 동일 경고를 한 번만 출력하기 위한 플래그
    warn_once = {'sparse_gt': True, 'dense_gt': True}

    for batch in tqdm(dataloader, desc='Export logits'):
        if n_export >= max_samples:
            break
        with torch.no_grad():
            out = runner.model.test_step(batch)

        # 다양한 반환 형식 정규화: None / 단일 샘플 / 중첩 리스트 → flat list
        samples = _unwrap_output(out)

        for sample in samples:
            if n_export >= max_samples:
                break
            if not _has_field(sample, 'occ_logits'):
                n_skip_no_logits += 1
                continue

            # sparse 모드 (SparseOcc): sparse_indices 존재 여부로 분기
            if _has_field(sample, 'sparse_indices'):
                ok = _process_sparse_sample(
                    sample, all_logits, all_gt, all_indices, warn_once,
                    per_sample_dir=per_sample_dir,
                    sample_id=n_export,
                    occ_size=occ_size,
                )
                if ok and has_sparse is None:
                    has_sparse = True
            else:
                ok = _process_dense_sample(sample, all_logits, all_gt, all_indices, warn_once)
                if ok and has_sparse is None:
                    has_sparse = False

            if ok:
                n_export += 1

    if n_skip_no_logits > 0:
        print(f'[INFO] occ_logits 없이 건너뛴 샘플: {n_skip_no_logits}개 '
              f'(모델이 export_occ_logits를 지원하는지 확인하세요)')

    if not all_logits:
        raise RuntimeError(
            'No samples with occ_logits found. '
            'Is export_occ_logits supported and enabled in this model?'
        )

    logits_all = np.concatenate(all_logits, axis=0)
    gt_all     = np.concatenate(all_gt,     axis=0)
    mask_all   = np.ones(gt_all.shape[0], dtype=bool)  # 이미 mask 적용 완료

    save_kwargs = dict(logits=logits_all, gt=gt_all, mask=mask_all)

    # sparse 모드: 공간 인덱스도 함께 저장 (train_temperature.py 거리별 분석에 사용)
    if has_sparse:
        indices_all = np.concatenate(all_indices, axis=0)  # [N_total, 3]
        save_kwargs['indices'] = indices_all
        print(f'[sparse mode] indices shape: {indices_all.shape}')

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    np.savez_compressed(args.output, **save_kwargs)
    print(f'Saved {logits_all.shape[0]:,} voxels to {args.output} '
          f'(num_classes={logits_all.shape[1]}, mode={"sparse" if has_sparse else "dense"})')


if __name__ == '__main__':
    main()
