"""
pip install onnx==1.20.1
pip install onnxruntime==1.23.2
pip install tensorrt --extra-index-url https://pypi.nvidia.com
    tensorrt                  10.3.0
    tensorrt-cu12             10.3.0
    tensorrt-cu12-bindings    10.3.0
    tensorrt-cu12-libs        10.3.0

STCOcc INT8 PTQ - 단계 1: Calibration 데이터 수집

TRT INT8 calibrator에 필요한 img_backbone 입력 텐서와
bev_encoder 입력 텐서를 캡처해 .npy 파일로 저장한다.

수집 데이터: **training set** — val_dataloader 구조 그대로 + ann_file만 train pkl로 교체.
(val 파이프라인·sampler를 그대로 써서 val_step 호출 가능하고,
 데이터는 train set에서 수집한다.)

사용법:
    python tools/stcocc_prepare_calib_data.py \\
        --config projects/STCOcc/configs/stcocc_r50_704x256_16f_occ3d_36e_miou_unified.py \\
        --checkpoint work_dirs_nas_1st/stcocc_r50_704x256_16f_occ3d_36e_miou_unified/iter_42192.pth \\
        --out-dir calib_data/stcocc \\
        --num-samples 200
"""

import argparse
import copy
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath('.'))


# ──────────────────────────────────────────────────────────────
# 캡처 훅
# ──────────────────────────────────────────────────────────────
_img_encoder_count: int = 0
_bev_encoder_count: int = 0
_img_encoder_shape = None
_bev_encoder_shape = None


def _make_img_encoder_hook(out_dir: str, max_samples: int):
    """img_backbone 직전 입력 텐서를 캡처하는 forward pre-hook."""
    global _img_encoder_count, _img_encoder_shape
    os.makedirs(os.path.join(out_dir, 'img_encoder'), exist_ok=True)

    def hook(module, args):
        global _img_encoder_count, _img_encoder_shape
        if _img_encoder_count >= max_samples:
            return
        img = args[0]           # [B*N, 3, H, W]
        arr = img.float().cpu().numpy()
        np.save(os.path.join(out_dir, 'img_encoder', f'{_img_encoder_count:05d}.npy'), arr)
        if _img_encoder_shape is None:
            _img_encoder_shape = arr.shape
        if _img_encoder_count % 10 == 0:
            print(f'  img_encoder samples: {_img_encoder_count+1}/{max_samples}  shape={arr.shape}')
        _img_encoder_count += 1
        del arr

    return hook


def _make_bev_encoder_hook(out_dir: str, max_samples: int):
    """img_bev_encoder_backbone 직전 BEV voxel 텐서를 캡처하는 forward pre-hook."""
    global _bev_encoder_count, _bev_encoder_shape
    os.makedirs(os.path.join(out_dir, 'bev_encoder'), exist_ok=True)

    def hook(module, args):
        global _bev_encoder_count, _bev_encoder_shape
        if _bev_encoder_count >= max_samples:
            return
        bev = args[0]           # [B, C, Z, H_bev, W_bev]
        arr = bev.float().cpu().numpy()
        np.save(os.path.join(out_dir, 'bev_encoder', f'{_bev_encoder_count:05d}.npy'), arr)
        if _bev_encoder_shape is None:
            _bev_encoder_shape = arr.shape
        if _bev_encoder_count % 10 == 0:
            print(f'  bev_encoder samples: {_bev_encoder_count+1}/{max_samples}  shape={arr.shape}')
        _bev_encoder_count += 1
        del arr

    return hook


# ──────────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description='STCOcc calibration data 수집')
    parser.add_argument('--config',     required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out-dir',    required=True)
    parser.add_argument('--num-samples', type=int, default=200,
                        help='수집할 calibration 샘플 수 (권장 100~500)')
    parser.add_argument('--launcher', default='none')
    return parser.parse_args()


def main():
    args = parse_args()

    # ── mmengine Runner 초기화 ──────────────────────────────────
    from mmengine.config import Config, DictAction
    from mmengine.runner import Runner

    import mmdet3d  # noqa: F401
    import projects.STCOcc  # noqa: F401

    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    cfg.load_from = args.checkpoint
    cfg.work_dir = '/tmp/stcocc_calib'

    # test 모드로 실행
    runner = Runner.from_cfg(cfg)
    runner.load_or_resume()

    model = runner.model
    model.eval()
    model.cuda()

    # ── 훅 등록 ────────────────────────────────────────────────
    fp = model.module.forward_projection if hasattr(model, 'module') \
        else model.forward_projection

    h1 = fp.img_backbone.register_forward_pre_hook(
        _make_img_encoder_hook(args.out_dir, args.num_samples)
    )
    h2 = fp.img_bev_encoder_backbone.register_forward_pre_hook(
        _make_bev_encoder_hook(args.out_dir, args.num_samples)
    )

    print(f'\n[STCOcc Calib] 데이터 수집 시작 (목표: {args.num_samples}개 샘플)')
    print(f'  저장 경로: {args.out_dir}\n')

    # val_dataloader 구조 그대로 + ann_file만 train pkl로 교체
    # → val 파이프라인·sampler 유지, 데이터는 train set 사용
    val_dl_cfg = copy.deepcopy(cfg.val_dataloader)
    val_dl_cfg.dataset.ann_file = cfg.train_dataloader.dataset.ann_file
    cfg.val_dataloader = val_dl_cfg
    dataloader = runner.build_dataloader(val_dl_cfg)

    with torch.no_grad():
        for batch in dataloader:
            if (_img_encoder_count >= args.num_samples and
                    _bev_encoder_count >= args.num_samples):
                break
            try:
                model.val_step(batch)
            except Exception as e:
                print(f'  [경고] 배치 처리 실패 (무시): {e}')

    h1.remove()
    h2.remove()

    print(f'\n[STCOcc Calib] 수집 완료')
    print(f'  img_encoder : {_img_encoder_count} 샘플 → {args.out_dir}/img_encoder/')
    print(f'  bev_encoder : {_bev_encoder_count} 샘플 → {args.out_dir}/bev_encoder/')

    # 수집된 텐서 shape 요약
    if _img_encoder_shape is not None:
        print(f'  img_encoder shape: {_img_encoder_shape}')
    if _bev_encoder_shape is not None:
        print(f'  bev_encoder shape: {_bev_encoder_shape}')


if __name__ == '__main__':
    main()
