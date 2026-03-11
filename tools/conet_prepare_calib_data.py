"""
CONet INT8 PTQ - 단계 1: Calibration 데이터 수집

TRT INT8 calibrator에 필요한 img_backbone 입력 텐서와
occ_encoder_backbone 입력 텐서를 캡처해 .npy 파일로 저장한다.

수집 데이터: training set — val_dataloader 구조 그대로 + ann_file만 train pkl로 교체.
(val 전처리 파이프라인·sampler를 그대로 써서 val_step 호출 가능하고,
 데이터는 train set에서 수집한다.)

STCOcc/FusionOcc 대비 주요 변경:
  - 모델 접근: model.img_backbone / model.occ_encoder_backbone (forward_projection 없음)
  - img_backbone 입력: [B*N, 3, 896, 1600]  (H=896, W=1600, N=6 카메라)
  - bev_encoder 입력:  [1, 80, 8, 100, 100]
    (occ_fuser 이후 fused voxel, numC_Trans=80, lss_downsample=[2,2,2] → 100×100×8)
  - BEV 모듈 이름: occ_encoder_backbone (img_bev_encoder_backbone 아님)
  - import: projects.CONet

사용법:
    python tools/conet_prepare_calib_data.py \\
        --config     projects/CONet/configs/multimodal_r50_img1600_cascade_x4_occ3d_unified.py \\
        --checkpoint <checkpoint.pth> \\
        --out-dir    calib_data/conet \\
        --num-samples 100
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
    """img_backbone 직전 입력 텐서를 캡처하는 forward pre-hook.

    CONet image_encoder() 내부 흐름:
        B, N, C, imH, imW = imgs.shape
        imgs_flat = imgs.view(B * N, C, imH, imW)
        backbone_feats = self.img_backbone(imgs_flat)
    → hook에서 캡처되는 텐서: [B*N, 3, 896, 1600] = [6, 3, 896, 1600]
    """
    global _img_encoder_count, _img_encoder_shape
    os.makedirs(os.path.join(out_dir, 'img_encoder'), exist_ok=True)

    def hook(module, args):
        global _img_encoder_count, _img_encoder_shape
        if _img_encoder_count >= max_samples:
            return
        img = args[0]   # [B*N, 3, H, W]
        arr = img.float().cpu().numpy()
        path = os.path.join(out_dir, 'img_encoder', f'{_img_encoder_count:05d}.npy')
        np.save(path, arr)
        if _img_encoder_shape is None:
            _img_encoder_shape = arr.shape
        if _img_encoder_count % 10 == 0:
            print(f'  img_encoder samples: {_img_encoder_count + 1}/{max_samples}'
                  f'  shape={arr.shape}')
        _img_encoder_count += 1
        del arr

    return hook


def _make_bev_encoder_hook(out_dir: str, max_samples: int):
    """occ_encoder_backbone 직전 fused BEV voxel 텐서를 캡처하는 forward pre-hook.

    CONet occ_encoder() 내부 흐름:
        voxel_feats = occ_fuser(img_voxel_feats, pts_voxel_feats)
        x = self.occ_encoder_backbone(voxel_feats)  ← 이 시점 캡처
    → BEV voxel shape: [B, numC_Trans, Z, H_bev, W_bev]
      numC_Trans  = 80
      grid: lss_downsample=[2,2,2], occ_size=[200,200,16]
        → Z    = 16/2 = 8
        → H_bev = W_bev = 200/2 = 100
    → [1, 80, 8, 100, 100]
    """
    global _bev_encoder_count, _bev_encoder_shape
    os.makedirs(os.path.join(out_dir, 'bev_encoder'), exist_ok=True)

    def hook(module, args):
        global _bev_encoder_count, _bev_encoder_shape
        if _bev_encoder_count >= max_samples:
            return
        bev = args[0]   # [B, C, Z, H_bev, W_bev]
        arr = bev.float().cpu().numpy()
        path = os.path.join(out_dir, 'bev_encoder', f'{_bev_encoder_count:05d}.npy')
        np.save(path, arr)
        if _bev_encoder_shape is None:
            _bev_encoder_shape = arr.shape
        if _bev_encoder_count % 10 == 0:
            print(f'  bev_encoder samples: {_bev_encoder_count + 1}/{max_samples}'
                  f'  shape={arr.shape}')
        _bev_encoder_count += 1
        del arr

    return hook


# ──────────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description='CONet INT8 PTQ - calibration data 수집')
    parser.add_argument('--config',      required=True,
                        help='CONet config 파일 경로')
    parser.add_argument('--checkpoint',  required=True,
                        help='모델 checkpoint (.pth)')
    parser.add_argument('--out-dir',     required=True,
                        help='수집 데이터 저장 디렉토리')
    parser.add_argument('--num-samples', type=int, default=100,
                        help='수집할 calibration 샘플 수 (권장 100~200)\n'
                             '주의: img_encoder 텐서 1개 ≈ 130 MB (fp32, 6×3×896×1600)')
    parser.add_argument('--launcher',    default='none')
    return parser.parse_args()


def main():
    args = parse_args()

    # ── mmengine Runner 초기화 ──────────────────────────────────
    from mmengine.config import Config
    from mmengine.runner import Runner

    import mmdet3d        # noqa: F401
    import projects.CONet  # noqa: F401

    cfg = Config.fromfile(args.config)
    cfg.launcher  = args.launcher
    cfg.load_from = args.checkpoint
    cfg.work_dir  = '/tmp/conet_calib'

    runner = Runner.from_cfg(cfg)
    runner.load_or_resume()

    model = runner.model
    model.eval()
    model.cuda()

    # ── 모델 컴포넌트 접근 ─────────────────────────────────────
    # CONet(OccNet)은 forward_projection 없이
    # img_backbone / occ_encoder_backbone에 직접 접근한다.
    m = model.module if hasattr(model, 'module') else model

    assert hasattr(m, 'img_backbone'), \
        'img_backbone 속성을 찾을 수 없습니다. 모델 타입을 확인하세요.'
    assert hasattr(m, 'occ_encoder_backbone'), \
        'occ_encoder_backbone 속성을 찾을 수 없습니다.'

    # ── forward pre-hook 등록 ──────────────────────────────────
    h1 = m.img_backbone.register_forward_pre_hook(
        _make_img_encoder_hook(args.out_dir, args.num_samples)
    )
    h2 = m.occ_encoder_backbone.register_forward_pre_hook(
        _make_bev_encoder_hook(args.out_dir, args.num_samples)
    )

    print(f'\n[CONet Calib] 데이터 수집 시작 (목표: {args.num_samples}개 샘플)')
    print(f'  저장 경로: {args.out_dir}\n')

    # ── val_dataloader 구조 + train ann_file 사용 ──────────────
    # val 전처리 파이프라인(aug 없음)을 그대로 쓰되 데이터는 train set에서 수집.
    val_dl_cfg = copy.deepcopy(cfg.val_dataloader)
    val_dl_cfg.dataset.ann_file = cfg.train_dataloader.dataset.ann_file

    # CONet의 커스텀 DistributedSampler는 __init__에서 즉시
    # torch.distributed.get_world_size()를 호출한다.
    # --launcher none(단일 프로세스) 환경에서는 init_process_group이 호출되지
    # 않으므로 ValueError가 발생한다.
    # → 단일 프로세스용 DefaultSampler로 교체한다.
    val_dl_cfg.sampler = dict(type='DefaultSampler', shuffle=False)

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

    print(f'\n[CONet Calib] 수집 완료')
    print(f'  img_encoder  : {_img_encoder_count} 샘플  →  {args.out_dir}/img_encoder/')
    print(f'  bev_encoder  : {_bev_encoder_count} 샘플  →  {args.out_dir}/bev_encoder/')

    if _img_encoder_shape is not None:
        print(f'  img_encoder shape : {_img_encoder_shape}')
    if _bev_encoder_shape is not None:
        print(f'  bev_encoder shape : {_bev_encoder_shape}')

    print(f'\n다음 단계: python tools/conet_export_onnx.py '
          f'--config {args.config} --checkpoint {args.checkpoint} '
          f'--out-dir onnx/conet')


if __name__ == '__main__':
    main()
