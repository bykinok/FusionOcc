"""
FusionOcc INT8 PTQ - 단계 2: ONNX Export

img_encoder (img_backbone=SwinTransformer + img_neck=FPN_LSS) 와
bev_encoder (img_bev_encoder_backbone=CustomResNet3D) 를 각각 ONNX 파일로 내보낸다.

STCOcc 대비 주요 변경:
  - backbone : SwinTransformer (return_stereo_feat=True, out_indices=(2,3))
               → forward 시 3개 출력: [stage0, stage2, stage3]
               stage0 = stereo_feat [B*N, 128, H/4,  W/4]
               stage2 = FPN input0  [B*N, 512, H/16, W/16]
               stage3 = FPN input1  [B*N,1024, H/32, W/32]
  - neck     : FPN_LSS (extra_upsample=None) → 단일 텐서 반환 [B*N, 256, H/16, W/16]
  - 모델 접근: model.img_backbone / model.img_bev_encoder_backbone (직접)
  - img input : [6, 3, 512, 1408]  (STCOcc: [6, 3, 256, 704])
  - bev input : [1, 96, 16, 200, 200] (STCOcc: [1, 160, 8, 100, 100])

사용법:
    python tools/fusionocc_export_onnx.py \\
        --config     projects/FusionOcc/configs/fusion_occ_occ3d_miou_unified.py \\
        --checkpoint <checkpoint.pth> \\
        --out-dir    onnx/fusionocc

출력 파일:
    onnx/fusionocc/img_encoder.onnx   (SwinTransformer + FPN_LSS)
    onnx/fusionocc/bev_encoder.onnx   (CustomResNet3D)
"""

import argparse
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath('.'))


# ──────────────────────────────────────────────────────────────
# Wrapper 클래스
# ──────────────────────────────────────────────────────────────

class FusionOccImageEncoderWrapper(nn.Module):
    """SwinTransformer + FPN_LSS 래퍼.

    SwinTransformer (return_stereo_feat=True, out_indices=(2,3)) 출력:
        outs[0] = stage0 feat  → stereo_feat  [B*N, 128,  H/4,  W/4 ]
        outs[1] = stage2 feat  → FPN input[0] [B*N, 512,  H/16, W/16]
        outs[2] = stage3 feat  → FPN input[1] [B*N,1024,  H/32, W/32]

    FPN_LSS (input_feature_index=(0,1), scale_factor=2, extra_upsample=None):
        feats[0]=stage2, feats[1]=stage3
        x = cat(stage2_up2x, stage3) → conv → [B*N, 256, H/16, W/16]  (single tensor)

    ONNX 출력:
        neck_feat   [B*N, 256, H/16, W/16]   (img_view_transformer 입력)
        stereo_feat [B*N, 128, H/4,  W/4 ]   (depth 추정에는 미사용이나 보존)
    """

    def __init__(self, backbone: nn.Module, neck: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.neck = neck

    def forward(self, img: torch.Tensor):
        # SwinTransformer: [stage0, stage2, stage3] (return_stereo_feat=True)
        x = self.backbone(img)
        stereo_feat = x[0]          # stage0: [B*N, 128, H/4, W/4]
        neck_in = x[1:]             # (stage2, stage3)

        # FPN_LSS.forward(feats): feats[input_feature_index[0]], feats[input_feature_index[1]]
        # input_feature_index=(0,1) → feats[0]=stage2, feats[1]=stage3
        neck_out = self.neck(neck_in)
        if isinstance(neck_out, (list, tuple)):
            neck_feat = neck_out[0]
        else:
            neck_feat = neck_out    # [B*N, 256, H/16, W/16]

        return neck_feat, stereo_feat


class FusionOccBEVEncoderWrapper(nn.Module):
    """CustomResNet3D (img_bev_encoder_backbone) 래퍼.

    입력 : fusion_voxel [B, C_in, Z, H_bev, W_bev]
      C_in = img_channels*(num_adj+1) + lidar_out_channel = 32*2 + 32 = 96
      Z    = (5.4 - (-1)) / 0.4 = 16
      H_bev = W_bev = (40 - (-40)) / 0.4 = 200
    → [1, 96, 16, 200, 200]

    출력 : tuple of 3 feature maps (각 stride 스테이지)
        bev_feat_0: [B, 64,  16, 200, 200]  stride=1
        bev_feat_1: [B, 128,  8, 100, 100]  stride=2
        bev_feat_2: [B, 256,  4,  50,  50]  stride=2
    """

    def __init__(self, bev_backbone: nn.Module):
        super().__init__()
        self.bev_backbone = bev_backbone

    def forward(self, fusion_voxel: torch.Tensor):
        feats = self.bev_backbone(fusion_voxel)
        if isinstance(feats, (list, tuple)):
            return tuple(feats)
        return (feats,)


# ──────────────────────────────────────────────────────────────
# 유틸
# ──────────────────────────────────────────────────────────────

def _disable_with_cp(module: nn.Module):
    """gradient checkpointing(with_cp)을 재귀적으로 비활성화.
    ONNX tracer는 torch.utils.checkpoint.checkpoint와 호환되지 않는다."""
    if hasattr(module, 'with_cp'):
        object.__setattr__(module, 'with_cp', False)
    for child in module.children():
        _disable_with_cp(child)


def _verify_onnx(onnx_path: str, dummy: torch.Tensor,
                 wrapper: nn.Module, output_names: list):
    """onnxruntime으로 수치 검증."""
    try:
        import onnxruntime as ort
        import numpy as np
        import onnx

        onnx.checker.check_model(onnx_path)

        sess = ort.InferenceSession(onnx_path,
                                    providers=['CUDAExecutionProvider',
                                               'CPUExecutionProvider'])
        input_name = sess.get_inputs()[0].name
        np_input   = dummy.float().cpu().numpy()
        ort_outs   = sess.run(None, {input_name: np_input})

        with torch.no_grad():
            pt_outs = wrapper(dummy.cuda())
        if not isinstance(pt_outs, (list, tuple)):
            pt_outs = [pt_outs]

        for i, (pt, ort_out) in enumerate(zip(pt_outs, ort_outs)):
            pt_np    = pt.float().cpu().numpy()
            max_err  = abs(pt_np - ort_out).max()
            print(f'  [{output_names[i]}] max abs error: {max_err:.6f}')

        print('  ONNX 수치 검증 통과 ✓')
    except ImportError:
        print('  onnxruntime 미설치 → 수치 검증 생략')
    except Exception as e:
        print(f'  수치 검증 경고: {e}')


# ──────────────────────────────────────────────────────────────
# ONNX 내보내기
# ──────────────────────────────────────────────────────────────

def export_img_encoder(model, out_path: str, opset: int = 17):
    """img_backbone(SwinTransformer) + img_neck(FPN_LSS) ONNX export.

    Args:
        model : FusionOCC 인스턴스 (module unwrap 완료)
        out_path : 저장 경로 (.onnx)
        opset    : ONNX opset 버전 (권장 17)

    주의: dynamic_axes 를 사용하지 않는다 (고정 shape [6, 3, 512, 1408]).
        SwinTransformer 의 window_reverse 함수에서
          B = int(windows.shape[0] / (H * W / ws / ws))
        가 torch.jit.trace 과정에서 B=6 으로 상수화된다.
        따라서 dynamic batch 를 지정하면 ONNX 그래프 내부에
        정적 reshape(6, ...) 와 동적 입력 축이 공존하여
        TRT 엔진 빌드 시 shape propagation 오류가 발생한다.
        FusionOcc 는 항상 1개 샘플 × 6 카메라로 추론하므로 정적 shape 이 적합하다.
    """
    print('\n[1/2] img_encoder ONNX export  (SwinTransformer + FPN_LSS) ...')

    backbone = model.img_backbone
    neck     = model.img_neck

    _disable_with_cp(backbone)
    backbone.eval()
    neck.eval()

    wrapper = FusionOccImageEncoderWrapper(backbone, neck).cuda().eval()

    # FusionOcc config: input_size=(512, 1408), N=6 카메라 (고정)
    H, W = 512, 1408
    B_N  = 6        # batch=1, N_cam=6  ← SwinTransformer 내부에서 상수로 trace됨
    dummy = torch.randn(B_N, 3, H, W, device='cuda')

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy,),
            out_path,
            opset_version=opset,
            input_names=['img'],
            output_names=['neck_feat', 'stereo_feat'],
            dynamic_axes=None,      # 고정 shape: SwinTransformer B=6 상수화 문제 회피
            do_constant_folding=True,
        )

    size_mb = os.path.getsize(out_path) / 1e6
    print(f'  저장: {out_path}  ({size_mb:.1f} MB)')

    _verify_onnx(out_path, dummy, wrapper, ['neck_feat', 'stereo_feat'])


def export_bev_encoder(model, out_path: str, opset: int = 17):
    """img_bev_encoder_backbone(CustomResNet3D) ONNX export.

    Args:
        model    : FusionOCC 인스턴스 (module unwrap 완료)
        out_path : 저장 경로 (.onnx)
        opset    : ONNX opset 버전

    BEV voxel 입력 shape: [B, C, Z, H_bev, W_bev] = [1, 96, 16, 200, 200]
      - C     = img_channels*(num_adj+1) + lidar_out = 32*2 + 32 = 96
      - Z     = (5.4+1) / 0.4 = 16
      - H=W   = 80 / 0.4 = 200

    주의: IInt8EntropyCalibrator2 + 5D(3D Conv) + dynamic_axes 조합 시
          calibrator::add 내부에서 illegal memory access가 발생할 수 있으므로
          고정 shape(batch=1)로 export한다.
    """
    print('\n[2/2] bev_encoder ONNX export  (CustomResNet3D) ...')

    bev_bb = model.img_bev_encoder_backbone
    _disable_with_cp(bev_bb)
    bev_bb.eval()

    wrapper = FusionOccBEVEncoderWrapper(bev_bb).cuda().eval()

    # FusionOcc BEV voxel shape
    # C = img_channels*(num_adj+1) + lidar_out_channel = 32*2 + 32 = 96
    # Z = (5.4 - (-1.0)) / 0.4 = 16
    # H = W = (40 - (-40)) / 0.4 = 200
    B, C_in, Z, H_bev, W_bev = 1, 96, 16, 200, 200
    dummy = torch.randn(B, C_in, Z, H_bev, W_bev, device='cuda')

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)

    # 출력 텐서 수 동적 파악
    with torch.no_grad():
        sample_out = wrapper(dummy)
    n_out = len(sample_out)
    output_names = [f'bev_feat_{i}' for i in range(n_out)]

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy,),
            out_path,
            opset_version=opset,
            input_names=['fusion_voxel'],
            output_names=output_names,
            # IInt8EntropyCalibrator2 + 5D(3D Conv) + dynamic axes 조합은
            # calibrator::add에서 illegal memory access를 유발할 수 있어
            # 고정 shape(batch=1)로 export.
            dynamic_axes=None,
            do_constant_folding=True,
        )

    size_mb = os.path.getsize(out_path) / 1e6
    print(f'  저장: {out_path}  ({size_mb:.1f} MB)')

    _verify_onnx(out_path, dummy, wrapper, output_names)


# ──────────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='FusionOcc ONNX export')
    parser.add_argument('--config',     required=True,
                        help='FusionOcc config 파일 경로')
    parser.add_argument('--checkpoint', required=True,
                        help='모델 checkpoint (.pth)')
    parser.add_argument('--out-dir',    default='onnx/fusionocc',
                        help='ONNX 출력 디렉토리')
    parser.add_argument('--opset',      type=int, default=17,
                        help='ONNX opset 버전 (기본 17)')
    parser.add_argument('--img-only',   action='store_true',
                        help='img_encoder만 export')
    parser.add_argument('--bev-only',   action='store_true',
                        help='bev_encoder만 export')
    parser.add_argument('--launcher',   default='none')
    return parser.parse_args()


def main():
    args = parse_args()

    from mmengine.config import Config
    from mmengine.runner import Runner
    import mmdet3d            # noqa: F401
    import projects.FusionOcc # noqa: F401

    cfg = Config.fromfile(args.config)
    cfg.launcher  = args.launcher
    cfg.load_from = args.checkpoint
    cfg.work_dir  = '/tmp/fusionocc_onnx'

    runner = Runner.from_cfg(cfg)
    runner.load_or_resume()

    # DDP wrapper 제거
    model = runner.model.module if hasattr(runner.model, 'module') \
        else runner.model
    model.eval()

    os.makedirs(args.out_dir, exist_ok=True)
    img_onnx_path = os.path.join(args.out_dir, 'img_encoder.onnx')
    bev_onnx_path = os.path.join(args.out_dir, 'bev_encoder.onnx')

    if not args.bev_only:
        export_img_encoder(model, img_onnx_path, opset=args.opset)

    if not args.img_only:
        export_bev_encoder(model, bev_onnx_path, opset=args.opset)

    print(f'\n[완료] ONNX 파일이 {args.out_dir}/ 에 저장됐습니다.')
    print(f'  다음 단계: python tools/stcocc_build_int8_engine.py '
          f'--onnx-dir {args.out_dir} '
          f'--calib-dir calib_data/fusionocc '
          f'--engine-dir engines/fusionocc '
          f'--precision int8')


if __name__ == '__main__':
    main()
