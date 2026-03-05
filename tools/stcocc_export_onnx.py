"""
STCOcc INT8 PTQ - 단계 2: ONNX Export

image_encoder (img_backbone + img_neck)와
bev_encoder  (img_bev_encoder_backbone) 를 각각 ONNX 파일로 내보낸다.

사용법:
    python tools/stcocc_export_onnx.py \\
        --config projects/STCOcc/configs/stcocc_r50_704x256_16f_occ3d_36e_miou_unified.py \\
        --checkpoint work_dirs_nas_1st/stcocc_r50_704x256_16f_occ3d_36e/iter_84384_ema.pth \\
        --out-dir onnx/stcocc

출력 파일:
    onnx/stcocc/img_encoder.onnx   (backbone + neck)
    onnx/stcocc/bev_encoder.onnx   (CustomResNet3D)
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

class ImageEncoderWrapper(nn.Module):
    """
    img_backbone + img_neck 래퍼.

    STCOcc는 stereo 모드에서 backbone 출력 중 첫 번째(layer1)를
    stereo_feat로 분리하고, 나머지를 FPN에 넘긴다.

    out_indices=(0, 2, 3):
        x[0] = layer1 output  → stereo_feat  [B*N, 256, H/4,  W/4]
        x[1] = layer3 output  → FPN input #0 [B*N,1024, H/16, W/16]
        x[2] = layer4 output  → FPN input #1 [B*N,2048, H/32, W/32]

    ONNX 출력:
        neck_feat   [B*N, 256, H/16, W/16]
        stereo_feat [B*N, 256, H/4,  W/4]  (stereo depth 추정용)
    """

    def __init__(self, backbone: nn.Module, neck: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.neck = neck

    def forward(self, img: torch.Tensor):
        x = self.backbone(img)          # tuple: (layer1, layer3, layer4)
        stereo_feat = x[0]              # layer1 출력
        neck_in = x[1:]                 # (layer3, layer4) → FPN
        neck_out = self.neck(neck_in)
        if isinstance(neck_out, (list, tuple)):
            neck_feat = neck_out[0]     # CustomFPN num_outs=1
        else:
            neck_feat = neck_out
        return neck_feat, stereo_feat


class BEVEncoderWrapper(nn.Module):
    """
    img_bev_encoder_backbone (CustomResNet3D) 래퍼.

    입력 : bev_voxel [B, C_in, Z, H_bev, W_bev]
    출력 : tuple of feature maps (각 스테이지 출력)
            [B, 96, Z, H, W], [B, 160, Z/2, H/2, W/2], [B, 320, Z/4, H/4, W/4]
            (adjust_number_channel 적용 후)
    """

    def __init__(self, bev_backbone: nn.Module):
        super().__init__()
        self.bev_backbone = bev_backbone

    def forward(self, bev_voxel: torch.Tensor):
        feats = self.bev_backbone(bev_voxel)
        if isinstance(feats, (list, tuple)):
            return tuple(feats)
        return (feats,)


# ──────────────────────────────────────────────────────────────
# ONNX 내보내기
# ──────────────────────────────────────────────────────────────

def _disable_with_cp(module: nn.Module):
    """gradient checkpointing(with_cp)을 재귀적으로 비활성화.
    ONNX tracer는 checkpoint.checkpoint과 호환되지 않는다."""
    if hasattr(module, 'with_cp'):
        object.__setattr__(module, 'with_cp', False)
    for child in module.children():
        _disable_with_cp(child)


def export_img_encoder(fp, out_path: str, opset: int = 17):
    """
    Args:
        fp: BEVDetStereoForwardProjection 인스턴스
        out_path: 저장 경로 (.onnx)
    """
    print('\n[1/2] img_encoder ONNX export ...')

    backbone = fp.img_backbone
    neck     = fp.img_neck
    _disable_with_cp(backbone)  # ONNX tracer 호환
    backbone.eval()
    neck.eval()

    wrapper = ImageEncoderWrapper(backbone, neck).cuda().eval()

    # STCOcc config: input_size=(256, 704), N=6 카메라
    H, W = 256, 704
    B_N  = 6          # batch=1, N_cam=6
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
            dynamic_axes={
                'img':         {0: 'batch_ncam'},
                'neck_feat':   {0: 'batch_ncam'},
                'stereo_feat': {0: 'batch_ncam'},
            },
            do_constant_folding=True,
        )

    size_mb = os.path.getsize(out_path) / 1e6
    print(f'  저장: {out_path}  ({size_mb:.1f} MB)')

    _verify_onnx(out_path, dummy, wrapper, ['neck_feat', 'stereo_feat'])


def export_bev_encoder(fp, out_path: str, opset: int = 17):
    """
    Args:
        fp: BEVDetStereoForwardProjection 인스턴스
        out_path: 저장 경로 (.onnx)
    """
    print('\n[2/2] bev_encoder ONNX export ...')

    bev_bb = fp.img_bev_encoder_backbone
    _disable_with_cp(bev_bb)
    bev_bb.eval()

    wrapper = BEVEncoderWrapper(bev_bb).cuda().eval()

    # BEV voxel shape:
    #   C_in = forward_numC_Trans * (num_adj+1) = 80 * 2 = 160
    #   Z    = (5.4 - (-1)) / 0.8 = 8
    #   H=W  = 100
    B, C_in, Z, H_bev, W_bev = 1, 160, 8, 100, 100
    dummy = torch.randn(B, C_in, Z, H_bev, W_bev, device='cuda')

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)

    # CustomResNet3D 출력 개수 파악
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
            input_names=['bev_voxel'],
            output_names=output_names,
            # bev_encoder는 dynamic axes 사용 안 함.
            # IInt8EntropyCalibrator2 + 5D 입력(3D Conv) + dynamic axes 조합 시
            # calibrator::add 내부에서 executeV2 호출이 illegal memory access를 유발.
            # 고정 shape(batch=1)로 export해 TRT calibration context가 shape를
            # 명확히 알 수 있도록 한다.
            dynamic_axes=None,
            do_constant_folding=True,
        )

    size_mb = os.path.getsize(out_path) / 1e6
    print(f'  저장: {out_path}  ({size_mb:.1f} MB)')

    _verify_onnx(out_path, dummy, wrapper, output_names)


def _verify_onnx(onnx_path: str, dummy: torch.Tensor, wrapper: nn.Module,
                 output_names: list[str]):
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
        np_input = dummy.float().cpu().numpy()
        ort_outs = sess.run(None, {input_name: np_input})

        with torch.no_grad():
            pt_outs = wrapper(dummy.cuda())
        if not isinstance(pt_outs, (list, tuple)):
            pt_outs = [pt_outs]

        for i, (pt, ort_out) in enumerate(zip(pt_outs, ort_outs)):
            pt_np = pt.float().cpu().numpy()
            max_err = abs(pt_np - ort_out).max()
            print(f'  [{output_names[i]}] max abs error: {max_err:.6f}')

        print('  ONNX 수치 검증 통과 ✓')
    except ImportError:
        print('  onnxruntime 미설치 → 수치 검증 생략')
    except Exception as e:
        print(f'  수치 검증 경고: {e}')


# ──────────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description='STCOcc ONNX export')
    parser.add_argument('--config',     required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out-dir',    default='onnx/stcocc')
    parser.add_argument('--opset',      type=int, default=17)
    parser.add_argument('--launcher',   default='none')
    return parser.parse_args()


def main():
    args = parse_args()

    from mmengine.config import Config
    from mmengine.runner import Runner
    import mmdet3d  # noqa: F401
    import projects.STCOcc  # noqa: F401

    cfg = Config.fromfile(args.config)
    cfg.launcher   = args.launcher
    cfg.load_from  = args.checkpoint
    cfg.work_dir   = '/tmp/stcocc_onnx'

    runner = Runner.from_cfg(cfg)
    runner.load_or_resume()

    model = runner.model.module if hasattr(runner.model, 'module') \
        else runner.model
    model.eval()

    fp = model.forward_projection

    os.makedirs(args.out_dir, exist_ok=True)
    img_onnx_path = os.path.join(args.out_dir, 'img_encoder.onnx')
    bev_onnx_path = os.path.join(args.out_dir, 'bev_encoder.onnx')

    export_img_encoder(fp, img_onnx_path, opset=args.opset)
    export_bev_encoder(fp, bev_onnx_path, opset=args.opset)

    print(f'\n[완료] ONNX 파일이 {args.out_dir}/ 에 저장됐습니다.')
    print(f'  다음 단계: python tools/stcocc_build_int8_engine.py --onnx-dir {args.out_dir} ...')


if __name__ == '__main__':
    main()
