"""
CONet INT8 PTQ - 단계 2: ONNX Export

img_backbone (ResNet-50 only, SECONDFPN neck 제외) 와
bev_encoder (occ_encoder_backbone=CustomResNet3D) 를 각각 ONNX 파일로 내보낸다.

STCOcc/FusionOcc 대비 주요 변경:
  - backbone : ResNet-50 (out_indices=(0,1,2,3)), DCN 없음
               → forward 시 4개 출력: [stage0, stage1, stage2, stage3]
                 stage0 [B*N, 256, H/4,  W/4 ]
                 stage1 [B*N, 512, H/8,  W/8 ]
                 stage2 [B*N,1024, H/16, W/16]
                 stage3 [B*N,2048, H/32, W/32]
  - neck     : SECONDFPN 은 TRT 에 포함하지 않음 → PyTorch FP16 유지
               이유: SECONDFPN 의 ConvTranspose2d(stride=2) 가 TRT INT8 에서
                     'cuTensor permutate execute failed' 를 유발하는 알려진 버그
                     (upsample_strides=[0.25, 0.5, 1, 2] 중 stride=2 분기)
  - BEV 모듈: occ_encoder_backbone (img_bev_encoder_backbone 아님)
  - img input : [6, 3, 896, 1600]  (static: B*N=6 고정)
  - bev input : [1, 80, 8, 100, 100]  (static)
  - import: projects.CONet

사용법:
    python tools/conet_export_onnx.py \\
        --config     projects/CONet/configs/multimodal_r50_img1600_cascade_x4_occ3d_unified.py \\
        --checkpoint <checkpoint.pth> \\
        --out-dir    onnx/conet

출력 파일:
    onnx/conet/img_encoder.onnx   (ResNet50 backbone only, 4 stage outputs)
    onnx/conet/bev_encoder.onnx   (CustomResNet3D)
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

class CONetBackboneWrapper(nn.Module):
    """ResNet50 backbone only 래퍼 (SECONDFPN neck 제외).

    SECONDFPN(ConvTranspose2d stride=2) 이 TRT INT8 에서 cuTENSOR 에러를 유발하므로
    backbone 만 TRT 로 내보내고 neck 은 PyTorch FP16 으로 유지한다.

    ResNet50 (out_indices=(0,1,2,3)) 출력 (H=896, W=1600 기준):
        stage0 [B*N, 256,  224, 400]
        stage1 [B*N, 512,  112, 200]
        stage2 [B*N,1024,  56,  100]
        stage3 [B*N,2048,  28,   50]

    ONNX 출력:
        stage0, stage1, stage2, stage3  (4개 feature map)
    """

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, img: torch.Tensor):
        feats = self.backbone(img)
        # ResNet50 out_indices=(0,1,2,3) → list of 4 tensors
        if isinstance(feats, (list, tuple)):
            return tuple(feats)
        return (feats,)


class CONetOccEncoderWrapper(nn.Module):
    """CustomResNet3D (occ_encoder_backbone) 래퍼.

    입력 : fused_voxel [B, numC_Trans, Z, H_bev, W_bev]
      numC_Trans = 80  (img_voxel + lidar_voxel 각 80ch 융합 후 80ch)
      lss_downsample=[2,2,2], occ_size=[200,200,16]
        → Z     = 16/2 = 8
        → H_bev = W_bev = 200/2 = 100
    → [1, 80, 8, 100, 100]

    출력 : tuple of 4 feature maps (voxel_out_indices=(0,1,2,3))
        feat_0: [B,  80, 8, 100, 100]  (block_inplanes[0]=80)
        feat_1: [B, 160, 4,  50,  50]  (block_inplanes[1]=160)
        feat_2: [B, 320, 2,  25,  25]  (block_inplanes[2]=320)
        feat_3: [B, 640, 1,  13,  13]  (block_inplanes[3]=640, ceil_mode 적용)
    """

    def __init__(self, occ_backbone: nn.Module):
        super().__init__()
        self.occ_backbone = occ_backbone

    def forward(self, fused_voxel: torch.Tensor):
        feats = self.occ_backbone(fused_voxel)
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
    """img_backbone(ResNet50 only) ONNX export.  SECONDFPN neck 제외.

    Args:
        model    : CONet OccNet 인스턴스 (module unwrap 완료)
        out_path : 저장 경로 (.onnx)
        opset    : ONNX opset 버전 (권장 17)

    SECONDFPN 을 포함하지 않는 이유:
        SECONDFPN 의 ConvTranspose2d(stride=2) 가 TRT INT8 빌드 시
        cuTENSOR permutation 커널을 선택하고, 이 커널이 대형 입력
        [6, 3, 896, 1600] 에서 'cuTensor permutate execute failed' +
        'illegal memory access' 를 유발한다.
        → backbone(ResNet-50) 만 INT8 TRT 로 변환하고,
          SECONDFPN 은 PyTorch FP16 autocast 하에서 실행한다.

    ONNX 출력: stage0~3 (4개 feature map)
        stage0 [6, 256,  224, 400]
        stage1 [6, 512,  112, 200]
        stage2 [6,1024,   56, 100]
        stage3 [6,2048,   28,  50]
    """
    print('\n[1/2] img_encoder ONNX export  (ResNet50 backbone only, neck 제외) ...')

    backbone = model.img_backbone
    _disable_with_cp(backbone)
    backbone.eval()

    wrapper = CONetBackboneWrapper(backbone).cuda().eval()

    # CONet config: input_size=(896, 1600), N=6 카메라
    H, W = 896, 1600
    B_N  = 6        # batch=1, N_cam=6 → B*N=6 (항상 고정)
    dummy = torch.randn(B_N, 3, H, W, device='cuda')

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy,),
            out_path,
            opset_version=opset,
            input_names=['img'],
            output_names=['stage0', 'stage1', 'stage2', 'stage3'],
            # ResNet backbone은 B*N=6으로 고정이므로 static shape으로 export.
            # dynamic_axes를 사용하면 TRT 빌드 시 cuTENSOR 전술이 선택될 수 있음.
            dynamic_axes=None,
            do_constant_folding=True,
        )

    size_mb = os.path.getsize(out_path) / 1e6
    print(f'  저장: {out_path}  ({size_mb:.1f} MB)')

    _verify_onnx(out_path, dummy, wrapper,
                 ['stage0', 'stage1', 'stage2', 'stage3'])


def export_bev_encoder(model, out_path: str, opset: int = 17):
    """occ_encoder_backbone(CustomResNet3D) ONNX export.

    Args:
        model    : CONet OccNet 인스턴스 (module unwrap 완료)
        out_path : 저장 경로 (.onnx)
        opset    : ONNX opset 버전

    BEV voxel 입력 shape: [B, numC_Trans, Z, H_bev, W_bev] = [1, 80, 8, 100, 100]
      - numC_Trans = 80
      - lss_downsample=[2,2,2], occ_size=[200,200,16]
          → Z=8, H_bev=W_bev=100

    주의: IInt8EntropyCalibrator2 + 5D(3D Conv) + dynamic_axes 조합 시
          calibrator::add 내부에서 illegal memory access가 발생할 수 있으므로
          고정 shape(batch=1)로 export한다.
    """
    print('\n[2/2] bev_encoder ONNX export  (CustomResNet3D) ...')

    occ_bb = model.occ_encoder_backbone
    _disable_with_cp(occ_bb)
    occ_bb.eval()

    wrapper = CONetOccEncoderWrapper(occ_bb).cuda().eval()

    # CONet BEV voxel shape:
    # numC_Trans=80, lss_downsample=[2,2,2], occ_size=[200,200,16]
    # → Z=16/2=8, H_bev=W_bev=200/2=100
    B, C_in, Z, H_bev, W_bev = 1, 80, 8, 100, 100
    dummy = torch.randn(B, C_in, Z, H_bev, W_bev, device='cuda')

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)

    # 출력 텐서 수 동적 파악 (voxel_out_indices=(0,1,2,3) → 4개)
    with torch.no_grad():
        sample_out = wrapper(dummy)
    n_out = len(sample_out)
    output_names = [f'occ_feat_{i}' for i in range(n_out)]

    print(f'  occ_encoder_backbone 출력 수: {n_out}개')
    for i, feat in enumerate(sample_out):
        print(f'    occ_feat_{i}: {tuple(feat.shape)}')

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy,),
            out_path,
            opset_version=opset,
            input_names=['fused_voxel'],
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
    parser = argparse.ArgumentParser(description='CONet ONNX export')
    parser.add_argument('--config',     required=True,
                        help='CONet config 파일 경로')
    parser.add_argument('--checkpoint', required=True,
                        help='모델 checkpoint (.pth)')
    parser.add_argument('--out-dir',    default='onnx/conet',
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
    import mmdet3d          # noqa: F401
    import projects.CONet   # noqa: F401

    cfg = Config.fromfile(args.config)
    cfg.launcher  = args.launcher
    cfg.load_from = args.checkpoint
    cfg.work_dir  = '/tmp/conet_onnx'

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
          f'--calib-dir calib_data/conet '
          f'--engine-dir engines/conet '
          f'--precision int8')


if __name__ == '__main__':
    main()
