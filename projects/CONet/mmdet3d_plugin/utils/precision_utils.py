"""CONet 정밀도(FP16 / INT8) 패치 유틸리티

inject_int8_engines_conet(model, eng_dir, TRTModule, output_fp16=False)
    occ_encoder_backbone 만 TRT INT8 엔진으로 교체.
    img_backbone + img_neck(SECONDFPN) 은 PyTorch FP16 으로 유지.

CONet (OccNet) 은 STCOcc 와 달리 forward_projection 속성이 없으며, 모델에
img_backbone / img_neck / occ_encoder_backbone 이 직접 등록된다.

  img_backbone TRT INT8 불가 이유 (6회 시도, 모두 동일 에러):
    ResNet-50 + INT8 + 입력 [6, 3, 896, 1600] 조합에서 TRT 10.x 가 내부적으로
    cuTENSOR permutation 연산(NCHW→NHWC 포맷 변환)을 삽입하고, 이 연산이
    workspace 포인터를 잘못 처리해 'cuTensor permutate execute failed' +
    'illegal memory access' 를 반복적으로 유발한다.
    시도한 워크어라운드:
      1) dedicated CUDA stream       – 동일 에러
      2) static batch ONNX           – 동일 에러
      3) set_tactic_sources 비활성화  – 동일 에러 (tactic 과 별개의 내부 연산)
      4) backbone-only ONNX          – 동일 에러 (SECONDFPN 이 원인 아님)
      5) allowed_formats = LINEAR    – 동일 에러 (INT8 모드에서 NHWC 강제됨)
    결론: 이 환경(GPU + TRT 버전)에서 img_backbone INT8 TRT 는 불가.

  현재 구조:
    img_backbone + img_neck(SECONDFPN) → PyTorch FP16 autocast
    occ_encoder_backbone               → TRT INT8 (정상 동작 확인됨)

  occ_encoder() 내부 흐름:
    x = self.occ_encoder_backbone(x)   ← [1, 80, 8, 100, 100] 입력
    x = self.occ_encoder_neck(x)
    → 주입 후 occ_encoder_backbone 이 TRT 출력 반환
"""

import os
import torch
import torch.nn as nn


def inject_int8_engines_conet(model: nn.Module, eng_dir: str, TRTModule,
                               output_fp16: bool = False) -> None:
    """CONet 모델의 occ_encoder_backbone 을 INT8 TRT 로 교체.

    img_backbone 은 cuTENSOR 버그로 TRT INT8 불가 → PyTorch FP16 유지.

    Args:
        model:       CONet OccNet 인스턴스 (DDP 래퍼 자동 해제)
        eng_dir:     .engine 파일들이 위치한 디렉토리 경로
        TRTModule:   stcocc_build_int8_engine.TRTModule 클래스 (런타임에 전달)
        output_fp16: True 이면 TRT 출력을 FP16 으로 변환해 반환.
                     나머지 모듈이 FP16 파이프라인일 때 사용.
    """
    m = model.module if hasattr(model, 'module') else model

    def _to_out(t: torch.Tensor) -> torch.Tensor:
        return t.half() if output_fp16 else t.float()

    bev_eng_path = os.path.join(eng_dir, 'bev_encoder_int8.engine')

    # ── img_backbone: TRT INT8 사용 안 함 ──────────────────────────────────
    # ResNet-50 + INT8 + [6, 3, 896, 1600] 조합에서 cuTENSOR NHWC permutation
    # 버그가 반복 재현되어 워크어라운드 불가 판정. PyTorch FP16 autocast 유지.
    print('==> [FP16] img_backbone + img_neck(SECONDFPN) → PyTorch FP16 유지')
    print('    (TRT INT8 cuTENSOR NHWC permutation 버그 회피)')

    # ── occ_encoder_backbone ────────────────────────────────────────────────
    if os.path.exists(bev_eng_path):
        trt_bev = TRTModule(bev_eng_path)

        class _TRTOccBackbone(nn.Module):
            """occ_encoder_backbone 을 TRT 엔진으로 교체.

            occnet.py occ_encoder():
              x = self.occ_encoder_backbone(x)   ← [1, 80, 8, 100, 100] 입력
            TRT 엔진 I/O (conet_export_onnx.py 기준):
              input : fused_voxel [1, 80, 8, 100, 100]  (static)
              output0~3: multi-scale occ features (voxel_out_indices=(0,1,2,3))
            """
            def forward(self, fused_voxel: torch.Tensor):
                feats = trt_bev(fused_voxel.float())
                if isinstance(feats, torch.Tensor):
                    feats = (feats,)
                return tuple(_to_out(f) for f in feats)

        m.occ_encoder_backbone = _TRTOccBackbone()
        print(f'==> [INT8] occ_encoder_backbone → TRT: {bev_eng_path}')
        if output_fp16:
            print('    출력 FP16 (나머지 모듈 FP16 파이프라인)')
    else:
        print(f'==> [INT8] bev_encoder 엔진 없음, 패치 생략: {bev_eng_path}')
