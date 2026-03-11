"""FusionOcc 정밀도(FP16 / INT8) 패치 유틸리티

inject_int8_engines_fusionocc(model, eng_dir, TRTModule, output_fp16=False)
    img_backbone+neck, img_bev_encoder_backbone 를 TRT INT8 엔진으로 교체.

FusionOcc 는 STCOcc 와 달리 forward_projection 속성이 없으며, 모델에
img_backbone / img_neck / img_bev_encoder_backbone 이 직접 등록된다.
TRT img_encoder 엔진은 backbone+neck 을 합친 단일 그래프이므로
아래 두 모듈을 쌍으로 교체한다:

  image_encoder() 내부 흐름 (fusion_occ.py):
    x = self.img_backbone(imgs)          ← [stage0, stage2, stage3]
    stereo_feat = x[0]  (if stereo)
    x = x[1:]                            ← neck 에 전달
    x = self.img_neck(x)                 ← neck_feat
    x = x[0]  (list/tuple 인 경우)

  주입 후:
    img_backbone: TRT 실행 → (neck_feat, stereo_feat) 캐싱
                  반환값 = [stereo_feat_trt, None, None]
                  x[0] = stereo_feat ✓, x[1:] = [None, None] (neck 에 전달, 무시됨)
    img_neck:     캐싱된 neck_feat 반환 (입력 무시)
"""

import os
import torch
import torch.nn as nn


def inject_int8_engines_fusionocc(model: nn.Module, eng_dir: str, TRTModule,
                                   output_fp16: bool = False) -> None:
    """FusionOcc 모델의 img_backbone+neck, img_bev_encoder_backbone을 INT8 TRT로 교체.

    Args:
        model:       FusionOcc 모델 인스턴스 (DDP 래퍼 자동 해제)
        eng_dir:     .engine 파일들이 위치한 디렉토리 경로
        TRTModule:   stcocc_build_int8_engine.TRTModule 클래스 (런타임에 전달)
        output_fp16: True 이면 TRT 출력을 FP16 으로 변환해 반환.
                     나머지 모듈이 FP16 파이프라인일 때 사용.
    """
    m = model.module if hasattr(model, 'module') else model

    def _to_out(t: torch.Tensor) -> torch.Tensor:
        return t.half() if output_fp16 else t.float()

    img_eng_path = os.path.join(eng_dir, 'img_encoder_int8.engine')
    bev_eng_path = os.path.join(eng_dir, 'bev_encoder_int8.engine')

    # ── img_backbone + img_neck ─────────────────────────────────────────────
    if os.path.exists(img_eng_path):
        trt_img = TRTModule(img_eng_path)
        _cache: dict = {'neck_feat': None}

        class _TRTBackbone(nn.Module):
            """img_backbone 을 TRT 엔진으로 교체.

            TRT 엔진 I/O:
              input : img [B*N, 3, 512, 1408]  (static, B*N=6)
              output0: neck_feat  [B*N, C, H, W]
              output1: stereo_feat [B*N, C2, H2, W2]

            반환값: [stereo_feat, None, None]
              x[0]  = stereo_feat → image_encoder 가 stereo=True 일 때 사용
              x[1:] = [None, None] → img_neck 에 전달되나 무시됨
            """
            def forward(self, imgs: torch.Tensor):
                neck_feat, stereo_feat = trt_img(imgs.float())
                _cache['neck_feat'] = _to_out(neck_feat)
                return [_to_out(stereo_feat), None, None]

        class _TRTNeck(nn.Module):
            """img_neck 을 캐시 반환으로 교체.

            _TRTBackbone.forward() 에서 이미 neck_feat 이 계산·캐싱되므로
            neck_in(=dummy) 을 무시하고 캐시를 반환한다.
            image_encoder 의 ``if type(x) in [list, tuple]: x = x[0]`` 처리를
            위해 리스트로 감싸 반환.
            """
            def forward(self, neck_in):
                cached = _cache['neck_feat']
                if cached is None:
                    raise RuntimeError(
                        '[TRTNeck] neck_feat 캐시가 비어 있습니다. '
                        'img_backbone(TRTBackbone) 이 먼저 호출되어야 합니다.'
                    )
                return [cached]

        m.img_backbone = _TRTBackbone()
        m.img_neck = _TRTNeck()
        m.with_img_neck = True  # image_encoder 내 neck 분기 활성화 유지

        print(f'==> [INT8] img_backbone + img_neck → TRT: {img_eng_path}')
        if output_fp16:
            print('    출력 FP16 (나머지 모듈 FP16 파이프라인)')
    else:
        print(f'==> [INT8] img_encoder 엔진 없음, 패치 생략: {img_eng_path}')

    # ── img_bev_encoder_backbone ────────────────────────────────────────────
    if os.path.exists(bev_eng_path):
        trt_bev = TRTModule(bev_eng_path)

        class _TRTBEVBackbone(nn.Module):
            """img_bev_encoder_backbone 을 TRT 엔진으로 교체.

            fusion_occ.py L513:
              x = self.img_bev_encoder_backbone(x)
            TRT 엔진 I/O:
              input : fusion_voxel [1, 96, 16, 200, 200]  (static)
              output0/1/2: multi-scale BEV features
            """
            def forward(self, fusion_voxel: torch.Tensor):
                feats = trt_bev(fusion_voxel.float())
                if isinstance(feats, torch.Tensor):
                    feats = (feats,)
                return tuple(_to_out(f) for f in feats)

        m.img_bev_encoder_backbone = _TRTBEVBackbone()
        print(f'==> [INT8] img_bev_encoder_backbone → TRT: {bev_eng_path}')
        if output_fp16:
            print('    출력 FP16 (나머지 모듈 FP16 파이프라인)')
    else:
        print(f'==> [INT8] bev_encoder 엔진 없음, 패치 생략: {bev_eng_path}')
