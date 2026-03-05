"""STCOcc 정밀도(FP16 / INT8) 패치 유틸리티

── FP16 전략 두 가지 ────────────────────────────────────────────────────────
patch_full_fp16(model)          [권장]
    전체 모델을 FP16 으로 변환하되, BatchNorm 은 FP32 유지.
    수치적으로 민감한 연산(softmax, layernorm, torch.inverse 등)은
    torch.cuda.amp.autocast() 및 모델 파일 내 명시적 .float() 캐스팅이 보장.
    호출 후 반드시 autocast 컨텍스트 안에서 추론해야 함.
    Deformable attention CUDA 커널도 FP16 버전으로 교체 (추가 ~160 MB 절감).
    → 대부분의 연산이 FP16 Tensor Core 로 실행 → 속도 향상 + 메모리 절감.

patch_selective_fp16(model)     [구버전 호환용]
    img_backbone / img_neck / img_bev_encoder_backbone 만 FP16 변환.
    나머지 모듈은 FP32 유지, 각 경계에서 명시적 FP32↔FP16 캐스팅.
    autocast 불필요. 메모리 절감은 미미하나 가장 안전한 방식.

── INT8 ──────────────────────────────────────────────────────────────────────
inject_int8_engines(model, eng_dir, TRTModule, output_fp16=False)
    img_backbone+neck, img_bev_encoder_backbone 를 TRT INT8 엔진으로 교체.
    output_fp16=False: 나머지 모듈 FP32 유지. TRT I/O FP32.
    output_fp16=True:  나머지 모듈 FP16 (호출 전 patch_full_fp16 등 적용). TRT 출력만 .half()로 변환해 전달.
    test.py --int8-engines 시 기본 FP16 + 해당 3개만 INT8 로 동작하도록 output_fp16=True 사용.
"""

import os
import torch
import torch.nn as nn


# ── 내부 헬퍼 ──────────────────────────────────────────────────────────────────

def _half_module_keep_bn_float(module: nn.Module) -> nn.Module:
    """모듈을 fp16으로 변환하되 BatchNorm 계층만 fp32를 유지."""
    module.half()
    for m in module.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.float()
    return module


# ── 공개 API ───────────────────────────────────────────────────────────────────

def patch_full_fp16(model: nn.Module) -> int:
    """STCOcc 모델 전체를 FP16 으로 변환 (BatchNorm 제외).

    PyTorch AMP(Automatic Mixed Precision) 방식:
      - 대부분의 연산(Conv, Linear, MatMul) → FP16 Tensor Core 실행
      - 수치 민감 연산은 자동/명시적으로 FP32 유지:
          · autocast 가 자동 처리: softmax, layer_norm, cross_entropy,
            log, exp, sum(complex) 등
          · BN: 본 함수에서 명시적으로 FP32 복원
          · torch.inverse(): 모델 파일 내 명시적 .float() 캐스팅으로 처리
          · MultiScaleDeformableAttnFunction_fp32 → fp16 버전으로 교체:
            @custom_fwd(cast_inputs=fp16) 로 CUDA 커널이 FP16 실행.
            apply() 출력 FP32(80MB)→FP16(40MB), 4회 호출 × ~160MB 절감.

    주의: 호출 후 반드시 torch.cuda.amp.autocast() 컨텍스트 안에서 추론해야
    함. SCA/TSA 파일의 .float() 캐스팅이 제거되어 FP16 텐서가 직접 FP16
    커널로 전달된다.

    STCOcc 데이터 파이프라인은 이미지를 FP32 로 로드하므로, forward_projection
    의 진입점에 FP32 → FP16 입력 캐스팅 패치도 함께 적용한다.

    forward_projection 속성이 없는 모델에서도 model.half() + BN float() 는
    정상 실행된다(no-op 패치).

    Returns:
        FP32 로 유지된 BN 레이어 수
    """
    # 1. 전체 모델 FP16 변환
    model.half()

    # 2. BN → FP32 복원
    #    autocast 는 BN 을 자동으로 FP32 로 올리지 않으므로 명시적으로 처리.
    bn_count = 0
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.float()
            bn_count += 1

    # 3. Deformable Attention CUDA 커널 → FP16 버전으로 교체 ─────────────────────
    #    MultiScaleDeformableAttnFunction_fp32 (@custom_fwd cast_inputs=fp32) 는
    #    autocast 를 무시하고 항상 FP32 로 강제 실행된다.
    #    model.half() 이후 attention 텐서가 FP16 이므로:
    #      · FP16 커널로 교체 시 apply() 출력: FP32 80 MB → FP16 40 MB
    #      · 4 회 호출(spatial×2, temporal×2) → ~160 MB 절감
    #      · SCA/TSA 파일의 .float() 입력 캐스팅을 제거했으므로
    #        FP16 텐서가 직접 FP16 커널로 전달 → 중간 FP32 복사본 없음
    #    FP32 모드: 이 함수 자체가 호출되지 않으므로 영향 없음.
    #    selective FP16 모드: attention 모듈이 FP32 이므로 이 교체 불필요.
    import sys as _sys
    _deform_swapped = False
    try:
        _msdaf16 = None
        for _key, _mod in list(_sys.modules.items()):
            if _mod is None:
                continue
            if ('multi_scale_deformable_attn_function' in _key
                    and 'STCOcc' in _key):
                _cls = getattr(_mod, 'MultiScaleDeformableAttnFunction_fp16',
                               None)
                if _cls is not None:
                    _msdaf16 = _cls
                    break
        if _msdaf16 is not None:
            for _key, _mod in list(_sys.modules.items()):
                if _mod is None:
                    continue
                if ('STCOcc' in _key and (
                        'spatial_cross_attention' in _key
                        or 'temporal_self_attention' in _key)):
                    if hasattr(_mod, 'MultiScaleDeformableAttnFunction_fp32'):
                        _mod.MultiScaleDeformableAttnFunction_fp32 = _msdaf16
                        _deform_swapped = True
        if _deform_swapped:
            print('  [FP16] Deformable attn FP32→FP16 CUDA 커널 교체 완료'
                  ' (spatial×2, temporal×2, ~160 MB 절감 예상)')
        else:
            print('  [FP16] Deformable attn swap: 대상 모듈을 찾지 못했습니다.')
    except Exception as _e:
        print(f'  [FP16] Deformable attn swap 실패: {_e}')

    # 4. STCOcc forward_projection 진입점 패치 ─────────────────────────────────
    #    데이터 파이프라인(to_float32=True)이 FP32 텐서를 넘기므로,
    #    FP16 backbone 에 맞게 입력을 캐스팅한다.
    fp = getattr(model, 'forward_projection', None)
    if fp is None:
        return bn_count

    # extract_img_feat: img[0](이미지 픽셀 텐서)만 FP16 으로 캐스팅
    if hasattr(fp, 'extract_img_feat'):
        _orig_eif = fp.extract_img_feat

        def _cast_img_fp16(img, img_metas, **kwargs):
            if isinstance(img, torch.Tensor):
                img = img.half()
            elif isinstance(img, (list, tuple)):
                img_list = list(img)
                if (img_list
                        and isinstance(img_list[0], torch.Tensor)
                        and img_list[0].is_floating_point()):
                    img_list[0] = img_list[0].half()
                img = type(img)(img_list)
            return _orig_eif(img, img_metas, **kwargs)

        fp.extract_img_feat = _cast_img_fp16

    # extract_stereo_ref_feat: extra_ref_frame 경로, 입력을 FP16 으로 캐스팅
    if hasattr(fp, 'extract_stereo_ref_feat'):
        _orig_srf = fp.extract_stereo_ref_feat

        def _cast_srf_fp16(x):
            return _orig_srf(x.half())

        fp.extract_stereo_ref_feat = _cast_srf_fp16

    # img_view_transformer: mlp_input(카메라 파라미터) 등이 FP32 로 들어오는
    # 경우를 위해 forward pre-hook 으로 모든 FP32 텐서를 FP16 으로 캐스팅.
    # autocast 가 활성화되어 있어도 hook 입력은 캐스팅되지 않으므로 필요.
    if hasattr(fp, 'img_view_transformer'):
        def _vt_fp16_hook(module, args):
            def _cast(x):
                if isinstance(x, torch.Tensor) and x.dtype == torch.float32:
                    return x.half()
                if isinstance(x, (list, tuple)):
                    return type(x)(_cast(v) for v in x)
                return x
            return tuple(_cast(a) for a in args)

        fp.img_view_transformer.register_forward_pre_hook(_vt_fp16_hook)

    return bn_count


def patch_selective_fp16(model: nn.Module) -> int:
    """STCOcc의 img_backbone, img_neck, img_bev_encoder_backbone만 FP16으로 변환.

    나머지 모듈(view_transformer, deformable attention 등)은 FP32를 유지하므로
    model.half() 나 autocast 없이도 dtype 불일치가 발생하지 않는다.

    각 모듈 진입점(image_encoder, extract_stereo_ref_feat, bev_encoder)을
    monkey-patch 하여 FP32 입력 → FP16 실행 → FP32 출력 흐름을 보장한다.

    forward_projection 속성이 없으면 no-op.

    Returns:
        FP16으로 변환된 서브모듈 수 (0 이면 패치 대상 없음)
    """
    fp = getattr(model, 'forward_projection', None)
    if fp is None:
        return 0

    count = 0
    for name in ('img_backbone', 'img_neck', 'img_bev_encoder_backbone'):
        m = getattr(fp, name, None)
        if m is not None:
            _half_module_keep_bn_float(m)
            count += 1
            print(f'  [FP16] {name} → fp16 (BN 은 fp32 유지)')

    if count == 0:
        return 0

    # ── image_encoder ──────────────────────────────────────────────────────────
    # 원본: img_backbone(fp16) + img_neck(fp16) 실행 후 [B,N,...] reshape
    # 패치: FP32 입력 → fp16 캐스팅 후 원본 호출 → 출력을 FP32로 캐스팅
    if hasattr(fp, 'image_encoder'):
        _orig_ie = fp.image_encoder

        def _fp16_image_encoder(img, stereo=False):
            feats, stereo_feat = _orig_ie(img.half(), stereo=stereo)
            feats = [f.float() for f in feats]
            if stereo_feat is not None:
                stereo_feat = stereo_feat.float()
            return feats, stereo_feat

        fp.image_encoder = _fp16_image_encoder

    # ── extract_stereo_ref_feat ────────────────────────────────────────────────
    # extra_ref_frame 경로: backbone layer1 출력만 필요
    # 패치: FP32 입력 → fp16 캐스팅 → FP32 출력
    # → image_encoder 와 동일한 fp16 품질로 stereo_feat 생성 (분포 통일)
    if hasattr(fp, 'extract_stereo_ref_feat'):
        _orig_srf = fp.extract_stereo_ref_feat

        def _fp16_extract_stereo_ref_feat(x):
            return _orig_srf(x.half()).float()

        fp.extract_stereo_ref_feat = _fp16_extract_stereo_ref_feat

    # ── bev_encoder ────────────────────────────────────────────────────────────
    # 원본 bev_encoder 메서드는 이미 내부에서 img_bev_encoder_backbone 의
    # weight dtype 으로 입력을 캐스팅하므로 (x = x.to(_ld)) 별도 입력 캐스팅
    # 불필요. 출력만 FP32 로 캐스팅하면 된다.
    if hasattr(fp, 'bev_encoder'):
        _orig_bev = fp.bev_encoder

        def _fp16_bev_encoder(x):
            result = _orig_bev(x)   # 내부에서 fp16으로 캐스팅 후 실행
            if isinstance(result, (list, tuple)):
                return type(result)(f.float() for f in result)
            return result.float()

        fp.bev_encoder = _fp16_bev_encoder

    return count


def inject_int8_engines(model: nn.Module, eng_dir: str, TRTModule,
                       output_fp16: bool = False) -> None:
    """STCOcc forward_projection 에 INT8 TRT 엔진을 주입한다.

    img_backbone+img_neck    → img_encoder_int8.engine (INT8)
    img_bev_encoder_backbone → bev_encoder_int8.engine (INT8)
    나머지 모듈: output_fp16=False 이면 FP32 유지, True 이면 FP16 파이프라인용으로
    TRT 출력을 FP16으로 변환해 전달 (나머지 모듈은 호출 전에 patch_full_fp16 등으로 FP16 적용된 상태).

    TRT 엔진 내부는 INT8, I/O 는 항상 FP32. output_fp16=True 시 반환 직전에 .half() 적용.
    forward_projection 속성이 없으면 no-op.

    Args:
        model:       DDP wrapper 가 해제된 STCOcc 모델 인스턴스
        eng_dir:     .engine 파일들이 위치한 디렉토리 경로
        TRTModule:   stcocc_build_int8_engine.TRTModule 클래스 (런타임에 전달)
        output_fp16: True 면 TRT 출력을 FP16으로 변환해 반환 (나머지 모듈이 FP16일 때 사용).
    """
    fp = getattr(model, 'forward_projection', None)
    if fp is None:
        return

    def _to_out(t):
        return t.half() if output_fp16 else t.float()

    img_eng_path = os.path.join(eng_dir, 'img_encoder_int8.engine')
    bev_eng_path = os.path.join(eng_dir, 'bev_encoder_int8.engine')

    # ── image_encoder + extract_stereo_ref_feat ────────────────────────────────
    if os.path.exists(img_eng_path):
        trt_img = TRTModule(img_eng_path)

        # image_encoder: 입력 FP32로 TRT 호출 → 출력을 output_fp16 여부에 따라 FP32/FP16 반환
        def _int8_image_encoder(img, stereo=False):
            B, N, C, imH, imW = img.shape
            img_bn = img.view(B * N, C, imH, imW).float()
            neck_feat, stereo_feat = trt_img(img_bn)
            neck_feats_reshape = [_to_out(neck_feat).view(B, N, *neck_feat.shape[1:])]
            return neck_feats_reshape, _to_out(stereo_feat)

        fp.image_encoder = _int8_image_encoder

        def _int8_extract_stereo_ref_feat(img):
            B, N, C, imH, imW = img.shape
            img_bn = img.view(B * N, C, imH, imW).float()
            _, stereo_feat = trt_img(img_bn)
            return _to_out(stereo_feat)

        fp.extract_stereo_ref_feat = _int8_extract_stereo_ref_feat

        print(f'==> [INT8] img_encoder + extract_stereo_ref_feat → TRT: {img_eng_path}')
        if output_fp16:
            print(f'    출력 FP16 (나머지 모듈 FP16 파이프라인)')
        else:
            print(f'    stereo_feat 통일: extra_ref_frame + 일반 프레임 모두 INT8')
    else:
        print(f'==> [INT8] img_encoder 엔진 없음, 패치 생략: {img_eng_path}')

    # ── bev_encoder ────────────────────────────────────────────────────────────
    if os.path.exists(bev_eng_path):
        trt_bev = TRTModule(bev_eng_path)

        def _int8_bev_encoder(bev_voxel):
            feats = trt_bev(bev_voxel.float())
            return tuple(_to_out(f) for f in feats)

        fp.bev_encoder = _int8_bev_encoder
        print(f'==> [INT8] bev_encoder → TRT: {bev_eng_path}')
        if output_fp16:
            print(f'    출력 FP16 (나머지 모듈 FP16 파이프라인)')
    else:
        print(f'==> [INT8] bev_encoder 엔진 없음, 패치 생략: {bev_eng_path}')
