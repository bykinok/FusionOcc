"""
STCOcc INT8 PTQ - 단계 3: TRT INT8 엔진 빌드

단계 2에서 생성한 ONNX 파일과
단계 1에서 수집한 calibration 데이터를 이용해
TensorRT INT8 엔진(.engine)을 빌드한다.

사용법:
    python tools/stcocc_build_int8_engine.py \\
        --onnx-dir    onnx/stcocc \\
        --calib-dir   calib_data/stcocc \\
        --engine-dir  engines/stcocc \\
        --precision   int8          # fp32 | fp16 | int8

출력 파일:
    engines/stcocc/img_encoder_int8.engine
    engines/stcocc/bev_encoder_int8.engine
"""

from __future__ import annotations

import argparse
import glob
import os
import struct

import numpy as np
import torch

try:
    import tensorrt as trt
    TRT_AVAILABLE = True
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
except ImportError:
    TRT_AVAILABLE = False
    print('[경고] tensorrt 패키지가 없습니다. 설치 후 실행하세요.')
    print('  pip install tensorrt --extra-index-url https://pypi.nvidia.com')


# ──────────────────────────────────────────────────────────────
# INT8 Calibrator
# ──────────────────────────────────────────────────────────────

class NpyFolderCalibrator(
        trt.IInt8EntropyCalibrator2 if TRT_AVAILABLE else object):
    """
    단계 1에서 저장한 .npy 파일 폴더에서 배치를 읽어
    TRT INT8 calibration에 제공한다.

    파일 하나 = 하나의 forward pass 입력 (배치 포함).
    """

    def __init__(self, npy_dir: str, cache_file: str, batch_size: int = 1):
        if TRT_AVAILABLE:
            super().__init__()
        self.npy_files  = sorted(glob.glob(os.path.join(npy_dir, '*.npy')))
        self.cache_file = cache_file
        self.batch_size = batch_size
        self._idx       = 0
        self._buf       = None

        if not self.npy_files:
            raise FileNotFoundError(f'No .npy files found in {npy_dir}')
        print(f'  Calibrator: {len(self.npy_files)} 배치, dir={npy_dir}')

    def get_batch_size(self) -> int:
        return self.batch_size

    def get_batch(self, names: list[str]):
        if self._idx >= len(self.npy_files):
            return None
        arr = np.load(self.npy_files[self._idx]).astype(np.float32)
        self._idx += 1

        # GPU 버퍼 할당 (shape 변화 시에도 재할당)
        if self._buf is None or tuple(self._buf.shape) != tuple(arr.shape):
            self._buf = torch.zeros(arr.shape, dtype=torch.float32,
                                    device='cuda')

        self._buf.copy_(torch.from_numpy(arr))
        # torch.cuda.synchronize()로 버퍼 복사 완료 후 포인터 반환
        torch.cuda.synchronize()
        return [int(self._buf.data_ptr())]

    def read_calibration_cache(self) -> bytes | None:
        # ONNX 구조가 변경된 경우(dynamic→static 등) 기존 캐시를 그대로 사용하면
        # 레이어 이름 불일치로 인해 잘못된 스케일이 적용될 수 있다.
        # 환경변수 TRT_DISCARD_CALIB_CACHE=1 로 캐시 무시를 강제할 수 있도록 지원.
        import os as _os
        if _os.environ.get('TRT_DISCARD_CALIB_CACHE', '0') == '1':
            print('  [TRT_DISCARD_CALIB_CACHE=1] 캐시 무시, 재보정 실행.')
            return None
        if os.path.exists(self.cache_file):
            print(f'  캐시 로딩: {self.cache_file}')
            with open(self.cache_file, 'rb') as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache: bytes):
        os.makedirs(os.path.dirname(self.cache_file) or '.', exist_ok=True)
        with open(self.cache_file, 'wb') as f:
            f.write(cache)
        print(f'  캐시 저장: {self.cache_file}')


# ──────────────────────────────────────────────────────────────
# 엔진 빌드
# ──────────────────────────────────────────────────────────────

def build_engine(onnx_path: str,
                 engine_path: str,
                 precision: str,
                 calib_dir: str | None = None,
                 cache_path: str | None = None,
                 workspace_gb: int = 4) -> None:
    """
    Args:
        onnx_path   : 입력 ONNX 파일
        engine_path : 출력 .engine 파일
        precision   : 'fp32' | 'fp16' | 'int8'
        calib_dir   : INT8 calibration .npy 폴더 (int8일 때 필수)
        cache_path  : INT8 calibration 캐시 파일 경로
        workspace_gb: TRT 작업 메모리 (GB)
    """
    assert TRT_AVAILABLE, 'tensorrt 패키지 필요'

    print(f'\n[TRT] 엔진 빌드: {os.path.basename(onnx_path)} → {precision.upper()}')

    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(
             1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
         ) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser, \
         builder.create_builder_config() as config:

        # 작업 메모리
        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE,
            workspace_gb * (1 << 30)
        )

        # ── cuTENSOR 전술(tactic) 비활성화 ─────────────────────────────
        # TRT 10.x 는 INT8 최적화 시 cuTENSOR 기반 커널을 선택할 수 있다.
        # 이 커널이 대형 입력(예: [6,3,896,1600])에서 내부 workspace 포인터를
        # 잘못 처리해 'Internal cuTensor permutate execute failed' +
        # 'illegal memory access' 를 유발하는 알려진 TRT 10.x 버그가 있다.
        #
        # 두 가지 방어 조치를 모두 적용:
        #  1) builder_optimization_level = 2 : 공격적 최적화를 억제해
        #     cuTENSOR 기반 전술 선택 빈도를 낮춤.
        #  2) set_tactic_sources(CUBLAS|CUBLAS_LT|CUDNN) :
        #     cuTENSOR 를 직접 사용하는 소스를 제외하고 cuDNN/cuBLAS 커널만 허용.
        #
        # 이 설정으로 빌드한 엔진은 cuDNN/cuBLAS 기반 커널을 사용하므로
        # 위 버그가 발생하지 않는다. (STCOcc 처럼 해상도가 낮은 경우에는
        # cuTENSOR 커널도 정상 동작하므로 기존 STCOcc 엔진에는 영향 없음)
        try:
            config.builder_optimization_level = 2
        except AttributeError:
            pass  # TRT < 8.6: 해당 API 없음 — 무시

        try:
            config.set_tactic_sources(
                1 << int(trt.TacticSource.CUBLAS)    |
                1 << int(trt.TacticSource.CUBLAS_LT) |
                1 << int(trt.TacticSource.CUDNN)
            )
        except (AttributeError, TypeError):
            pass  # TRT 버전에 따라 TacticSource 가 없을 수 있음 — 무시
        # ────────────────────────────────────────────────────────────────

        # 정밀도 설정
        if precision == 'fp16':
            assert builder.platform_has_fast_fp16, 'fp16 미지원 GPU'
            config.set_flag(trt.BuilderFlag.FP16)

        elif precision == 'int8':
            assert builder.platform_has_fast_int8, 'INT8 미지원 GPU'
            config.set_flag(trt.BuilderFlag.INT8)
            # fp16 폴백 허용: INT8 미지원 레이어는 fp16으로 자동 fallback
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)

            assert calib_dir, 'INT8 모드에는 --calib-dir 필요'
            calibrator = NpyFolderCalibrator(
                npy_dir=calib_dir,
                cache_file=cache_path or (engine_path + '.calib_cache'),
            )
            config.int8_calibrator = calibrator

        # ONNX 파싱
        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print(f'  ONNX 파싱 오류: {parser.get_error(i)}')
                raise RuntimeError('ONNX 파싱 실패')

        # ── 전체 텐서 LINEAR(NCHW) 포맷 강제 ──────────────────────────────────
        # 문제 원인:
        #   TRT INT8 빌드 시, 대형 입력([6, 3, 896, 1600])에 대해 TRT 가 내부적으로
        #   NHWC 포맷 변환(NCHW→NHWC) 을 위한 cuTENSOR permutation 연산을 삽입한다.
        #   이 연산은 'tactic' 이 아닌 TRT 내부 format-transformation layer 로,
        #   set_tactic_sources(CUBLAS|CUDNN) 로는 제어할 수 없다.
        #   TRT 10.x 에서 이 cuTENSOR permutation 커널이 workspace 포인터를 잘못
        #   처리해 'cuTensor permutate execute failed' + 'illegal memory access' 를 유발.
        #
        # 해결:
        #   파싱 직후, 네트워크의 모든 ITensor 에 allowed_formats = LINEAR 를 설정.
        #   TRT 가 NHWC 포맷으로 변환할 수 없게 되므로 cuTENSOR permutation 자체가
        #   선택되지 않는다. cuDNN 은 NCHW 모드 INT8 conv 커널을 제공하므로
        #   빌드는 계속 가능하다. (STCOcc 처럼 해상도가 낮은 경우엔 NHWC 가 더 빠르지만,
        #   CONet 처럼 [6, 3, 896, 1600] 대형 입력에선 이 설정이 안정성을 우선한다.)
        try:
            _linear_fmt = 1 << int(trt.TensorFormat.LINEAR)
            for _i in range(network.num_inputs):
                network.get_input(_i).allowed_formats = _linear_fmt
            for _i in range(network.num_outputs):
                network.get_output(_i).allowed_formats = _linear_fmt
            for _i in range(network.num_layers):
                _lyr = network.get_layer(_i)
                for _j in range(_lyr.num_outputs):
                    _lyr.get_output(_j).allowed_formats = _linear_fmt
            print('  [NCHW 강제] 전체 텐서 → LINEAR(NCHW) 포맷으로 제한'
                  ' (cuTENSOR NHWC permutation 방지)')
        except Exception as _fe:
            print(f'  [경고] NCHW 강제 설정 실패 (무시): {_fe}')
        # ──────────────────────────────────────────────────────────────────────

        # dynamic shape profile 설정
        # bev_encoder는 고정 shape(dynamic_axes=None으로 export) → profile 불필요.
        # STCOcc img_encoder: batch_ncam이 dynamic(-1) → profile 필요.
        # FusionOcc img_encoder: SwinTransformer window_reverse 내부에서
        #   B = int(windows.shape[0] / (H*W/ws/ws)) 가 tracing 시 B=6 으로 상수화됨.
        #   따라서 dynamic_axes=None 으로 재export 해야 하지만, 혹시 과거 dynamic 버전의
        #   ONNX 가 남아 있는 경우를 대비해 input_shape[0]==-1 이어도 min=opt=max=6 으로
        #   profile 을 고정한다. STCOcc 는 N_cam 이 진정 가변이므로 기존 범위(1~12)를 유지.
        input_tensor = network.get_input(0)
        input_name   = input_tensor.name
        input_shape  = input_tensor.shape  # dynamic dim은 -1로 표시됨

        if input_shape[0] == -1:
            profile = builder.create_optimization_profile()
            # SwinTransformer 기반 img_encoder (FusionOcc)는 내부에서 B=6 상수화.
            # 캘리브레이션 데이터 파일명(calib_dir)으로 모델 종류를 추정한다.
            _calib_dir_str = str(calib_dir) if calib_dir else ''
            _is_swin_encoder = ('fusion' in _calib_dir_str.lower() or
                                'fusionocc' in _calib_dir_str.lower())
            if _is_swin_encoder:
                # SwinTransformer: window_reverse B=6 상수화 → profile 고정
                _ncam = int(input_shape[1]) if input_shape[0] == -1 else input_shape[0]
                _ncam = 6  # FusionOcc 고정 카메라 수
                min_sh = (_ncam,) + tuple(input_shape[1:])
                opt_sh = (_ncam,) + tuple(input_shape[1:])
                max_sh = (_ncam,) + tuple(input_shape[1:])
                print(f'  [FusionOcc] SwinTransformer 내부 B=6 상수 → profile 고정: {opt_sh}')
            else:
                # STCOcc 등 ResNet 계열: batch_ncam 진정 가변
                min_sh  = (1,)  + tuple(input_shape[1:])
                opt_sh  = (6,)  + tuple(input_shape[1:])   # 6 카메라가 기본
                max_sh  = (12,) + tuple(input_shape[1:])
            profile.set_shape(input_name, min=min_sh, opt=opt_sh, max=max_sh)
            config.add_optimization_profile(profile)
        else:
            # 고정 shape (bev_encoder, 또는 dynamic_axes=None 으로 export된 img_encoder)
            # IInt8EntropyCalibrator2 + 5D(3D Conv) + dynamic shape 조합이
            # calibrator::add에서 illegal memory access를 유발하므로 static shape 사용.
            pass

        # 빌드
        print(f'  빌드 중 (약 5~15분 소요) ...')
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError('TRT 네트워크 직렬화 실패')

        os.makedirs(os.path.dirname(engine_path) or '.', exist_ok=True)
        with open(engine_path, 'wb') as f:
            f.write(serialized)

        size_mb = os.path.getsize(engine_path) / 1e6
        print(f'  저장: {engine_path}  ({size_mb:.1f} MB)')


# ──────────────────────────────────────────────────────────────
# TRT 엔진 런타임 래퍼 (추론 시 사용)
# ──────────────────────────────────────────────────────────────

class TRTModule(torch.nn.Module):
    """
    저장된 TRT .engine 파일을 로딩해 추론을 실행하는 래퍼.

    사용 예:
        trt_img = TRTModule('engines/stcocc/img_encoder_int8.engine')
        neck_feat, stereo_feat = trt_img(img_tensor)

        trt_bev = TRTModule('engines/stcocc/bev_encoder_int8.engine')
        bev_feats = trt_bev(bev_voxel)
    """

    def __init__(self, engine_path: str):
        super().__init__()
        assert TRT_AVAILABLE, 'tensorrt 패키지 필요'

        with open(engine_path, 'rb') as f, \
             trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.engine_path = engine_path

        # TRT 전용 non-default CUDA 스트림.
        # torch.cuda.current_stream().cuda_stream 이 fresh 프로세스에서 0
        # (CUDA null/default stream)을 반환하는 경우, TRT 는 내부적으로 추가
        # cudaStreamSynchronize(0) 을 삽입한다. default stream 은 CUDA 전역
        # 동기화 의미(all-stream barrier)를 가지므로 autocast 가 같은 스트림에
        # 올린 FP16 커널과 충돌하여 cuTENSOR permutate execute failed 가 발생한다.
        # 전용 스트림을 사용하면 이 문제가 해결된다.
        self._trt_stream = torch.cuda.Stream()

        # 입출력 바인딩 이름 수집
        self.input_names  = []
        self.output_names = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

        # TRT 실행 시 필요한 device memory 크기
        # (활성화 버퍼 + cuDNN/cuBLAS workspace — create_execution_context() 시점에 pre-allocate)
        # PyTorch 할당자 외부에서 관리되므로 torch.cuda.memory_allocated()에 미반영.
        # 논문 보고용 "순수 활성화 메모리"를 구하려면 이 값을 PyTorch 측 측정치에 더해야 함.
        try:
            self.device_memory_bytes: int = self.engine.device_memory_size
        except AttributeError:
            # TRT 버전에 따라 API 이름이 다를 수 있음 — 0으로 폴백
            self.device_memory_bytes = 0

        print(f'[TRTModule] 로딩: {engine_path}')
        print(f'  inputs : {self.input_names}')
        print(f'  outputs: {self.output_names}')
        print(f'  TRT device memory (pre-alloc): {self.device_memory_bytes / 1024**2:.1f} MB'
              f'  ← 활성화 버퍼+workspace (PyTorch 외부 관리)')

    def forward(self, *inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        assert len(inputs) == len(self.input_names)

        current_stream = torch.cuda.current_stream()

        # TRT 전용 스트림이 PyTorch current stream의 작업이 모두 완료된 후 실행되도록 대기.
        # (입력 텐서가 current stream에서 준비된 후 TRT가 읽어야 하므로 필수)
        self._trt_stream.wait_stream(current_stream)

        with torch.cuda.stream(self._trt_stream):
            # 입력 shape 설정 (static 또는 dynamic axes 처리)
            for name, tensor in zip(self.input_names, inputs):
                self.context.set_input_shape(name, tuple(tensor.shape))
                self.context.set_tensor_address(name, tensor.data_ptr())

            # 출력 버퍼 할당
            output_tensors = []
            for name in self.output_names:
                shape = tuple(self.context.get_tensor_shape(name))
                dtype = trt.nptype(self.engine.get_tensor_dtype(name))
                pt_dtype = {
                    np.float32: torch.float32,
                    np.float16: torch.float16,
                    np.int32:   torch.int32,
                }.get(dtype, torch.float32)
                out = torch.empty(shape, dtype=pt_dtype, device='cuda')
                self.context.set_tensor_address(name, out.data_ptr())
                output_tensors.append(out)

            # 전용 스트림으로 비동기 실행.
            # non-default 스트림이므로 TRT 내부의 추가 cudaStreamSynchronize(0) 삽입이
            # 발생하지 않아 cuTENSOR permutate 충돌을 방지한다.
            self.context.execute_async_v3(
                stream_handle=self._trt_stream.cuda_stream
            )

        # PyTorch current stream이 TRT 스트림 완료 후에 출력 텐서를 사용하도록 대기.
        # (TRT 완료 전에 .half() 등을 호출하면 illegal memory access 발생)
        current_stream.wait_stream(self._trt_stream)

        return tuple(output_tensors)


# ──────────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='STCOcc TRT 엔진 빌드')
    parser.add_argument('--onnx-dir',    required=True,
                        help='stcocc_export_onnx.py 출력 디렉토리')
    parser.add_argument('--calib-dir',   default=None,
                        help='stcocc_prepare_calib_data.py 출력 디렉토리')
    parser.add_argument('--engine-dir',  default='engines/stcocc')
    parser.add_argument('--precision',   default='int8',
                        choices=['fp32', 'fp16', 'int8'])
    parser.add_argument('--workspace-gb', type=int, default=4)
    parser.add_argument('--img-only',    action='store_true',
                        help='img_encoder만 빌드')
    parser.add_argument('--bev-only',    action='store_true',
                        help='bev_encoder만 빌드')
    return parser.parse_args()


def main():
    args = parse_args()

    if not TRT_AVAILABLE:
        return

    os.makedirs(args.engine_dir, exist_ok=True)
    prec = args.precision

    if not args.bev_only:
        build_engine(
            onnx_path   = os.path.join(args.onnx_dir,   'img_encoder.onnx'),
            engine_path = os.path.join(args.engine_dir, f'img_encoder_{prec}.engine'),
            precision   = prec,
            calib_dir   = os.path.join(args.calib_dir,  'img_encoder') if args.calib_dir else None,
            cache_path  = os.path.join(args.engine_dir, 'img_encoder_calib.cache'),
            workspace_gb= args.workspace_gb,
        )

    if not args.img_only:
        build_engine(
            onnx_path   = os.path.join(args.onnx_dir,   'bev_encoder.onnx'),
            engine_path = os.path.join(args.engine_dir, f'bev_encoder_{prec}.engine'),
            precision   = prec,
            calib_dir   = os.path.join(args.calib_dir,  'bev_encoder') if args.calib_dir else None,
            cache_path  = os.path.join(args.engine_dir, 'bev_encoder_calib.cache'),
            workspace_gb= args.workspace_gb,
        )

    print(f'\n[완료] 엔진 파일이 {args.engine_dir}/ 에 저장됐습니다.')
    print('다음 단계: test.py에 --int8-engines 옵션을 추가해 추론합니다.')
    print(f'  python tools/test.py ... --int8-engines {args.engine_dir}')


if __name__ == '__main__':
    main()
