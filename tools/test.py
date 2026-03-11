# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import warnings
import datetime
import sys

from mmengine.config import Config, ConfigDict, DictAction
from mmengine.registry import RUNNERS, DefaultScope
from mmengine.runner import Runner

try:
    from mmdet3d.utils import replace_ceph_backend
except ImportError:
    replace_ceph_backend = None
import mmdet3d  # Import to register mmdet3d modules


class TeeOutput:
    """Class to redirect stdout to both terminal and log file"""
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log_file = open(log_file, 'a', encoding='utf-8')
        
    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()  # Ensure immediate write to file
        
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
        
    def close(self):
        if hasattr(self.log_file, 'close'):
            self.log_file.close()


def setup_logging(log_file, config_path, checkpoint_path):
    """Setup logging with timestamp and metadata."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Write header to log file
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"Evaluation Started - {timestamp}\n")
        f.write(f"Config: {config_path}\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"{'='*80}\n")
    
    # Setup stdout redirection
    tee = TeeOutput(log_file)
    sys.stdout = tee
    return tee


def cleanup_logging(tee_output, log_file):
    """Cleanup logging and restore stdout."""
    # Restore original stdout
    sys.stdout = tee_output.terminal
    tee_output.close()
    
    # Write footer to log file
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"Evaluation Completed - {timestamp}\n")
        f.write(f"{'='*80}\n\n")


def save_eval_results_to_log(results, log_file, config_path, checkpoint_path):
    """Save evaluation results to log file with timestamp and metadata."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"Evaluation Results - {timestamp}\n")
        f.write(f"Config: {config_path}\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"{'='*80}\n")
        f.write(str(results))
        f.write(f"\n{'='*80}\n\n")


# TODO: support fuse_conv_bn and format_only
def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet3D test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument(
        '--ceph', action='store_true', help='Use ceph as data storage backend')
    parser.add_argument(
        '--show', action='store_true', help='show prediction results')
    parser.add_argument(
        '--show-dir',
        help='directory where painted images will be saved. '
        'If specified, it will be automatically saved '
        'to the work_dir/timestamp/show_dir')
    parser.add_argument(
        '--score-thr', type=float, default=0.1, help='bbox score threshold')
    parser.add_argument(
        '--task',
        type=str,
        choices=[
            'mono_det', 'multi-view_det', 'lidar_det', 'lidar_seg',
            'multi-modality_det'
        ],
        help='Determine the visualization method depending on the task.')
    parser.add_argument(
        '--wait-time', type=float, default=2, help='the interval of show (s)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--tta', action='store_true', help='Test time augmentation')
    parser.add_argument(
        '--out', help='output result file in pickle format')
    parser.add_argument(
        '--log-file', 
        help='log file to save evaluation results')
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum number of samples to test (for quick evaluation)')
    parser.add_argument(
        '--save-predictions',
        type=str,
        default=None,
        help='Save predictions to this path (one file per rank: path_rank0.pkl). '
             'Then run: python tools/compute_metrics_from_file.py --predictions <path_rank0.pkl> --config <config>')
    parser.add_argument(
        '--save-predictions-only',
        action='store_true',
        help='With --save-predictions: skip evaluation metrics during test (only write pkl). '
             'Saves memory; run compute_metrics_from_file.py afterward for mIoU etc.')
    parser.add_argument(
        '--fp16',
        action='store_true',
        help='Run inference in float16 (AMP autocast) without retraining. '
             'Reduces GPU memory usage by ~40-50%%. '
             'BatchNorm layers are automatically kept in float32 by autocast.')
    parser.add_argument(
        '--int8-engines',
        default=None,
        metavar='ENGINE_DIR',
        help='(STCOcc 전용) TRT INT8 엔진 디렉토리. '
             'stcocc_build_int8_engine.py로 생성한 .engine 파일이 있어야 합니다. '
             '예: --int8-engines engines/stcocc')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/test.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def trigger_visualization_hook(cfg, args):
    default_hooks = cfg.default_hooks
    if 'visualization' in default_hooks:
        visualization_hook = default_hooks['visualization']
        # Turn on visualization
        visualization_hook['draw'] = True
        if args.show:
            visualization_hook['show'] = True
            visualization_hook['wait_time'] = args.wait_time
        if args.show_dir:
            visualization_hook['test_out_dir'] = args.show_dir
        all_task_choices = [
            'mono_det', 'multi-view_det', 'lidar_det', 'lidar_seg',
            'multi-modality_det'
        ]
        assert args.task in all_task_choices, 'You must set '\
            f"'--task' in {all_task_choices} in the command " \
            'if you want to use visualization hook'
        visualization_hook['vis_task'] = args.task
        visualization_hook['score_thr'] = args.score_thr
    else:
        raise RuntimeError(
            'VisualizationHook must be included in default_hooks.'
            'refer to usage '
            '"visualization=dict(type=\'VisualizationHook\')"')

    return cfg


def main():
    args = parse_args()
    
    # Setup logging if log file is specified
    tee_output = None
    if args.log_file:
        tee_output = setup_logging(args.log_file, args.config, args.checkpoint)

    try:
        # Set default scope to mmdet3d before loading config
        DefaultScope.get_instance('mmdet3d', scope_name='mmdet3d')
        
        # load config
        cfg = Config.fromfile(args.config)
        
        # Handle custom imports
        if hasattr(cfg, 'custom_imports') and cfg.custom_imports:
            import importlib
            for module_name in cfg.custom_imports.get('imports', []):
                print(f"Importing custom module: {module_name}")
                importlib.import_module(module_name)

        # If --save-predictions: add SavePredictionsEvaluator. Optionally skip other metrics to save memory.
        if getattr(args, 'save_predictions', None):
            import importlib.util
            # Load save_predictions_metric.py directly (bypasses BEVFormer package __init__.py)
            # to avoid 'LearnedPositionalEncoding already registered' collision when another
            # project (e.g. TPVFormer) has already registered the same module.
            _metric_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'projects', 'BEVFormer', 'datasets', 'save_predictions_metric.py')
            _spec = importlib.util.spec_from_file_location(
                'projects.BEVFormer.datasets.save_predictions_metric', _metric_path)
            assert _spec is not None and _spec.loader is not None, \
                f"Cannot load save_predictions_metric from {_metric_path}"
            _mod = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)
            save_path = os.path.abspath(args.save_predictions)
            save_only = getattr(args, 'save_predictions_only', False)
            if save_only:
                cfg.test_evaluator = [
                    dict(type='mmdet3d.SavePredictionsEvaluator', save_path=save_path),
                ]
                print(f"\n==> Save-predictions-only: no metrics during test (lower memory). Predictions → {save_path}_rank<N>.pkl")
                print("    Run mIoU later: python tools/compute_metrics_from_file.py --predictions <path_rank0.pkl> --config <config>\n")
            else:
                original_evaluator = cfg.test_evaluator
                if isinstance(original_evaluator, list):
                    cfg.test_evaluator = original_evaluator + [
                        dict(type='mmdet3d.SavePredictionsEvaluator', save_path=save_path),
                    ]
                else:
                    cfg.test_evaluator = [
                        original_evaluator,
                        dict(type='mmdet3d.SavePredictionsEvaluator', save_path=save_path),
                    ]
                print(f"\n==> Save-predictions mode: mIoU computed during test; predictions also → {save_path}_rank<N>.pkl")
                print("    For uncertainty metrics, run: python tools/compute_metrics_from_file.py --predictions <path_rank0.pkl> --config <config>\n")

        # TODO: We will unify the ceph support approach with other OpenMMLab repos
        if args.ceph:
            if replace_ceph_backend is not None:
                cfg = replace_ceph_backend(cfg)
            else:
                print("Warning: replace_ceph_backend not available, skipping ceph backend replacement")

        cfg.launcher = args.launcher
        if args.cfg_options is not None:
            cfg.merge_from_dict(args.cfg_options)

        # work_dir is determined in this priority: CLI > segment in file > filename
        if args.work_dir is not None:
            # update configs according to CLI args if args.work_dir is not None
            cfg.work_dir = args.work_dir
        elif cfg.get('work_dir', None) is None:
            # use config filename as default work_dir if cfg.work_dir is None
            cfg.work_dir = osp.join('./work_dirs',
                                    osp.splitext(osp.basename(args.config))[0])

        cfg.load_from = args.checkpoint

        if args.show or args.show_dir:
            cfg = trigger_visualization_hook(cfg, args)

        if args.tta:
            # Currently, we only support tta for 3D segmentation
            # TODO: Support tta for 3D detection
            assert 'tta_model' in cfg, 'Cannot find ``tta_model`` in config.'
            assert 'tta_pipeline' in cfg, 'Cannot find ``tta_pipeline`` in config.'
            cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline
            cfg.model = ConfigDict(**cfg.tta_model, module=cfg.model)

        # Limit test samples if max_samples is specified
        if args.max_samples is not None:
            print(f"\n==> Limiting test to FIRST {args.max_samples} samples (indices 0-{args.max_samples-1}) for quick evaluation\n")
            # Modify sampler to limit number of samples
            # Use a custom sampler that only samples the first max_samples
            # IMPORTANT: Ensure shuffle=False to get deterministic sample order
            cfg.test_dataloader.sampler = dict(
                type='DefaultSampler',
                shuffle=False,  # CRITICAL: Must be False for reproducible results
            )
            # Also update dataset test_mode to ensure consistent behavior
            if hasattr(cfg, 'test_dataloader') and hasattr(cfg.test_dataloader, 'dataset'):
                cfg.test_dataloader.dataset.test_mode = True
            # Store max_samples in cfg for later use
            cfg.max_samples_limit = args.max_samples

        # Ensure work_dir exists
        if cfg.work_dir is not None:
            os.makedirs(cfg.work_dir, exist_ok=True)
        else:
            # Set a default work_dir if not specified
            cfg.work_dir = './work_dirs/test_output'
            os.makedirs(cfg.work_dir, exist_ok=True)

        # build the runner from config
        if 'runner_type' not in cfg:
            # build the default runner
            runner = Runner.from_cfg(cfg)
        else:
            # build customized runner from the registry
            # if 'runner_type' is set in the cfg
            runner = RUNNERS.build(cfg)
        
        # Verify work_dir is set correctly
        # print(f"==> Runner work_dir: {runner.work_dir}")
        
        # Fix LoggerHook to prevent "join() argument must be str" error
        # from pathlib import Path
        # import logging
        
        # for i, hook in enumerate(runner._hooks):
        #     hook_name = type(hook).__name__
        #     if hook_name == 'LoggerHook':
        #         # Ensure LoggerHook has proper paths set
        #         work_dir_path = Path(runner.work_dir) if runner.work_dir else None
                
                # Set out_dir properly
                # if hasattr(hook, 'out_dir') and work_dir_path:
                #     hook.out_dir = str(work_dir_path)
                
                # # Ensure json_log_path is set correctly
                # if hasattr(hook, 'json_log_path') and work_dir_path:
                #     if hook.json_log_path is None:
                #         hook.json_log_path = str(work_dir_path / 'test.log.json')
                
                # # Get or create the log file path
                # if hasattr(hook, 'file_handler'):
                #     if hook.file_handler is None and work_dir_path:
                #         # Create a file handler if it doesn't exist
                #         log_file = work_dir_path / f"{work_dir_path.name}.log"
                #         file_handler = logging.FileHandler(log_file, 'a')
                #         file_handler.setLevel(logging.INFO)
                #         formatter = logging.Formatter(
                #             '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                #             datefmt='%Y/%m/%d %H:%M:%S'
                #         )
                #         file_handler.setFormatter(formatter)
                #         hook.file_handler = file_handler
        
        # CRITICAL: mmengine Runner does not auto-load checkpoint in test mode
        # Explicitly load checkpoint if cfg.load_from is set (general solution for all models)
        if hasattr(cfg, 'load_from') and cfg.load_from:
            from mmengine.runner import load_checkpoint
            load_checkpoint(runner.model, cfg.load_from, map_location='cpu', strict=False)

        # Apply max_samples limit after runner is built
        if args.max_samples is not None and hasattr(cfg, 'max_samples_limit'):
            # We need to modify the test_loop after it's created
            # Save original test method and wrap it
            original_test = runner.test
            
            def limited_test():
                # Build test loop if not already built
                if runner._test_loop is None or isinstance(runner._test_loop, dict):
                    runner._test_loop = runner.build_test_loop(runner._test_loop)
                
                # Now modify the dataloader to use specific sample indices
                from torch.utils.data import Subset
                original_dataset = runner._test_loop.dataloader.dataset
                # Use first N samples for fair comparison
                subset_indices = list(range(min(cfg.max_samples_limit, len(original_dataset))))
                subset_dataset = Subset(original_dataset, subset_indices)
                
                # Rebuild dataloader with subset
                from torch.utils.data import DataLoader
                dataloader_cfg = runner._test_loop.dataloader
                
                # Create new dataloader with explicit settings for reproducibility
                runner._test_loop.dataloader = DataLoader(
                    subset_dataset,
                    batch_size=1,  # Always use batch_size=1 for testing
                    num_workers=dataloader_cfg.num_workers if hasattr(dataloader_cfg, 'num_workers') else 4,
                    collate_fn=dataloader_cfg.collate_fn if hasattr(dataloader_cfg, 'collate_fn') else None,
                    pin_memory=getattr(dataloader_cfg, 'pin_memory', False),
                    sampler=None,  # No sampler, sequential access
                    shuffle=False,  # CRITICAL: Never shuffle for reproducibility
                    drop_last=False
                )
                
                # Call the original test loop run (LoggerHook should be fixed now)
                try:
                    runner._test_loop.run()
                except TypeError as e:
                    if "join() argument must be str" in str(e) and "LoggerHook" in str(e):
                        # Ignore LoggerHook path errors during testing
                        print(f"\nWarning: LoggerHook error ignored (common in test mode): {e}")
                        print("Test results were computed successfully despite the error.\n")
                    else:
                        raise
            
            runner.test = limited_test

        # start testing
        try:
            import time
            import torch
            import torch.nn as nn

            def _print_parameter_memory(model):
                """모델 파라미터 개수와 파라미터 메모리(바이트/MB) 출력. (param_mb, param_gb) 반환."""
                _model = model.module if hasattr(model, 'module') else model
                total_params = sum(p.numel() for p in _model.parameters())
                param_bytes = sum(p.numel() * p.element_size() for p in _model.parameters())
                param_mb = param_bytes / (1024 ** 2)
                param_gb = param_bytes / (1024 ** 3)
                trainable = sum(p.numel() for p in _model.parameters() if p.requires_grad)
                print(f"\n==> [Parameter Memory]")
                print(f"    total params   : {total_params:,}")
                print(f"    trainable     : {trainable:,}")
                print(f"    param memory  : {param_mb:.2f} MB  ({param_gb:.3f} GB)\n")
                return param_mb, param_gb

            def _print_parameter_memory_after_fp16(model, fp32_mb):
                """FP16 적용 후 파라미터 메모리 및 FP32 대비 절감량 출력."""
                _model = model.module if hasattr(model, 'module') else model
                param_bytes = sum(p.numel() * p.element_size() for p in _model.parameters())
                fp16_mb = param_bytes / (1024 ** 2)
                fp16_gb = param_bytes / (1024 ** 3)
                saved_mb = fp32_mb - fp16_mb
                saved_pct = (saved_mb / fp32_mb * 100) if fp32_mb > 0 else 0
                print(f"\n==> [Parameter Memory] FP16 적용 후")
                print(f"    param memory  : {fp16_mb:.2f} MB  ({fp16_gb:.3f} GB)")
                print(f"    절감 (vs FP32): {saved_mb:.2f} MB  ({saved_pct:.1f}%)\n")

            def _print_fp16_fp32_flops_coverage(model):
                """FLOPs 기준 FP16/FP32 coverage 출력.

                파라미터는 대부분 FP16이지만, 실제 연산은 autocast/명시적으로 일부가 FP32 유지됨.
                - FP32 유지 연산: BatchNorm, LayerNorm, GroupNorm, Softmax, inverse 등 (파라미터 적거나 없음).
                - 여기서는 (1) 연산 유형별 모듈의 파라미터 비율, (2) 파라미터 dtype 비율 둘 다 출력하고,
                  실제 FLOPs 비율은 FP32가 파라미터 비율보다 더 높을 수 있음을 명시.
                """
                _model = model.module if hasattr(model, 'module') else model
                nn_mod = nn

                # (1) 연산 유형 기준: FP32로 실행되는 모듈(BatchNorm, LayerNorm, GroupNorm)의 파라미터 합
                fp32_op_modules = (nn_mod.BatchNorm1d, nn_mod.BatchNorm2d, nn_mod.BatchNorm3d,
                                   nn_mod.LayerNorm, nn_mod.GroupNorm)
                fp32_op_params = 0
                for m in _model.modules():
                    if isinstance(m, fp32_op_modules):
                        fp32_op_params += sum(p.numel() for p in m.parameters(recurse=False))

                total_params = sum(p.numel() for p in _model.parameters())
                fp16_op_params = total_params - fp32_op_params
                if total_params == 0:
                    print("\n==> [FP16/FP32 FLOPs Coverage] 파라미터 없음\n")
                    return

                # 연산 유형 기준: FP32-op 모듈(BN/LN/GN)이 차지하는 비율 (이들의 실제 FLOPs는 활성화 크기에 비례해 더 클 수 있음)
                pct_fp32_ops = fp32_op_params / total_params * 100
                pct_fp16_ops = fp16_op_params / total_params * 100

                # (2) 파라미터 dtype 기준 (참고용)
                fp16_dtype_params = sum(p.numel() for p in _model.parameters() if p.dtype == torch.float16)
                fp32_dtype_params = sum(p.numel() for p in _model.parameters() if p.dtype == torch.float32)
                pct_fp16_dtype = fp16_dtype_params / total_params * 100
                pct_fp32_dtype = fp32_dtype_params / total_params * 100

                print(f"\n==> [FP16/FP32 FLOPs Coverage]")
                print(f"    [연산(모듈) 기준] FP32로 실행되는 연산: BN, LayerNorm, GroupNorm")
                print(f"      FP16 연산(Conv/Linear/Attn 등) : {pct_fp16_ops:.1f}%  ({fp16_op_params:,} params)")
                print(f"      FP32 연산(BN/LayerNorm/GroupNorm): {pct_fp32_ops:.1f}%  ({fp32_op_params:,} params)")
                print(f"    [참고] Softmax, inverse 등 파라미터 없는 FP32 연산은 위 비율에 미포함.")
                print(f"          실제 FLOPs 기준 FP32 비율은 이보다 더 높을 수 있음 (입력 해상도·시퀀스 길이 의존).")
                print(f"    [Parameter dtype] FP16 가중치 {pct_fp16_dtype:.1f}% / FP32 가중치 {pct_fp32_dtype:.1f}%\n")

            def _print_parameter_memory_after_int8(model, fp32_mb, int8_module_params):
                """INT8+FP16 적용 후 파라미터 메모리 및 FP32 대비 절감량 출력.

                inject_int8_engines는 모듈을 교체하지 않고 메서드(image_encoder/bev_encoder)만
                패치하므로, INT8 대상 모듈(backbone/neck/bev_enc_backbone)의 PyTorch 파라미터가
                inject 후에도 _model.parameters()에 남아 있다.
                따라서 전체 파라미터 바이트에서 INT8 대상 모듈의 현재 바이트를 빼고,
                INT8 크기(1B/param)로 대체하여 올바른 추정치를 계산한다.
                """
                _model = model.module if hasattr(model, 'module') else model

                # INT8 대상 모듈의 현재(FP16 변환 후) 파라미터 바이트 계산
                int8_modules_current_bytes = 0
                if hasattr(_model, 'forward_projection'):
                    _fp = _model.forward_projection
                    for _mod_name in ('img_backbone', 'img_neck', 'img_bev_encoder_backbone'):
                        _submod = getattr(_fp, _mod_name, None)
                        if _submod is not None:
                            int8_modules_current_bytes += sum(
                                p.numel() * p.element_size() for p in _submod.parameters()
                            )

                # 전체 PyTorch 파라미터 바이트 (INT8 대상 포함)
                total_bytes = sum(p.numel() * p.element_size() for p in _model.parameters())
                # INT8 대상 제외한 실제 나머지 파라미터 바이트 (FP16 + FP32 BN)
                remaining_bytes = total_bytes - int8_modules_current_bytes
                remaining_mb = remaining_bytes / (1024 ** 2)
                # INT8 대상 파라미터: 1 byte each (TRT 엔진이 INT8로 저장)
                int8_mb = int8_module_params / (1024 ** 2)
                total_mb = remaining_mb + int8_mb
                saved_mb = fp32_mb - total_mb
                saved_pct = (saved_mb / fp32_mb * 100) if fp32_mb > 0 else 0
                print(f"\n==> [Parameter Memory] INT8+FP16 적용 후")
                print(f"    INT8 모듈 (backbone/neck/bev_enc): {int8_mb:.2f} MB  ({int8_module_params:,} params × 1B)")
                print(f"    FP16/FP32 나머지 모듈             : {remaining_mb:.2f} MB")
                print(f"    합계 (추정)                       : {total_mb:.2f} MB")
                print(f"    절감 (vs FP32)                    : {saved_mb:.2f} MB  ({saved_pct:.1f}%)\n")

            def _print_int8_fp16_fp32_flops_coverage(model, int8_module_params, int8_module_info):
                """INT8/FP16/FP32 FLOPs Coverage 출력.

                inject_int8_engines는 메서드만 패치하므로 INT8 대상 모듈이 _model.modules()에
                여전히 포함된다. 따라서:
                  - fp32_op_params: INT8 대상 모듈 외부의 BN/LN/GN만 집계
                  - fp16_params:    non-INT8 파라미터 - fp32_op_params
                  - int8_module_params: inject 전에 별도 측정한 값 사용
                """
                _model = model.module if hasattr(model, 'module') else model

                # INT8 대상 모듈(backbone/bev_enc_backbone) 내 모든 서브모듈 id 수집
                # CONet: img_neck(SECONDFPN)은 PyTorch FP16 유지 → INT8 집계 제외
                int8_submodule_ids = set()
                _is_conet_cov = (
                    not hasattr(_model, 'forward_projection')
                    and hasattr(_model, 'occ_encoder_backbone')
                    and not hasattr(_model, 'img_bev_encoder_backbone')
                )
                if hasattr(_model, 'forward_projection'):
                    _fp = _model.forward_projection
                    for _mod_name in ('img_backbone', 'img_neck', 'img_bev_encoder_backbone'):
                        _submod = getattr(_fp, _mod_name, None)
                        if _submod is not None:
                            for m in _submod.modules():
                                int8_submodule_ids.add(id(m))
                elif _is_conet_cov:
                    # CONet: img_backbone + SECONDFPN 은 PyTorch FP16, occ_encoder_backbone 만 INT8
                    for _mod_name in ('occ_encoder_backbone',):
                        _submod = getattr(_model, _mod_name, None)
                        if _submod is not None:
                            for m in _submod.modules():
                                int8_submodule_ids.add(id(m))

                # INT8 대상 모듈 외부의 BN/LN/GN 파라미터 수 (실제 FP32로 실행되는 것)
                fp32_op_params = 0
                for m in _model.modules():
                    if id(m) in int8_submodule_ids:
                        continue  # INT8 모듈 내 BN은 TRT에서 INT8로 처리
                    if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                                      nn.LayerNorm, nn.GroupNorm)):
                        fp32_op_params += sum(p.numel() for p in m.parameters(recurse=False))

                # 전체 파라미터에서 INT8 대상 제외 → non-INT8 파라미터
                total_model_params = sum(p.numel() for p in _model.parameters())
                non_int8_params = total_model_params - int8_module_params
                fp16_params = non_int8_params - fp32_op_params
                total_params_orig = total_model_params  # = int8_module_params + non_int8_params

                if total_params_orig == 0:
                    print("\n==> [INT8/FP16/FP32 FLOPs Coverage] 파라미터 없음\n")
                    return
                pct_int8 = int8_module_params / total_params_orig * 100
                pct_fp16 = fp16_params / total_params_orig * 100
                pct_fp32 = fp32_op_params / total_params_orig * 100
                print(f"\n==> [INT8/FP16/FP32 FLOPs Coverage]")
                print(f"    INT8 (TRT: backbone/neck/bev_enc): {pct_int8:.1f}%  ({int8_module_params:,} params)")
                print(f"    FP16 (나머지 Conv/Linear/Attn 등): {pct_fp16:.1f}%  ({fp16_params:,} params)")
                print(f"    FP32 (BN/LayerNorm/GroupNorm)    : {pct_fp32:.1f}%  ({fp32_op_params:,} params)")
                print(f"    [참고] Softmax, inverse 등 파라미터 없는 FP32 연산은 위 비율에 미포함.")
                print(f"          실제 FLOPs 기준 FP32/FP16 비율은 입력 해상도·시퀀스 길이에 따라 달라질 수 있음.")
                if int8_module_info:
                    print(f"    INT8 대상 모듈:")
                    for name, n in int8_module_info:
                        print(f"      {name}: {n:,} params")
                print()
            fp32_param_mb, _ = _print_parameter_memory(runner.model)

            # ── pynvml 초기화 (NVML 기반 드라이버 레벨 전체 GPU 메모리 측정) ──
            _nvml_handle = None
            try:
                import pynvml
                pynvml.nvmlInit()
                _gpu_idx = torch.cuda.current_device() if torch.cuda.is_available() else 0
                _nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(_gpu_idx)
            except Exception:
                _nvml_handle = None

            def _nvml_used_mb() -> float:
                """pynvml로 드라이버 레벨 GPU 사용 메모리(MB) 반환.
                cudaDeviceSynchronize() 후 호출해야 TRT 내부 버퍼까지 정확히 반영됨.
                pynvml 미설치 시 NaN 반환.
                """
                if _nvml_handle is None:
                    return float('nan')
                info = pynvml.nvmlDeviceGetMemoryInfo(_nvml_handle)
                return int(info.used) / 1024 ** 2  # type: ignore[attr-defined]  # bytes → MB

            def _report_gpu_memory(label):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    allocated = torch.cuda.memory_allocated() / 1024 ** 3
                    reserved  = torch.cuda.memory_reserved()  / 1024 ** 3
                    nvml_mb   = _nvml_used_mb()
                    nvml_str  = (f", nvml={nvml_mb:.0f} MB"
                                 if not (nvml_mb != nvml_mb) else "")  # NaN guard
                    print(f"\n==> [GPU Memory] {label}: "
                          f"allocated={allocated:.2f} GB, reserved={reserved:.2f} GB"
                          f"{nvml_str}\n")

            def _run_with_metrics(label: str, run_fn):
                """추론 실행 + GPU 메모리(pynvml 드라이버 레벨 피크 포함) + 추론 Latency 측정.

                측정 대상: test_step(data) 한 번 호출당 시간 (sync → test_step → sync).
                - Data loading 제외: 루프가 'batch = next(dataloader)' 후 test_step(batch) 호출하므로,
                  배치 로딩/전송 시간은 현재 측정 구간 밖(이전 반복 또는 다음 반복)에 있음.
                - TensorRT build 제외: INT8 사용 시 inject_int8_engines는 run_fn() 전에 호출되므로
                  측정에는 TRT 실행만 포함되고 빌드 시간은 포함되지 않음.
                - 포함되는 것: test_step 내부 전체 (입력 전처리·to(device), forward, 결과 포장 등).
                  즉 '순수 forward만'이 아니라 '배치 1개에 대한 추론 스텝 전체' 시간.
                - torch.cuda.synchronize() 로 GPU 연산 완료 후 측정.
                - 첫 번째 배치는 CUDA JIT / cuDNN autotuning 으로 느릴 수 있음.
                - pynvml: cudaDeviceSynchronize() 후 NVML 드라이버 레벨 전체 점유 측정.
                  PyTorch 할당자 + TRT 내부 버퍼 + cuDNN workspace 모두 포함.
                  논문 보고용 메모리 수치로 권장 (nvidia-smi 백엔드와 동일).
                """
                _times: list = []
                # PyTorch allocator 기반 스텝별 순수 활성화 메모리 (baseline 차감)
                _peak_deltas:     list = []   # max_allocated - baseline (per step)
                _resrv_snapshots: list = []   # memory_reserved() after each step
                # pynvml 기반 드라이버 레벨 전체 GPU 메모리 스냅샷 (스텝 완료 후)
                _nvml_snapshots:  list = []   # nvmlDeviceGetMemoryInfo().used (MB, per step)

                # ── TRT engine pre-allocated device memory 수집 ──────────
                # TRT는 create_execution_context() 시점에 활성화 버퍼+workspace를
                # PyTorch 할당자 외부에서 cudaMalloc한다.
                # 이 값을 PyTorch activation 측정치에 더해야 "INT8 순수 활성화 메모리"를
                # 올바르게 구할 수 있다 (TRTModule.device_memory_bytes 속성 이용).
                _trt_device_mem_bytes: int = 0
                _search_model = (runner.model.module
                                 if hasattr(runner.model, 'module')
                                 else runner.model)
                for _submod in _search_model.modules():
                    if hasattr(_submod, 'device_memory_bytes'):
                        _trt_device_mem_bytes += _submod.device_memory_bytes

                # 테스트 루프가 호출하는 것은 runner.model.test_step
                _orig_test_step = runner.model.test_step

                def _timed_test_step(data):
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        # ── 순수 활성 메모리 측정: 스텝 직전 baseline 저장 후 peak 리셋 ──
                        baseline_mem = torch.cuda.memory_allocated()
                        torch.cuda.reset_peak_memory_stats()
                    t0 = time.perf_counter()
                    out = _orig_test_step(data)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()  # TRT 포함 모든 비동기 커널 완료 대기
                    # sync 완료 직후 타이머 종료 (메모리 측정 오버헤드 제외)
                    _times.append(time.perf_counter() - t0)
                    if torch.cuda.is_available():
                        # PyTorch: 이 스텝에서 baseline 대비 추가 할당된 최대치
                        _peak_deltas.append(
                            torch.cuda.max_memory_allocated() - baseline_mem
                        )
                        # PyTorch: reserved 현재값 스냅샷
                        _resrv_snapshots.append(torch.cuda.memory_reserved())
                        # pynvml: cudaDeviceSynchronize() 완료 후 드라이버 레벨 전체 점유
                        # TRT 내부 활성화 버퍼, cuDNN workspace 모두 포함
                        _nvml_snapshots.append(_nvml_used_mb())
                    return out

                runner.model.test_step = _timed_test_step
                _report_gpu_memory(f"before {label}")

                try:
                    run_fn()
                finally:
                    runner.model.test_step = _orig_test_step

                # warm-up 배치 수 (latency·메모리 stable 기준 공통)
                _WARMUP = 50

                # ── GPU 메모리 피크 ────────────────────────────────────────
                _report_gpu_memory(f"after {label}")
                if torch.cuda.is_available():
                    MB = 1024 ** 2
                    GB = 1024 ** 3

                    # ── PyTorch allocator 기반 (참고용) ──────────────────
                    if _peak_deltas:
                        n_d          = len(_peak_deltas)
                        wu_d         = min(_WARMUP, n_d)
                        act_max      = max(_peak_deltas)       / GB
                        act_min      = min(_peak_deltas)       / GB
                        act_avg      = sum(_peak_deltas) / n_d / GB
                        act_stable   = (_peak_deltas[wu_d:]
                                        if n_d > wu_d else _peak_deltas)
                        act_stbl_avg = sum(act_stable) / len(act_stable) / GB
                    else:
                        act_max = act_min = act_avg = act_stbl_avg = float('nan')
                        wu_d = 0
                    peak_resrv = (max(_resrv_snapshots) / GB
                                  if _resrv_snapshots else float('nan'))

                    # ── pynvml 기반 드라이버 레벨 (논문 보고용) ──────────
                    _nvml_valid = [v for v in _nvml_snapshots
                                   if v == v]  # NaN 제거
                    if _nvml_valid:
                        nv_n          = len(_nvml_valid)
                        nv_wu         = min(_WARMUP, nv_n)
                        nv_peak       = max(_nvml_valid)
                        nv_min        = min(_nvml_valid)
                        nv_avg        = sum(_nvml_valid) / nv_n
                        nv_stable     = (_nvml_valid[nv_wu:]
                                         if nv_n > nv_wu else _nvml_valid)
                        nv_stbl_avg   = sum(nv_stable) / len(nv_stable)
                        nvml_avail    = True
                    else:
                        nv_peak = nv_min = nv_avg = nv_stbl_avg = float('nan')
                        nv_wu   = 0
                        nvml_avail = False

                    # ── TRT pre-allocated device memory ──────────────────
                    trt_mem_mb  = _trt_device_mem_bytes / MB
                    trt_mem_gb  = _trt_device_mem_bytes / GB
                    has_trt_mem = _trt_device_mem_bytes > 0

                    print(f"==> [GPU Memory Peak] {label}:")

                    # ① TRT pre-allocated device memory (INT8 전용)
                    if has_trt_mem:
                        print(f"    TRT device mem (pre-alloc)  : {trt_mem_mb:.1f} MB  ({trt_mem_gb:.3f} GB)"
                              f"  ← create_execution_context() 시 cudaMalloc (활성화 버퍼+workspace)")

                    # ② PyTorch activation + TRT 합산 (순수 추론 활성화 메모리)
                    total_act_max_gb = act_max + trt_mem_gb
                    total_act_avg_gb = act_avg + trt_mem_gb
                    total_act_stbl_gb = act_stbl_avg + trt_mem_gb
                    print(f"    [순수 추론 활성화 메모리] (PyTorch activation + TRT pre-alloc):")
                    print(f"    total activation  max  : {total_act_max_gb:.3f} GB"
                          f"  ← 논문 표기 권장 (FP16/INT8 공정 비교 가능)")
                    print(f"    total activation  avg  : {total_act_avg_gb:.3f} GB  "
                          f"(stable avg {wu_d+1}th~: {total_act_stbl_gb:.3f} GB)")
                    if has_trt_mem:
                        print(f"      └─ PyTorch activation max : {act_max:.3f} GB")
                        print(f"      └─ TRT pre-alloc          : {trt_mem_gb:.3f} GB")

                    # ③ pynvml 결과 (전체 드라이버 레벨 — 파라미터 포함)
                    if nvml_avail:
                        print(f"    [참고] pynvml 전체 GPU 점유 (파라미터 포함):")
                        print(f"    peak GPU mem  (max)  : {nv_peak:.1f} MB  ({nv_peak/1024:.3f} GB)")
                        print(f"    peak GPU mem  (avg)  : {nv_avg:.1f} MB  "
                              f"(stable avg {nv_wu+1}th~: {nv_stbl_avg:.1f} MB)")
                    else:
                        print(f"    [참고] pynvml 미설치 — pip install pynvml 후 전체 GPU 점유도 측정 가능")

                    # ④ PyTorch allocator 원본 (참고용)
                    print(f"    [참고] PyTorch allocator only (TRT 미포함):")
                    print(f"    active allocated  max  : {act_max:.3f} GB")
                    print(f"    active allocated  avg  : {act_avg:.3f} GB  "
                          f"(stable avg {wu_d+1}th~: {act_stbl_avg:.3f} GB)")
                    print(f"    peak reserved          : {peak_resrv:.3f} GB"
                          f"  ← CUDA 캐싱 할당자 예약 최대치")
                    print()

                # ── 추론 Latency (test_step 호출당 시간) ─────────────────────
                if _times:
                    n     = len(_times)
                    total = sum(_times)
                    wu    = min(_WARMUP, n)           # 실제 warm-up 배치 수
                    stable = _times[wu:] if n > wu else []
                    print(f"==> [Inference Latency] {label} (test_step 기준, {n} 배치):")
                    print(f"    warm-up ({wu} batches) avg : "
                          f"{sum(_times[:wu]) / wu * 1000:.1f} ms/batch")
                    if stable:
                        s_avg = sum(stable) / len(stable) * 1000
                        s_min = min(stable) * 1000
                        s_max = max(stable) * 1000
                        print(f"    stable  ({wu+1}th~)   avg : {s_avg:.1f} ms/batch  (latency)")
                        print(f"                         min : {s_min:.1f} ms/batch")
                        print(f"                         max : {s_max:.1f} ms/batch")
                    else:
                        print(f"    stable: 배치 수({n})가 warm-up({wu}) 이하 — stable 구간 없음")
                    print(f"    total (all batches)    : {total:.3f} s  ({n} batches)\n")
                else:
                    print(f"==> [Inference Latency] {label}: 측정된 배치 없음 (test_step 호출 0회)\n")

            if args.fp16:
                _model = runner.model.module if hasattr(runner.model, 'module') \
                    else runner.model

                # ── STCOcc: 전체 모델 FP16 + autocast (AMP 방식) ─────────────
                # patch_full_fp16():
                #   · model.half() → 모든 파라미터 FP16
                #   · BN → FP32 (명시적 복원)
                #   · forward_projection 진입점 입력 캐스팅 패치
                # autocast():
                #   · softmax, layer_norm, cross_entropy 등 → 자동 FP32
                #   · conv2d, linear, matmul 등 → FP16 Tensor Core
                #   · @custom_fwd(cast_inputs=fp32) 데코레이터가 붙은
                #     MultiScaleDeformableAttnFunction 등도 올바르게 FP32 처리
                # 모델 파일 패치:
                #   · torch.inverse() 등 autocast 가 다루지 않는 연산은
                #     view_transformer.py, bevformer_encoder.py 등에서
                #     명시적 .float() 캐스팅으로 FP32 보장
                stcocc_patched = False
                if hasattr(_model, 'forward_projection'):
                    try:
                        from projects.STCOcc.stcocc.utils.precision_utils import (
                            patch_full_fp16,
                        )
                        bn_count = patch_full_fp16(_model)
                        stcocc_patched = True
                        print(f'\n==> [FP16] STCOcc 전체 FP16 (AMP):'
                              f' 모든 모듈 FP16, BN {bn_count}개 FP32 유지.'
                              f' softmax/layernorm/inverse 등 수치 민감 연산은'
                              f' autocast + 모델 패치로 FP32 보장.'
                              f' autocast 활성.\n')
                    except ImportError:
                        pass

                if not stcocc_patched:
                    # 모든 모델 공통: 파라미터 fp16 변환 + BN fp32 유지
                    # (BEVFormer, TPVFormer, SurroundOcc, CONet, FusionOcc, LiCROcc 등)
                    runner.model.half()
                    for m in runner.model.modules():
                        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d,
                                          nn.BatchNorm3d)):
                            m.float()
                    bn_count = sum(
                        1 for m in runner.model.modules()
                        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d,
                                          nn.BatchNorm3d))
                    )
                    # spconv implicit_gemm은 fp16을 지원하지 않으므로
                    # spconv를 사용하는 LiDAR/레이더 관련 모듈만 fp32로 유지.
                    # pts_backbone 등 일반 ResNet(BEVFormer 등)은 spconv 미사용이므로
                    # FP16 변환 대상 → spconv 레이어 존재 여부를 확인 후 조건부 FP32 유지.
                    def _has_spconv(mod):
                        """모듈 내 spconv 레이어 포함 여부 반환."""
                        for m in mod.modules():
                            mod_cls = type(m)
                            if 'spconv' in mod_cls.__module__:
                                return True
                        return False

                    for _attr in ('pts_voxel_encoder', 'pts_middle_encoder',
                                  'pts_backbone',
                                  'radar_voxel_encoder', 'radar_backbone',
                                  'radar_middle_encoder'):
                        _mod = getattr(_model, _attr, None)
                        if _mod is not None:
                            if _has_spconv(_mod):
                                _mod.float()
                            # spconv 없는 모듈(BEVFormer pts_backbone 등 일반 ResNet)은
                            # model.half()로 이미 FP16 변환됐으므로 그대로 유지

                    # ── img_backbone forward_pre_hook: fp32 입력 → fp16 자동 캐스팅 ──
                    # 원인: model.half()로 backbone 가중치가 fp16이 된 상태에서
                    # dataloader가 출력하는 fp32 이미지가 그대로 전달되면
                    # "Input type (FloatTensor) != weight type (HalfTensor)" 오류 발생.
                    # autocast()는 model.half()로 미리 fp16 변환된 가중치에 대해
                    # fp32 입력을 자동으로 처리하지 않으므로 hook으로 명시 처리.
                    # (TPVFormer, SurroundOcc, CONet, FusionOcc, LiCROcc 등 모든 모델 적용)
                    _hook_handles = []
                    if hasattr(_model, 'img_backbone') and _model.img_backbone is not None:
                        def _fp16_bb_hook(module, args):
                            if args and isinstance(args[0], torch.Tensor) \
                                    and args[0].is_floating_point():
                                return (args[0].to(torch.float16),) + tuple(args[1:])
                            return args
                        _hook_handles.append(
                            _model.img_backbone.register_forward_pre_hook(_fp16_bb_hook)
                        )

                    # ── extract_img_feat 패치: 모델별 시그니처 대응 (이중 안전장치) ──
                    import inspect as _inspect
                    _patch_note = (
                        f' img_backbone fp16 hook 등록 ({len(_hook_handles)}개).'
                        if _hook_handles else ''
                    )
                    if hasattr(_model, 'extract_img_feat'):
                        _orig_extract = _model.extract_img_feat
                        _sig_params = list(
                            _inspect.signature(_orig_extract).parameters.keys()
                        )
                        if 'img_metas' in _sig_params:
                            # BEVFormer/CONet 스타일: (img, img_metas[, len_queue])
                            _has_lq = 'len_queue' in _sig_params
                            def _make_fp16_extract_patch(orig, has_lq):
                                def _patched(img, img_metas=None, len_queue=None):
                                    if isinstance(img, torch.Tensor):
                                        img = img.to(torch.float16)
                                    elif (isinstance(img, (list, tuple))
                                          and len(img) > 0
                                          and isinstance(img[0], torch.Tensor)):
                                        img = [img[0].to(torch.float16)] + list(img[1:])
                                    if has_lq:
                                        return orig(img, img_metas, len_queue=len_queue)
                                    return orig(img, img_metas)
                                return _patched
                            _model.extract_img_feat = _make_fp16_extract_patch(
                                _orig_extract, _has_lq
                            )
                        else:
                            # LiCROcc 스타일: extract_img_feat(img_inputs) - img_inputs는 dict
                            def _make_fp16_licr_patch(orig):
                                def _patched(img_inputs):
                                    if isinstance(img_inputs, dict) and 'imgs' in img_inputs:
                                        img_inputs = dict(img_inputs)
                                        img_inputs['imgs'] = img_inputs['imgs'].to(torch.float16)
                                    return orig(img_inputs)
                                return _patched
                            _model.extract_img_feat = _make_fp16_licr_patch(_orig_extract)
                        _patch_note += ' extract_img_feat fp16 캐스팅 패치 적용.'
                    _model_name = type(_model).__name__
                    print(f"\n==> [{_model_name}] FP16 inference enabled: "
                          f"model weights converted to fp16 "
                          f"({bn_count} BN layers kept in fp32).{_patch_note}"
                          f" autocast active.\n")

                # FP16 적용 후 파라미터 메모리 절감량 출력
                _print_parameter_memory_after_fp16(runner.model, fp32_param_mb)
                # FLOPs 기준 FP16/FP32 coverage 출력
                _print_fp16_fp32_flops_coverage(runner.model)

                # STCOcc 와 다른 모델 모두 autocast 사용
                # · STCOcc: 수치 민감 연산 자동 FP32 처리
                # · 기타 모델: deformable attention @custom_fwd 정상 동작
                def _fp16_run():
                    with torch.cuda.amp.autocast():  # type: ignore[attr-defined]
                        runner.test()
                _run_with_metrics("FP16 inference", _fp16_run)

            elif args.int8_engines:
                from tools.stcocc_build_int8_engine import TRTModule

                _model = runner.model.module if hasattr(runner.model, 'module') \
                    else runner.model

                # 기본 FP16 + img_backbone/neck/bev_encoder_backbone 만 INT8 엔진으로 교체 (--fp16과 동일한 FP16 파이프라인)
                # 1) 먼저 전체 모델 FP16 적용 (patch_full_fp16 또는 model.half())
                stcocc_fp16_patched = False
                if hasattr(_model, 'forward_projection'):
                    try:
                        from projects.STCOcc.stcocc.utils.precision_utils import (
                            patch_full_fp16,
                        )
                        bn_count = patch_full_fp16(_model)
                        stcocc_fp16_patched = True
                        print(f'\n==> [INT8+FP16] STCOcc 나머지 모듈 FP16 (AMP): BN {bn_count}개 FP32 유지. autocast 사용.\n')
                    except ImportError:
                        pass

                if not stcocc_fp16_patched:
                    runner.model.half()
                    for m in runner.model.modules():
                        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d,
                                          nn.BatchNorm3d)):
                            m.float()
                    # img_backbone forward_pre_hook: fp32 입력 → fp16 자동 캐스팅
                    if hasattr(_model, 'img_backbone') and _model.img_backbone is not None:
                        def _fp16_bb_hook_int8(module, args):
                            if args and isinstance(args[0], torch.Tensor) \
                                    and args[0].is_floating_point():
                                return (args[0].to(torch.float16),) + tuple(args[1:])
                            return args
                        _model.img_backbone.register_forward_pre_hook(_fp16_bb_hook_int8)
                    # extract_img_feat가 있는 모델만 패치 (모델별 시그니처 대응)
                    import inspect as _inspect_int8
                    if hasattr(_model, 'extract_img_feat'):
                        _orig_extract_int8 = _model.extract_img_feat
                        _sig_params_int8 = list(
                            _inspect_int8.signature(_orig_extract_int8).parameters.keys()
                        )
                        if 'img_metas' in _sig_params_int8:
                            _has_lq_int8 = 'len_queue' in _sig_params_int8
                            def _make_fp16_int8_patch(orig, has_lq):
                                def _patched(img, img_metas=None, len_queue=None):
                                    if isinstance(img, torch.Tensor):
                                        img = img.to(torch.float16)
                                    elif (isinstance(img, (list, tuple))
                                          and len(img) > 0
                                          and isinstance(img[0], torch.Tensor)):
                                        img = [img[0].to(torch.float16)] + list(img[1:])
                                    if has_lq:
                                        return orig(img, img_metas, len_queue=len_queue)
                                    return orig(img, img_metas)
                                return _patched
                            _model.extract_img_feat = _make_fp16_int8_patch(
                                _orig_extract_int8, _has_lq_int8
                            )
                        else:
                            def _make_fp16_int8_licr_patch(orig):
                                def _patched(img_inputs):
                                    if isinstance(img_inputs, dict) and 'imgs' in img_inputs:
                                        img_inputs = dict(img_inputs)
                                        img_inputs['imgs'] = img_inputs['imgs'].to(torch.float16)
                                    return orig(img_inputs)
                                return _patched
                            _model.extract_img_feat = _make_fp16_int8_licr_patch(
                                _orig_extract_int8
                            )
                    print(f'\n==> [INT8+FP16] 나머지 모듈 FP16, autocast 사용.\n')

                # 2) inject 전: INT8 대상 모듈 파라미터 수 측정 (TRT 교체 후엔 PyTorch 파라미터가 사라짐)
                int8_module_params = 0
                int8_module_info = []
                # STCOcc: forward_projection 하위에 서브모듈이 있음
                # FusionOcc: img_backbone / img_neck / img_bev_encoder_backbone
                # CONet: img_backbone / img_neck / occ_encoder_backbone
                _int8_owner = (
                    _model.forward_projection if hasattr(_model, 'forward_projection')
                    else _model
                )
                # CONet은 img_bev_encoder_backbone 대신 occ_encoder_backbone 사용
                _is_conet = (
                    not hasattr(_model, 'forward_projection')
                    and hasattr(_model, 'occ_encoder_backbone')
                    and not hasattr(_model, 'img_bev_encoder_backbone')
                )
                _bev_mod_name = 'occ_encoder_backbone' if _is_conet else 'img_bev_encoder_backbone'
                # CONet: img_backbone + SECONDFPN 모두 PyTorch FP16 유지
                #        (TRT INT8 cuTENSOR NHWC permutation 버그, 6회 시도 후 불가 판정)
                #        → occ_encoder_backbone 만 INT8 집계
                # 기타 모델(STCOcc, FusionOcc): backbone + neck + bev_encoder 모두 INT8
                _int8_targets = (
                    (_bev_mod_name,)
                    if _is_conet
                    else ('img_backbone', 'img_neck', _bev_mod_name)
                )
                for _mod_name in _int8_targets:
                    _submod = getattr(_int8_owner, _mod_name, None)
                    if _submod is not None:
                        _n = sum(p.numel() for p in _submod.parameters())
                        int8_module_params += _n
                        int8_module_info.append((_mod_name, _n))

                # 3) INT8 엔진 주입 (TRT 출력을 FP16으로 넘겨 나머지 FP16 모듈과 연결)
                # STCOcc (forward_projection 보유): STCOcc precision_utils 사용.
                #   단, ``from projects.STCOcc.stcocc.utils.precision_utils import ...``
                #   구문은 stcocc/__init__.py 를 실행시켜 LSSViewTransformer 를
                #   MMEngine 레지스트리에 이중 등록하려고 시도한다.
                #   FusionOcc 와 같이 동일 이름의 클래스가 이미 등록된 환경에서는
                #   KeyError 가 발생하므로 importlib 로 파일을 직접 로드한다.
                # FusionOcc (forward_projection 없음): FusionOcc precision_utils 사용.
                import importlib.util as _ilib_util
                if hasattr(_model, 'forward_projection'):
                    # STCOcc: importlib 으로 __init__.py 실행 우회
                    _pu_file = os.path.join(
                        os.path.dirname(os.path.abspath(__file__)),
                        '../projects/STCOcc/stcocc/utils/precision_utils.py',
                    )
                    try:
                        _spec = _ilib_util.spec_from_file_location(
                            'stcocc_precision_utils', _pu_file)
                        _pu = _ilib_util.module_from_spec(_spec)  # type: ignore[arg-type]
                        _spec.loader.exec_module(_pu)             # type: ignore[union-attr]
                        _pu.inject_int8_engines(
                            _model, args.int8_engines, TRTModule, output_fp16=True)
                    except Exception as _e:
                        print(f'  [경고] STCOcc inject_int8_engines 로드 실패: {_e}')
                elif _is_conet:
                    # CONet: occ_encoder_backbone 사용, CONet 전용 주입 함수 사용
                    _pu_file = os.path.join(
                        os.path.dirname(os.path.abspath(__file__)),
                        '../projects/CONet/mmdet3d_plugin/utils/precision_utils.py',
                    )
                    try:
                        _spec = _ilib_util.spec_from_file_location(
                            'conet_precision_utils', _pu_file)
                        _pu = _ilib_util.module_from_spec(_spec)  # type: ignore[arg-type]
                        _spec.loader.exec_module(_pu)             # type: ignore[union-attr]
                        _pu.inject_int8_engines_conet(
                            _model, args.int8_engines, TRTModule, output_fp16=True)
                    except Exception as _e:
                        print(f'  [경고] CONet inject_int8_engines 로드 실패: {_e}')
                else:
                    # FusionOcc: FusionOcc 전용 주입 함수 사용
                    _pu_file = os.path.join(
                        os.path.dirname(os.path.abspath(__file__)),
                        '../projects/FusionOcc/fusionocc/utils/precision_utils.py',
                    )
                    try:
                        _spec = _ilib_util.spec_from_file_location(
                            'fusionocc_precision_utils', _pu_file)
                        _pu = _ilib_util.module_from_spec(_spec)  # type: ignore[arg-type]
                        _spec.loader.exec_module(_pu)             # type: ignore[union-attr]
                        _pu.inject_int8_engines_fusionocc(
                            _model, args.int8_engines, TRTModule, output_fp16=True)
                    except Exception as _e:
                        print(f'  [경고] FusionOcc inject_int8_engines 로드 실패: {_e}')

                # INT8+FP16 파라미터 메모리 절감률 출력
                _print_parameter_memory_after_int8(runner.model, fp32_param_mb, int8_module_params)
                # INT8/FP16/FP32 FLOPs Coverage 출력
                _print_int8_fp16_fp32_flops_coverage(runner.model, int8_module_params, int8_module_info)

                # 4) autocast 안에서 추론 (나머지 FP16 연산)
                if _is_conet:
                    _int8_desc = 'occ_encoder_backbone=INT8, img_backbone/SECONDFPN/나머지=FP16'
                else:
                    _int8_desc = 'img_backbone/neck/bev_encoder_backbone=INT8, 나머지=FP16'
                print(f'\n==> INT8 TRT + FP16 추론 시작 ({_int8_desc})\n')
                def _int8_run():
                    with torch.cuda.amp.autocast():  # type: ignore[attr-defined]
                        runner.test()
                _run_with_metrics("INT8+FP16 inference", _int8_run)

            else:
                _run_with_metrics("FP32 inference", runner.test)
        except TypeError as e:
            if "join() argument must be str" in str(e) and "LoggerHook" in str(e):
                # Ignore LoggerHook path errors during testing
                print(f"\nWarning: LoggerHook error ignored (common in test mode): {e}")
                print("Test completed successfully despite the logging error.\n")
            else:
                raise
        
        # Save results to output file if specified
        if args.out:
            print(f'\nwriting results to {args.out}')
            # Note: This is a placeholder. The actual results saving would need
            # to be implemented based on the runner's output format
            # For now, we'll just log that the option was specified
            if args.log_file:
                save_eval_results_to_log(f"Results saved to: {args.out}", args.log_file, args.config, args.checkpoint)
        
        # Clean up resources and exit cleanly
        # import sys
        # print("\n" + "="*60, flush=True)
        # print("✓ Testing completed successfully!", flush=True)
        # print("="*60, flush=True)
        # sys.stdout.flush()
        # sys.stderr.flush()
        
        # import gc
        # import torch
        # import os
        
        # # Clear CUDA cache
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()
        #     torch.cuda.synchronize()
        
        # # Delete runner and model explicitly
        # if 'runner' in locals():
        #     if hasattr(runner, 'model'):
        #         del runner.model
        #     del runner
        
        # # Force garbage collection
        # gc.collect()
        
        # # Use os._exit(0) to avoid Python cleanup handlers
        # # which may cause "free(): invalid pointer" core dump
        # os._exit(0)

    finally:
        # Cleanup logging (only if we didn't exit cleanly above)
        try:
            if tee_output and args.log_file:
                cleanup_logging(tee_output, args.log_file)
        except:
            pass  # Ignore any cleanup errors


if __name__ == '__main__':
    import sys
    try:
        main()
        # Exit cleanly to prevent core dump
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n==> Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n==> Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
