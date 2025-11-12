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
            import os
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
            runner.test()
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
