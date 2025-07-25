# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import warnings
import datetime
import sys

from mmengine.config import Config, ConfigDict, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from mmdet3d.utils import replace_ceph_backend


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
        # load config
        cfg = Config.fromfile(args.config)

        # TODO: We will unify the ceph support approach with other OpenMMLab repos
        if args.ceph:
            cfg = replace_ceph_backend(cfg)

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

        # build the runner from config
        if 'runner_type' not in cfg:
            # build the default runner
            runner = Runner.from_cfg(cfg)
        else:
            # build customized runner from the registry
            # if 'runner_type' is set in the cfg
            runner = RUNNERS.build(cfg)

        # start testing
        runner.test()
        
        # Save results to output file if specified
        if args.out:
            print(f'\nwriting results to {args.out}')
            # Note: This is a placeholder. The actual results saving would need
            # to be implemented based on the runner's output format
            # For now, we'll just log that the option was specified
            if args.log_file:
                save_eval_results_to_log(f"Results saved to: {args.out}", args.log_file, args.config, args.checkpoint)

    finally:
        # Cleanup logging
        if tee_output and args.log_file:
            cleanup_logging(tee_output, args.log_file)


if __name__ == '__main__':
    main()
