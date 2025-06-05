# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import warnings
import datetime
import sys

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

import mmdet
from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet.apis import multi_gpu_test, set_random_seed
from mmdet.datasets import replace_ImageToTensor

if mmdet.__version__ > '2.23.0':
    # If mmdet version > 2.23.0, setup_multi_processes would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import setup_multi_processes
else:
    from mmdet3d.utils import setup_multi_processes

try:
    # If mmdet version > 2.23.0, compat_cfg would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import compat_cfg
except ImportError:
    from mmdet3d.utils import compat_cfg


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


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--load-results', 
        help='load pre-computed results from pickle file for evaluation only')
    parser.add_argument(
        '--log-file', 
        help='log file to save evaluation results')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        default='mAP',
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where results will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--no-aavt',
        action='store_true',
        help='Do not align after view transformer.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
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
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both specified, '
            '--options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def main():
    args = parse_args()
    
    # Setup logging if log file is specified
    tee_output = None
    if args.log_file:
        checkpoint_path = args.load_results if args.load_results else args.checkpoint
        tee_output = setup_logging(args.log_file, args.config, checkpoint_path)

    try:
        # If load-results is specified, skip model inference and only do evaluation
        if args.load_results:
            if not os.path.exists(args.load_results):
                raise FileNotFoundError(f"Results file not found: {args.load_results}")
            
            print(f"Loading pre-computed results from {args.load_results}")
            outputs = mmcv.load(args.load_results)
            
            # We still need to build dataset for evaluation
            cfg = Config.fromfile(args.config)
            if args.cfg_options is not None:
                cfg.merge_from_dict(args.cfg_options)
            cfg = compat_cfg(cfg)
            
            # Build dataset for evaluation
            if isinstance(cfg.data.test, dict):
                cfg.data.test.test_mode = True
            elif isinstance(cfg.data.test, list):
                for ds_cfg in cfg.data.test:
                    ds_cfg.test_mode = True
            
            dataset = build_dataset(cfg.data.test)
            
            # Perform evaluation
            kwargs = {} if args.eval_options is None else args.eval_options
            if args.format_only:
                dataset.format_results(outputs, **kwargs)
            if args.eval:
                eval_kwargs = cfg.get('evaluation', {}).copy()
                # hard-code way to remove EvalHook args
                for key in [
                        'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                        'rule'
                ]:
                    eval_kwargs.pop(key, None)
                eval_kwargs.update(dict(metric=args.eval, **kwargs))
                
                # Add show_dir parameter for visualization
                if args.show_dir:
                    eval_kwargs['show_dir'] = args.show_dir
                    
                print("Evaluation Results:")
                results = dataset.evaluate(outputs, **eval_kwargs)
                print(results)
                
                if args.log_file:
                    print(f"Evaluation results saved to: {args.log_file}")
            return

        assert args.out or args.eval or args.format_only or args.show \
            or args.show_dir, \
            ('Please specify at least one operation (save/eval/format/show the '
             'results / save the results) with the argument "--out", "--eval"'
             ', "--format-only", "--show" or "--show-dir"')

        if args.eval and args.format_only:
            raise ValueError('--eval and --format_only cannot be both specified')

        if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
            raise ValueError('The output file must be a pkl file.')

        cfg = Config.fromfile(args.config)
        cfg.model.img_view_transformer.is_train = False    # use full sparse depth map during inference
        if args.cfg_options is not None:
            cfg.merge_from_dict(args.cfg_options)
        cfg = compat_cfg(cfg)

        # set multi-process settings
        setup_multi_processes(cfg)

        # set cudnn_benchmark
        if cfg.get('cudnn_benchmark', False):
            torch.backends.cudnn.benchmark = True

        cfg.model.pretrained = None

        if args.gpu_ids is not None:
            cfg.gpu_ids = args.gpu_ids[0:1]
            warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                          'Because we only support single GPU mode in '
                          'non-distributed testing. Use the first GPU '
                          'in `gpu_ids` now.')
        else:
            cfg.gpu_ids = [args.gpu_id]

        # init distributed env first, since logger depends on the dist info.
        if args.launcher == 'none':
            distributed = False
        else:
            distributed = True
            init_dist(args.launcher, **cfg.dist_params)

        test_dataloader_default_args = dict(
            samples_per_gpu=1, workers_per_gpu=2, dist=distributed, shuffle=False)

        # in case the test dataset is concatenated
        if isinstance(cfg.data.test, dict):
            cfg.data.test.test_mode = True
            if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
                # Replace 'ImageToTensor' to 'DefaultFormatBundle'
                cfg.data.test.pipeline = replace_ImageToTensor(
                    cfg.data.test.pipeline)
        elif isinstance(cfg.data.test, list):
            for ds_cfg in cfg.data.test:
                ds_cfg.test_mode = True
            if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
                for ds_cfg in cfg.data.test:
                    ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

        test_loader_cfg = {
            **test_dataloader_default_args,
            **cfg.data.get('test_dataloader', {})
        }

        # set random seeds
        if args.seed is not None:
            set_random_seed(args.seed, deterministic=args.deterministic)

        # build the dataloader
        dataset = build_dataset(cfg.data.test)
        data_loader = build_dataloader(dataset, **test_loader_cfg)

        # build the model and load checkpoint
        if not args.no_aavt:
            if '4D' in cfg.model.type:
                cfg.model.align_after_view_transfromation=True
        if 'num_proposals_test' in cfg and cfg.model.type=='DAL':
            cfg.model.pts_bbox_head.num_proposals=cfg.num_proposals_test
        cfg.model.train_cfg = None
        model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
        fp16_cfg = cfg.get('fp16', None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)
        checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
        if args.fuse_conv_bn:
            model = fuse_conv_bn(model)
        # old versions did not save class info in checkpoints, this walkaround is
        # for backward compatibility
        if 'CLASSES' in checkpoint.get('meta', {}):
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            model.CLASSES = dataset.CLASSES
        # palette for visualization in segmentation tasks
        if 'PALETTE' in checkpoint.get('meta', {}):
            model.PALETTE = checkpoint['meta']['PALETTE']
        elif hasattr(dataset, 'PALETTE'):
            # segmentation dataset has `PALETTE` attribute
            model.PALETTE = dataset.PALETTE

        if not distributed:
            model = MMDataParallel(model, device_ids=cfg.gpu_ids)
            outputs = single_gpu_test(model, data_loader, args.show, args.show_dir)
        else:
            model = MMDistributedDataParallel(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False)
            outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                     args.gpu_collect)

        rank, _ = get_dist_info()
        if rank == 0:
            if args.out:
                print(f'\nwriting results to {args.out}')
                mmcv.dump(outputs, args.out)
            kwargs = {} if args.eval_options is None else args.eval_options
            if args.format_only:
                dataset.format_results(outputs, **kwargs)
            if args.eval:
                eval_kwargs = cfg.get('evaluation', {}).copy()
                # hard-code way to remove EvalHook args
                for key in [
                        'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                        'rule'
                ]:
                    eval_kwargs.pop(key, None)
                eval_kwargs.update(dict(metric=args.eval, **kwargs))
                
                # Add show_dir parameter for visualization
                if args.show_dir:
                    eval_kwargs['show_dir'] = args.show_dir
                    
                results = dataset.evaluate(outputs, **eval_kwargs)
                print(results)
                
                if args.log_file:
                    print(f"Evaluation results saved to: {args.log_file}")

    finally:
        # Cleanup logging
        if tee_output and args.log_file:
            cleanup_logging(tee_output, args.log_file)


if __name__ == '__main__':
    main()
