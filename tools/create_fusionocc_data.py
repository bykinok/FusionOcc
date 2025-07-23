# Copyright (c) OpenMMLab. All rights reserved.
"""Create FusionOcc dataset.

This script creates the FusionOcc dataset from nuScenes data.
"""

import argparse
import os
from os import path as osp

from tools.dataset_converters import fusionocc_converter


def fusionocc_data_prep(root_path,
                        info_prefix,
                        version,
                        max_sweeps=10):
    """Prepare data related to FusionOcc dataset.

    Related data consists of '.pkl' files recording basic infos.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        max_sweeps (int, optional): Number of input consecutive frames.
            Default: 10.
    """
    fusionocc_converter.create_fusionocc_infos(
        root_path, info_prefix, version=version, max_sweeps=max_sweeps)
    
    # Add annotation information for FusionOcc
    fusionocc_converter.add_ann_adj_info(info_prefix, root_path)


def main():
    parser = argparse.ArgumentParser(description='Create FusionOcc dataset')
    parser.add_argument(
        '--root-path',
        type=str,
        default='./data/nuscenes',
        help='specify the root path of dataset')
    parser.add_argument(
        '--version',
        type=str,
        default='v1.0-trainval',
        help='specify the dataset version')
    parser.add_argument(
        '--max-sweeps',
        type=int,
        default=10,
        help='specify sweeps of lidar per example')
    parser.add_argument(
        '--extra-tag',
        type=str,
        default='fusionocc-nuscenes',
        help='extra tag for the dataset')
    args = parser.parse_args()

    fusionocc_data_prep(
        root_path=args.root_path,
        info_prefix=args.extra_tag,
        version=args.version,
        max_sweeps=args.max_sweeps)


if __name__ == '__main__':
    main() 