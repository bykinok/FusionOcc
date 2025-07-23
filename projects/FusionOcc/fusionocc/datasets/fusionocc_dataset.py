# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from mmdet3d.datasets import NuScenesDataset
from mmdet3d.registry import DATASETS


@DATASETS.register_module()
class FusionOccDataset(NuScenesDataset):
    """FusionOcc dataset for 3D occupancy prediction.
    
    This dataset is based on nuScenes dataset but adds occupancy-specific
    annotations and processing.
    """
    
    def __init__(self, 
                 data_root,
                 ann_file,
                 pipeline=None,
                 classes=None,
                 modality=None,
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 test_mode=False,
                 use_valid_flag=False,
                 **kwargs):
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            use_valid_flag=use_valid_flag,
            **kwargs)
    
    def get_data_info(self, index):
        """Get data info according to the given index.
        
        Args:
            index (int): Index of the sample data to get.
            
        Returns:
            dict: Data information that will be passed to the data
                preprocessing pipelines.
        """
        data_info = super().get_data_info(index)
        
        # Add FusionOcc specific information
        if 'ann_infos' in data_info:
            # Add occupancy ground truth path
            if 'occ_path' in data_info:
                data_info['occ_gt_path'] = data_info['occ_path']
            
            # Add annotation information for occupancy
            if 'ann_infos' in data_info:
                data_info['gt_boxes_3d'] = data_info['ann_infos'][0]  # boxes
                data_info['gt_labels_3d'] = data_info['ann_infos'][1]  # labels
        
        return data_info
    
    def prepare_train_data(self, index):
        """Training data preparation.
        
        Args:
            index (int): Index for accessing the target data.
            
        Returns:
            dict: Training data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        
        # Add FusionOcc specific data preparation
        if self.modality is None:
            input_dict['modality'] = dict(use_lidar=True, use_camera=True)
        
        # Add occupancy-specific keys
        input_dict['voxel_semantics'] = None  # Will be loaded by pipeline
        input_dict['mask_camera'] = None  # Will be loaded by pipeline
        
        data = self.pipeline(input_dict)
        return data
    
    def prepare_test_data(self, index):
        """Testing data preparation.
        
        Args:
            index (int): Index for accessing the target data.
            
        Returns:
            dict: Testing data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        
        # Add FusionOcc specific data preparation
        if self.modality is None:
            input_dict['modality'] = dict(use_lidar=True, use_camera=True)
        
        data = self.pipeline(input_dict)
        return data 