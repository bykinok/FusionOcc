
# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from mmcv import BaseTransform

from mmdet3d.registry import TRANSFORMS as PIPELINES
from mmdet3d.datasets.transforms.formating import to_tensor
from mmengine.structures import InstanceData, BaseDataElement


# Define OccDataSample at module level for pickling
class OccDataSample(BaseDataElement):
    """Data sample for occupancy prediction."""
    pass


@PIPELINES.register_module()
class OccDefaultFormatBundle3D(BaseTransform):
    """Default formatting bundle.
    It simplifies the pipeline of formatting common fields for voxels,
    including "proposals", "gt_bboxes", "gt_labels", "gt_masks" and
    "gt_semantic_seg".
    These fields are formatted as follows.
    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - proposals: (1)to tensor, (2)to DataContainer
    - gt_bboxes: (1)to tensor, (2)to DataContainer
    - gt_bboxes_ignore: (1)to tensor, (2)to DataContainer
    - gt_labels: (1)to tensor, (2)to DataContainer
    """
    def __init__(self, *args, **kwargs):
        super().__init__()

        
    def transform(self, results):
        """Transform function to format common fields in results.
        Args:
            results (dict): Result dict contains the data to convert.
        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """
        if 'gt_occ' in results.keys():
            results['gt_occ'] = to_tensor(results['gt_occ'])

        if 'gt_vel' in results.keys():
            results['gt_vel'] = to_tensor(results['gt_vel'])

        import torch
        # Note: Batch dimension will be added by the custom collate function (conet_collate_fn)
        # We keep img_inputs as-is here without adding batch dimension
        # The collate function will properly stack tensors across the batch
        if 'img_inputs' in results:
            # Convert img_inputs to list if it's a tuple for easier handling
            if isinstance(results['img_inputs'], tuple):
                results['img_inputs'] = list(results['img_inputs'])
                
        return results


@PIPELINES.register_module(force=True)
class Collect3D(BaseTransform):
    """Collect data from the loader relevant to the specific task.
    
    This is compatible with the new MMDetection3D data structure.
    
    Args:
        keys (Sequence[str]): Keys of results to be collected in ``data``.
        meta_keys (Sequence[str], optional): Meta keys to be converted to
            ``mmengine.DataContainer`` and collected in ``data[img_metas]``.
            Default: ('filename', 'ori_shape', 'img_shape', 'lidar2img',
            'depth2img', 'cam2img', 'pad_shape', 'scale_factor', 'flip',
            'pcd_horizontal_flip', 'pcd_vertical_flip', 'box_mode_3d',
            'box_type_3d', 'img_norm_cfg', 'pcd_trans', 'sample_idx',
            'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
            'transformation_3d_flow', 'scene_token', 'can_bus')
    """

    def __init__(self,
                 keys,
                 meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                           'depth2img', 'cam2img', 'pad_shape',
                           'scale_factor', 'flip', 'pcd_horizontal_flip',
                           'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                           'img_norm_cfg', 'pcd_trans', 'sample_idx',
                           'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                           'transformation_3d_flow', 'scene_token', 'can_bus',
                           'lidar_token', 'pc_range', 'occ_size')):
        self.keys = keys
        self.meta_keys = meta_keys

    def transform(self, results):
        """Transform function to collect keys in results.
        
        The keys in ``keys`` will be collected to ``inputs``.
        The keys in ``meta_keys`` will be collected to ``data_samples``.
        
        Args:
            results (dict): Result dict contains the data to convert.
            
        Returns:
            dict: The result dict contains the following keys
                - inputs (dict): The data dict with keys specified in ``keys``.
                - data_samples (obj:`Det3DDataSample`): The annotation info
                    of the sample.
        """
        # Collect data keys
        data = {}
        for key in self.keys:
            if key in results:
                data[key] = results[key]
        
        # Collect meta keys
        img_metas = {}
        for key in self.meta_keys:
            if key in results:
                img_metas[key] = results[key]
        
        # Create data_samples structure compatible with new MMDet3D
        data_sample = OccDataSample()
        data_sample.set_metainfo(img_metas)
        
        # Add gt_occ to data_sample if exists
        if 'gt_occ' in data:
            data_sample.gt_occ = data['gt_occ']
        
        # Store img_inputs in data_sample for model access
        if 'img_inputs' in data:
            data_sample.img_inputs = data['img_inputs']
        
        # Return in new MMDet3D format
        return {
            'inputs': {
                'points': data.get('points'),
                'img_inputs': data.get('img_inputs'),
            },
            'data_samples': data_sample
        }