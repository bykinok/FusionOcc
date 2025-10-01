
# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from mmcv import BaseTransform
# MMEngine does not use DataContainer in the same way

from mmdet3d.registry import TRANSFORMS as PIPELINES
from mmdet3d.datasets.transforms.formating import to_tensor


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

        # Keep img_inputs as is for CONNet model compatibility

        return results