
# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
try:
    from mmcv.parallel import DataContainer as DC
except ImportError:
    # DataContainer is deprecated in newer versions, create a simple wrapper
    class DC:
        def __init__(self, data, **kwargs):
            self.data = data
            self._kwargs = kwargs
        def __repr__(self):
            return f'DC({self.data})'

try:
    from mmdet3d.core.bbox import BaseInstance3DBoxes
    from mmdet3d.core.points import BasePoints
except ImportError:
    from mmdet3d.structures.bbox_3d import BaseInstance3DBoxes
    from mmdet3d.structures.points import BasePoints
try:
    from mmdet.datasets.builder import PIPELINES
    from mmdet.datasets.pipelines import to_tensor
    from mmdet3d.datasets.pipelines import DefaultFormatBundle3D
except ImportError:
    from mmdet3d.registry import TRANSFORMS as PIPELINES
    from mmcv.transforms import to_tensor
    # DefaultFormatBundle3D is no longer available in new versions, use object as base
    DefaultFormatBundle3D = object

@PIPELINES.register_module(name='DefaultFormatBundle3D')
@PIPELINES.register_module()
class CustomDefaultFormatBundle3D:
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
    
    def __init__(self, class_names=None, with_gt=True, with_label=True):
        self.class_names = class_names
        self.with_gt = with_gt
        self.with_label = with_label

    def __call__(self, results):
        """Call function to transform and format common fields in results.
        Args:
            results (dict): Result dict contains the data to convert.
        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """
        # Format 3D data
        if isinstance(results.get('img', None), list):
            # Multiple images
            results['img'] = DC(
                [to_tensor(img.transpose(2, 0, 1)) for img in results['img']], stack=True, padding_value=0)
        elif 'img' in results:
            # Single image
            if len(results['img'].shape) < 3:
                results['img'] = np.expand_dims(results['img'], -1)
            results['img'] = DC(
                to_tensor(results['img'].transpose(2, 0, 1)), stack=True)
        
        for key in ['proposals', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels']:
            if key not in results:
                continue
            results[key] = DC(to_tensor(results[key]))
        
        if 'gt_bboxes_3d' in results:
            results['gt_bboxes_3d'] = DC(results['gt_bboxes_3d'], cpu_only=True)
        if 'gt_labels_3d' in results:
            results['gt_labels_3d'] = DC(to_tensor(results['gt_labels_3d']))
        if 'gt_map_masks' in results:
            results['gt_map_masks'] = DC(
                to_tensor(results['gt_map_masks']), stack=True)
        
        return results