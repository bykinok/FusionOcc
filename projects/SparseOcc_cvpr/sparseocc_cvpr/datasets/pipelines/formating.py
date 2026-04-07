# Copyright (c) OpenMMLab. All rights reserved.
from ...compat import DC
from ...compat import PIPELINES
from ...compat import to_tensor
from ...compat import DefaultFormatBundle3D

@PIPELINES.register_module()
class OccDefaultFormatBundle3D(DefaultFormatBundle3D):
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
        super().__init__(*args, **kwargs)

    def __call__(self, results):
        """Call function to transform and format common fields in results.
        Args:
            results (dict): Result dict contains the data to convert.
        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """
        # Format 3D data
        results = super(OccDefaultFormatBundle3D, self).__call__(results)

        if 'gt_occ' in results.keys() and results['gt_occ'] is not None:
            if type(results['gt_occ']) is list:
                results['gt_occ'] = tuple([DC(to_tensor(x), stack=True) for x in results['gt_occ']])
            else:
                results['gt_occ'] = DC(to_tensor(results['gt_occ']), stack=True)

        if 'points_occ' in results.keys():
            results['points_occ'] = DC(to_tensor(results['points_occ']), stack=False)
        
        if 'points_uv' in results.keys():
            results['points_uv'] = DC(to_tensor(results['points_uv']), stack=False)
            
        return results


@PIPELINES.register_module()
class Collect3D:
    """구버전 mmdet3d Collect3D 호환 클래스.

    지정된 키만 결과 dict에 남기고, meta_keys는 img_metas에 묶습니다.
    """
    def __init__(self,
                 keys,
                 meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                            'depth2img', 'cam2img', 'pad_shape', 'scale_factor',
                            'flip', 'pcd_horizontal_flip', 'pcd_vertical_flip',
                            'box_mode_3d', 'box_type_3d', 'img_norm_cfg',
                            'pcd_trans', 'sample_idx', 'pcd_scale_factor',
                            'pcd_rotation', 'pcd_rotation_angle', 'pts_filename',
                            'transformation_3d_flow', 'trans_mat', 'affine_aug',
                            'occ_size', 'pc_range', 'scene_token', 'lidar_token')):
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        data = {}
        img_metas = {}
        for key in self.meta_keys:
            if key in results:
                img_metas[key] = results[key]
        data['img_metas'] = DC(img_metas, cpu_only=True)
        for key in self.keys:
            data[key] = results[key]
        return data

    def __repr__(self):
        return f'{self.__class__.__name__}(keys={self.keys}, meta_keys={self.meta_keys})'