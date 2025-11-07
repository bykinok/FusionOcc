import numpy as np
import mmcv
# mmdet3d와 mmengine 모두의 TRANSFORMS에 등록해야 합니다
from mmdet3d.registry import TRANSFORMS as TRANSFORMS_MMDET3D
from mmengine.registry import TRANSFORMS as TRANSFORMS_MMENGINE
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


# 두 레지스트리 모두에 등록 (force=True로 기본 구현 덮어쓰기)
@TRANSFORMS_MMDET3D.register_module(force=True)
@TRANSFORMS_MMENGINE.register_module(force=True)
class LoadMultiViewImageFromFiles(object):
    """Load multi channel images from a list of separate channel files.
    
    빠른 이미지 로딩을 위해 mmcv.imread를 직접 사용합니다.
    이것은 원본 BEVFormer의 구현과 동일한 방식입니다.
    
    Note: force=True를 사용하여 mmdet3d의 기본 구현을 덮어씁니다.
    이렇게 하면 원본 BEVFormer와 동일한 빠른 로딩 방식을 사용할 수 있습니다.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self, to_float32=False, color_type='unchanged'):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data.
        """
        filename = results['img_filename']
        
        # 원본 BEVFormer 방식: mmcv.imread를 직접 사용 (빠름)
        imgs = []
        for name in filename:
            img = mmcv.imread(name, self.color_type)
            if self.to_float32:
                img = img.astype(np.float32)
            imgs.append(img)
        
        # 이미지를 스택하지 않고 리스트로 유지
        # 이것이 원본 BEVFormer의 방식입니다
        results['filename'] = filename
        results['img'] = imgs
        results['img_shape'] = [img.shape for img in imgs]
        results['ori_shape'] = [img.shape for img in imgs]
        results['pad_shape'] = [img.shape for img in imgs]
        
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f"color_type='{self.color_type}')"
        return repr_str

