"""Custom transforms for TPVFormer."""

import mmcv
import numpy as np
from mmdet3d.registry import TRANSFORMS


@TRANSFORMS.register_module()
class MultiViewImageNormalize:
    """Normalize multi-view images with specified mean and std.
    
    This matches the original TPVFormer normalization:
    - mean=[103.530, 116.280, 123.675] 
    - std=[1.0, 1.0, 1.0]
    - to_rgb=False (BGR format)
    
    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB.
            Default: False (keep BGR as in original).
    """

    def __init__(self, mean, std, to_rgb=False):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        """Normalize multi-view images.
        
        Args:
            results (dict): Result dict containing 'img' key with list of images.
            
        Returns:
            dict: Result dict with normalized images.
        """
        if 'img' not in results:
            return results
        
        # Normalize each view
        normalized_imgs = []
        for img in results['img']:
            # mmcv.imnormalize handles the normalization
            # (img - mean) / std, with optional RGB conversion
            normalized_img = mmcv.imnormalize(
                img, self.mean, self.std, self.to_rgb)
            normalized_imgs.append(normalized_img)
        
        results['img'] = normalized_imgs
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})'
        return repr_str

