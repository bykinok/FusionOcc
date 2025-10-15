"""Custom transforms for TPVFormer."""

import mmcv
import numpy as np
from numpy import random
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


@TRANSFORMS.register_module()
class PhotoMetricDistortionMultiViewImage:
    """Apply photometric distortion to multi-view images sequentially.
    
    Following original TPVFormer implementation for data augmentation during training.
    Every transformation is applied with a probability of 0.5. 
    The position of random contrast is in second or second to last.
    
    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels
    
    Args:
        brightness_delta (int): delta of brightness. Default: 32.
        contrast_range (tuple): range of contrast. Default: (0.5, 1.5).
        saturation_range (tuple): range of saturation. Default: (0.5, 1.5).
        hue_delta (int): delta of hue. Default: 18.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, results):
        """Call function to perform photometric distortion on images.
        
        Args:
            results (dict): Result dict from loading pipeline.
            
        Returns:
            dict: Result dict with images distorted.
        """
        if 'img' not in results:
            return results
        
        imgs = results['img']
        new_imgs = []
        
        for img in imgs:
            assert img.dtype == np.float32, \
                'PhotoMetricDistortionMultiViewImage needs the input image of dtype ' \
                'np.float32, please set "to_float32=True" in "BEVLoadMultiViewImageFromFiles" pipeline'
            
            # random brightness
            if random.randint(2):
                delta = random.uniform(-self.brightness_delta,
                                      self.brightness_delta)
                img += delta

            # mode == 0 --> do random contrast first
            # mode == 1 --> do random contrast last
            mode = random.randint(2)
            if mode == 1:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower,
                                          self.contrast_upper)
                    img *= alpha

            # convert color from BGR to HSV
            img = mmcv.bgr2hsv(img)

            # random saturation
            if random.randint(2):
                img[..., 1] *= random.uniform(self.saturation_lower,
                                              self.saturation_upper)

            # random hue
            if random.randint(2):
                img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
                img[..., 0][img[..., 0] > 360] -= 360
                img[..., 0][img[..., 0] < 0] += 360

            # convert color from HSV to BGR
            img = mmcv.hsv2bgr(img)

            # random contrast
            if mode == 0:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower,
                                          self.contrast_upper)
                    img *= alpha

            # randomly swap channels
            if random.randint(2):
                img = img[..., random.permutation(3)]
            
            new_imgs.append(img)
        
        results['img'] = new_imgs
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(\nbrightness_delta={self.brightness_delta},\n'
        repr_str += 'contrast_range='
        repr_str += f'{(self.contrast_lower, self.contrast_upper)},\n'
        repr_str += 'saturation_range='
        repr_str += f'{(self.saturation_lower, self.saturation_upper)},\n'
        repr_str += f'hue_delta={self.hue_delta})'
        return repr_str

