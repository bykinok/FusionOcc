import mmcv
import numpy as np
from mmcv.transforms import BaseTransform
from mmengine.registry import TRANSFORMS as ENGINE_TRANSFORMS
from mmdet3d.registry import TRANSFORMS as DET3D_TRANSFORMS


@ENGINE_TRANSFORMS.register_module()
@DET3D_TRANSFORMS.register_module()
class LoadMultiViewImageFromFilesFullRes(BaseTransform):
    """Load multi-view images from files at full resolution.
    
    This transform loads images without resizing them, maintaining the original resolution.
    
    Args:
        to_float32 (bool): Whether to convert the loaded image to float32. Default: False.
        color_type (str): Color type of the image. Default: 'color'.
        backend_args (dict, optional): Arguments to instantiate the file backend.
            Default: None.
    """
    
    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 backend_args=None):
        self.to_float32 = to_float32
        self.color_type = color_type
        # backend_args is ignored for compatibility - we use direct file loading
        self.backend_args = backend_args
    
    def transform(self, results):
        """Transform function to load multi-view images.
        
        Args:
            results (dict): Result dict containing image filename info.
            
        Returns:
            dict: The result dict with loaded images.
        """
        # print(f"DEBUG LoadMultiViewImageFromFilesFullRes: CALLED")
        
        # Get image filenames - check multiple possible keys
        filename = None
        if 'img_filename' in results:
            filename = results['img_filename']
        elif 'img_path' in results:
            filename = results['img_path']
        elif 'cams' in results:
            # Extract image paths from camera dict
            filename = []
            for cam_key, cam_info in results['cams'].items():
                if 'data_path' in cam_info:
                    filename.append(cam_info['data_path'])
                elif 'img_path' in cam_info:
                    filename.append(cam_info['img_path'])
        
        if filename is None:
            raise ValueError("No image filename found in results")
        
        # Handle single image or list of images
        if not isinstance(filename, list):
            filename = [filename]
        
        # Load images
        imgs = []
        for name in filename:
            # Handle relative paths
            if name.startswith('./'):
                name = name[2:]
            
            # Load image using direct file reading (backend_args not supported in mmcv 2.x)
            img = mmcv.imread(name, self.color_type)
            
            if img is None:
                raise ValueError(f"Failed to load image: {name}")
            
            if self.to_float32:
                img = img.astype(np.float32)
            
            imgs.append(img)
        
        # Store images in results
        results['img'] = imgs
        results['img_shape'] = [img.shape for img in imgs]
        results['ori_shape'] = [img.shape for img in imgs]
        
        # Update other metadata
        results['filename'] = filename
        results['img_fields'] = ['img']
        
        # Print image shapes for debugging
        # if len(imgs) > 0:
        #     print(f"DEBUG: Loaded {len(imgs)} images, first image shape: {imgs[0].shape}")
        
        return results
    
    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f'color_type={self.color_type})'
        return repr_str

