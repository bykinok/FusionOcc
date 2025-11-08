# Copyright (c) OpenMMLab. All rights reserved.
import torch
import numpy as np
from mmengine.structures import PixelData
from mmdet3d.structures import Det3DDataSample
from mmcv.transforms import to_tensor
from mmdet3d.datasets.transforms import Pack3DDetInputs

from mmengine.registry import TRANSFORMS as ENGINE_TRANSFORMS
from mmdet3d.registry import TRANSFORMS as DET3D_TRANSFORMS


@ENGINE_TRANSFORMS.register_module()
@DET3D_TRANSFORMS.register_module()
class OccDefaultFormatBundle3D(Pack3DDetInputs):
    """Custom formatting bundle for occupancy prediction.
    
    This transform formats the gt_occ field properly for occupancy prediction.
    It extends Pack3DDetInputs to handle occupancy-specific data.
    
    Args:
        keys (Sequence[str]): Keys of results to be collected.
        meta_keys (Sequence[str], optional): Meta keys to be collected.
    """
    
    def __init__(self,
                 keys=('img', 'gt_occ'),
                 meta_keys=('img_path', 'ori_shape', 'img_shape', 'lidar2img',
                            'depth2img', 'cam2img', 'pad_shape', 'lidar2cam',
                            'scale_factor', 'flip', 'pcd_horizontal_flip',
                            'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                            'img_norm_cfg', 'sample_idx', 'prev_idx', 'next_idx',
                            'scene_token', 'pc_range', 'occ_size', 'occ_path')):
        # Initialize keys
        self.keys = keys
        self.meta_keys = meta_keys
        
        # For compatibility with Pack3DDetInputs
        super().__init__(
            keys=keys,
            meta_keys=meta_keys
        )
    
    def transform(self, results: dict) -> dict:
        """Transform function to format occupancy data.
        
        Args:
            results (dict): Result dict contains the data to convert.
            
        Returns:
            dict: The result dict with formatted data.
        """
        # Create data sample
        data_sample = Det3DDataSample()
        
        # Handle gt_occ
        if 'gt_occ' in results:
            gt_occ = results['gt_occ']
            # Convert to tensor if needed
            if not isinstance(gt_occ, (torch.Tensor, np.ndarray)):
                gt_occ = np.array(gt_occ)
            if isinstance(gt_occ, np.ndarray):
                gt_occ = torch.from_numpy(gt_occ)
            
            # Store in data sample
            data_sample.gt_occ = gt_occ
        
        # Handle images
        if 'img' in results:
            imgs = results['img']
            if isinstance(imgs, list):
                # Stack multi-view images
                if isinstance(imgs[0], np.ndarray):
                    # Convert to tensors: HWC -> CHW
                    img_tensors = [torch.from_numpy(img).permute(2, 0, 1) for img in imgs]
                    imgs = torch.stack(img_tensors, dim=0)  # [N, C, H, W]
                elif isinstance(imgs[0], torch.Tensor):
                    # Ensure correct shape
                    if imgs[0].dim() == 3 and imgs[0].shape[-1] in [3, 1]:  # HWC
                        img_tensors = [img.permute(2, 0, 1) for img in imgs]
                        imgs = torch.stack(img_tensors, dim=0)
                    else:
                        imgs = torch.stack(imgs, dim=0)
            
            # Store image data
            if not isinstance(imgs, dict):
                inputs = {'imgs': imgs}
            else:
                inputs = imgs
        else:
            inputs = {}
        
        # Collect metainfo
        metainfo = {}
        for key in self.meta_keys:
            if key in results:
                metainfo[key] = results[key]
        
        data_sample.set_metainfo(metainfo)
        
        # Return packed data
        packed_results = {
            'inputs': inputs,
            'data_samples': data_sample
        }
        
        return packed_results

