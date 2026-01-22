# Copyright (c) OpenMMLab. All rights reserved.
from mmdet3d.datasets.transforms import Pack3DDetInputs
from mmdet3d.registry import TRANSFORMS


@TRANSFORMS.register_module()
class TPVPack3DDetInputs(Pack3DDetInputs):
    """Extended Pack3DDetInputs for TPVFormer with occ3d support.
    
    This extends the INPUTS_KEYS to include occ_3d and occ_3d_masked,
    and adds scene_name and scene_token to default meta_keys.
    """
    
    # Extend INPUTS_KEYS to include occ3d data
    INPUTS_KEYS = ['points', 'img', 'occ_3d', 'occ_3d_masked']
    
    def __init__(self, meta_keys=None, **kwargs):
        """Initialize TPVPack3DDetInputs.
        
        Args:
            meta_keys (tuple[str], optional): Meta keys to be collected in metainfo.
                If None, default meta keys will be used with token, scene_name and scene_token added.
            **kwargs: Other arguments for Pack3DDetInputs.
        """
        # Add token, scene_name and scene_token to default meta_keys if not provided
        if meta_keys is None:
            # Use default meta_keys from parent class and add token, scene_name, scene_token
            super().__init__(**kwargs)
            # Extend meta_keys with token, scene_name and scene_token
            if not hasattr(self, 'meta_keys') or self.meta_keys is None:
                self.meta_keys = ('lidar2img', 'lidar_path', 'sample_idx', 'pts_filename', 
                                  'img_shape', 'token', 'scene_name', 'scene_token')
            else:
                # Convert to list, add new keys, convert back to tuple
                meta_keys_list = list(self.meta_keys)
                if 'token' not in meta_keys_list:
                    meta_keys_list.append('token')
                if 'scene_name' not in meta_keys_list:
                    meta_keys_list.append('scene_name')
                if 'scene_token' not in meta_keys_list:
                    meta_keys_list.append('scene_token')
                self.meta_keys = tuple(meta_keys_list)
        else:
            # Use provided meta_keys
            super().__init__(meta_keys=meta_keys, **kwargs)

