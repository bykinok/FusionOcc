# Copyright (c) OpenMMLab. All rights reserved.
from mmdet3d.datasets.transforms import Pack3DDetInputs
from mmdet3d.registry import TRANSFORMS


@TRANSFORMS.register_module()
class TPVPack3DDetInputs(Pack3DDetInputs):
    """Extended Pack3DDetInputs for TPVFormer with occ3d support.
    
    This simply extends the INPUTS_KEYS to include occ_3d and occ_3d_masked.
    All other functionality is inherited from the base Pack3DDetInputs.
    """
    
    # Extend INPUTS_KEYS to include occ3d data
    INPUTS_KEYS = ['points', 'img', 'occ_3d', 'occ_3d_masked']

