from mmengine.registry import Registry, DATASETS as ENGINE_DATASETS, FUNCTIONS
from mmdet3d.registry import DATASETS as DET3D_DATASETS

# GaussianFormer_ori's own dataset/transform registries (dataset/__init__.py),
# kept unchanged: NuScenesDataset builds its pipeline through
# OPENOCC_TRANSFORMS itself (see nuscenes_dataset.py), so these never need to
# be visible to mmengine/mmdet3d's own TRANSFORMS registry.
OPENOCC_DATASET = Registry('openocc_dataset')
OPENOCC_DATAWRAPPER = Registry('openocc_datawrapper')
OPENOCC_TRANSFORMS = Registry('openocc_transforms')

from .nuscenes_dataset import NuScenesDataset
from .transform_3d import *
from .sampler import CustomDistributedSampler
from .utils import custom_collate_fn_temporal

# Dual-register NuScenesDataset and its collate_fn so a standard mmengine
# `train_dataloader`/`val_dataloader` config (dataset=dict(type=
# 'GaussianFormerNuScenesDataset', ...), collate_fn=dict(type=
# 'custom_collate_fn_temporal')) can build them through the Runner, instead
# of GaussianFormer_ori's own bespoke `get_dataloader()` (dataset/__init__.py)
# which this framework's Runner does not call. Registered under a distinct
# name -- 'NuScenesDataset' is already taken by mmdet3d's own (differently
# shaped) dataset class in this registry.
if 'GaussianFormerNuScenesDataset' not in DET3D_DATASETS.module_dict:
    DET3D_DATASETS.register_module(name='GaussianFormerNuScenesDataset', module=NuScenesDataset)
if 'GaussianFormerNuScenesDataset' not in ENGINE_DATASETS.module_dict:
    ENGINE_DATASETS.register_module(name='GaussianFormerNuScenesDataset', module=NuScenesDataset)
if 'custom_collate_fn_temporal' not in FUNCTIONS.module_dict:
    FUNCTIONS.register_module(name='custom_collate_fn_temporal', module=custom_collate_fn_temporal)
