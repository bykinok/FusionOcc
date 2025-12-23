from mmdet3d.registry import DATA_SAMPLERS
from mmengine.registry import build_from_cfg

# For backward compatibility
SAMPLER = DATA_SAMPLERS


def build_sampler(cfg, default_args):
    return build_from_cfg(cfg, DATA_SAMPLERS, default_args)
