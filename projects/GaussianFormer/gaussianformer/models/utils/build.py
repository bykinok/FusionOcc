from mmdet3d.registry import MODELS

# mmdet's ResNet/FPN are not cross-registered into mmdet3d's MODELS registry
# in this framework (mmdet3d.registry.MODELS and mmdet.registry.MODELS are
# sibling scopes, not parent/child) -- other ported projects in this repo
# hit the same issue and work around it by instantiating these two classes
# directly instead of going through `MODELS.build` (see e.g.
# projects/SurroundOcc/surroundocc/detectors/surroundocc.py, which
# special-cases `if img_backbone['type'] == 'ResNet'`). Reproduced here for
# GaussianFormer_ori's img_backbone (ResNet+DCN) / img_neck (FPN), which are
# otherwise built unchanged.
_MMDET_ONLY_TYPES = {}


def _mmdet_resnet(**kwargs):
    from mmdet.models.backbones import ResNet
    return ResNet(**kwargs)


def _mmdet_fpn(**kwargs):
    from mmdet.models.necks import FPN
    return FPN(**kwargs)


_MMDET_ONLY_TYPES['ResNet'] = _mmdet_resnet
_MMDET_ONLY_TYPES['FPN'] = _mmdet_fpn


def build_model(cfg):
    if cfg is None:
        return None
    cfg = dict(cfg)
    obj_type = cfg.pop('type')
    if obj_type in _MMDET_ONLY_TYPES:
        return _MMDET_ONLY_TYPES[obj_type](**cfg)
    cfg['type'] = obj_type
    return MODELS.build(cfg)
