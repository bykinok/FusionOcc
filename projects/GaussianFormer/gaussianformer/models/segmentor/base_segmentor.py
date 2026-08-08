from mmdet3d.registry import MODELS
from mmengine.model import BaseModel

from ..utils.build import build_model


@MODELS.register_module()
class CustomBaseSegmentor(BaseModel):
    """Ported from GaussianFormer_ori model/segmentor/base_segmentor.py.

    Original built submodules through mmseg's ``builder``/``SEGMENTORS``
    registry (mmseg is not a dependency of this framework); building them
    all through ``mmdet3d.registry.MODELS`` instead is behaviourally
    identical, since GaussianFormer_ori's own registries are children of the
    same mmengine root MODELS registry that mmseg's builder ultimately calls
    into. Base class changed from ``mmengine.model.BaseModule`` to
    ``BaseModel`` so ``BEVSegmentor`` can plug into mmengine's Runner
    (train_step/val_step/parse_losses) without touching its forward logic.
    """

    def __init__(
        self,
        img_backbone=None,
        img_neck=None,
        lifter=None,
        encoder=None,
        head=None,
        data_preprocessor=None,
        init_cfg=None,
        **kwargs,
    ):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        if img_backbone is not None:
            self.img_backbone = build_model(img_backbone)
        if img_neck is not None:
            self.img_neck = build_model(img_neck)
        if lifter is not None:
            self.lifter = MODELS.build(lifter)
        if encoder is not None:
            self.encoder = MODELS.build(encoder)
        if head is not None:
            self.head = MODELS.build(head)

    def extract_img_feat(self, imgs, **kwargs):
        """Extract features of images."""
        B = imgs.size(0)

        B, N, C, H, W = imgs.size()
        imgs = imgs.reshape(B * N, C, H, W)
        img_feats = self.img_backbone(imgs)
        if isinstance(img_feats, dict):
            img_feats = list(img_feats.values())
        img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return {'ms_img_feats': img_feats_reshaped}

    def forward(
        self,
        imgs,
        metas,
        **kwargs
    ):
        pass
