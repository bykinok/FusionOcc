from typing import Optional, Union

from torch import nn

from mmdet3d.models import Base3DSegmentor
from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import SampleList


@MODELS.register_module()
class TPVFormerOccupancy(Base3DSegmentor):
    """TPVFormer for 3D occupancy prediction.
    
    This is a modified version of TPVFormer that supports occupancy prediction
    instead of segmentation. It uses TPVAggregator instead of TPVFormerDecoder.
    """

    def __init__(self,
                 data_preprocessor: Optional[Union[dict, nn.Module]] = None,
                 backbone=None,
                 neck=None,
                 encoder=None,
                 tpv_aggregator=None,
                 use_grid_mask: bool = False):

        super().__init__(data_preprocessor=data_preprocessor)

        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        self.encoder = MODELS.build(encoder)
        self.tpv_aggregator = MODELS.build(tpv_aggregator)
        self.use_grid_mask = use_grid_mask

    def extract_feat(self, img):
        """Extract features of images."""
        B, N, C, H, W = img.size()
        img = img.view(B * N, C, H, W)
        img_feats = self.backbone(img)

        if hasattr(self, 'neck'):
            img_feats = self.neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            _, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, N, C, H, W))
        return img_feats_reshaped

    def _forward(self, batch_inputs, batch_data_samples):
        """Forward training function."""
        img_feats = self.extract_feat(batch_inputs['imgs'])
        tpv_queries = self.encoder(img_feats, batch_data_samples)
        occupancy_logits = self.tpv_aggregator(tpv_queries, batch_inputs['voxels']['coors'])
        return occupancy_logits

    def loss(self, batch_inputs: dict,
             batch_data_samples: SampleList) -> SampleList:
        """Compute loss for occupancy prediction."""
        img_feats = self.extract_feat(batch_inputs['imgs'])
        tpv_queries = self.encoder(img_feats, batch_data_samples)
        losses = self.tpv_aggregator.loss(tpv_queries, batch_data_samples)
        return losses

    def predict(self, batch_inputs: dict,
                batch_data_samples: SampleList) -> SampleList:
        """Forward predict function."""
        img_feats = self.extract_feat(batch_inputs['imgs'])
        tpv_queries = self.encoder(img_feats, batch_data_samples)
        occupancy_preds = self.tpv_aggregator.predict(tpv_queries, batch_data_samples)
        
        # Convert predictions to data samples
        for i, data_sample in enumerate(batch_data_samples):
            data_sample.pred_occ_sem_seg = occupancy_preds[i]

        return batch_data_samples

    def aug_test(self, batch_inputs, batch_data_samples):
        """Augmented test function."""
        # Placeholder for augmented testing
        return self.predict(batch_inputs, batch_data_samples)

    def encode_decode(self, batch_inputs: dict,
                      batch_data_samples: SampleList) -> SampleList:
        """Encode and decode function."""
        # Placeholder for encode-decode
        return self.predict(batch_inputs, batch_data_samples)
