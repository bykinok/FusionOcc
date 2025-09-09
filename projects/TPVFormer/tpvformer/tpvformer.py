from typing import Optional, Union

import torch
from torch import nn

from mmdet3d.models import Base3DSegmentor
from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import SampleList


@MODELS.register_module()
class TPVFormer(Base3DSegmentor):

    def __init__(self,
                 data_preprocessor: Optional[Union[dict, nn.Module]] = None,
                 img_backbone=None,
                 img_neck=None,
                 tpv_head=None,
                 tpv_aggregator=None,
                 use_grid_mask=False):

        super().__init__(data_preprocessor=data_preprocessor)

        self.backbone = MODELS.build(img_backbone)
        if img_neck is not None:
            self.neck = MODELS.build(img_neck)
        self.tpv_head = MODELS.build(tpv_head)
        if tpv_aggregator is not None:
            self.tpv_aggregator = MODELS.build(tpv_aggregator)
        self.use_grid_mask = use_grid_mask

    def extract_feat(self, img):
        """Extract features of images."""
        # img가 리스트인 경우 처리
        if isinstance(img, list):
            # 리스트를 스택으로 변환
            img = torch.stack(img, dim=0)
        
        # 이미지를 float32로 변환
        if img.dtype != torch.float32:
            img = img.float()
        
        B, N, C, H, W = img.size()
        img = img.view(B * N, C, H, W)
        
        # Apply GridMask if enabled
        if hasattr(self, 'use_grid_mask') and self.use_grid_mask:
            img = self._apply_grid_mask(img)
        
        img_feats = self.backbone(img)

        if hasattr(self, 'neck'):
            img_feats = self.neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            _, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, N, C, H, W))
        return img_feats_reshaped

    def _apply_grid_mask(self, img):
        """Apply GridMask augmentation to input images.
        
        Args:
            img (torch.Tensor): Input images of shape (B*N, C, H, W)
            
        Returns:
            torch.Tensor: Images with GridMask applied
        """
        if not self.use_grid_mask:
            return img
            
        # GridMask 파라미터 (설정에서 조정 가능)
        grid_size = 64  # Grid 크기
        prob = 0.5      # 적용 확률
        
        # 랜덤하게 GridMask 적용
        if torch.rand(1).item() < prob:
            B, C, H, W = img.shape
            
            # Grid 패턴 생성
            mask = torch.ones_like(img)
            
            # 랜덤한 시작 위치
            start_h = torch.randint(0, H - grid_size, (1,)).item()
            start_w = torch.randint(0, W - grid_size, (1,)).item()
            
            # Grid 패턴 적용 (일부 영역을 0으로 마스킹)
            for i in range(0, grid_size, grid_size // 4):
                for j in range(0, grid_size, grid_size // 4):
                    h_start = start_h + i
                    w_start = start_w + j
                    h_end = min(h_start + grid_size // 4, H)
                    w_end = min(w_start + grid_size // 4, W)
                    
                    if h_start < H and w_start < W:
                        mask[:, :, h_start:h_end, w_start:w_end] = 0
            
            # 마스크 적용
            img = img * mask
            
        return img

    def _forward(self, batch_inputs, batch_data_samples):
        """Forward training function."""
        img_feats = self.extract_feat(batch_inputs['img'])
        tpv_queries = self.tpv_head(img_feats, batch_data_samples)
        
        if hasattr(self, 'tpv_aggregator'):
            # TPVAggregator를 사용하여 occupancy prediction 수행
            occupancy_logits = self.tpv_aggregator(tpv_queries, batch_inputs.get('voxels', {}).get('coors', None))
            return occupancy_logits
        else:
            return tpv_queries

    def loss(self, batch_inputs: dict,
             batch_data_samples: SampleList) -> dict:
        img_feats = self.extract_feat(batch_inputs['img'])
        tpv_queries = self.tpv_head(img_feats, batch_data_samples)
        
        if hasattr(self, 'tpv_aggregator'):
            # TPVAggregator를 사용하여 loss 계산
            losses = self.tpv_aggregator.loss(tpv_queries, batch_data_samples)
            return losses
        else:
            # TPVFormerHead만 있는 경우 dummy loss 반환
            import torch
            losses = {'loss_dummy': torch.tensor(0.0, device=img_feats[0].device)}
            return losses

    def predict(self, batch_inputs: dict,
                batch_data_samples: SampleList) -> SampleList:
        """Forward predict function."""
        img_feats = self.extract_feat(batch_inputs['img'])
        tpv_queries = self.tpv_head(img_feats, batch_data_samples)
        
        if hasattr(self, 'tpv_aggregator'):
            # TPVAggregator를 사용하여 occupancy prediction 수행
            occupancy_preds = self.tpv_aggregator.predict(tpv_queries, batch_data_samples)
            
            # 결과를 data samples에 추가
            for i, data_sample in enumerate(batch_data_samples):
                data_sample.pred_occ_sem_seg = occupancy_preds[i]
            
            return batch_data_samples
        else:
            # TPVFormerHead만 있는 경우 dummy predictions 생성
            import torch
            batch_size = len(batch_data_samples)
            device = img_feats[0].device
            
            # Dummy occupancy predictions 생성 (기본값 사용)
            occupancy_preds = []
            for i in range(batch_size):
                # (H, W, Z) 형태의 occupancy grid 생성
                pred = torch.randint(0, 18, (100, 100, 8), device=device)  # 기본값 사용
                occupancy_preds.append(pred)
            
            # 결과를 data samples에 추가
            for i, data_sample in enumerate(batch_data_samples):
                data_sample.pred_occ_sem_seg = occupancy_preds[i]
            
            return batch_data_samples

    def aug_test(self, batch_inputs, batch_data_samples):
        """Augmented test function."""
        return self.predict(batch_inputs, batch_data_samples)

    def encode_decode(self, batch_inputs: dict,
                      batch_data_samples: SampleList) -> SampleList:
        """Encode and decode function."""
        return self.predict(batch_inputs, batch_data_samples)
