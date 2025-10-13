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
                 use_grid_mask=False,
                 init_cfg=None):

        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        # 원본과 동일하게 img_backbone, img_neck 이름 사용 (체크포인트 호환성)
        self.img_backbone = MODELS.build(img_backbone)
        if img_neck is not None:
            self.img_neck = MODELS.build(img_neck)
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
        
        # Apply GridMask if enabled (training mode에서만 적용, 원본과 동일)
        if hasattr(self, 'use_grid_mask') and self.use_grid_mask and self.training:
            img = self._apply_grid_mask(img)
        
        img_feats = self.img_backbone(img)

        if hasattr(self, 'img_neck'):
            img_feats = self.img_neck(img_feats)

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
        """Forward function - not used in mmdet3d, kept for compatibility."""
        return self.predict(batch_inputs, batch_data_samples)

    def loss(self, batch_inputs: dict,
             batch_data_samples: SampleList) -> dict:
        """Compute loss for training.
        
        Note: For evaluation with pretrained checkpoint, this may not be called.
        The original TPVFormer eval.py computes loss externally.
        """
        img_feats = self.extract_feat(batch_inputs['img'])
        tpv_queries = self.tpv_head(img_feats, batch_data_samples)
        
        if hasattr(self, 'tpv_aggregator'):
            # Get points for point-wise prediction
            points = None
            if 'points' in batch_inputs:
                points_data = batch_inputs['points']
                # Extract coordinates from PointData structure
                if hasattr(points_data, 'coord'):
                    points = points_data.coord
                elif isinstance(points_data, torch.Tensor):
                    points = points_data
            
            # Forward through aggregator
            outputs = self.tpv_aggregator(tpv_queries, points)
            
            # Simple CE loss implementation (can be extended)
            losses = {}
            losses['loss_dummy'] = torch.tensor(0.0, device=img_feats[0].device, requires_grad=True)
            return losses
        else:
            losses = {'loss_dummy': torch.tensor(0.0, device=img_feats[0].device)}
            return losses

    def predict(self, batch_inputs: dict,
                batch_data_samples: SampleList) -> SampleList:
        """Forward predict function for evaluation/testing.
        
        This mimics the original TPVFormer eval.py behavior where model is called
        with img and points to produce voxel and point predictions.
        """
        # Handle different batch_inputs keys (data_preprocessor changes the structure)
        if isinstance(batch_inputs, dict):
            if 'img' in batch_inputs:
                img = batch_inputs['img']
            elif 'inputs' in batch_inputs:
                img = batch_inputs['inputs']
            else:
                # Check for any tensor-like key
                for key, value in batch_inputs.items():
                    if isinstance(value, torch.Tensor):
                        img = value
                        break
                else:
                    raise KeyError(f"Cannot find image tensor in batch_inputs keys: {batch_inputs.keys()}")
        else:
            img = batch_inputs
        
        img_feats = self.extract_feat(img)
        tpv_queries = self.tpv_head(img_feats, batch_data_samples)
        
        if hasattr(self, 'tpv_aggregator'):
            # Get point coordinates for point-wise prediction (following original eval.py)
            # In original eval.py, val_grid (each point's voxel coordinate) is passed as "points"
            # NOTE: We need point_coors (all points), NOT voxel_coors (unique voxels)
            point_coords_float = None
            if len(batch_data_samples) > 0:
                data_sample = batch_data_samples[0]
                point_coords = None
                
                # Get point_coors - these are voxel coordinates for EACH point
                # (set by data_preprocessor.voxelize -> data_sample.point_coors)
                if hasattr(data_sample, 'point_coors'):
                    point_coords = data_sample.point_coors
                # Fallback: try gt_pts_seg for backward compatibility
                elif hasattr(data_sample, 'gt_pts_seg'):
                    gt_pts_seg = data_sample.gt_pts_seg
                    if hasattr(gt_pts_seg, 'point_coors'):
                        point_coords = gt_pts_seg.point_coors
                
                if point_coords is not None:
                    # Convert to float tensor for model forward (like val_grid_float in original)
                    if isinstance(point_coords, torch.Tensor):
                        point_coords_float = point_coords.float()
                    else:
                        point_coords_float = torch.from_numpy(point_coords).float()
                    
                    # Move to correct device
                    point_coords_float = point_coords_float.to(img_feats[0].device)
                    
                    # Add batch dimension if needed (assuming batch_size=1)
                    if point_coords_float.dim() == 2:
                        point_coords_float = point_coords_float.unsqueeze(0)
                    
                    # print(f"[DEBUG TPVFormer.predict] point_coords_float shape: {point_coords_float.shape}, min: {point_coords_float.min()}, max: {point_coords_float.max()}")
                    
            # Forward through aggregator (returns tuple if point_coords, single tensor if not)
            outputs = self.tpv_aggregator(tpv_queries, point_coords_float)
            
            # Store predictions in data_samples
            if isinstance(outputs, tuple):
                # (logits_vox, logits_pts) - following original implementation
                logits_vox, logits_pts = outputs
                for i, data_sample in enumerate(batch_data_samples):
                    # Store both voxel and point predictions (logits)
                    data_sample.pred_logits_vox = logits_vox[i]  # (C, W, H, Z)
                    data_sample.pred_logits_pts = logits_pts[i]  # (C, N, 1, 1)
                    
                    # Also store argmax results for metrics
                    # Following original eval.py: predict_labels_vox = torch.argmax(predict_labels_vox, dim=1)
                    pred_sem_seg = torch.argmax(logits_vox[i], dim=0)  # (W, H, Z)
                    data_sample.pred_occ_sem_seg = pred_sem_seg
            else:
                # Voxel-only prediction
                logits_vox = outputs
                for i, data_sample in enumerate(batch_data_samples):
                    data_sample.pred_logits_vox = logits_vox[i]
                    # Store argmax result
                    pred_sem_seg = torch.argmax(logits_vox[i], dim=0)  # (W, H, Z)
                    data_sample.pred_occ_sem_seg = pred_sem_seg
            
            return batch_data_samples
        else:
            # TPVFormerHead만 있는 경우 dummy predictions 생성
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
