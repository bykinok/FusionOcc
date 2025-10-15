from typing import Optional, Union

import torch
from torch import nn
import torch.nn.functional as F

from mmdet3d.models import Base3DSegmentor
from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import SampleList
from .lovasz_losses import lovasz_softmax


@MODELS.register_module()
class TPVFormer(Base3DSegmentor):

    def __init__(self,
                 data_preprocessor: Optional[Union[dict, nn.Module]] = None,
                 img_backbone=None,
                 img_neck=None,
                 tpv_head=None,
                 tpv_aggregator=None,
                 use_grid_mask=False,
                 ignore_label=0,
                 lovasz_input='voxel',
                 ce_input='voxel',
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
        
        # Loss configuration (following original TPVFormer)
        self.ignore_label = ignore_label
        self.lovasz_input = lovasz_input  # 'voxel' or 'points'
        self.ce_input = ce_input  # 'voxel' or 'points'
        self.ce_loss_func = nn.CrossEntropyLoss(ignore_index=ignore_label)

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
        """Compute loss for training (following original TPVFormer implementation).
        
        Original TPVFormer uses Lovasz Softmax + Cross Entropy loss.
        """
        # Handle different batch_inputs keys (same as predict function)
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
        
        # Extract features
        img_feats = self.extract_feat(img)
        tpv_queries = self.tpv_head(img_feats, batch_data_samples)
        
        if hasattr(self, 'tpv_aggregator'):
            # Get voxel coordinates and labels from data samples
            # Following original train.py structure
            voxel_coords = None
            voxel_labels = None
            point_labels = None
            
            # Extract ground truth from batch_data_samples
            # Process all samples in the batch
            voxel_coords_list = []
            voxel_labels_list = []
            point_labels_list = []
            
            for data_sample in batch_data_samples:
                # Get voxel coordinates (for point-wise prediction)
                sample_voxel_coords = None
                if hasattr(data_sample, 'point_coors'):
                    sample_voxel_coords = data_sample.point_coors.float()
                elif hasattr(data_sample, 'gt_pts_seg') and hasattr(data_sample.gt_pts_seg, 'point_coors'):
                    sample_voxel_coords = data_sample.gt_pts_seg.point_coors.float()
                
                # Get ground truth labels
                if hasattr(data_sample, 'gt_pts_seg'):
                    gt_pts_seg = data_sample.gt_pts_seg
                    
                    # Voxel-level labels
                    if hasattr(gt_pts_seg, 'voxel_semantic_mask'):
                        sample_voxel_labels = gt_pts_seg.voxel_semantic_mask
                        if not isinstance(sample_voxel_labels, torch.Tensor):
                            sample_voxel_labels = torch.from_numpy(sample_voxel_labels)
                        sample_voxel_labels = sample_voxel_labels.long().to(img_feats[0].device)
                        voxel_labels_list.append(sample_voxel_labels)
                    
                    # Point-level labels
                    if hasattr(gt_pts_seg, 'pts_semantic_mask'):
                        sample_point_labels = gt_pts_seg.pts_semantic_mask
                        if not isinstance(sample_point_labels, torch.Tensor):
                            sample_point_labels = torch.from_numpy(sample_point_labels)
                        sample_point_labels = sample_point_labels.long().to(img_feats[0].device)
                        point_labels_list.append(sample_point_labels)
                
                # Add voxel_coords
                if sample_voxel_coords is not None:
                    if not isinstance(sample_voxel_coords, torch.Tensor):
                        sample_voxel_coords = torch.from_numpy(sample_voxel_coords)
                    sample_voxel_coords = sample_voxel_coords.float().to(img_feats[0].device)
                    voxel_coords_list.append(sample_voxel_coords)
            
            # Keep as lists since each sample has different number of voxels (sparse)
            voxel_coords = voxel_coords_list if voxel_coords_list else None  # List of (N_i, 3)
            voxel_labels = voxel_labels_list if voxel_labels_list else None  # List of (N_i,) or (W, H, Z)
            point_labels = point_labels_list if point_labels_list else None  # List of (N_i,)
            
            # Pad voxel_coords to tensor for tpv_aggregator
            # tpv_aggregator expects (B, N, 3) tensor
            voxel_coords_padded = None
            if voxel_coords is not None and len(voxel_coords) > 0:
                max_points = max(coords.size(0) for coords in voxel_coords)
                B = len(voxel_coords)
                voxel_coords_padded = torch.zeros(B, max_points, 3, 
                                                   device=voxel_coords[0].device,
                                                   dtype=voxel_coords[0].dtype)
                for i, coords in enumerate(voxel_coords):
                    voxel_coords_padded[i, :coords.size(0)] = coords
            
            # Forward through aggregator (returns voxel and point predictions)
            outputs = self.tpv_aggregator(tpv_queries, voxel_coords_padded)
            
            if isinstance(outputs, tuple):
                outputs_vox, outputs_pts = outputs
            else:
                outputs_vox = outputs
                outputs_pts = None
            
            # Compute loss following original TPVFormer
            losses = {}
            total_loss = 0.0
            
            # Debug: print shapes (first iteration only)
            # if not hasattr(self, '_debug_printed'):
            #     print(f"[DEBUG LOSS] outputs_vox shape: {outputs_vox.shape if outputs_vox is not None else None}")
            #     print(f"[DEBUG LOSS] outputs_pts shape: {outputs_pts.shape if outputs_pts is not None else None}")
            #     print(f"[DEBUG LOSS] voxel_labels: {type(voxel_labels)}, len: {len(voxel_labels) if isinstance(voxel_labels, list) else 'N/A'}")
            #     print(f"[DEBUG LOSS] point_labels: {type(point_labels)}, len: {len(point_labels) if isinstance(point_labels, list) else 'N/A'}")
            #     if point_labels is not None and isinstance(point_labels, list) and len(point_labels) > 0:
            #         print(f"[DEBUG LOSS] point_labels[0] shape: {point_labels[0].shape}")
            #         print(f"[DEBUG LOSS] point_labels[0] unique: {torch.unique(point_labels[0])}")
            #     print(f"[DEBUG LOSS] voxel_coords: {type(voxel_coords)}, len: {len(voxel_coords) if isinstance(voxel_coords, list) else 'N/A'}")
            #     print(f"[DEBUG LOSS] lovasz_input: {self.lovasz_input}, ce_input: {self.ce_input}")
            #     print(f"[DEBUG LOSS] ignore_label: {self.ignore_label}")
            #     self._debug_printed = True
            
            # Determine which predictions and labels to use
            # For voxel-based loss: sample from voxel grid at point locations, use point labels
            # For point-based loss: use point predictions directly
            if self.lovasz_input == 'voxel':
                lovasz_input_tensor = outputs_vox
                lovasz_label_tensor = point_labels  # Use point labels (voxel_labels not available)
            else:  # 'points'
                lovasz_input_tensor = outputs_pts
                lovasz_label_tensor = point_labels
            
            if self.ce_input == 'voxel':
                ce_input_tensor = outputs_vox
                ce_label_tensor = point_labels  # Use point labels (voxel_labels not available)
            else:  # 'points'
                ce_input_tensor = outputs_pts
                ce_label_tensor = point_labels
            
            # Compute Lovasz loss
            if lovasz_input_tensor is not None and lovasz_label_tensor is not None:
                if self.lovasz_input == 'voxel' and voxel_coords is not None:
                    # Voxel-based: sample voxel predictions at point locations
                    # voxel_coords and lovasz_label_tensor are now lists
                    B, C, W, H, Z = lovasz_input_tensor.shape
                    
                    # Sample predictions at point locations
                    lovasz_input_list = []
                    for b in range(B):
                        coords = voxel_coords[b].long()  # List indexing: (N_b, 3)
                        x_idx = coords[:, 0].clamp(0, W-1)
                        y_idx = coords[:, 1].clamp(0, H-1)
                        z_idx = coords[:, 2].clamp(0, Z-1)
                        
                        # Extract predictions: (C, N_b)
                        sampled_preds = lovasz_input_tensor[b, :, x_idx, y_idx, z_idx]
                        lovasz_input_list.append(sampled_preds)
                    
                    # Concatenate all samples: [(C, N_0), (C, N_1), ...] -> (C, N_total)
                    lovasz_input_concat = torch.cat(lovasz_input_list, dim=1)  # (C, N_total)
                    
                    # Concatenate labels: [(N_0,), (N_1,), ...] -> (N_total,)
                    lovasz_label_concat = torch.cat(lovasz_label_tensor, dim=0)  # (N_total,)
                    
                    # Reshape to 4D for lovasz_softmax: (C, N_total) -> (1, C, N_total, 1)
                    lovasz_input_4d = lovasz_input_concat.unsqueeze(0).unsqueeze(-1)  # (1, C, N_total, 1)
                    lovasz_label_2d = lovasz_label_concat.unsqueeze(0)  # (1, N_total)
                    
                    # Apply softmax before lovasz_softmax
                    lovasz_probas_4d = F.softmax(lovasz_input_4d, dim=1)  # (1, C, N_total, 1)
                    
                    lovasz_loss = lovasz_softmax(lovasz_probas_4d, lovasz_label_2d, ignore=self.ignore_label)
                    
                    # if not hasattr(self, '_debug_lovasz_loss'):
                    #     print(f"[DEBUG LOVASZ] lovasz_loss: {lovasz_loss.item() if hasattr(lovasz_loss, 'item') else lovasz_loss}")
                    #     self._debug_lovasz_loss = True
                else:
                    # Point-based: (B, C, N, 1, 1) -> (B, C, N)
                    # lovasz_label_tensor is a list, concatenate it
                    lovasz_input_flat = lovasz_input_tensor.squeeze(-1).squeeze(-1)  # (B, C, N)
                    B, C, N = lovasz_input_flat.shape
                    
                    # Concatenate labels across batch
                    lovasz_label_concat = torch.cat(lovasz_label_tensor, dim=0)  # (B*N,)
                    
                    # Reshape input: (B, C, N) -> (1, C, B*N, 1) for lovasz_softmax
                    lovasz_input_4d = lovasz_input_flat.reshape(1, C, -1, 1)  # (1, C, B*N, 1)
                    lovasz_label_2d = lovasz_label_concat.unsqueeze(0)  # (1, B*N)
                    
                    # Apply softmax before lovasz_softmax
                    lovasz_probas_4d = F.softmax(lovasz_input_4d, dim=1)  # (1, C, B*N, 1)
                    
                    lovasz_loss = lovasz_softmax(lovasz_probas_4d, lovasz_label_2d, ignore=self.ignore_label)
                
                losses['loss_lovasz'] = lovasz_loss
                total_loss += lovasz_loss
            
            # Compute CE loss
            if ce_input_tensor is not None and ce_label_tensor is not None:
                if self.ce_input == 'voxel' and voxel_coords is not None:
                    # Voxel-based: sample voxel predictions at point locations
                    # voxel_coords and ce_label_tensor are now lists
                    B, C, W, H, Z = ce_input_tensor.shape
                    
                    # Index into voxel grid: for each batch, extract predictions at point locations
                    ce_input_list = []
                    for b in range(B):
                        # Get coordinates for this batch (list indexing)
                        coords = voxel_coords[b].long()  # (N_b, 3)
                        x_idx = coords[:, 0].clamp(0, W-1)
                        y_idx = coords[:, 1].clamp(0, H-1)
                        z_idx = coords[:, 2].clamp(0, Z-1)
                        
                        # Extract predictions at these locations: (C, N_b)
                        sampled_preds = ce_input_tensor[b, :, x_idx, y_idx, z_idx]  # (C, N_b)
                        ce_input_list.append(sampled_preds.permute(1, 0))  # (N_b, C)
                    
                    ce_input_flat = torch.cat(ce_input_list, dim=0)  # (N_total, C)
                    
                    # Concatenate labels: [(N_0,), (N_1,), ...] -> (N_total,)
                    ce_label_flat = torch.cat(ce_label_tensor, dim=0)
                else:
                    # Point-based: (B, C, N, 1, 1) -> (N_total, C) and (N_total,)
                    # ce_label_tensor is a list, concatenate it
                    ce_input_flat = ce_input_tensor.squeeze(-1).squeeze(-1)  # (B, C, N)
                    ce_input_flat = ce_input_flat.permute(0, 2, 1).contiguous().view(-1, ce_input_flat.size(1))  # (B*N, C)
                    ce_label_flat = torch.cat(ce_label_tensor, dim=0)  # (N_total,)
                
                ce_loss = self.ce_loss_func(ce_input_flat, ce_label_flat)
                losses['loss_ce'] = ce_loss
                total_loss += ce_loss
            
            losses['loss'] = total_loss
            return losses
        else:
            # No aggregator, return dummy loss
            losses = {'loss': torch.tensor(0.0, device=img_feats[0].device, requires_grad=True)}
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
