from typing import Optional, Union

import torch
from torch import nn
import torch.nn.functional as F

from mmdet3d.models import Base3DSegmentor
from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import SampleList
from .lovasz_losses import lovasz_softmax
from .grid_mask import GridMask

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
                 dataset_name=None,  # 'occ3d' or None for traditional GT
                 save_results=False,  # Save prediction results to disk
                 depth_supervision=None,  # dict(enabled=True, grid_config=..., downsample=..., loss_weight=..., feature_level=...)
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
        self.dataset_name = dataset_name  # Store dataset_name for predict method
        self.save_results = save_results  # Save prediction results to disk

        # Auxiliary depth supervision (ego-frame consistent with TPVFormer lidar2img=ego2img)
        self.depth_supervision = depth_supervision or dict(enabled=False)
        self.depth_head = None
        self._depth_feature_level = 1
        if self.depth_supervision.get('enabled'):
            depth_cfg = dict(
                type='AuxiliaryDepthHead',
                in_channels=256,  # FPN out_channels
                grid_config=self.depth_supervision['grid_config'],
                downsample=self.depth_supervision['downsample'],
                loss_weight=self.depth_supervision.get('loss_weight', 0.5),
            )
            self.depth_head = MODELS.build(depth_cfg)
            self._depth_feature_level = self.depth_supervision.get('feature_level', 1)

        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
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
        
        # breakpoint()

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
            
        # breakpoint()

        return self.grid_mask(img)

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
        
        # breakpoint()

        # Extract features
        img_feats = self.extract_feat(img)
        tpv_queries = self.tpv_head(img_feats, batch_data_samples)

        # Auxiliary depth supervision (ego-frame gt_depth; only when gt_depth in batch)
        loss_depth = None
        if isinstance(batch_inputs, dict):
            gt_depth = batch_inputs.get('gt_depth') or (batch_inputs.get('inputs') or {}).get('gt_depth')
        else:
            gt_depth = None
        if self.depth_head is not None and gt_depth is not None:
            level = min(self._depth_feature_level, len(img_feats) - 1)
            depth_logits = self.depth_head(img_feats[level])
            # gt_depth may be a list of tensors when collated (per-sample); stack and move to device
            if isinstance(gt_depth, (list, tuple)):
                gt_depth = torch.stack([t if torch.is_tensor(t) else torch.as_tensor(t) for t in gt_depth])
            gt_depth = gt_depth.to(depth_logits.device)
            loss_depth = self.depth_head.get_depth_loss(gt_depth, depth_logits)

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
                        sample_voxel_labels = sample_voxel_labels.long().to(img_feats[0].device)
                        voxel_labels_list.append(sample_voxel_labels)
                    
                    # Point-level labels
                    if hasattr(gt_pts_seg, 'pts_semantic_mask'):
                        sample_point_labels = gt_pts_seg.pts_semantic_mask
                        sample_point_labels = sample_point_labels.long().to(img_feats[0].device)
                        point_labels_list.append(sample_point_labels)
                
                # Add voxel_coords
                if sample_voxel_coords is not None:
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
            
            # Stack voxel_labels to batch tensor (원본과 동일)
            # voxel_labels: List of (W, H, Z) -> (B, W, H, Z)
            if voxel_labels is not None and len(voxel_labels) > 0:
                voxel_label_tensor = torch.stack(voxel_labels, dim=0)  # (B, W, H, Z)
            else:
                voxel_label_tensor = None
            
            # Stack point_labels to batch tensor (원본과 동일)
            # point_labels: List of (N_i,) -> stack 필요시
            if point_labels is not None and len(point_labels) > 0:
                # point_labels는 각 샘플마다 포인트 수가 다를 수 있으므로 스택 안함
                point_label_list = point_labels
            else:
                point_label_list = None
            
            # Determine which predictions and labels to use (원본과 동일)
            if self.lovasz_input == 'voxel':
                lovasz_input_tensor = outputs_vox
                lovasz_label_tensor = voxel_label_tensor  # Dense voxel labels (B, W, H, Z)
            else:  # 'points'
                lovasz_input_tensor = outputs_pts
                # point_labels는 list이므로 그대로 유지
                lovasz_label_tensor = point_label_list
            
            if self.ce_input == 'voxel':
                ce_input_tensor = outputs_vox
                ce_label_tensor = voxel_label_tensor  # Dense voxel labels (B, W, H, Z)
            else:  # 'points'
                ce_input_tensor = outputs_pts
                ce_label_tensor = point_label_list

            # breakpoint()
            
            # Compute Lovasz loss (원본과 완전히 동일)
            if lovasz_input_tensor is not None and lovasz_label_tensor is not None:
                # Apply softmax before lovasz_softmax (원본과 동일)
                lovasz_probas = F.softmax(lovasz_input_tensor, dim=1)
                
                # Compute lovasz loss (원본과 동일)
                # lovasz_input_tensor: (B, C, W, H, Z) for voxel
                # lovasz_label_tensor: (B, W, H, Z) for voxel
                lovasz_loss = lovasz_softmax(
                    lovasz_probas, 
                    lovasz_label_tensor, 
                    per_image=False,  # 원본과 동일 (전체 배치를 한번에 처리)
                    ignore=self.ignore_label
                )
                
                losses['loss_lovasz'] = lovasz_loss
                total_loss += lovasz_loss
            
            # Compute CE loss (원본과 완전히 동일)
            if ce_input_tensor is not None and ce_label_tensor is not None:
                # PyTorch CrossEntropyLoss는 multi-dimensional input을 자동으로 처리
                # Input: (N, C, d1, d2, ..., dk)
                # Target: (N, d1, d2, ..., dk)
                # 원본과 동일하게 flatten 없이 직접 전달
                ce_loss = self.ce_loss_func(ce_input_tensor, ce_label_tensor)
                
                losses['loss_ce'] = ce_loss
                total_loss += ce_loss

            if loss_depth is not None:
                losses['loss_depth'] = loss_depth
                total_loss = total_loss + loss_depth
            
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
        
        # breakpoint()

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

            # breakpoint()
   
            # Forward through aggregator (returns tuple if point_coords, single tensor if not)
            outputs = self.tpv_aggregator(tpv_queries, point_coords_float)
            
            # Check if we should use occ3d format (STCOcc compatible)
            use_occ3d_format = (self.dataset_name == 'occ3d')
            
            if use_occ3d_format:
                # STCOcc 형식으로 반환 (occ3d용)
                import numpy as np
                import os
                
                if isinstance(outputs, tuple):
                    logits_vox, logits_pts = outputs
                else:
                    logits_vox = outputs
                
                # Get batch size
                batch_size = logits_vox.shape[0]
                
                # Convert to numpy and get argmax
                pred_occ = torch.argmax(logits_vox, dim=1)  # (B, W, H, Z)
                pred_occ_np = pred_occ.cpu().numpy().astype(np.uint8)
                
                # Get indices from img_metas
                img_metas = []
                for data_sample in batch_data_samples:
                    if hasattr(data_sample, 'metainfo'):
                        img_metas.append(data_sample.metainfo)
                    else:
                        img_metas.append({})
                
                # Create STCOcc-compatible return format
                return_dict = dict()
                return_dict['occ_results'] = pred_occ_np  # (B, W, H, Z) numpy array
                # CRITICAL: Use sample_idx if index is not available
                return_dict['index'] = [
                    img_meta.get('index', img_meta.get('sample_idx', i)) if isinstance(img_meta, dict) else i
                    for i, img_meta in enumerate(img_metas)
                ]
                
                # Save results to disk (following STCOcc implementation)
                if self.save_results:
                    # Use token (sample token) as filename, not sample_idx
                    sample_token = [
                        img_meta.get('token', img_meta.get('sample_idx', i)) if isinstance(img_meta, dict) else i
                        for i, img_meta in enumerate(img_metas)
                    ]
                    scene_name = [
                        img_meta.get('scene_name', 'unknown') if isinstance(img_meta, dict) else 'unknown'
                        for img_meta in img_metas
                    ]
                    
                    # Create save directories
                    for name in scene_name:
                        save_dir = f'results/TPVFormer/{name}'
                        if not os.path.exists(save_dir):
                            os.makedirs(save_dir)
                    
                    # Save predictions as npz files (use token as filename)
                    for i, token in enumerate(sample_token):
                        save_path = f'results/TPVFormer/{scene_name[i]}/{token}.npz'
                        np.savez(save_path, semantics=return_dict['occ_results'][i])
                
                # Also store in data_samples for backward compatibility
                for i, data_sample in enumerate(batch_data_samples):
                    if isinstance(outputs, tuple):
                        data_sample.pred_logits_vox = logits_vox[i]
                        data_sample.pred_logits_pts = logits_pts[i]
                    else:
                        data_sample.pred_logits_vox = logits_vox[i]
                    data_sample.pred_occ_sem_seg = pred_occ[i]
                
                return [return_dict]  # Return list of dict (STCOcc format)
            else:
                # 기존 형식으로 반환 (traditional GT용)
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
