import torch
import collections 
import torch.nn.functional as F

from mmdet3d.registry import MODELS as DETECTORS
from mmengine.model import BaseModel
from .bevdepth import BEVDepth
from mmdet3d.registry import MODELS

import numpy as np
import time
import copy

@DETECTORS.register_module(force=True)
class OccNet(BEVDepth):
    def __init__(self, 
            loss_cfg=None,
            disable_loss_depth=False,
            empty_idx=0,
            occ_fuser=None,
            occ_encoder_backbone=None,
            occ_encoder_neck=None,
            loss_norm=False,
            pts_voxel_encoder=None,
            pts_middle_encoder=None,
            pts_voxel_layer=None,
            **kwargs):
        # Store lidar-specific components before calling super().__init__
        self.pts_voxel_encoder_cfg = pts_voxel_encoder  
        self.pts_middle_encoder_cfg = pts_middle_encoder
        self.pts_voxel_layer_cfg = pts_voxel_layer
        
        super().__init__(**kwargs)
                
        self.loss_cfg = loss_cfg
        self.disable_loss_depth = disable_loss_depth
        self.loss_norm = loss_norm
        
        self.record_time = False
        self.time_stats = collections.defaultdict(list)
        self.empty_idx = empty_idx
        self.occ_encoder_backbone = MODELS.build(occ_encoder_backbone)
        self.occ_encoder_neck = MODELS.build(occ_encoder_neck)
        self.occ_fuser = MODELS.build(occ_fuser) if occ_fuser is not None else None
        
        # Build LiDAR-specific components
        if self.pts_voxel_encoder_cfg is not None:
            self.pts_voxel_encoder = MODELS.build(self.pts_voxel_encoder_cfg)
        if self.pts_middle_encoder_cfg is not None:
            self.pts_middle_encoder = MODELS.build(self.pts_middle_encoder_cfg)
        # pts_voxel_layer is config only, not a module
        self.pts_voxel_layer = self.pts_voxel_layer_cfg
            

    def image_encoder(self, img):
        imgs = img
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)
        
        backbone_feats = self.img_backbone(imgs)
        if self.with_img_neck:
            x = self.img_neck(backbone_feats)
            if type(x) in [list, tuple]:
                x = x[0]
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)
        
        return {'x': x,
                'img_feats': [x.clone()]}
    
    # @force_fp32()  # Removed for mmengine compatibility
    def occ_encoder(self, x):
        x = self.occ_encoder_backbone(x)
        x = self.occ_encoder_neck(x)
        return x
    def train_step(self, data, optim_wrapper):
        """Training step for MMEngine compatibility."""
        # Extract gt_occ directly from the data dict (this is where it actually is!)
        gt_occ = data.get('gt_occ', None)
        points = data.get('points', None)
        img_metas = data.get('img_metas', None)
        
        
        # Prepare kwargs for forward pass - pass the original data structure
        kwargs = {
            'mode': 'loss',
            'points': points,
            'gt_occ': gt_occ,
            'img_metas': img_metas
        }
        
        # Remove None values
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        
        # Call forward pass
        losses = self.forward(**kwargs)
        
        # Format losses for MMEngine
        parsed_losses, log_vars = self.parse_losses(losses)
        optim_wrapper.update_params(parsed_losses)
        
        return log_vars

    def parse_losses(self, losses):
        """Parse losses for MMEngine compatibility."""
        import torch
        
        
        log_vars = {}
        parsed_losses = []
        
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                if loss_value.requires_grad:
                    parsed_losses.append(loss_value)
                log_vars[loss_name] = loss_value.item()
            else:
                log_vars[loss_name] = loss_value
                
        # Sum all losses
        if parsed_losses:
            total_loss = sum(parsed_losses)
        else:
            total_loss = sum(losses.values())
            
        log_vars['loss'] = total_loss.item() if hasattr(total_loss, 'item') else total_loss
        
        
        return total_loss, log_vars

    def loss(self, data_dict):
        """Calculate loss for training.
        
        Args:
            data_dict: Input data containing gt_occ, points, img_metas, etc.
        """
        
        # Extract and parse data
        gt_occ = data_dict.get('gt_occ')
        points = data_dict.get('points')
        img_metas = data_dict.get('img_metas')
        
        
        # Extract features
        data_parsed = self._parse_inputs(data_dict)
        feats = self.extract_feat(**data_parsed)
        
        # Call pts_bbox_head.loss
        
        loss_dict = self.pts_bbox_head.loss(
            output_voxels=feats.get('output_voxels'),
            output_coords_fine=feats.get('output_coords_fine'),
            output_voxels_fine=feats.get('output_voxels_fine'),
            target_voxels=gt_occ
        )
        
        
        return loss_dict

    def forward(self, mode='tensor', **kwargs):
        """Forward method for MMDetection3D v1.4+ compatibility.
        
        Args:
            mode (str): Forward mode - 'loss', 'predict', or 'tensor'
            **kwargs: All input data including 'img_inputs', 'gt_occ', etc.
        """
        if mode == 'loss':
            # Check if we have direct data format (from train_step)
            if 'gt_occ' in kwargs and 'points' in kwargs:
                
                # Extract data directly
                gt_occ = kwargs.get('gt_occ')
                points = kwargs.get('points')
                img_metas = kwargs.get('img_metas')
                
                # Ensure gt_occ is on the correct device
                if isinstance(gt_occ, list) and len(gt_occ) > 0:
                    if hasattr(gt_occ[0], 'cuda'):
                        gt_occ = [item.cuda() for item in gt_occ]
                elif hasattr(gt_occ, 'cuda'):
                    gt_occ = gt_occ.cuda()
                
                
                # Extract features directly
                data_parsed = self._parse_inputs(kwargs)
                # For LiDAR-only mode, extract_feat needs points, img=None, img_metas
                feats = self.extract_feat(
                    points=data_parsed['points'], 
                    img=None, 
                    img_metas=data_parsed['img_metas']
                )
                
                
                # Unpack the features tuple (voxel_feats_enc, img_feats, pts_feats, depth)
                voxel_feats_enc, img_feats, pts_feats, depth = feats
                
                # Call pts_bbox_head to get outputs
                outs = self.pts_bbox_head(
                    voxel_feats=voxel_feats_enc,
                    points=None,  # points_occ (not used in this case)
                    img_metas=data_parsed['img_metas'],
                    img_feats=img_feats,
                    pts_feats=pts_feats,
                    transform=None,
                )
                
                
                # Call pts_bbox_head.loss directly
                loss_dict = self.pts_bbox_head.loss(
                    output_voxels=outs['output_voxels'],
                    output_coords_fine=outs['output_coords_fine'],
                    output_voxels_fine=outs['output_voxels_fine'],
                    target_voxels=gt_occ
                )
                
                
                return loss_dict
            else:
                # Use MMEngine format parsing
                result = self.loss(kwargs)
                return result
        elif mode == 'predict':
            return self.predict(kwargs) 
        else:
            # Extract input data
            data_dict = self._parse_inputs(kwargs)
            return self.extract_feat(**data_dict)
    
    def _parse_inputs(self, kwargs):
        """Parse inputs from MMDetection3D data structure"""
        
        # Extract data from inputs dictionary (MMDetection3D v1.1+)
        if 'inputs' in kwargs:
            inputs = kwargs['inputs']
                
            if isinstance(inputs, dict):
                # Extract points and img from inputs
                if 'points' in inputs:
                    kwargs['points'] = inputs['points']
                if 'img' in inputs:
                    kwargs['img_inputs'] = inputs['img']
                # Check for gt_occ in inputs
                if 'gt_occ' in inputs:
                    kwargs['gt_occ'] = inputs['gt_occ']
                # Check inside voxels dict for gt_occ
                elif 'voxels' in inputs and isinstance(inputs['voxels'], dict):
                    voxels_dict = inputs['voxels']
                    if 'gt_occ' in voxels_dict:
                        kwargs['gt_occ'] = voxels_dict['gt_occ']
                    
        # Extract data_samples info
        if 'data_samples' in kwargs:
            data_samples = kwargs['data_samples']
            if data_samples and len(data_samples) > 0:
                # Try to extract gt_occ from data_samples
                sample = data_samples[0]
                
                # Check various possible attribute names
                for attr in ['gt_occ', 'gt_occ_1_1', 'gt_occupancy', 'occ_gt', 'gt_seg_3d']:
                    if hasattr(sample, attr):
                        val = getattr(sample, attr)
                        if kwargs.get('gt_occ') is None:
                            kwargs['gt_occ'] = val
                            
                if hasattr(sample, 'metainfo'):
                    kwargs['img_metas'] = [sample.metainfo]
                    
        # Check for gt_occ directly in kwargs (this is where it should be in MMDetection3D)
        if 'gt_occ' in kwargs:
            pass
        else:
            # Try alternative keys
            for key in ['gt_occ_1_1', 'gt_occupancy', 'occ_gt']:
                if key in kwargs and kwargs[key] is not None:
                    kwargs['gt_occ'] = kwargs[key]
                    break
                    
        return kwargs
    
    def loss(self, data_dict, **extra_kwargs):
        """Loss forward function."""
        # Check if data is already in direct format (from train_step)
        if 'gt_occ' in data_dict and 'points' in data_dict:
            # Data is already parsed, use as-is
            pass
        else:
            # Parse inputs from MMDetection3D data structure
            data_dict = self._parse_inputs(data_dict)
        
        # Ensure gt_occ is present
        if data_dict.get('gt_occ') is None:
            raise ValueError("gt_occ must be provided for training. Check your data pipeline and LoadOccupancy configuration.")
        
        # Handle case where gt_occ might be a list (batch of tensors)
        if isinstance(data_dict['gt_occ'], list):
            # Stack list of tensors into a batch tensor
            import torch as torch_mod
            data_dict['gt_occ'] = torch_mod.stack(data_dict['gt_occ'], dim=0)
        
        # Ensure gt_occ has batch dimension if not already present
        if data_dict['gt_occ'].dim() == 3:
            data_dict['gt_occ'] = data_dict['gt_occ'].unsqueeze(0)  # Add batch dimension
        
        # Move gt_occ to same device as model
        device = next(self.parameters()).device
        data_dict['gt_occ'] = data_dict['gt_occ'].to(device)
        
        
        # Extract data from inputs dict
        img_inputs = data_dict.get('img_inputs')
        gt_occ = data_dict.get('gt_occ')
        img_metas = data_dict.get('img_metas', [{}] * len(img_inputs) if img_inputs is not None else [{}])
        
        return self.forward_train(
            points=data_dict.get('points'),
            img_inputs=img_inputs,
            img_metas=img_metas,
            gt_occ=gt_occ,
            **extra_kwargs
        )
    
    def predict(self, data_dict, **extra_kwargs):
        """Predict forward function."""
        # Extract data from inputs dict
        img_inputs = data_dict.get('img_inputs')
        img_metas = data_dict.get('img_metas', [{}] * len(img_inputs) if img_inputs is not None else [{}])
        
        return self.forward_test(
            img_inputs=img_inputs,
            img_metas=img_metas,
            **extra_kwargs
        )
    
    def _forward(self, data_dict, **extra_kwargs):
        """Simple forward function."""
        # Extract data from inputs dict  
        img_inputs = data_dict.get('img_inputs')
        img_metas = data_dict.get('img_metas', [{}] * len(img_inputs) if img_inputs is not None else [{}])
        
        # Extract features only
        voxel_feats, img_feats, pts_feats, depth = self.extract_feat(
            points=None, img=img_inputs, img_metas=img_metas)
        return voxel_feats

    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        
        if self.record_time:
            torch.cuda.synchronize()
            t0 = time.time()
        
        # Handle different input formats from DataLoader
        import torch
        
        # Extract all components from img_inputs tuple
        # img_inputs structure: (imgs, rots, trans, intrins, post_rots, post_trans, bda_rot, img_shape, gt_depths, sensor2sensors)
        if isinstance(img, list):
            # Take first sample from batch - this might be a wrapped tensor
            if len(img) == 1 and isinstance(img[0], torch.Tensor):
                # Case: img_inputs was converted to a single tensor wrapped in a list
                imgs = img[0]
                img_tuple = None
            else:
                # Case: img_inputs is a proper list with tuple inside
                img_tuple = img[0]
        elif isinstance(img, tuple):
            img_tuple = img
        else:
            # If img is already processed, handle differently
            imgs = img
            img_tuple = None
        
        if img_tuple is not None:
            # Debug: Print tuple structure
            
            # img_inputs tuple structure after LoadAnnotationsBEVDepth:
            # (imgs, rots, trans, intrins, post_rots, post_trans, bda_rot, img_shape, gt_depths, sensor2sensors)
            imgs = img_tuple[0]
            rots = img_tuple[1] if len(img_tuple) > 1 else None
            trans = img_tuple[2] if len(img_tuple) > 2 else None
            intrins = img_tuple[3] if len(img_tuple) > 3 else None
            post_rots = img_tuple[4] if len(img_tuple) > 4 else None
            post_trans = img_tuple[5] if len(img_tuple) > 5 else None
            bda = img_tuple[6] if len(img_tuple) > 6 else None
            # img_tuple[7] is img_shape
            gt_depths = img_tuple[8] if len(img_tuple) > 8 else None
            sensor2sensors = img_tuple[9] if len(img_tuple) > 9 else None
        else:
            # Handle case where img_inputs is just a tensor (fallback)
            rots = None
            trans = None  
            intrins = None
            post_rots = None
            post_trans = None
            bda = None
            gt_depths = None
            sensor2sensors = None
        
        # Ensure imgs is a tensor
        if isinstance(imgs, list):
            imgs = torch.stack(imgs)
        
        # Debug output
        
        # Add batch dimension if missing (B, N, C, H, W)
        if len(imgs.shape) == 4:  # [N, C, H, W]
            imgs = imgs.unsqueeze(0)  # [1, N, C, H, W]
                
        img_enc_feats = self.image_encoder(imgs)
        x = img_enc_feats['x']
        img_feats = img_enc_feats['img_feats']
        
        if self.record_time:
            torch.cuda.synchronize()
            t1 = time.time()
            self.time_stats['img_encoder'].append(t1 - t0)

        # Use already extracted variables from img_inputs tuple
        # Check if we have valid camera parameters
        if rots is not None and trans is not None and intrins is not None:
            mlp_input = self.img_view_transformer.get_mlp_input(rots, trans, intrins, post_rots, post_trans, bda)
            geo_inputs = [rots, trans, intrins, post_rots, post_trans, bda, mlp_input]
            x, depth = self.img_view_transformer([x] + geo_inputs)
        else:
            # Create dummy camera parameters for fallback
            B, N, C, H, W = imgs.shape
            device = imgs.device
            rots = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1)
            trans = torch.zeros(B, N, 3, device=device)
            intrins = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1)
            post_rots = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1)
            post_trans = torch.zeros(B, N, 3, device=device)
            bda = torch.eye(3, device=device).unsqueeze(0)
            
            mlp_input = self.img_view_transformer.get_mlp_input(rots, trans, intrins, post_rots, post_trans, bda)
            geo_inputs = [rots, trans, intrins, post_rots, post_trans, bda, mlp_input]
            x, depth = self.img_view_transformer([x] + geo_inputs)

        if self.record_time:
            torch.cuda.synchronize()
            t2 = time.time()
            self.time_stats['view_transformer'].append(t2 - t1)
        
        return x, depth, img_feats

    def extract_pts_feat(self, pts):
        if self.record_time:
            torch.cuda.synchronize()
            t0 = time.time()
        
        
        # Check if pts_voxel_encoder and pts_middle_encoder exist
        if not hasattr(self, 'pts_voxel_encoder') or self.pts_voxel_encoder is None:
            print("ERROR: pts_voxel_encoder is not initialized!")
            return None, None
        if not hasattr(self, 'pts_middle_encoder') or self.pts_middle_encoder is None:
            print("ERROR: pts_middle_encoder is not initialized!")
            return None, None
            
        # Use data_preprocessor for voxelization
        if hasattr(self, 'data_preprocessor') and hasattr(self.data_preprocessor, 'voxelize'):
            # Create dummy data_samples for voxelization 
            data_samples = [{}] * len(pts) if isinstance(pts, list) else [{}]
            voxel_dict = self.data_preprocessor.voxelize(pts, data_samples)
            voxels = voxel_dict['voxels']
            num_points = voxel_dict['num_points']  
            coors = voxel_dict['coors']
        else:
            # data_preprocessor is required for voxelization
            raise NotImplementedError("data_preprocessor with voxelize method is required for LiDAR point processing")
        # Move tensors to the same device as model
        device = next(self.parameters()).device
        voxels = voxels.to(device)
        num_points = num_points.to(device)
        coors = coors.to(device)
        
        
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        
        batch_size = coors[-1, 0] + 1
        pts_enc_feats = self.pts_middle_encoder(voxel_features, coors, batch_size)
        
        if self.record_time:
            torch.cuda.synchronize()
            t1 = time.time()
            self.time_stats['pts_encoder'].append(t1 - t0)
        
        pts_feats = pts_enc_feats['pts_feats']
        return pts_enc_feats['x'], pts_feats

    def extract_feat(self, points, img, img_metas):
        """Extract features from images and points."""
        
        img_voxel_feats = None
        pts_voxel_feats, pts_feats = None, None
        depth, img_feats = None, None
        if img is not None:
            img_voxel_feats, depth, img_feats = self.extract_img_feat(img, img_metas)
        if points is not None:
            pts_voxel_feats, pts_feats = self.extract_pts_feat(points)

        if self.record_time:
            torch.cuda.synchronize()
            t0 = time.time()

        
        if self.occ_fuser is not None:
            voxel_feats = self.occ_fuser(img_voxel_feats, pts_voxel_feats)
        else:
            assert (img_voxel_feats is None) or (pts_voxel_feats is None)
            voxel_feats = img_voxel_feats if pts_voxel_feats is None else pts_voxel_feats
            

        if self.record_time:
            torch.cuda.synchronize()
            t1 = time.time()
            self.time_stats['occ_fuser'].append(t1 - t0)

        voxel_feats_enc = self.occ_encoder(voxel_feats)
        if type(voxel_feats_enc) is not list:
            voxel_feats_enc = [voxel_feats_enc]

        if self.record_time:
            torch.cuda.synchronize()
            t2 = time.time()
            self.time_stats['occ_encoder'].append(t2 - t1)

        return (voxel_feats_enc, img_feats, pts_feats, depth)
    
    # @force_fp32(apply_to=('voxel_feats'))  # Removed for mmengine compatibility
    def forward_pts_train(
            self,
            voxel_feats,
            gt_occ=None,
            points_occ=None,
            img_metas=None,
            transform=None,
            img_feats=None,
            pts_feats=None,
            visible_mask=None,
        ):
        
        if self.record_time:
            torch.cuda.synchronize()
            t0 = time.time()
        
        outs = self.pts_bbox_head(
            voxel_feats=voxel_feats,
            points=points_occ,
            img_metas=img_metas,
            img_feats=img_feats,
            pts_feats=pts_feats,
            transform=transform,
        )
        
        if self.record_time:
            torch.cuda.synchronize()
            t1 = time.time()
            self.time_stats['occ_head'].append(t1 - t0)
        
        losses = self.pts_bbox_head.loss(
            output_voxels=outs['output_voxels'],
            output_voxels_fine=outs['output_voxels_fine'],
            output_coords_fine=outs['output_coords_fine'],
            target_voxels=gt_occ,
            target_points=points_occ,
            img_metas=img_metas,
            visible_mask=visible_mask,
        )
        
        if self.record_time:
            torch.cuda.synchronize()
            t2 = time.time()
            self.time_stats['loss_occ'].append(t2 - t1)
        
        return losses
    
    def forward_train(self,
            points=None,
            img_metas=None,
            img_inputs=None,
            gt_occ=None,
            points_occ=None,
            visible_mask=None,
            **kwargs,
        ):

        # extract bird-eye-view features from perspective images
        voxel_feats, img_feats, pts_feats, depth = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas)
        
        # training losses
        losses = dict()
        
        if self.record_time:        
            torch.cuda.synchronize()
            t0 = time.time()
        
        if not self.disable_loss_depth and depth is not None:
            # Handle fallback case where img_inputs might not be a tuple
            if isinstance(img_inputs, (list, tuple)) and len(img_inputs) > 8:
                gt_depths_raw = img_inputs[-2]  # Extract gt_depths from tuple
                
                # Check if gt_depths_raw is itself a tuple/list and extract tensor
                if isinstance(gt_depths_raw, (list, tuple)):
                    # If it's a tuple/list, try to get the first tensor element
                    for item in gt_depths_raw:
                        if hasattr(item, 'shape'):  # It's a tensor
                            gt_depths = item
                            # Add batch dimension if missing
                            if len(gt_depths.shape) == 3:  # [N, H, W]
                                gt_depths = gt_depths.unsqueeze(0)  # [1, N, H, W]
                            break
                    else:
                        # No tensor found, create dummy
                        B, N = 1, 6
                        device = depth.device if hasattr(depth, 'device') else 'cpu'
                        gt_depths = torch.zeros((B, N, depth.shape[-2], depth.shape[-1]), device=device)
                elif hasattr(gt_depths_raw, 'shape'):
                    # It's already a tensor
                    gt_depths = gt_depths_raw
                    # Add batch dimension if missing
                    if len(gt_depths.shape) == 3:  # [N, H, W]
                        gt_depths = gt_depths.unsqueeze(0)  # [1, N, H, W]
                else:
                    # Unknown type, create dummy
                    B, N = 1, 6
                    device = depth.device if hasattr(depth, 'device') else 'cpu'
                    gt_depths = torch.zeros((B, N, depth.shape[-2], depth.shape[-1]), device=device)
            else:
                # Fallback: create dummy gt_depths for loss calculation
                B, N = 1, 6  # Typical values
                device = depth.device if hasattr(depth, 'device') else 'cpu'
                gt_depths = torch.zeros((B, N, depth.shape[-2], depth.shape[-1]), device=device)
            
            losses['loss_depth'] = self.img_view_transformer.get_depth_loss(gt_depths, depth)
        
        if self.record_time:
            torch.cuda.synchronize()
            t1 = time.time()
            self.time_stats['loss_depth'].append(t1 - t0)
        
        # Handle transform extraction with fallback
        if img_inputs is not None:
            if isinstance(img_inputs, (list, tuple)) and len(img_inputs) > 7:
                transform = img_inputs[1:8]  # Extract transform from tuple
            else:
                # Fallback: no transform available
                transform = None
        else:
            transform = None
        losses_occupancy = self.forward_pts_train(voxel_feats, gt_occ,
                        points_occ, img_metas, img_feats=img_feats, pts_feats=pts_feats, transform=transform, 
                        visible_mask=visible_mask)
        losses.update(losses_occupancy)
        if self.loss_norm:
            for loss_key in losses.keys():
                if loss_key.startswith('loss'):
                    losses[loss_key] = losses[loss_key] / (losses[loss_key].detach() + 1e-9)

        def logging_latencies():
            # logging latencies
            avg_time = {key: sum(val) / len(val) for key, val in self.time_stats.items()}
            sum_time = sum(list(avg_time.values()))
            out_res = ''
            for key, val in avg_time.items():
                out_res += '{}: {:.4f}, {:.1f}, '.format(key, val, val / sum_time)
            
            print(out_res)
        
        if self.record_time:
            logging_latencies()
        
        return losses
        
    def forward_test(self,
            points=None,
            img_metas=None,
            img_inputs=None,
            gt_occ=None,
            visible_mask=None,
            **kwargs,
        ):
        return self.simple_test(img_metas, img_inputs, points, gt_occ=gt_occ, visible_mask=visible_mask, **kwargs)
    
    def simple_test(self, img_metas, img=None, points=None, rescale=False, points_occ=None, 
            gt_occ=None, visible_mask=None):
        
        voxel_feats, img_feats, pts_feats, depth = self.extract_feat(points, img=img, img_metas=img_metas)

        transform = img[1:8] if img is not None else None
        output = self.pts_bbox_head(
            voxel_feats=voxel_feats,
            points=points_occ,
            img_metas=img_metas,
            img_feats=img_feats,
            pts_feats=pts_feats,
            transform=transform,
        )

        pred_c = output['output_voxels'][0]
        SC_metric, _ = self.evaluation_semantic(pred_c, gt_occ, eval_type='SC', visible_mask=visible_mask)
        SSC_metric, SSC_occ_metric = self.evaluation_semantic(pred_c, gt_occ, eval_type='SSC', visible_mask=visible_mask)

        pred_f = None
        SSC_metric_fine = None
        if output['output_voxels_fine'] is not None:
            if output['output_coords_fine'] is not None:
                fine_pred = output['output_voxels_fine'][0]  # N ncls
                fine_coord = output['output_coords_fine'][0]  # 3 N
                pred_f = self.empty_idx * torch.ones_like(gt_occ)[:, None].repeat(1, fine_pred.shape[1], 1, 1, 1).float()
                pred_f[:, :, fine_coord[0], fine_coord[1], fine_coord[2]] = fine_pred.permute(1, 0)[None]
            else:
                pred_f = output['output_voxels_fine'][0]
            SC_metric, _ = self.evaluation_semantic(pred_f, gt_occ, eval_type='SC', visible_mask=visible_mask)
            SSC_metric_fine, SSC_occ_metric_fine = self.evaluation_semantic(pred_f, gt_occ, eval_type='SSC', visible_mask=visible_mask)

        test_output = {
            'SC_metric': SC_metric,
            'SSC_metric': SSC_metric,
            'pred_c': pred_c,
            'pred_f': pred_f,
        }

        if SSC_metric_fine is not None:
            test_output['SSC_metric_fine'] = SSC_metric_fine

        return test_output


    def evaluation_semantic(self, pred, gt, eval_type, visible_mask=None):
        _, H, W, D = gt.shape
        pred = F.interpolate(pred, size=[H, W, D], mode='trilinear', align_corners=False).contiguous()
        pred = torch.argmax(pred[0], dim=0).cpu().numpy()
        gt = gt[0].cpu().numpy()
        gt = gt.astype(np.int)

        # ignore noise
        noise_mask = gt != 255

        if eval_type == 'SC':
            # 0 1 split
            gt[gt != self.empty_idx] = 1
            pred[pred != self.empty_idx] = 1
            return fast_hist(pred[noise_mask], gt[noise_mask], max_label=2), None


        if eval_type == 'SSC':
            hist_occ = None
            if visible_mask is not None:
                visible_mask = visible_mask[0].cpu().numpy()
                mask = noise_mask & (visible_mask!=0)
                hist_occ = fast_hist(pred[mask], gt[mask], max_label=17)

            hist = fast_hist(pred[noise_mask], gt[noise_mask], max_label=17)
            return hist, hist_occ
    
    def forward_dummy(self,
            points=None,
            img_metas=None,
            img_inputs=None,
            points_occ=None,
            **kwargs,
        ):

        voxel_feats, img_feats, pts_feats, depth = self.extract_feat(points, img=img_inputs, img_metas=img_metas)

        transform = img_inputs[1:8] if img_inputs is not None else None
        output = self.pts_bbox_head(
            voxel_feats=voxel_feats,
            points=points_occ,
            img_metas=img_metas,
            img_feats=img_feats,
            pts_feats=pts_feats,
            transform=transform,
        )
        
        return output
    
    
def fast_hist(pred, label, max_label=18):
    pred = copy.deepcopy(pred.flatten())
    label = copy.deepcopy(label.flatten())
    bin_count = np.bincount(max_label * label.astype(int) + pred, minlength=max_label ** 2)
    return bin_count[:max_label ** 2].reshape(max_label, max_label)
