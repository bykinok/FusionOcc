# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Xiaoyu Tian
# ---------------------------------------------

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear
try:
    from mmcv.cnn import bias_init_with_prob
except ImportError:
    # In newer versions, this might be in mmengine or removed
    import math
    def bias_init_with_prob(prior_prob=0.01):
        """initialize conv/fc bias value according to a given probability value."""
        bias_init = float(-math.log((1 - prior_prob) / prior_prob))
        return bias_init
try:
    from mmcv.utils import TORCH_VERSION, digit_version
except ImportError:
    # In newer versions, use torch version directly
    import torch
    TORCH_VERSION = torch.__version__
    def digit_version(version_str):
        from packaging import version
        return version.parse(version_str)

# Import utilities with compatibility layer
try:
    from mmdet.core import multi_apply, reduce_mean
except ImportError:
    try:
        from mmengine.utils import multi_apply
    except ImportError:
        from torch.nn.parallel import parallel_apply
        multi_apply = parallel_apply
    try:
        from mmengine.dist import reduce_mean
    except ImportError:
        def reduce_mean(tensor):
            import torch.distributed as dist
            if not dist.is_available() or not dist.is_initialized():
                return tensor
            tensor = tensor.clone()
            dist.all_reduce(tensor.div_(dist.get_world_size()))
            return tensor

try:
    from mmdet.models.utils.transformer import inverse_sigmoid
except ImportError:
    def inverse_sigmoid(x, eps=1e-5):
        import torch
        x = x.clamp(min=0, max=1)
        x1 = x.clamp(min=eps)
        x2 = (1 - x).clamp(min=eps)
        return torch.log(x1 / x2)

try:
    from mmdet.models import HEADS
except ImportError:
    from mmdet3d.registry import MODELS as HEADS

try:
    from mmdet.models.dense_heads import DETRHead
except ImportError:
    try:
        from mmdet.models.dense_heads.detr_head import DETRHead
    except ImportError:
        # Create a minimal base class if DETRHead is not available
        from mmengine.model import BaseModule
        DETRHead = BaseModule

try:
    from mmdet3d.core.bbox.coders import build_bbox_coder
except ImportError:
    # In newer versions, use MODELS registry
    from mmdet3d.registry import MODELS
    def build_bbox_coder(cfg):
        return MODELS.build(cfg)

from projects.BEVFormer.utils.bbox_util import normalize_bbox

try:
    from mmcv.cnn.bricks.transformer import build_positional_encoding
except ImportError:
    try:
        from mmdet.models.layers import build_positional_encoding
    except ImportError:
        from mmdet3d.registry import MODELS
        def build_positional_encoding(cfg):
            return MODELS.build(cfg)

try:
    from mmcv.runner import force_fp32, auto_fp16
except ImportError:
    # Create dummy decorators
    def force_fp32(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        def decorator(func):
            return func
        return decorator
    def auto_fp16(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        def decorator(func):
            return func
        return decorator
from projects.BEVFormer.utils.bricks import run_time
import numpy as np
import mmcv
import cv2 as cv
from projects.BEVFormer.utils.visual import save_tensor
from mmcv.cnn.bricks.transformer import build_positional_encoding
try:
    from mmdet.models.utils import build_transformer
except ImportError:
    try:
        from mmdet.models.layers import build_transformer
    except ImportError:
        from mmdet3d.registry import MODELS
        def build_transformer(cfg):
            return MODELS.build(cfg)
try:
    from mmdet.models.builder import build_loss
except ImportError:
    # In newer versions, use MODELS registry
    from mmdet3d.registry import MODELS
    def build_loss(cfg):
        return MODELS.build(cfg)
try:
    from mmcv.runner import BaseModule
except ImportError:
    from mmengine.model import BaseModule
# force_fp32 is already defined above

@HEADS.register_module()
class BEVFormerOccHead(BaseModule):
    """Head of Detr3D.
    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
        bev_h, bev_w (int): spatial shape of BEV queries.
    """

    def __init__(self,
                 *args,
                 with_box_refine=False,
                 as_two_stage=False,
                 transformer=None,
                 bbox_coder=None,
                 num_cls_fcs=2,
                 code_weights=None,
                 pc_range=[-40, -40, -1.0, 40, 40, 5.4],
                 bev_h=30,
                 bev_w=30,
                 loss_occ=None,
                 use_mask=False,
                 positional_encoding=None,
                 **kwargs):

        self.bev_h = bev_h
        self.bev_w = bev_w
        self.fp16_enabled = False
        self.num_classes=kwargs['num_classes']
        self.use_mask=use_mask

        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage


        self.pc_range = pc_range
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.num_cls_fcs = num_cls_fcs - 1
        super(BEVFormerOccHead, self).__init__()

        self.loss_occ = build_loss(loss_occ)
        self.positional_encoding = build_positional_encoding(
            positional_encoding)
        self.transformer = build_transformer(transformer)
        self.embed_dims = self.transformer.embed_dims

        if not self.as_two_stage:
            self.bev_embedding = nn.Embedding(
                self.bev_h * self.bev_w, self.embed_dims)

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
        # if self.loss_cls.use_sigmoid:
        #     bias_init = bias_init_with_prob(0.01)
        #     for m in self.cls_branches:
        #         nn.init.constant_(m[-1].bias, bias_init)

    @auto_fp16(apply_to=('mlvl_feats'))
    def forward(self, mlvl_feats, img_metas, prev_bev=None, only_bev=False, test=False):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
            prev_bev: previous bev featues
            only_bev: only compute BEV features with encoder.
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        device = mlvl_feats[0].device
        object_query_embeds = None
        
        # Ensure all modules are on the correct device
        if hasattr(self, 'bev_embedding'):
            self.bev_embedding = self.bev_embedding.to(device)
        if hasattr(self, 'positional_encoding'):
            self.positional_encoding = self.positional_encoding.to(device)
        if hasattr(self, 'transformer'):
            self.transformer = self.transformer.to(device)
        
        # breakpoint()

        bev_queries = self.bev_embedding.weight.to(device=device, dtype=dtype)

        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                               device=device, dtype=dtype)
        bev_pos = self.positional_encoding(bev_mask).to(device=device, dtype=dtype)

        if only_bev:  # only use encoder to obtain BEV features, TODO: refine the workaround
            return self.transformer.get_bev_features(
                mlvl_feats,
                bev_queries,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h,
                             self.real_w / self.bev_w),
                bev_pos=bev_pos,
                img_metas=img_metas,
                prev_bev=prev_bev,
            )
        else:
            outputs = self.transformer(
                mlvl_feats,
                bev_queries,
                object_query_embeds,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h,
                             self.real_w / self.bev_w),
                bev_pos=bev_pos,
                reg_branches=None,  # noqa:E501
                cls_branches=None,
                img_metas=img_metas,
                prev_bev=prev_bev
            )
        bev_embed, occ_outs = outputs

        outs = {
            'bev_embed': bev_embed,
            'occ':occ_outs,
        }

        return outs

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             # gt_bboxes_list,
             # gt_labels_list,
             voxel_semantics,
             mask_camera,
             preds_dicts,
             gt_bboxes_ignore=None,
             img_metas=None):

        loss_dict=dict()
        occ=preds_dicts['occ']
        
        # voxel_semantics and mask_camera should already be batch tensors from forward_train
        # No need to unwrap from list - they are already proper batch tensors
        # Just ensure they are tensors, not lists
        if isinstance(voxel_semantics, list):
            # This shouldn't happen with proper collate_fn, but handle it
            if len(voxel_semantics) == 1:
                voxel_semantics = voxel_semantics[0]
            else:
                # Multiple items - they should already be stacked in forward_train
                raise ValueError(f"voxel_semantics should be a tensor, got list of {len(voxel_semantics)} items")
        
        if isinstance(mask_camera, list):
            # This shouldn't happen with proper collate_fn, but handle it
            if len(mask_camera) == 1:
                mask_camera = mask_camera[0]
            else:
                # Multiple items - they should already be stacked in forward_train
                raise ValueError(f"mask_camera should be a tensor, got list of {len(mask_camera)} items")
        
        # Validate shape and range
        assert voxel_semantics.min()>=0 and voxel_semantics.max()<=17, \
            f"voxel_semantics range error: min={voxel_semantics.min()}, max={voxel_semantics.max()}"
        
        losses = self.loss_single(voxel_semantics,mask_camera,occ)
        loss_dict['loss_occ']=losses
        return loss_dict

    def loss_single(self,voxel_semantics,mask_camera,preds):
        # Convert numpy arrays to torch tensors if needed
        if isinstance(voxel_semantics, np.ndarray):
            voxel_semantics = torch.from_numpy(voxel_semantics)
        if isinstance(mask_camera, np.ndarray):
            mask_camera = torch.from_numpy(mask_camera)
        
        # Move tensors to the same device as preds
        device = preds.device
        voxel_semantics = voxel_semantics.to(device)
        mask_camera = mask_camera.to(device)
        
        voxel_semantics=voxel_semantics.long()
        if self.use_mask:
            voxel_semantics=voxel_semantics.reshape(-1)
            preds=preds.reshape(-1,self.num_classes)
            mask_camera=mask_camera.reshape(-1)
            num_total_samples=mask_camera.sum()
            loss_occ=self.loss_occ(preds,voxel_semantics,mask_camera, avg_factor=num_total_samples)
        else:
            voxel_semantics = voxel_semantics.reshape(-1)
            preds = preds.reshape(-1, self.num_classes)
            loss_occ = self.loss_occ(preds, voxel_semantics,)
        return loss_occ

    @force_fp32(apply_to=('preds'))
    def get_occ(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            predss : occ results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        # return self.transformer.get_occ(
        #     preds_dicts, img_metas, rescale=rescale)
        # print(img_metas[0].keys())
        occ_out=preds_dicts['occ']
        occ_score=occ_out.softmax(-1)
        occ_score=occ_score.argmax(-1)


        return occ_score
