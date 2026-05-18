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
                 temperature=None,
                 ray_aux_loss=None,
                 **kwargs):

        self.bev_h = bev_h
        self.bev_w = bev_w
        self.fp16_enabled = False
        self.num_classes=kwargs['num_classes']
        self.use_mask=use_mask
        # Temperature scaling for calibration (T > 1 softens softmax). Set via config or load_state_dict.
        self.temperature = float(temperature) if temperature is not None else None

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

        # Optional ray-aligned auxiliary loss.
        # Disabled (self.ray_aux_loss = None) when ray_aux_loss is not set in config,
        # preserving full backward-compatibility with existing checkpoints and configs.
        if ray_aux_loss is not None:
            self.ray_aux_loss = build_loss(ray_aux_loss)
        else:
            self.ray_aux_loss = None

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
        """Compute occupancy losses.

        Primary loss  : voxel-wise cross-entropy (loss_occ).
        Auxiliary loss: ray-aligned loss (loss_ray) – only when
                        ray_aux_loss is configured in the model config.

        Args:
            voxel_semantics : (B, X, Y, Z) int GT labels [0..17], or list of 1.
            mask_camera     : (B, X, Y, Z) uint8 visibility mask, or list of 1.
            preds_dicts     : dict with key 'occ' → (B, X, Y, Z, C) logits.
            gt_bboxes_ignore: unused, kept for API compatibility.
            img_metas       : list[dict] of length B.  Used to extract per-sample
                              lidar origins via img_metas[b]['ego2lidar'].
                              ego2lidar is the ego→lidar 4×4 matrix (built with
                              inverse=True in nuscenes_occ.py).
                              Lidar origin in ego frame = inv(ego2lidar)[:3, 3].

        Returns:
            loss_dict (dict):
                'loss_occ'             – primary CE loss (always present)
                'loss_ray'             – ray loss total  (only when enabled)
                'ray_pre_free'         – pre-hit BCE value   (logging only)
                'ray_hit_occ'          – first-hit BCE value (logging only)
                'ray_hit_sem'          – first-hit CE value  (logging only)
                'num_valid_rays'       – logging scalar      (only when enabled)
                'num_pre_hit_voxels'   – logging scalar      (only when enabled)
                'num_hit_voxels'       – logging scalar      (only when enabled)
                'valid_ray_ratio'      – logging scalar      (only when enabled)
                'avg_voxels_per_ray'   – logging scalar      (only when enabled)
        """
        loss_dict = dict()
        occ = preds_dicts['occ']
        # occ shape: (B, X=200, Y=200, Z=16, C=18)
        # See TransformerOcc.forward (use_3d=True): permute(0,4,3,2,1) →
        # (B, bev_w, bev_h, pillar_h, out_dim) → predicter → (B, X, Y, Z, C)

        # Unwrap list wrappers coming from DataContainer collation
        if isinstance(voxel_semantics, list):
            voxel_semantics = voxel_semantics[0]
        if isinstance(mask_camera, list):
            mask_camera = mask_camera[0]

        assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17, (
            f"voxel_semantics out of expected range [0,17]: "
            f"min={voxel_semantics.min()}, max={voxel_semantics.max()}"
        )

        # ── Primary voxel-wise CE loss ───────────────────────────────────
        loss_dict['loss_occ'] = self.loss_single(voxel_semantics, mask_camera, occ)

        # ── Optional ray-aligned auxiliary loss ─────────────────────────
        # Enabled only when `ray_aux_loss` is specified in config;
        # zero impact on training otherwise (self.ray_aux_loss is None).
        if self.ray_aux_loss is not None:
            device = occ.device

            # Prepare tensors: convert numpy → Tensor, move to device
            if isinstance(voxel_semantics, np.ndarray):
                voxel_semantics = torch.from_numpy(voxel_semantics)
            if isinstance(mask_camera, np.ndarray):
                mask_camera = torch.from_numpy(mask_camera)

            vs_for_ray = voxel_semantics.to(device)
            mc_for_ray = mask_camera.to(device)

            # DataContainer collation unpacks the list to a single element,
            # so when B=1, the shape can be (X, Y, Z) instead of (1, X, Y, Z).
            # Restore the batch dim so ray_aux_loss always sees (B, X, Y, Z).
            if vs_for_ray.dim() == 3:
                vs_for_ray = vs_for_ray.unsqueeze(0)   # (1, X, Y, Z)
            if mc_for_ray.dim() == 3:
                mc_for_ray = mc_for_ray.unsqueeze(0)   # (1, X, Y, Z)

            # ── Extract per-sample lidar origin from img_metas ──────────
            # img_metas is list[dict] of length B (one dict per sample).
            # ego2lidar is the ego→lidar 4×4 matrix (nuscenes_occ.py, inverse=True).
            # Lidar origin in ego frame = inv(ego2lidar)[:3, 3]
            #   = lidar2ego_translation  (the sensor mount offset in ego frame)
            #
            # Coordinate frame consistency:
            #   Both the lidar origin and the occupancy grid (pc_range) are in
            #   the NuScenes ego frame (x-forward, y-left, z-up).  No extra
            #   rotation/translation is required.
            lidar_origins = None
            if img_metas is not None and len(img_metas) > 0:
                origins_np = []
                for meta in img_metas:
                    if meta is not None and 'ego2lidar' in meta:
                        ego2lidar = np.array(meta['ego2lidar'], dtype=np.float64)
                        # inv(ego2lidar) = lidar2ego; its translation column is
                        # the lidar sensor origin in ego frame.
                        lidar2ego = np.linalg.inv(ego2lidar)
                        origin    = lidar2ego[:3, 3].astype(np.float32)
                        origins_np.append(origin)
                    else:
                        # Fallback when ego2lidar is missing (should not happen)
                        origins_np.append(
                            np.array([0.9858, 0.0, 1.8402], dtype=np.float32)
                        )
                if origins_np:
                    lidar_origins = torch.from_numpy(
                        np.stack(origins_np, axis=0)
                    ).to(device)   # (B, 3)

            # occ is already (B, X, Y, Z, C) on `device`
            ray_loss_dict = self.ray_aux_loss(
                occ_logits=occ,
                voxel_semantics=vs_for_ray,
                mask_camera=mc_for_ray,
                lidar_origins=lidar_origins,
            )

            # Merge into loss_dict; logged automatically by mmengine LoggerHook.
            # Keys without 'loss_' prefix are logged but NOT summed into
            # the total backward loss (mmengine only sums 'loss_*' keys).
            loss_dict['loss_ray']           = ray_loss_dict['loss_ray']
            loss_dict['ray_pre_free']       = ray_loss_dict['ray_pre_free']
            loss_dict['ray_hit_occ']        = ray_loss_dict['ray_hit_occ']
            loss_dict['ray_hit_sem']        = ray_loss_dict['ray_hit_sem']
            loss_dict['num_valid_rays']     = ray_loss_dict['num_valid_rays']
            loss_dict['num_pre_hit_voxels'] = ray_loss_dict['num_pre_hit_voxels']
            loss_dict['num_hit_voxels']     = ray_loss_dict['num_hit_voxels']
            loss_dict['valid_ray_ratio']    = ray_loss_dict['valid_ray_ratio']
            loss_dict['avg_voxels_per_ray'] = ray_loss_dict['avg_voxels_per_ray']

            # Optional debug metrics (present only when ray_loss_debug=True)
            for _dbg_key in (
                'dbg_p_occ_pre',
                'dbg_p_occ_hit',
                'dbg_no_hit_frac',
                'dbg_hit_depth',
                'dbg_hit_class',
            ):
                if _dbg_key in ray_loss_dict:
                    loss_dict[_dbg_key] = ray_loss_dict[_dbg_key]

        return loss_dict

    def set_epoch(self, epoch: int) -> None:
        """Propagate the current training epoch to the ray auxiliary loss.

        Called by RayLossEpochHook at the start of each epoch.
        No-op when ray_aux_loss is disabled.
        """
        if self.ray_aux_loss is not None and hasattr(self.ray_aux_loss, 'set_epoch'):
            self.ray_aux_loss.set_epoch(epoch)

    def loss_single(self,voxel_semantics,mask_camera,preds):
        # Convert numpy arrays to torch tensors if needed
        if isinstance(voxel_semantics, np.ndarray):
            voxel_semantics = torch.from_numpy(voxel_semantics)
        
        # Move tensors to the same device as preds
        device = preds.device
        voxel_semantics = voxel_semantics.to(device)
        
        voxel_semantics=voxel_semantics.long()
        if self.use_mask:
            # mask_camera가 제공된 경우에만 mask 처리
            if isinstance(mask_camera, np.ndarray):
                mask_camera = torch.from_numpy(mask_camera)
            mask_camera = mask_camera.to(device)
            voxel_semantics=voxel_semantics.reshape(-1)
            preds=preds.reshape(-1,self.num_classes)
            mask_camera=mask_camera.reshape(-1)
            num_total_samples=mask_camera.sum()
            loss_occ=self.loss_occ(preds,voxel_semantics,mask_camera, avg_factor=num_total_samples)
        else:
            # use_mask=False일 때는 mask_camera 없이 전체 GT로 학습
            voxel_semantics = voxel_semantics.reshape(-1)
            preds = preds.reshape(-1, self.num_classes)
            loss_occ = self.loss_occ(preds, voxel_semantics,)
        return loss_occ

    @force_fp32(apply_to=('preds', 'occ_logits'))
    def compute_occ_uncertainty(self, occ_logits):
        """Compute per-voxel uncertainty scores from occupancy logits.

        Args:
            occ_logits (Tensor): Raw logits, shape (B, H, W, Z, num_classes).

        Returns:
            tuple[Tensor]: (uncertainty_msp, uncertainty_entropy)
                - uncertainty_msp: 1 - max_prob, shape (B, H, W, Z). Higher = more uncertain.
                - uncertainty_entropy: predictive entropy -sum(p*log(p)), shape (B, H, W, Z).
        """
        probs = F.softmax(occ_logits, dim=-1)
        # Maximum Softmax Probability (MSP): uncertainty = 1 - max_c p(c|x)
        max_prob = probs.max(dim=-1)[0]
        uncertainty_msp = 1.0 - max_prob
        # Predictive entropy: H = -sum_c p(c|x) log p(c|x)
        eps = 1e-10
        entropy = -(probs * (probs.clamp(min=eps).log())).sum(dim=-1)
        uncertainty_entropy = entropy
        return uncertainty_msp, uncertainty_entropy

    @force_fp32(apply_to=('preds'))
    def get_occ(self, preds_dicts, img_metas, rescale=False, return_uncertainty=False):
        """Generate occupancy prediction from head logits; optionally return uncertainty maps.

        Args:
            preds_dicts: dict with 'occ' (logits), shape (B, H, W, Z, num_classes).
            img_metas (list[dict]): Meta info (unused, for API compatibility).
            rescale (bool): Unused, for API compatibility.
            return_uncertainty (bool): If True, return dict with pred_occ, uncertainty_msp, uncertainty_entropy.

        Returns:
            If return_uncertainty is False: Tensor of predicted class indices (B, H, W, Z).
            If return_uncertainty is True: dict with keys pred_occ, uncertainty_msp, uncertainty_entropy.
        """
        # breakpoint()
        occ_out = preds_dicts['occ']
        if self.temperature is not None and self.temperature != 1.0:
            occ_out = occ_out / self.temperature
        occ_score = occ_out.softmax(-1)
        pred_occ = occ_score.argmax(-1)

        # breakpoint()

        if return_uncertainty:
            uncertainty_msp, uncertainty_entropy = self.compute_occ_uncertainty(occ_out)
            return {
                'pred_occ': pred_occ,
                'uncertainty_msp': uncertainty_msp,
                'uncertainty_entropy': uncertainty_entropy,
                'softmax_probs': occ_score,  # (B,H,W,Z,num_classes) for ECE/NLL
            }
        return pred_occ
