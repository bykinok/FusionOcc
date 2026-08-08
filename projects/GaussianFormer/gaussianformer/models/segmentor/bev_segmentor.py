from mmdet3d.registry import MODELS
from mmengine.logging import MessageHub
import numpy as np
import torch, time

from .base_segmentor import CustomBaseSegmentor
from ..utils.checkpoint_util import refine_load_from_sd
from ...losses import OPENOCC_LOSS


@MODELS.register_module()
class BEVSegmentor(CustomBaseSegmentor):
    """Ported from GaussianFormer_ori model/segmentor/bev_segmentor.py.

    ``extract_img_feat`` and ``forward_bev`` below are the original
    ``BEVSegmentor.forward`` logic, unchanged. GaussianFormer_ori's own
    train.py drives training with a bespoke loop: it calls
    ``my_model(imgs=..., metas=...)``, remaps the returned dict through
    ``cfg.loss_input_convertion`` and feeds it to a separately-built
    ``OPENOCC_LOSS`` module. To run under this framework's mmengine
    ``Runner`` instead, ``loss()``/``predict()``/``_forward()`` and the
    ``forward()`` dispatcher below reproduce exactly that glue (including
    ``loss_input_convertion`` and the ``OPENOCC_LOSS.build(loss)`` call)
    inside the model, so ``BaseModel.train_step``/``val_step`` can drive it.
    """

    def __init__(
        self,
        freeze_img_backbone=False,
        freeze_img_neck=False,
        freeze_lifter=False,
        img_backbone_out_indices=[1, 2, 3],
        extra_img_backbone=None,
        loss=None,
        loss_input_convertion=None,
        temperature=None,
        export_occ_logits=False,
        compute_uncertainty=False,
        # use_post_fusion=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # self.fp16_enabled = False
        self.freeze_img_backbone = freeze_img_backbone
        self.freeze_img_neck = freeze_img_neck
        self.img_backbone_out_indices = img_backbone_out_indices
        # self.use_post_fusion = use_post_fusion

        if freeze_img_backbone:
            self.img_backbone.requires_grad_(False)
        if freeze_img_neck:
            self.img_neck.requires_grad_(False)
        if freeze_lifter:
            self.lifter.requires_grad_(False)
            if hasattr(self.lifter, "random_anchors"):
                self.lifter.random_anchors.requires_grad = True
        if extra_img_backbone is not None:
            self.extra_img_backbone = MODELS.build(extra_img_backbone)

        # === training glue that GaussianFormer_ori's train.py handled
        # externally (loss_func = OPENOCC_LOSS.build(cfg.loss); loss_input
        # built from cfg.loss_input_convertion) is kept here verbatim so
        # this model works with mmengine's BaseModel.train_step. ===
        self.loss_input_convertion = loss_input_convertion or {}
        self.loss_func = OPENOCC_LOSS.build(loss) if loss is not None else None

        # === confidence calibration (temperature scaling), fit offline via
        # tools/train_temperature.py on a held-out split -- see predict(). ===
        self.temperature = temperature
        self.export_occ_logits = export_occ_logits
        self.compute_uncertainty = compute_uncertainty

    def extract_img_feat(self, imgs, **kwargs):
        """Extract features of images."""
        B = imgs.size(0)
        result = {}

        B, N, C, H, W = imgs.size()
        imgs = imgs.reshape(B * N, C, H, W)
        img_feats_backbone = self.img_backbone(imgs)
        if isinstance(img_feats_backbone, dict):
            img_feats_backbone = list(img_feats_backbone.values())
        img_feats = []
        for idx in self.img_backbone_out_indices:
            img_feats.append(img_feats_backbone[idx])
        img_feats = self.img_neck(img_feats)
        if isinstance(img_feats, dict):
            secondfpn_out = img_feats["secondfpn_out"][0]
            BN, C, H, W = secondfpn_out.shape
            secondfpn_out = secondfpn_out.view(B, int(BN / B), C, H, W)
            img_feats = img_feats["fpn_out"]
            result.update({"secondfpn_out": secondfpn_out})

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            # if self.use_post_fusion:
            #     img_feats_reshaped.append(img_feat.unsqueeze(1))
            # else:
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        result.update({'ms_img_feats': img_feats_reshaped})
        return result

    def forward_extra_img_backbone(self, imgs, **kwargs):
        """Extract features of images."""
        B, N, C, H, W = imgs.size()
        imgs = imgs.reshape(B * N, C, H, W)
        img_feats_backbone = self.extra_img_backbone(imgs)

        if isinstance(img_feats_backbone, dict):
            img_feats_backbone = list(img_feats_backbone.values())

        img_feats_backbone_reshaped = []
        for img_feat_backbone in img_feats_backbone:
            BN, C, H, W = img_feat_backbone.size()
            img_feats_backbone_reshaped.append(
                img_feat_backbone.view(B, int(BN / B), C, H, W))
        return img_feats_backbone_reshaped

    def forward_bev(self,
                imgs=None,
                metas=None,
                points=None,
                extra_backbone=False,
                occ_only=False,
                rep_only=False,
                **kwargs,
        ):
        """Forward training function. (verbatim BEVSegmentor.forward body)
        """
        if extra_backbone:
            return self.forward_extra_img_backbone(imgs=imgs)

        results = {
            'imgs': imgs,
            'metas': metas,
            'points': points
        }
        results.update(kwargs)
        outs = self.extract_img_feat(**results)
        results.update(outs)

        # torch.cuda.synchronize()
        # start_time = time.perf_counter()
        outs = self.lifter(**results)
        # torch.cuda.synchronize()
        # elapsed = time.perf_counter() - start_time
        # results.update({"lifter_time": elapsed})

        results.update(outs)
        outs = self.encoder(**results)
        if rep_only:
            return outs['representation']
        results.update(outs)
        if occ_only and hasattr(self.head, "forward_occ"):
            outs = self.head.forward_occ(**results)
        else:
            outs = self.head(**results)
        results.update(outs)
        return results

    # ------------------------------------------------------------------
    # mmengine Runner entry points (train_step/val_step/test_step call
    # forward(**data, mode=...) via BaseModel._run_forward). The dataset's
    # collate_fn (dataset/utils.py::custom_collate_fn_temporal, unchanged)
    # produces a flat dict keyed 'img'/'projection_mat'/'occ_label'/... —
    # exactly what GaussianFormer_ori's train.py did with
    # `imgs = data.pop('img'); metas = data` each iteration, reproduced here.
    # ------------------------------------------------------------------
    def _split_batch(self, kwargs):
        imgs = kwargs.pop('img')
        metas = kwargs
        return imgs, metas

    def forward(self, mode='tensor', **kwargs):
        imgs, metas = self._split_batch(kwargs)
        results = self.forward_bev(imgs=imgs, metas=metas)
        if mode == 'loss':
            return self.loss(results, metas)
        elif mode == 'predict':
            return self.predict(results, metas)
        return results

    def loss(self, results, metas):
        assert self.loss_func is not None, \
            "BEVSegmentor was built without a `loss` config; cannot train."
        loss_input = {'metas': metas}
        for dst_key, src_key in self.loss_input_convertion.items():
            loss_input[dst_key] = results[src_key]
        tot_loss, loss_dict = self.loss_func(loss_input)
        # MultiLoss.forward() (unchanged) returns per-component *floats*
        # (via .detach().item()) for its own tensorboard logging, so they
        # can't be returned as-is from loss() (mmengine's parse_losses
        # requires tensors) -- surface them through MessageHub instead.
        message_hub = MessageHub.get_current_instance()
        for name, value in loss_dict.items():
            message_hub.update_scalar(f'train/{name}', value)
        return {'loss': tot_loss}

    def predict(self, results, metas):
        # === temperature scaling (calibration axis) ===
        # GaussianHead's CUDA aggregator only exposes a post-aggregation
        # class *probability* distribution (`pred_occ[-1]`, verified to sum
        # to 1 over classes -- see GaussianHead.prepare_gaussian_args/
        # forward), not pre-softmax logits like SurroundOcc/CONet's
        # detectors divide by `temperature`. softmax(logit/T) is exactly
        # equal to normalize(p ** (1/T)) when p = softmax(logit) (the shared
        # normalizing constant cancels), so this rescales in probability
        # space instead -- same effect, no logit tensor needed. `p` (T=1 if
        # no temperature set) is also what `occ_logits`/`softmax_probs`
        # below are derived from, matching CONet's own convention of
        # computing uncertainty/exported-logits from the
        # *already-temperature-scaled* distribution.
        p = results['pred_occ'][-1].clamp_min(1e-12)
        if self.temperature is not None and self.temperature != 1.0:
            p = p.pow(1.0 / self.temperature)
            p = p / p.sum(dim=1, keepdim=True)
        final_occ = p.argmax(dim=1)

        batch_size = final_occ.shape[0]
        occ_mask = results.get('occ_mask')
        lidar_origin = metas.get('lidar_origin') if metas else None
        mask_lidar = metas.get('mask_lidar') if metas else None
        index = metas.get('index') if metas else None
        out = []
        for i in range(batch_size):
            sample = dict(
                final_occ=final_occ[i],
                sampled_label=results['sampled_label'][i],
                occ_mask=occ_mask[i] if occ_mask is not None else None,
            )
            # dense (200,200,16) grid form for grid-based metrics (RayIoU,
            # GaussianFormerOcc3DMetric) and tools/export_occ_logits.py --
            # valid because this config's occ_xyz/sampled_label/occ_mask are
            # the flattened full grid, not a subsampled point set. Returned
            # as numpy (not `final_occ`/`sampled_label`/`occ_mask` above,
            # which GaussianFormerOccupancyMetric -- used by the
            # SurroundOcc-GT baseline config -- still consumes as tensors):
            # tools/export_occ_logits.py's `_process_dense_sample` calls
            # `np.asarray(...)` directly with no `.cpu()` step, matching
            # SurroundOcc/CONet's own detectors, which return
            # `occ_logits`/`voxel_semantics`/`mask_camera` as
            # `.cpu().numpy()` from predict() for the same reason.
            sample['occ_results'] = sample['final_occ'].reshape(200, 200, 16).cpu().numpy()
            sample['voxel_semantics'] = sample['sampled_label'].reshape(200, 200, 16).cpu().numpy()
            if occ_mask is not None:
                sample['mask_camera'] = occ_mask[i].reshape(200, 200, 16).cpu().numpy()
            if mask_lidar is not None:
                sample['mask_lidar'] = mask_lidar[i].reshape(200, 200, 16).cpu().numpy()
            if lidar_origin is not None:
                sample['lidar_origin'] = lidar_origin[i].cpu().numpy()
            if index is not None:
                # SavePredictionsEvaluator / STCOcc's OccupancyMetric key
                # predictions back to GT by this integer (see
                # dataset/nuscenes_dataset.py::__getitem__).
                sample['index'] = int(index[i])

            p_i = p[i].transpose(0, 1).reshape(200, 200, 16, -1)  # (200,200,16,C)
            if self.export_occ_logits:
                # `train_temperature.py` computes `softmax(logits / T)`
                # internally, so exporting `p` itself (already a
                # probability) would double-softmax. log(p) makes
                # softmax(log(p) / T) == normalize(p ** (1/T)) exactly
                # (softmax(log(p)) == p when T=1, since exp(log(p))==p and
                # p already sums to 1) -- i.e. mathematically equivalent to
                # a true pre-softmax logit for this purpose.
                sample['occ_logits'] = torch.log(p_i.clamp_min(1e-12)).cpu().numpy()
            if self.compute_uncertainty:
                probs_np = p_i.cpu().numpy().astype('float32')
                eps = 1e-8
                sample['softmax_probs'] = probs_np
                sample['uncertainty_msp'] = (1.0 - probs_np.max(axis=-1)).astype('float32')
                sample['uncertainty_entropy'] = (
                    -(probs_np * np.log(probs_np + eps)).sum(axis=-1)
                ).astype('float32')
            out.append(sample)
        return out

    def _forward(self, **kwargs):
        kwargs.pop('mode', None)
        imgs, metas = self._split_batch(kwargs)
        return self.forward_bev(imgs=imgs, metas=metas)

    def load_state_dict(self, state_dict, strict=True):
        """Mirrors GaussianFormer_ori train.py/eval.py's checkpoint loading:
        try a plain non-strict load first, and only if that raises fall back
        to stripping img_neck.*/lifter.anchor keys (misc/checkpoint_util.py
        ::refine_load_from_sd, ported unchanged) before retrying.
        """
        try:
            return super().load_state_dict(state_dict, False)
        except Exception:
            state_dict = refine_load_from_sd(dict(state_dict))
            return super().load_state_dict(state_dict, False)
