"""
Ray-Aligned Auxiliary Loss for 3D Occupancy Prediction.

VERSION 4 — Depth-ordered BCE + First-hit CE
            (replaces V3 transmittance-NLL as default)

Changes from V3
───────────────
1. New default ray_loss_mode = "ordered_bce"
   The old NeRF-style transmittance chain produced harmful gradients on
   pre-first-hit voxels and caused mIoU collapse at loss_weight ≥ 0.05.
   The new formulation is fully discriminative:

   For each valid ray, let s = step index, k = GT first-hit step:
     s < k  (pre-hit, vmask):   BCE(p_occ[s], target=0)
                                = -log(softmax[free_class_id][s])
     s == k (first-hit, valid): BCE(p_occ[k], target=1)
                                = -log(1 - softmax[free_class_id][k])
                              + CE(logits[k], GT_semantic_class)
     s > k  (post-hit):         ignored – may be occluded

2. Old transmittance NLL preserved behind ray_loss_mode = "transmittance_nll"
   For ablation comparison.  Not the default.

3. New __init__ parameters
   ray_loss_mode : str   "ordered_bce" (default) | "transmittance_nll"
   w_bce_pre     : float per-step pre-hit BCE weight    (default 1.0)
   w_bce_hit     : float per-ray  first-hit BCE weight  (default 1.0)
   w_ce_hit      : float per-ray  first-hit CE weight   (default 1.0)

4. New logged metrics (non-loss_ prefix → not summed into total loss)
   ray_pre_free, ray_hit_occ, ray_hit_sem  – unweighted component values
   num_pre_hit_voxels, num_hit_voxels      – supervision counts

Previous V3 geometry retained unchanged
   - Per-sample lidar origin from img_metas['ego2lidar']
   - DDA-style voxel deduplication
   - GPU-computed voxel sequences

──────────────────────────────────────────────────────────────────────────────
GEOMETRY NOTES (unchanged from V3)
──────────────────────────────────────────────────────────────────────────────

1. Ray directions
   Reproduced from generate_lidar_rays() in ray_metrics_occ3d.py.
   pitch_angles : ≈ −0.54 rad (steep down) → +0.21 rad (up).
   azimuth      : 0° – 359° in 1° steps.
   Unit vector  :
       x = cos(pitch) * cos(az)   ← ego +X (forward)
       y = cos(pitch) * sin(az)   ← ego +Y (left)
       z = sin(pitch)             ← ego +Z (up)

2. Lidar origin in ego frame  (per-sample)
   Source: img_metas[b]['ego2lidar']  (4×4 numpy array)

   NAMING NOTE:  Despite the field name 'ego2lidar', this matrix is built in
   nuscenes_occ.py line 284–285 as:
       ego2lidar = transform_matrix(lidar2ego_translation,
                                    lidar2ego_rotation,
                                    inverse=True)
   'inverse=True' correctly produces the ego→lidar transform.

   Formula for lidar origin in ego frame:
       lidar_origin_ego = np.linalg.inv(ego2lidar)[:3, 3]
                        = lidar2ego_translation  (calibration offset)

3. Voxel index conversion
   ix = floor((x_ego − pc_range[0]) / voxel_size)
   iy = floor((y_ego − pc_range[1]) / voxel_size)
   iz = floor((z_ego − pc_range[2]) / voxel_size)

4. Model output layout (verified against TransformerOcc)
   occ_logits : (B, X=200, Y=200, Z=16, C=18)
       dim-4 = C = class logits (LAST dim, not dim-1)
       free/empty class : index 17

──────────────────────────────────────────────────────────────────────────────
REMAINING ASSUMPTIONS (post-V4)
──────────────────────────────────────────────────────────────────────────────
| Assumption           | Description                           | Impact    |
|----------------------|---------------------------------------|-----------|
| Uniform stepping     | dt=voxel_size/max|d|; dedup applied   | Minimal   |
| Single timestamp     | T=1 only                              | By design |
| Ray subsample        | Random sample of num_rays_train rays  | Stochastic|
| mask_camera=1        | Camera-visible voxels                 | Dataset   |
| Post-hit ignored     | Occluded voxels receive no gradient   | By design |
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mmdet3d.registry import MODELS as _REGISTRY
except ImportError:
    try:
        from mmdet.models.builder import LOSSES as _REGISTRY
    except ImportError:
        _REGISTRY = None


def _register(cls):
    if _REGISTRY is not None:
        try:
            _REGISTRY.register_module(module=cls)
        except Exception:
            pass
    return cls


# ──────────────────────────────────────────────────────────────────────────────
# Ray direction generator (exact reproduction of ray_metrics_occ3d.py)
# ──────────────────────────────────────────────────────────────────────────────

def _generate_lidar_rays() -> np.ndarray:
    """Reproduce generate_lidar_rays() from ray_metrics_occ3d.py.

    Returns:
        rays : (N_rays, 3) float32 unit vectors in ego frame.
               x = cos(pitch)*cos(az), y = cos(pitch)*sin(az), z = sin(pitch)
    """
    pitch_angles = []
    for k in range(10):
        angle = math.pi / 2 - math.atan(k + 1)
        pitch_angles.append(-angle)
    # Extend upward to match NuScenes LIDAR_TOP upper FOV limit (~0.21 rad)
    while pitch_angles[-1] < 0.21:
        delta = pitch_angles[-1] - pitch_angles[-2]
        pitch_angles.append(pitch_angles[-1] + delta)

    rays = []
    for pitch in pitch_angles:
        for az_deg in range(360):
            az = math.radians(az_deg)
            rays.append((
                math.cos(pitch) * math.cos(az),
                math.cos(pitch) * math.sin(az),
                math.sin(pitch),
            ))
    return np.array(rays, dtype=np.float32)  # (N_rays, 3)


# ──────────────────────────────────────────────────────────────────────────────
# Loss module
# ──────────────────────────────────────────────────────────────────────────────

@_register
class RayAlignedLoss(nn.Module):
    """Ray-aligned auxiliary loss for 3D occupancy (V4: ordered-BCE default).

    Args:
        pc_range (list):
            [x_min, y_min, z_min, x_max, y_max, z_max] in metres.
            Default: [-40, -40, -1.0, 40, 40, 5.4] (NuScenes Occ3D).
        voxel_size (float):
            Voxel edge length in metres. Default: 0.4.
        grid_size (list):
            [X, Y, Z] voxel counts. Default: [200, 200, 16].
        lidar_origin (list):
            Fallback lidar sensor origin in ego frame [x, y, z].
            Used only when per-sample origins are not provided.
        free_class_id (int):
            Class index for free/empty space. Occ3D: 17. Default: 17.
        loss_weight (float):
            Scalar weight applied to the final combined ray loss. Default: 1.0.
        eps (float):
            Numerical stability clamp for log. Default: 1e-6.
        num_rays_train (int or None):
            Subsample this many rays per forward call. None = use all.
        ray_max_dist (float):
            Maximum ray length in metres. Default: 60.0.
        use_mask (bool):
            If True, require the GT first-hit voxel to be camera-visible
            (mask_camera == 1) before counting a ray as valid.
        ray_loss_mode (str):
            "ordered_bce"      – (DEFAULT) Depth-ordered BCE + first-hit CE.
            "transmittance_nll"– Legacy NeRF-style NLL (V3 behaviour).
        w_bce_pre (float):
            Weight for the pre-hit free-space BCE term. Default: 1.0.
        w_bce_hit (float):
            Weight for the first-hit occupancy BCE term. Default: 1.0.
        w_ce_hit (float):
            Weight for the first-hit semantic CE term. Default: 1.0.
    """

    def __init__(
        self,
        pc_range: list = None,
        voxel_size: float = 0.4,
        grid_size: list = None,
        lidar_origin: list = None,
        free_class_id: int = 17,
        loss_weight: float = 1.0,
        eps: float = 1e-6,
        num_rays_train: int = 2000,
        ray_max_dist: float = 60.0,
        use_mask: bool = True,
        ray_loss_mode: str = "ordered_bce",
        w_bce_pre: float = 1.0,
        w_bce_hit: float = 1.0,
        w_ce_hit: float = 1.0,
        # ── Delayed activation & warmup ─────────────────────────────────────
        ray_loss_start_epoch: int = 0,
        ray_loss_warmup_epochs: int = 0,
        # ── Per-sample loss normalization ────────────────────────────────────
        normalize_per_sample: bool = True,
        # ── Debug diagnostics ────────────────────────────────────────────────
        ray_loss_debug: bool = False,
    ):
        super().__init__()

        if pc_range is None:
            pc_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
        if grid_size is None:
            grid_size = [200, 200, 16]
        if lidar_origin is None:
            lidar_origin = [0.9858, 0.0, 1.8402]  # NuScenes LIDAR_TOP

        assert ray_loss_mode in ("ordered_bce", "transmittance_nll"), (
            f"ray_loss_mode must be 'ordered_bce' or 'transmittance_nll', "
            f"got '{ray_loss_mode}'"
        )
        assert ray_loss_start_epoch >= 0, "ray_loss_start_epoch must be >= 0"
        assert ray_loss_warmup_epochs >= 0, "ray_loss_warmup_epochs must be >= 0"

        self.free_class_id           = free_class_id
        self.loss_weight             = loss_weight
        self.eps                     = eps
        self.num_rays_train          = num_rays_train
        self.use_mask                = use_mask
        self.ray_max_dist            = float(ray_max_dist)
        self.voxel_size              = float(voxel_size)
        self.grid_size               = list(grid_size)
        self.ray_loss_mode           = ray_loss_mode
        self.w_bce_pre               = float(w_bce_pre)
        self.w_bce_hit               = float(w_bce_hit)
        self.w_ce_hit                = float(w_ce_hit)
        self.ray_loss_start_epoch    = int(ray_loss_start_epoch)
        self.ray_loss_warmup_epochs  = int(ray_loss_warmup_epochs)
        self.normalize_per_sample    = bool(normalize_per_sample)
        self.ray_loss_debug          = bool(ray_loss_debug)
        # Current epoch — updated externally via set_epoch(); never a buffer
        # (no need to move to GPU; used as a Python scalar only)
        self._current_epoch: int     = 0

        self.register_buffer(
            'pc_range_min',
            torch.tensor(pc_range[:3], dtype=torch.float32),
        )
        self.register_buffer(
            'default_origin',
            torch.tensor(lidar_origin, dtype=torch.float32),
        )

        # ── Precompute ray directions and per-ray step sizes ─────────────────
        rays    = _generate_lidar_rays()                            # (N_rays, 3)
        max_abs = np.abs(rays).max(axis=1) + 1e-8                  # (N_rays,)
        dt_np   = (voxel_size / max_abs).astype(np.float32)        # (N_rays,)

        S_max = int(min(np.ceil(ray_max_dist / dt_np.min()), 512))
        self.S_max = S_max

        self.register_buffer('ray_dirs',   torch.from_numpy(rays))   # (N_rays, 3)
        self.register_buffer('dt_per_ray', torch.from_numpy(dt_np))  # (N_rays,)

        N_rays = len(rays)
        print(
            f"[RayAlignedLoss v4] {N_rays} ray directions, "
            f"S_max={S_max}, max_dist={ray_max_dist}m, "
            f"voxel_size={voxel_size}m, mode={ray_loss_mode}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Epoch-based warmup helpers
    # ─────────────────────────────────────────────────────────────────────────

    def set_epoch(self, epoch: int) -> None:
        """Update the current training epoch.

        Must be called at the start of every training epoch, e.g. via
        RayLossEpochHook.  If never called the loss is active from step 0
        (equivalent to ray_loss_start_epoch=0, ray_loss_warmup_epochs=0).
        """
        self._current_epoch = int(epoch)

    def _epoch_scale(self) -> float:
        """Return a [0.0, 1.0] multiplier based on the current training epoch.

        Timeline:
            epoch < ray_loss_start_epoch             → 0.0  (loss disabled)
            ray_loss_start_epoch ≤ epoch
              < start + warmup_epochs                → linear ramp 0 → 1
            epoch ≥ start + warmup_epochs            → 1.0  (full weight)

        If ray_loss_warmup_epochs == 0, activation is a hard step at
        ray_loss_start_epoch (0 before, 1 at and after).
        """
        if self._current_epoch < self.ray_loss_start_epoch:
            return 0.0
        elapsed = self._current_epoch - self.ray_loss_start_epoch
        if self.ray_loss_warmup_epochs <= 0:
            return 1.0
        return min(1.0, elapsed / self.ray_loss_warmup_epochs)

    # ─────────────────────────────────────────────────────────────────────────
    # Loss helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_termination_prob(
        p_occ: torch.Tensor,  # (B, N, S)  — must be pre-masked with vmask
        eps: float,
    ) -> torch.Tensor:        # (B, N, S)
        """Absorptive ray model: q[s] = p_occ[s] * prod_{j<s}(1 − p_occ[j]).

        Used only in transmittance_nll mode (legacy V3).
        """
        p_safe  = p_occ.clamp(min=eps, max=1.0 - eps)
        pf      = 1.0 - p_safe
        ones    = torch.ones(*pf.shape[:-1], 1, dtype=pf.dtype, device=pf.device)
        shifted = torch.cat([ones, pf[..., :-1]], dim=-1)   # (B, N, S)
        T       = torch.cumprod(shifted, dim=-1)             # (B, N, S)
        return p_safe * T

    def _zero_result(
        self,
        ref: torch.Tensor,
        num_valid: int = 0,
        total: int = 1,
    ) -> dict:
        z = ref.new_tensor(0.0)
        result = {
            "loss_ray":           z.clone(),
            "ray_pre_free":       z.clone(),
            "ray_hit_occ":        z.clone(),
            "ray_hit_sem":        z.clone(),
            "num_valid_rays":     ref.new_tensor(float(num_valid)),
            "num_pre_hit_voxels": z.clone(),
            "num_hit_voxels":     z.clone(),
            "valid_ray_ratio":    ref.new_tensor(float(num_valid) / max(total, 1)),
            "avg_voxels_per_ray": z.clone(),
        }
        if self.ray_loss_debug:
            # Sentinel values that are visually distinct in logs:
            #   p_occ stats = 0.0, depth = 0.0, class = 0.0,
            #   no_hit_frac = 1.0 (all rays have "no hit" when result is zero)
            result.update({
                "dbg_p_occ_pre":   z.clone(),
                "dbg_p_occ_hit":   z.clone(),
                "dbg_no_hit_frac": ref.new_tensor(1.0),
                "dbg_hit_depth":   z.clone(),
                "dbg_hit_class":   z.clone(),
            })
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Mode-specific loss computations
    # ─────────────────────────────────────────────────────────────────────────

    def _ordered_bce_loss(
        self,
        occ_logits: torch.Tensor,       # (B, X, Y, Z, C)  — for new_tensor ref
        logits_along: torch.Tensor,     # (B, N, S, C)
        softmax_probs: torch.Tensor,    # (B, N, S, C)
        p_occ: torch.Tensor,            # (B, N, S)    raw, not vmask-multiplied
        vmask: torch.Tensor,            # (B, N, S)    bool
        gt_along: torch.Tensor,         # (B, N, S)    GT semantic class per step
        first_hit: torch.Tensor,        # (B, N)       GT first-hit step index
        ray_valid: torch.Tensor,        # (B, N)       bool
        num_valid: int,
        total_possible: int,
        avg_vox_per_ray: float,
        device: torch.device,
        N_used: int,
        C: int,
        epoch_scale: float = 1.0,
    ) -> dict:
        """Depth-ordered BCE + first-hit CE.

        Tensor layout
        -------------
        B  : batch size
        N  : num sampled rays  (N_used)
        S  : steps per ray     (S_max)
        C  : num semantic classes (18)

        Normalization (normalize_per_sample=True, default)
        ---------------------------------------------------
        Each sample contributes equally to every loss term regardless of how
        many valid rays or pre-hit voxels it contributes.  This prevents
        samples with dense occupancy from dominating the gradient and makes
        the loss scale stable across batches.

        Global normalization (normalize_per_sample=False)
        -------------------------------------------------
        Original V4 behaviour: mean over all valid elements globally.
        """
        B = ray_valid.shape[0]

        # ── Step index tensor for positional comparison ────────────────────
        # steps[s] = s,  first_hit[b,n] = k.
        # Broadcast: steps (1,1,S) vs first_hit (B,N,1) → (B,N,S).
        steps = torch.arange(self.S_max, device=device)[None, None, :]  # (1, 1, S)

        # ── Pre-hit free-space mask ────────────────────────────────────────
        # Strict s < k: first-hit step is NOT included.
        # ray_valid guard: skip the entire ray if no confirmed first-hit.
        pre_hit_mask = (
            vmask
            & (steps < first_hit.unsqueeze(-1))   # s < k  (strict <)
            & ray_valid.unsqueeze(-1)              # only rays with a valid hit
        )  # (B, N, S) bool

        # ── Pre-hit BCE loss ───────────────────────────────────────────────
        # BCE(p_occ, target=0) = -log(1 - p_occ) = -log(softmax[free_class_id])
        # Computing via softmax directly is more numerically stable.
        log_free         = torch.log(
            softmax_probs[..., self.free_class_id].clamp(min=self.eps)
        )                                          # (B, N, S)
        bce_pre_per_step = -log_free               # (B, N, S)

        # ── First-hit BCE & CE shared tensors ─────────────────────────────
        ray_valid_f = ray_valid.float()   # (B, N)

        p_at_hit = p_occ.gather(
            2, first_hit.unsqueeze(-1)
        ).squeeze(-1)                     # (B, N)
        bce_hit_per_ray = -torch.log(p_at_hit.clamp(min=self.eps))  # (B, N)

        # Gather logits and GT class at the first-hit step
        first_hit_exp = (
            first_hit.unsqueeze(-1)               # (B, N, 1)
                      .unsqueeze(-1)               # (B, N, 1, 1)
                      .expand(-1, -1, 1, C)        # (B, N, 1, C)
        )
        logits_at_hit = logits_along.gather(2, first_hit_exp).squeeze(2)  # (B, N, C)
        gt_sem_at_hit = gt_along.gather(
            2, first_hit.unsqueeze(-1)
        ).squeeze(-1)                             # (B, N)

        # ── Reduction: per-sample or global ───────────────────────────────
        if self.normalize_per_sample:
            # ── Per-sample path ────────────────────────────────────────────
            # Each sample contributes one value (its own mean) to the final
            # batch mean, regardless of how many voxels/rays it contains.
            # Samples with zero valid elements are excluded from the average.

            # Pre-hit BCE — per-sample
            n_pre_per_s      = pre_hit_mask.float().sum(dim=(1, 2))          # (B,)
            bce_pre_per_s    = (bce_pre_per_step * pre_hit_mask.float()).sum(dim=(1, 2))  # (B,)
            valid_pre_s      = (n_pre_per_s > 0).float()                     # (B,)
            n_valid_pre_s    = valid_pre_s.sum().clamp(min=1.0)
            loss_bce_pre     = (
                (bce_pre_per_s / n_pre_per_s.clamp(min=1.0)) * valid_pre_s
            ).sum() / n_valid_pre_s

            # First-hit BCE — per-sample
            n_hit_per_s      = ray_valid_f.sum(dim=1)                        # (B,)
            bce_hit_per_s    = (bce_hit_per_ray * ray_valid_f).sum(dim=1)    # (B,)
            valid_hit_s      = (n_hit_per_s > 0).float()                     # (B,)
            n_valid_hit_s    = valid_hit_s.sum().clamp(min=1.0)
            loss_bce_hit     = (
                (bce_hit_per_s / n_hit_per_s.clamp(min=1.0)) * valid_hit_s
            ).sum() / n_valid_hit_s

            # First-hit CE — vectorized per-sample
            # Compute CE for ALL B*N rays; invalid rays are masked to 0 below.
            # gt_sem_at_hit for invalid rays may be any class in [0,17] — this
            # is safe because the result is zeroed out by ray_valid_f.
            loss_ce_per_ray_all = F.cross_entropy(
                logits_at_hit.reshape(-1, C),         # (B*N, C)
                gt_sem_at_hit.reshape(-1).long(),      # (B*N,)
                reduction='none',
            ).reshape(B, N_used)                       # (B, N)
            ce_hit_per_s = (loss_ce_per_ray_all * ray_valid_f).sum(dim=1)    # (B,)
            loss_ce_hit  = (
                (ce_hit_per_s / n_hit_per_s.clamp(min=1.0)) * valid_hit_s
            ).sum() / n_valid_hit_s

            n_pre = n_pre_per_s.sum()  # total pre-hit voxels for logging

        else:
            # ── Global path (original behaviour) ──────────────────────────
            n_pre        = pre_hit_mask.float().sum()                         # scalar
            loss_bce_pre = (
                (bce_pre_per_step * pre_hit_mask.float()).sum()
                / n_pre.clamp(min=1.0)
            )

            loss_bce_hit = (
                (bce_hit_per_ray * ray_valid_f).sum()
                / ray_valid_f.sum().clamp(min=1.0)
            )

            valid_flat  = ray_valid.reshape(-1)                               # (B*N,)
            logits_flat = logits_at_hit.reshape(-1, C)[valid_flat]            # (num_valid, C)
            labels_flat = gt_sem_at_hit.reshape(-1)[valid_flat].long()        # (num_valid,)
            loss_ce_hit = F.cross_entropy(logits_flat, labels_flat, reduction='mean')

        # ── Combine and apply epoch warmup scale ──────────────────────────
        loss_ray_unscaled = (
            self.w_bce_pre * loss_bce_pre
            + self.w_bce_hit * loss_bce_hit
            + self.w_ce_hit  * loss_ce_hit
        ) * self.loss_weight
        # epoch_scale in [0.0, 1.0]: ramps from 0 to 1 during warmup.
        # The component values in the return dict are NOT scaled — they reflect
        # actual loss magnitudes regardless of where we are in the warmup.
        loss_ray = loss_ray_unscaled * epoch_scale

        if not torch.isfinite(loss_ray):
            return self._zero_result(occ_logits, num_valid, total_possible)

        # ── Build result dict ──────────────────────────────────────────────
        result = {
            # Only 'loss_ray' starts with 'loss_': mmengine sums it into total
            "loss_ray":           loss_ray,
            # Component values: logged but NOT summed into total training loss
            "ray_pre_free":       loss_bce_pre.detach(),
            "ray_hit_occ":        loss_bce_hit.detach(),
            "ray_hit_sem":        loss_ce_hit.detach(),
            # Counts (always logged)
            "num_valid_rays":     occ_logits.new_tensor(float(num_valid)),
            "num_pre_hit_voxels": occ_logits.new_tensor(float(int(n_pre.item()))),
            "num_hit_voxels":     occ_logits.new_tensor(float(num_valid)),
            "valid_ray_ratio":    occ_logits.new_tensor(
                float(num_valid) / total_possible
            ),
            "avg_voxels_per_ray": occ_logits.new_tensor(avg_vox_per_ray),
        }

        # ── Optional debug metrics (ray_loss_debug=True) ───────────────────
        # All operations below reuse already-computed tensors; no new gathers.
        if self.ray_loss_debug:
            n_pre_f      = n_pre.clamp(min=1.0)
            n_hit_f      = ray_valid_f.sum().clamp(min=1.0)

            # Mean occupied probability over valid pre-hit voxels.
            # Healthy trend: decreases toward 0 as training progresses.
            # Warning: if stable > 0.5, pre-hit BCE is not suppressing p_occ.
            dbg_p_occ_pre = (
                (p_occ * pre_hit_mask.float()).sum() / n_pre_f
            )

            # Mean occupied probability at GT first-hit voxels (valid rays).
            # Healthy trend: increases toward 1 as training progresses.
            # Warning: if < 0.3, model is collapsing pre-hit and hit alike.
            dbg_p_occ_hit = (
                (p_at_hit * ray_valid_f).sum() / n_hit_f
            )

            # Fraction of sampled rays with NO valid visible GT first hit.
            # = 1 - valid_ray_ratio.  Large value (> 0.8) may indicate:
            #   (a) mask_camera is overly restrictive, or
            #   (b) scene is very sparse / mostly free space.
            dbg_no_hit_frac = occ_logits.new_tensor(
                1.0 - float(num_valid) / max(total_possible, 1)
            )

            # Mean step index of first-hit voxels for valid rays.
            # Low  → objects on average close to the sensor.
            # High → objects far away; fewer pre-hit voxels per ray on average.
            dbg_hit_depth = (
                (first_hit.float() * ray_valid_f).sum() / n_hit_f
            )

            # Mean GT semantic class index at first-hit voxels.
            # Valid range: [0, 16]  (17 = free never appears for valid rays).
            # Low  (0–4)  → mostly foreground / dynamic objects at first hit.
            # Mid  (5–12) → mostly static background classes.
            # If near 17  → bug: free-space voxels are being used as first hits.
            dbg_hit_class = (
                (gt_sem_at_hit.float() * ray_valid_f).sum() / n_hit_f
            )

            result.update({
                "dbg_p_occ_pre":   dbg_p_occ_pre.detach(),
                "dbg_p_occ_hit":   dbg_p_occ_hit.detach(),
                "dbg_no_hit_frac": dbg_no_hit_frac,
                "dbg_hit_depth":   dbg_hit_depth.detach(),
                "dbg_hit_class":   dbg_hit_class.detach(),
            })

        return result

    def _transmittance_nll_loss(
        self,
        occ_logits: torch.Tensor,   # reference tensor for device/dtype
        p_occ: torch.Tensor,        # (B, N, S)  raw, not yet vmask-multiplied
        vmask: torch.Tensor,        # (B, N, S)
        first_hit: torch.Tensor,    # (B, N)
        ray_valid: torch.Tensor,    # (B, N)
        num_valid: int,
        total_possible: int,
        avg_vox_per_ray: float,
        epoch_scale: float = 1.0,
    ) -> dict:
        """Legacy V3 transmittance-NLL loss (ablation only).

        q[k] = p_occ[k] * prod_{j<k}(1 - p_occ[j])
        loss = -log(q[first_hit])

        NOTE: This formulation pushes p_occ towards 0 for all pre-first-hit
        voxels, conflicting with the semantic CE loss for occupied background
        classes.  Use for ablation comparison only.
        """
        # vmask applied here per V3 convention
        p_occ_masked = p_occ * vmask.float()
        q = self._compute_termination_prob(p_occ_masked, self.eps)  # (B, N, S)

        q_gt = q.gather(2, first_hit.unsqueeze(-1)).squeeze(-1)     # (B, N)
        nll  = -torch.log(q_gt.clamp(min=self.eps))                 # (B, N)

        ray_valid_f = ray_valid.float()
        loss_mean   = (nll * ray_valid_f).sum() / ray_valid_f.sum().clamp(min=1.0)

        if not torch.isfinite(loss_mean):
            return self._zero_result(occ_logits, num_valid, total_possible)

        z = occ_logits.new_tensor(0.0)
        return {
            "loss_ray":           loss_mean * self.loss_weight * epoch_scale,
            # Logging stubs (transmittance mode does not split into components)
            "ray_pre_free":       z.clone(),
            "ray_hit_occ":        z.clone(),
            "ray_hit_sem":        z.clone(),
            "num_valid_rays":     occ_logits.new_tensor(float(num_valid)),
            "num_pre_hit_voxels": z.clone(),
            "num_hit_voxels":     occ_logits.new_tensor(float(num_valid)),
            "valid_ray_ratio":    occ_logits.new_tensor(
                float(num_valid) / total_possible
            ),
            "avg_voxels_per_ray": occ_logits.new_tensor(avg_vox_per_ray),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Forward
    # ─────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        occ_logits: torch.Tensor,            # (B, X, Y, Z, C)
        voxel_semantics: torch.Tensor,       # (B, X, Y, Z) int [0..17]
        mask_camera: torch.Tensor = None,    # (B, X, Y, Z) uint8/bool, optional
        lidar_origins: torch.Tensor = None,  # (B, 3) lidar origin in ego frame
    ) -> dict:
        """Compute the ray-aligned auxiliary loss.

        Shared geometry (both modes)
        ----------------------------
        1. Determine per-sample lidar origins (per-sample or fallback default).
        2. Build voxel-index sequences for subsampled rays on GPU.
        3. Compute validity mask vmask (in-bounds, deduplicated, within-range).
        4. Gather logits and GT labels along each ray.
        5. Determine GT first-hit step and ray validity.
        6. Dispatch to _ordered_bce_loss or _transmittance_nll_loss.

        Returns
        -------
        dict with keys:
            loss_ray           – scalar Tensor (gradient, loss_weight applied)
            ray_pre_free       – detached pre-hit BCE value (logging only)
            ray_hit_occ        – detached first-hit BCE value (logging only)
            ray_hit_sem        – detached first-hit CE value (logging only)
            num_valid_rays     – float Tensor (logging)
            num_pre_hit_voxels – float Tensor (logging)
            num_hit_voxels     – float Tensor (logging)
            valid_ray_ratio    – float Tensor (logging)
            avg_voxels_per_ray – float Tensor (logging)
        """
        device = occ_logits.device
        B, X, Y, Z, C = occ_logits.shape

        # ── Epoch-based delayed activation ────────────────────────────────────
        # Compute the warmup scale early.  If scale == 0 (before start_epoch),
        # skip all tensor work and return zero immediately — saves GPU memory
        # for the large (B, N, S, C) ray tensors.
        epoch_scale = self._epoch_scale()
        if epoch_scale == 0.0:
            N_est = (
                self.num_rays_train
                if self.num_rays_train is not None
                else self.ray_dirs.shape[0]
            )
            return self._zero_result(occ_logits, 0, B * N_est)

        assert voxel_semantics.shape == (B, X, Y, Z), (
            f"[RayAlignedLoss] shape mismatch: "
            f"occ_logits={occ_logits.shape}, "
            f"voxel_semantics={voxel_semantics.shape}"
        )
        assert 0 <= self.free_class_id < C, (
            f"[RayAlignedLoss] free_class_id={self.free_class_id} out of range C={C}"
        )

        # ── 1. Per-sample lidar origin in ego frame ───────────────────────────
        if lidar_origins is not None:
            origins = lidar_origins.to(device=device, dtype=torch.float32)  # (B, 3)
        else:
            origins = self.default_origin.unsqueeze(0).expand(B, 3)         # (B, 3)

        pc_min     = self.pc_range_min.to(device)                       # (3,)
        origin_vox = (origins - pc_min) / self.voxel_size               # (B, 3) float

        # ── 2. Subsample rays ─────────────────────────────────────────────────
        N_rays_total = self.ray_dirs.shape[0]
        if self.num_rays_train is not None and N_rays_total > self.num_rays_train:
            ray_ids = torch.randperm(N_rays_total, device=device)[:self.num_rays_train]
        else:
            ray_ids = torch.arange(N_rays_total, device=device)
        N_used = ray_ids.numel()

        ray_dirs_used = self.ray_dirs[ray_ids]    # (N_used, 3)
        dt_used       = self.dt_per_ray[ray_ids]  # (N_used,)

        # ── 3. Build per-sample voxel sequences on GPU ────────────────────────
        # t_vals[n, s] = dt[n] * (s+1) metres along ray n at step s
        step_idx = torch.arange(
            1, self.S_max + 1, device=device, dtype=torch.float32,
        )                                                    # (S_max,)
        t_vals = dt_used[:, None] * step_idx[None, :]       # (N_used, S_max)

        # Direction in voxel-index space: Δvox/m = ray_dir / voxel_size
        dir_vox = ray_dirs_used / self.voxel_size            # (N_used, 3)

        # Relative voxel-float offset from lidar origin
        ray_pts_rel = (
            t_vals[:, :, None]          # (N_used, S_max, 1)
            * dir_vox[:, None, :]       # (N_used, 1, 3)
        )                               # (N_used, S_max, 3)

        # Absolute voxel float coordinates: origin + offset
        pts_vox = (
            origin_vox[:, None, None, :]    # (B,     1,      1,     3)
            + ray_pts_rel[None, :, :, :]    # (1,     N_used, S_max, 3)
        )                                   # (B, N_used, S_max, 3)

        # Integer voxel indices by flooring
        vox_idx = pts_vox.floor().long()    # (B, N_used, S_max, 3)
        ix, iy, iz = vox_idx[..., 0], vox_idx[..., 1], vox_idx[..., 2]

        # ── 4. Distance mask ──────────────────────────────────────────────────
        n_steps_f = (
            torch.tensor(self.ray_max_dist, dtype=torch.float32, device=device)
            / dt_used
        )  # (N_used,) max valid step count per ray
        s_arange  = torch.arange(self.S_max, device=device)[None, :]  # (1, S_max)
        in_dist   = s_arange.float() < n_steps_f[:, None]             # (N_used, S_max)
        in_dist_b = in_dist.unsqueeze(0).expand(B, N_used, self.S_max)  # (B, N, S)

        # ── 5. In-bounds mask ─────────────────────────────────────────────────
        in_bounds = (
            (ix >= 0) & (ix < X)
            & (iy >= 0) & (iy < Y)
            & (iz >= 0) & (iz < Z)
        )  # (B, N_used, S_max)

        # ── 6. DDA-style deduplication ────────────────────────────────────────
        # Mark step s as "new" when at least one voxel coordinate changed vs s-1.
        # Step 0 is always new (sentinel = vox[0] - 1 forces a difference).
        prev_vox = torch.cat([
            vox_idx[:, :, :1, :] - 1,     # (B, N, 1,       3) always-new sentinel
            vox_idx[:, :, :-1, :],         # (B, N, S_max-1, 3) previous step
        ], dim=2)                          # (B, N, S_max,   3)
        is_new_voxel = (vox_idx != prev_vox).any(dim=-1)   # (B, N, S) bool

        # Combined validity: in-bounds, new voxel, within max distance
        vmask = in_bounds & is_new_voxel & in_dist_b       # (B, N, S)

        # ── 7. Safe indices for gather (clamp OOB indices to avoid CUDA errors) ─
        ix_safe = ix.clamp(0, X - 1)
        iy_safe = iy.clamp(0, Y - 1)
        iz_safe = iz.clamp(0, Z - 1)

        # b_idx[b, n, s] = b  →  used for correct batched advanced indexing
        b_idx = (
            torch.arange(B, device=device)[:, None, None]
            .expand(B, N_used, self.S_max)
        )  # (B, N_used, S_max)

        # ── 8. Gather logits and GT along rays ────────────────────────────────
        # logits_along[b, n, s, :] = occ_logits[b, ix[b,n,s], iy[b,n,s], iz[b,n,s], :]
        logits_along = occ_logits[b_idx, ix_safe, iy_safe, iz_safe, :]
        # shape: (B, N_used, S_max, C)

        # ── 9. Binary occupancy probability ──────────────────────────────────
        # p_occ[s] = 1 - P(free | voxel_s)
        # p_occ = 1 means certain occupied; p_occ = 0 means certain free.
        # Class dim is the LAST dim (verified: occ_logits shape = (B,X,Y,Z,C)).
        softmax_probs = F.softmax(logits_along, dim=-1)             # (B, N, S, C)
        p_occ         = 1.0 - softmax_probs[..., self.free_class_id]  # (B, N, S)
        # NOTE: p_occ is NOT multiplied by vmask here.
        # ordered_bce mode uses explicit per-term masks.
        # transmittance_nll mode applies vmask inside _transmittance_nll_loss.

        # ── 10. GT occupancy flag and first-hit step ──────────────────────────
        gt_sem   = voxel_semantics.long().to(device)
        gt_along = gt_sem[b_idx, ix_safe, iy_safe, iz_safe]        # (B, N, S)
        # A voxel is GT-occupied iff its semantic class ≠ free_class_id
        # and the step is valid (in-bounds, new, within range).
        gt_occ   = (gt_along != self.free_class_id) & vmask        # (B, N, S) bool
        has_hit  = gt_occ.any(dim=-1)                              # (B, N) bool
        # argmax on bool tensor: returns index of first True; returns 0 if none
        # (has_hit guards against the 0-case below)
        first_hit = gt_occ.long().argmax(dim=-1)                   # (B, N)

        # ── 11. Ray validity: has GT first-hit AND first-hit is camera-visible ─
        ray_valid = has_hit  # (B, N) bool

        if self.use_mask and mask_camera is not None:
            mc       = mask_camera.to(device).bool()               # (B, X, Y, Z)
            mc_along = mc[b_idx, ix_safe, iy_safe, iz_safe]        # (B, N, S)
            # Check camera visibility specifically at the first-hit voxel
            cam_at_hit = mc_along.gather(
                2, first_hit.unsqueeze(-1)
            ).squeeze(-1).bool()                                   # (B, N)
            ray_valid = ray_valid & cam_at_hit

        num_valid      = int(ray_valid.sum().item())
        total_possible = B * N_used

        # Debug stat: average deduplicated voxel count per ray (all rays)
        avg_vox_per_ray = vmask.float().sum(dim=-1).mean().item()

        if num_valid == 0:
            return self._zero_result(occ_logits, num_valid, total_possible)

        # ── 12. Dispatch to loss mode ─────────────────────────────────────────
        if self.ray_loss_mode == "ordered_bce":
            return self._ordered_bce_loss(
                occ_logits=occ_logits,
                logits_along=logits_along,
                softmax_probs=softmax_probs,
                p_occ=p_occ,
                vmask=vmask,
                gt_along=gt_along,
                first_hit=first_hit,
                ray_valid=ray_valid,
                num_valid=num_valid,
                total_possible=total_possible,
                avg_vox_per_ray=avg_vox_per_ray,
                device=device,
                N_used=N_used,
                C=C,
                epoch_scale=epoch_scale,
            )
        else:  # "transmittance_nll"
            return self._transmittance_nll_loss(
                occ_logits=occ_logits,
                p_occ=p_occ,
                vmask=vmask,
                first_hit=first_hit,
                ray_valid=ray_valid,
                num_valid=num_valid,
                total_possible=total_possible,
                avg_vox_per_ray=avg_vox_per_ray,
                epoch_scale=epoch_scale,
            )


# ──────────────────────────────────────────────────────────────────────────────
# Runner hook — propagates current epoch to RayAlignedLoss
# ──────────────────────────────────────────────────────────────────────────────

try:
    from mmengine.hooks import Hook
    from mmengine.registry import HOOKS as _HOOKS

    @_HOOKS.register_module()
    class RayLossEpochHook(Hook):
        """Calls set_epoch() on BEVFormerOccHead.ray_aux_loss at epoch start.

        Add to your config:
            custom_hooks = [dict(type='RayLossEpochHook', priority='NORMAL')]

        This hook navigates: runner.model → (DDP unwrap) → pts_bbox_head
        → set_epoch(runner.epoch).  Works with both single-GPU and DDP.

        The hook is registered as soon as ray_aligned_loss.py is imported
        (which happens automatically when the model config is built), so no
        additional import is required.
        """

        priority = 'NORMAL'

        def before_train_epoch(self, runner) -> None:
            model = runner.model
            # Unwrap DDP / DataParallel if present
            if hasattr(model, 'module'):
                model = model.module
            head = getattr(model, 'pts_bbox_head', None)
            if head is not None and hasattr(head, 'set_epoch'):
                head.set_epoch(runner.epoch)

except ImportError:
    # mmengine not available — hook silently skipped.
    # set_epoch() can still be called manually if needed.
    pass
