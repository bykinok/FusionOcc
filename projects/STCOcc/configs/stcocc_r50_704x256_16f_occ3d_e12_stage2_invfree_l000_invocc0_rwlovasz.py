_base_ = ['./stcocc_r50_704x256_16f_occ3d_e12_stage2_invfree_l000_invocc0.py']

# reweight_lovasz=True on top of lambda_inv_occupied=0.0 (2026-10, research_v2
# decomposition ablation): now EVERY invisible voxel (free AND occupied) is
# excluded from ALL FOUR loss terms (CE/sem_scal/geo_scal via weight=0, Lovasz via
# build_lovasz_target's stochastic exclusion, deterministic at weight=0) -- the
# lambda-mechanism's closest reachable approximation to the w/mask hard-baseline's
# ignore_index=255 relabeling (which also excludes all invisible voxels from all
# four losses at the label level). Compare against w/mask (41.17/0.3351 EMA): if
# this closes most of the remaining gap, the lambda_inv_free + lambda_inv_occupied +
# reweight_lovasz combination reproduces w/mask's effective supervision; any
# residual gap is attributable to something this decomposition still misses
# (e.g. the hard relabel vs. soft/stochastic weighting mechanism itself).
model = dict(reweight_lovasz=True)
