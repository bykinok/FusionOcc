_base_ = ['./stcocc_r50_704x256_16f_occ3d_e12_stage2_invfree_l000.py']

# reweight_lovasz=True (2026-10-02, research_v2 follow-up to the raw/EMA checkpoint
# investigation): extends lambda_inv_free to the Lovasz term via stochastic per-voxel
# exclusion (see STCOcc.build_lovasz_target docstring). Everything else (lambda_inv_free=0.0,
# 12-epoch e12_stage recipe, data, optimizer/schedule) is identical to the base config this
# inherits from -- only this one field differs, so any mIoU/RayIoU change vs. the existing
# l000 EMA result (39.41/0.3455) is attributable to this change alone.
model = dict(reweight_lovasz=True)
