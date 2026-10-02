_base_ = ['./stcocc_r50_704x256_16f_occ3d_e12_stage3_adaptive_r20_n025_f010.py']

# reweight_lovasz=True (2026-10-02, research_v2 follow-up) -- see l000_rwlovasz sibling
# config for the full rationale. Only this one field differs from the base config
# (lambda_inv_free=dict(mode='radius_piecewise', boundary_m=20, near=0.25, far=0.10), the
# current best non-dominated E3 point), so any mIoU/RayIoU change vs. the existing EMA
# result (38.75/0.3930) is attributable to this change alone.
model = dict(reweight_lovasz=True)
