_base_ = ['./stcocc_r50_704x256_16f_occ3d_e12_stage2_invfree_l025.py']

# reweight_lovasz=True (2026-10-02, research_v2 follow-up) -- see l000_rwlovasz sibling
# config for the full rationale. Only this one field differs from the base l025 config
# (lambda_inv_free=0.25, the flat-sweep optimum), so any mIoU/RayIoU change vs. the existing
# l025 EMA result (37.66/0.3928) is attributable to this change alone.
model = dict(reweight_lovasz=True)
