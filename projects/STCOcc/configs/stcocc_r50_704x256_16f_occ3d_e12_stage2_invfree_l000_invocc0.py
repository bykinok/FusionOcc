_base_ = ['./stcocc_r50_704x256_16f_occ3d_e12_stage2_invfree_l000.py']

# lambda_inv_occupied=0.0 (2026-10, research_v2 decomposition ablation): isolates the
# effect of invisible-OCCUPIED voxel supervision on the w/mask-vs-lambda=0 mIoU gap.
# l000 (lambda_inv_free=0.0) already excludes invisible-free from CE/sem_scal/geo_scal;
# this ALSO excludes invisible-occupied from the same 3 losses (full weight=0, same
# mechanism as lambda_inv_free -- see STCOcc.build_inv_free_voxel_weight). Lovasz is
# left untouched (reweight_lovasz stays False), so this isolates the invisible-occupied
# effect alone, holding the Lovasz-term confound fixed at l000's existing behavior --
# compare against l000 (39.41/0.3455 EMA) to read off invisible-occupied's own effect.
model = dict(lambda_inv_occupied=0.0)
