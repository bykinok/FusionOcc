# Calibration axis, step 2/3: pre-calibration baseline (temperature=1.0,
# i.e. a no-op) evaluated on the held-out eval split (disjoint from the
# calib split _calib_train.py runs on).
_base_ = ['./nuscenes_gs25600_occ3d_ori_setting.py']

val_dataloader = dict(dataset=dict(
    imageset='projects/GaussianFormer/data/nuscenes_cam/nuscenes_infos_val_sweeps_occ_eval.pkl'))
test_dataloader = val_dataloader

# Point at the eval-split flat companion pkl (not the standard val one),
# matching the imageset override above.
val_evaluator = dict(
    ann_file='projects/GaussianFormer/data/nuscenes_cam/nuscenes_infos_val_sweeps_occ_eval_flat.pkl')
test_evaluator = val_evaluator

model = dict(temperature=1.0)
