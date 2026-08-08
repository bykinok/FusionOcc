# Calibration axis, step 1/3: run inference on the held-out calibration
# split (see projects/GaussianFormer/tools/split_val_calib_eval.py) and
# export per-voxel logits with tools/export_occ_logits.py
# (--cfg-options model.export_occ_logits=True), then fit a temperature with
# tools/train_temperature.py. See _calib_eval_before.py / _calib_eval.py
# for the two before/after eval configs on the disjoint eval split.
_base_ = ['./nuscenes_gs25600_occ3d_ori_setting.py']

val_dataloader = dict(dataset=dict(
    imageset='projects/GaussianFormer/data/nuscenes_cam/nuscenes_infos_val_sweeps_occ_calib.pkl'))
test_dataloader = val_dataloader
