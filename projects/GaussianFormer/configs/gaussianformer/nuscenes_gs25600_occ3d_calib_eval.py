# Calibration axis, step 3/3: post-calibration evaluation on the same held-
# out eval split as _calib_eval_before.py, with the fitted temperature.
#
# `temperature` below is a PLACEHOLDER (1.0, i.e. currently a no-op) --
# tools/train_temperature.py was not actually run against this model (that
# requires a full inference pass over the calib split via
# tools/export_occ_logits.py first, which needs a GPU allocation beyond
# this port's smoke-testing scope). Update this value after running:
#   python tools/export_occ_logits.py \
#     projects/GaussianFormer/configs/gaussianformer/nuscenes_gs25600_occ3d_calib_train.py \
#     <checkpoint> --output work_dirs/gaussianformer_occ_logits_calib.npz \
#     --cfg-options model.export_occ_logits=True
#   python tools/train_temperature.py work_dirs/gaussianformer_occ_logits_calib.npz
_base_ = ['./nuscenes_gs25600_occ3d_ori_setting.py']

val_dataloader = dict(dataset=dict(
    imageset='projects/GaussianFormer/data/nuscenes_cam/nuscenes_infos_val_sweeps_occ_eval.pkl'))
test_dataloader = val_dataloader

# Point at the eval-split flat companion pkl (not the standard val one),
# matching the imageset override above.
val_evaluator = dict(
    ann_file='projects/GaussianFormer/data/nuscenes_cam/nuscenes_infos_val_sweeps_occ_eval_flat.pkl')
test_evaluator = val_evaluator

model = dict(temperature=1.0)  # TODO: replace with the fitted value
