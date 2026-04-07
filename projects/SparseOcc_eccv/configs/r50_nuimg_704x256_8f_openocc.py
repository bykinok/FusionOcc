"""SparseOcc_ori r50 NuScenes 704x256 8frames OpenOcc 변형 설정.

원본: Ref/SparseOcc_ori/configs/r50_nuimg_704x256_8f_openocc.py
"""
_base_ = ['./r50_nuimg_704x256_8f.py']

occ_gt_root = 'data/nuscenes/openocc_v2'

occ_class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier',
    'driveable_surface', 'other_flat', 'sidewalk', 'terrain',
    'manmade', 'vegetation', 'free'
]

model = dict(
    pts_bbox_head=dict(
        class_names=occ_class_names,
        transformer=dict(
            num_classes=len(occ_class_names),
        ),
        loss_cfgs=dict(
            loss_mask2former=dict(
                num_classes=len(occ_class_names),
            ),
        ),
    ),
)

train_dataloader = dict(
    dataset=dict(
        occ_gt_root=occ_gt_root,
        classes=occ_class_names,
    ),
)

val_dataloader = dict(
    dataset=dict(
        occ_gt_root=occ_gt_root,
        classes=occ_class_names,
    ),
)

test_dataloader = val_dataloader

work_dir = 'work_dirs/sparseocc_eccv_r50_256x704_8f_openocc'
