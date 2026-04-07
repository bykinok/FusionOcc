"""SparseOcc_ori r50 NuScenes 704x256 8frames Panoptic 변형 설정.

원본: Ref/SparseOcc_ori/configs/r50_nuimg_704x256_8f_pano.py
"""
_base_ = ['./r50_nuimg_704x256_8f.py']

occ_gt_root = 'data/nuscenes/occ3d_panoptic'

model = dict(
    pts_bbox_head=dict(
        panoptic=True,
    ),
)

train_dataloader = dict(
    dataset=dict(
        occ_gt_root=occ_gt_root,
    ),
)

val_dataloader = dict(
    dataset=dict(
        occ_gt_root=occ_gt_root,
    ),
)

test_dataloader = val_dataloader

work_dir = 'work_dirs/sparseocc_eccv_r50_256x704_8f_pano'
