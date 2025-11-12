# 인덱스 1270부터 시작하는 subset config
_base_ = ['./stcocc_r50_704x256_16f_occ3d_36e.py']

# Subset annotation 파일 사용
test_dataloader = dict(
    dataset=dict(
        ann_file='data/nuscenes/stcocc-nuscenes_infos_val_subset_1270.pkl',
    )
)

val_dataloader = test_dataloader

val_evaluator = dict(
    type='OccupancyMetric',
    ann_file='data/nuscenes/stcocc-nuscenes_infos_val_subset_1270.pkl',
    data_root='',
    dataset_name='occ3d',
    num_classes=18,
    use_image_mask=True)

test_evaluator = val_evaluator

