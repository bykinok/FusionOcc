# Copyright (c) OpenMMLab. All rights reserved.

# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
metainfo = dict(classes=class_names)
dataset_type = 'NuscOCCDataset'
data_root = 'data/nuscenes/'

# Input modality configuration
input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

# Occupancy configuration
occ_path = "./data/nuScenes-Occupancy"
occ_size = [512, 512, 40]
empty_idx = 0  # noise 0-->255
num_cls = 17  # 0 free, 1-16 obj
visible_mask = False

file_client_args = dict(backend='disk')
