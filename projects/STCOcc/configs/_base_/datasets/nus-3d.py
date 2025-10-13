# dataset settings
dataset_type = 'NuScenesDatasetOccpancy'
data_root = 'data/nuscenes/'

backend_args = None

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

# For nuScenes occupancy prediction we use different class names
class_names = ['car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle', 'motorcycle', 'pedestrian',
               'traffic_cone', 'barrier', 'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
               'vegetation', 'free']
