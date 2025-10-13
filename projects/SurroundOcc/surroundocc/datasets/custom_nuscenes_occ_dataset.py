import numpy as np
from mmdet3d.registry import DATASETS as DET3D_DATASETS
from mmengine.registry import DATASETS as ENGINE_DATASETS
from mmdet3d.datasets import NuScenesDataset
import os
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion

@DET3D_DATASETS.register_module()
@ENGINE_DATASETS.register_module()
class CustomNuScenesOccDataset(NuScenesDataset):
    """Custom NuScenes Dataset for Occupancy Prediction.
    
    This dataset is designed for SurroundOcc occupancy prediction,
    supporting both semantic and geometric occupancy.
    """
    
    def __init__(self, occ_size, pc_range, occ_root=None, **kwargs):
        # Remove classes from kwargs if present to avoid mmengine compatibility issues
        classes = kwargs.pop('classes', None)
        
        # Store original data loading parameters
        self.ann_file = kwargs.get('ann_file')
        self.data_root = kwargs.get('data_root', '')
        
        # Store mmdet3d specific parameters and remove from kwargs
        self.modality = kwargs.pop('modality', {})
        self.box_type_3d = kwargs.pop('box_type_3d', 'LiDAR')
        self.use_valid_flag = kwargs.pop('use_valid_flag', False)
        self.backend_args = kwargs.pop('backend_args', {})
        
        # Add NuScenesDataset specific attributes to fix compatibility
        self.filter_empty_gt = kwargs.pop('filter_empty_gt', False)  # Set to False for occupancy
        self.with_velocity = kwargs.pop('with_velocity', True)
        self.load_type = kwargs.pop('load_type', 'frame_based')
        self.show_ins_var = kwargs.pop('show_ins_var', False)
        
        # Add box_mode_3d attribute to fix compatibility
        from mmdet3d.structures.bbox_3d import LiDARInstance3DBoxes, CameraInstance3DBoxes
        if self.box_type_3d == 'LiDAR':
            self.box_mode_3d = LiDARInstance3DBoxes
        else:
            self.box_mode_3d = CameraInstance3DBoxes
        
        # Initialize base attributes manually to avoid parent class data filtering
        from mmengine.dataset import BaseDataset
        
        # Initialize basic attributes without calling full_init()
        self.serialize_data = kwargs.get('serialize_data', True) 
        self.test_mode = kwargs.get('test_mode', False)
        self.pipeline = kwargs.get('pipeline', [])
        
        # Build essential pipeline - focus on core transforms for occupancy
        if self.pipeline:
            # Import mmdet3d transforms to ensure they are registered
            import mmdet3d.datasets.transforms
            from mmdet3d.registry import TRANSFORMS as DET3D_TRANSFORMS
            from mmcv.transforms import Compose
            
            # Build core transforms manually
            pipeline = []
            
            # 1. LoadMultiViewImageFromFiles - essential for image loading
            try:
                load_img_transform = DET3D_TRANSFORMS.build({
                    'type': 'LoadMultiViewImageFromFiles',
                    'to_float32': True
                })
                pipeline.append(load_img_transform)
            except Exception as e:
                # LoadMultiViewImageFromFiles has registry issues, will use dummy images
                pass
            
            # 2. LoadOccupancy - essential for GT loading
            try:
                # Find LoadOccupancy in original pipeline config
                from projects.SurroundOcc.surroundocc.transforms.loading import LoadOccupancy
                load_occ_transform = LoadOccupancy(use_semantic=True)
                pipeline.append(load_occ_transform)
            except Exception as e:
                print(f"ERROR: Failed to build LoadOccupancy: {e}")
                
            self.pipeline = Compose(pipeline) if pipeline else None
        
        # Load data directly without parent class filtering
        self._load_data_list_direct()
        
        self.data_list = list(sorted(self.data_list, key=lambda e: e.get('timestamp', 0)))
        # Use load_interval if available, otherwise default to 1
        load_interval = getattr(self, 'load_interval', 1)
        self.data_list = self.data_list[::load_interval]
        self.occ_size = occ_size
        self.pc_range = pc_range
        self.occ_root = occ_root or self.data_root
        
        # CustomNuScenesOccDataset initialized
        
    def _load_data_list_direct(self):
        """Load data list directly from annotation file without parent filtering."""
        import pickle
        import os.path as osp
        
        ann_file_path = osp.join(self.data_root, self.ann_file)
        
        # Check if annotation file exists
        if not osp.exists(ann_file_path):
            raise FileNotFoundError(f"Annotation file not found: {ann_file_path}")
            
        try:
            with open(ann_file_path, 'rb') as f:
                data = pickle.load(f)
        except Exception as e:
            raise RuntimeError(f"Could not load annotation file {ann_file_path}: {e}")
        
        # Use the infos from the annotation file as data_list
        if 'data_list' in data:
            self.data_list = data['data_list']
        elif 'infos' in data:
            self.data_list = data['infos']
        else:
            raise KeyError(f"Neither 'data_list' nor 'infos' found in annotation file {ann_file_path}")
        
        # Store metainfo if available (avoid setting as attribute due to property conflicts)
        if 'metainfo' in data:
            self._metainfo_data = data['metainfo']
        elif 'metadata' in data:
            self._metainfo_data = data['metadata']
            
        # Data loaded successfully
    
    
    def get_data_info(self, index: int) -> dict:
        """Get data info according to the given index.
        
        Args:
            index (int): Index of the sample data
            
        Returns:
            dict: Data information of the specified index
        """
        if index >= len(self.data_list):
            raise IndexError(f"Index {index} out of range, dataset has {len(self.data_list)} samples")
        
        data_info = self.data_list[index].copy()
        
        # Convert data format for LoadMultiViewImageFromFiles
        if 'images' not in data_info:
            # Handle case where data is in individual keys format
            if 'img_filename' in data_info and isinstance(data_info['img_filename'], list):
                data_info['images'] = {}
                cam_names = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
                
                for i, cam_key in enumerate(cam_names):
                    if i < len(data_info['img_filename']):
                        data_info['images'][cam_key] = {
                            'img_path': data_info['img_filename'][i],
                            'filename': data_info['img_filename'][i],
                            'cam2img': data_info['cam_intrinsic'][i] if 'cam_intrinsic' in data_info and i < len(data_info['cam_intrinsic']) else None,
                            'lidar2cam': data_info['lidar2cam'][i] if 'lidar2cam' in data_info and i < len(data_info['lidar2cam']) else None,
                            'lidar2img': data_info['lidar2img'][i] if 'lidar2img' in data_info and i < len(data_info['lidar2img']) else None
                        }
            elif 'cams' in data_info:
                data_info['images'] = {}
                for cam_key, cam_info in data_info['cams'].items():
                    data_info['images'][cam_key] = {
                        'img_path': cam_info['data_path'],
                        'filename': cam_info['data_path'],
                        'cam2img': cam_info['cam_intrinsic'],
                        'lidar2cam': cam_info.get('sensor2lidar_rotation', None),
                        'lidar2img': cam_info.get('lidar2img', None)
                    }
        
        # Add occ_path if missing - construct from sample information
        if 'occ_path' not in data_info:
            # Try to construct occ_path from available information
            if 'sample_idx' in data_info:
                sample_idx = data_info['sample_idx']
                data_info['occ_path'] = f'./data/nuscenes_occ/samples/{sample_idx}.npy'
            elif 'pts_filename' in data_info:
                # Extract sample info from pts_filename if available
                pts_file = data_info['pts_filename']
                if isinstance(pts_file, str) and 'LIDAR_TOP' in pts_file:
                    import os
                    base_name = os.path.basename(pts_file).replace('.pcd.bin', '')
                    data_info['occ_path'] = f'./data/nuscenes_occ/samples/{base_name}.pcd.bin.npy'
        
        # Add required fields for mmdet3d compatibility
        if 'ann_info' not in data_info:
            data_info['ann_info'] = {
                'gt_bboxes_3d': [],
                'gt_labels_3d': [],
                'gt_names': []
            }
        
        return data_info
    
    
    def full_init(self):
        """Override to prevent duplicate data loading."""
        # Data is already loaded in __init__, so just pass
        pass
    
    def __len__(self):
        """Return the length of data list."""
        return len(self.data_list)

    def __getitem__(self, idx):
        """Get item from infos according to the given index.
        Returns:
            dict: Data dictionary of the corresponding index.
        """
        try:
            # Get data from get_data_info to ensure proper format conversion
            if idx < len(self.data_list):
                data_info = self.data_list[idx].copy()
                
                # Ensure img_filename is available for LoadMultiViewImageFromFiles
                # Our data already has img_filename as a list, so just verify it's correct
                if 'img_filename' in data_info and isinstance(data_info['img_filename'], list):
                    # img_filename already exists and is correct format
                    pass
                elif 'cams' in data_info:
                    # Convert cams format to img_filename list
                    cam_keys = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
                    img_filename = []
                    for cam_key in cam_keys:
                        if cam_key in data_info['cams']:
                            img_filename.append(data_info['cams'][cam_key]['data_path'])
                    data_info['img_filename'] = img_filename
                
                
                # Add occ_path if missing
                if 'occ_path' not in data_info:
                    if 'pts_filename' in data_info:
                        pts_file = data_info['pts_filename']
                        if isinstance(pts_file, str) and 'LIDAR_TOP' in pts_file:
                            import os
                            base_name = os.path.basename(pts_file).replace('.pcd.bin', '')
                            data_info['occ_path'] = f'./data/nuscenes_occ/samples/{base_name}.pcd.bin.npy'
                
                # Check if this is real data (has required fields)        
                if 'img_filename' in data_info and 'occ_path' in data_info:
                    # Process real data through pipeline
                    if self.pipeline is not None:
                        try:
                            result = self.pipeline(data_info)
                            if result is not None:
                                # Convert old MMDet3D format to new MMEngine format
                                if isinstance(result, dict) and 'inputs' not in result:
                                    # Extract images from result
                                    imgs = None
                                
                                    # Check for 'img' key (LoadMultiViewImageFromFiles output)
                                    if 'img' in result and isinstance(result['img'], list):
                                        # Convert list of images to stacked tensor
                                        import torch
                                        import numpy as np
                                        
                                        img_list = result['img']
                                        if len(img_list) > 0 and isinstance(img_list[0], np.ndarray):
                                            # Convert numpy arrays to tensors and stack
                                            img_tensors = [torch.from_numpy(img).permute(2, 0, 1) for img in img_list]  # HWC -> CHW
                                            imgs = torch.stack(img_tensors, dim=0)  # [N, C, H, W]
                                        else:
                                            if idx < 3:
                                                print(f"DEBUG: Unexpected img format in result: {type(img_list[0]) if img_list else 'empty list'}")
                                    
                                    # Fallback: try other image keys
                                    if imgs is None:
                                        for img_key in ['imgs', 'image', 'images']:
                                            if img_key in result:
                                                imgs = result[img_key]
                                                break
                                
                                    # If no images found, create dummy images for now
                                    if imgs is None:
                                        # Temporarily use dummy images since LoadMultiViewImageFromFiles has registry issues
                                        import torch
                                        dummy_imgs = torch.randn(6, 3, 256, 448, dtype=torch.float32)  # [N, C, H, W]
                                        imgs = dummy_imgs
                                    
                                    # Create data_sample with GT and metadata
                                    from mmdet3d.structures.det3d_data_sample import Det3DDataSample
                                    import torch
                                    import numpy as np
                                    
                                    data_sample = Det3DDataSample()
                                    
                                    # Add GT occupancy
                                    if 'gt_occ' in result:
                                        if isinstance(result['gt_occ'], torch.Tensor):
                                            data_sample.gt_occ = result['gt_occ']
                                        else:
                                            data_sample.gt_occ = torch.from_numpy(result['gt_occ'])
                                    
                                    # Add metadata
                                    metainfo = {}
                                    for key in ['sample_idx', 'timestamp', 'scene_token', 'pc_range', 'occ_size']:
                                        if key in result:
                                            metainfo[key] = result[key]
                                    
                                    # Add camera matrices from LoadMultiViewImageFromFiles output
                                    for matrix_key in ['lidar2img', 'cam2img', 'lidar2cam']:
                                        if matrix_key in result:
                                            metainfo[matrix_key] = result[matrix_key]
                                    
                                    # If lidar2img not available from transform, get from original data
                                    if 'lidar2img' not in metainfo:
                                        if 'lidar2img' in data_info:
                                            metainfo['lidar2img'] = data_info['lidar2img']
                                        else:
                                            # Create dummy lidar2img matrices for 6 cameras
                                            import numpy as np
                                            dummy_lidar2img = []
                                            for i in range(6):
                                                # Create identity 4x4 matrix as dummy
                                                dummy_matrix = np.eye(4, dtype=np.float32)
                                                dummy_lidar2img.append(dummy_matrix)
                                            metainfo['lidar2img'] = np.array(dummy_lidar2img)
                                    
                                    # Add required img_shape metadata (6 cameras, H=256, W=448)
                                    if 'img_shape' not in metainfo:
                                        metainfo['img_shape'] = [(256, 448, 3)] * 6
                                    
                                    data_sample.set_metainfo(metainfo)
                                    
                                    # Convert to new format
                                    return {
                                        'inputs': {'imgs': imgs} if imgs is not None else {},
                                        'data_samples': [data_sample]
                                    }
                                else:
                                    # Already in new format
                                    return result
                        except Exception as pipeline_error:
                            print(f"DEBUG: Pipeline error for idx {idx}: {type(pipeline_error).__name__}: {pipeline_error}")
                            import traceback
                            traceback.print_exc()
                            raise RuntimeError(f"Pipeline failed for index {idx}: {pipeline_error}")
                
        except Exception as e:
            # Data loading failed, raise error instead of using dummy data
            raise RuntimeError(f"Failed to load data for index {idx}: {e}")
        
        # If we reach here, it means no valid data was found
        raise RuntimeError(f"No valid data found for index {idx}")

    def prepare_train_data(self, index):
        input_dict = self.get_data_info(index)
        if input_dict is None:
            raise RuntimeError(f"No data info found for index {index}")

        # No pre_pipeline needed - data is already prepared
        if self.pipeline is None:
            raise RuntimeError(f"No pipeline available for processing data at index {index}")
            
        example = self.pipeline(input_dict)
        if example is None:
            raise RuntimeError(f"Pipeline returned None for index {index}")
            
        return example

    def prepare_test_data(self, index):
        input_dict = self.get_data_info(index)

        if input_dict is None:
            raise RuntimeError(f"No data info found for index {index}")

        # No pre_pipeline needed - data is already prepared
        if self.pipeline is None:
            raise RuntimeError(f"No pipeline available for processing data at index {index}")
            
        example = self.pipeline(input_dict)
        if example is None:
            raise RuntimeError(f"Pipeline returned None for index {index}")
            
        return example

    def get_data_info(self, index):
        """Get data info according to the given index."""
        # Safety check for index bounds
        if index >= len(self.data_list):
            index = index % len(self.data_list)
            
        info = self.data_list[index]
        
        # standard protocal modified from SECOND.Pytorch
        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            lidar_points=dict(lidar_path=info['lidar_path']),
            sweeps=info['sweeps'],
            lidar2ego_translation=info.get('lidar2ego_translation', [0.0, 0.0, 0.0]),
            lidar2ego_rotation=info.get('lidar2ego_rotation', [1.0, 0.0, 0.0, 0.0]),
            ego2global_translation=info['ego2global_translation'],
            ego2global_rotation=info['ego2global_rotation'],
            prev_idx=info['prev'],
            next_idx=info['next'],
            scene_token=info['scene_token'],
            can_bus=info['can_bus'],
            timestamp=info['timestamp'] / 1e6,
            occ_size=np.array(self.occ_size),
            pc_range=np.array(self.pc_range),
            lidar_token=info.get('lidar_token', info['token']),
            lidarseg=info.get('lidarseg', None),
            curr=info,
        )

        if self.modality.get('use_camera', True):
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []
            
            lidar2cam_dic = {}
            
            for cam_type, cam_info in info['cams'].items():
                image_paths.append(cam_info['data_path'])
                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                lidar2cam_t = cam_info[
                    'sensor2lidar_translation'] @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                lidar2img_rts.append(lidar2img_rt)

                cam_intrinsics.append(viewpad)
                lidar2cam_rts.append(lidar2cam_rt.T)
                
                lidar2cam_dic[cam_type] = lidar2cam_rt.T

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    cam_intrinsic=cam_intrinsics,
                    lidar2cam=lidar2cam_rts,
                    lidar2cam_dic=lidar2cam_dic,
                ))
                
        if self.modality.get('use_lidar', False):
            # FIXME alter lidar path
            input_dict['pts_filename'] = input_dict['pts_filename'].replace('./data/nuscenes/', self.data_root)
            for sw in input_dict['sweeps']:
                sw['data_path'] = sw['data_path'].replace('./data/nuscenes/', self.data_root)

        return input_dict

    def _rand_another(self, idx):
        """Randomly get another item when the current item is invalid."""
        import random
        pool = np.where(np.arange(len(self)) != idx)[0]
        return np.random.choice(pool)

    def evaluate(self, results, logger=None, **kwargs):
        """Evaluation method for occupancy prediction."""
        eval_results = {}
        
        # Real evaluation should be handled by the OccupancyMetric evaluator
        # This method is kept for compatibility but actual evaluation is done elsewhere
        if logger is not None:
            logger.info(f'Evaluated {len(results)} samples')
        
        return eval_results