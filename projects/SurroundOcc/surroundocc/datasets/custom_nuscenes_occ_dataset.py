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
    
    def __init__(self, occ_size, pc_range, occ_root=None, use_ego_frame=None, **kwargs):
        # Remove classes from kwargs if present to avoid mmengine compatibility issues
        classes = kwargs.pop('classes', None)
        # When True, pass ego2img as lidar2img so that model output is in Ego frame (for Occ3D GT)
        self.use_ego_frame = (use_ego_frame if use_ego_frame is not None
                              else kwargs.pop('use_ego_frame', False))

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
        pipeline_cfg = kwargs.get('pipeline', [])
        
        # Build pipeline from config, skipping legacy transforms
        if pipeline_cfg:
            from mmcv.transforms import Compose
            from mmengine.registry import TRANSFORMS as ENGINE_TRANSFORMS
            from mmdet3d.registry import TRANSFORMS as DET3D_TRANSFORMS
            
            # Import transforms to ensure they are registered
            import mmdet3d.datasets.transforms
            from projects.SurroundOcc.surroundocc import transforms as custom_transforms
            
            # Legacy transforms that should be skipped in mmengine environment
            SKIP_TRANSFORMS = ['DefaultFormatBundle3D', 'Collect3D']
            
            # Build pipeline from config
            transforms = []
            for transform_cfg in pipeline_cfg:
                if isinstance(transform_cfg, dict):
                    transform_type = transform_cfg.get('type', '')
                    
                    # Skip legacy transforms
                    if transform_type in SKIP_TRANSFORMS:
                        print(f"INFO: Skipping legacy transform: {transform_type}")
                        continue
                    
                    # Try ENGINE_TRANSFORMS first, then DET3D_TRANSFORMS
                    try:
                        transform = ENGINE_TRANSFORMS.build(transform_cfg)
                        transforms.append(transform)
                    except Exception as e1:
                        try:
                            transform = DET3D_TRANSFORMS.build(transform_cfg)
                            transforms.append(transform)
                        except Exception as e2:
                            print(f"WARNING: Failed to build transform {transform_type}: {e2}")
                            continue
                else:
                    transforms.append(transform_cfg)
            
            self.pipeline = Compose(transforms)
        else:
            self.pipeline = None
        
        # Load data directly without parent class filtering
        self._load_data_list_direct()
        
        # self.data_list = list(sorted(self.data_list, key=lambda e: e.get('timestamp', 0)))
        # Use load_interval if available, otherwise default to 1
        load_interval = getattr(self, 'load_interval', 1)
        self.data_list = self.data_list[::load_interval]
        self.occ_size = occ_size
        self.pc_range = pc_range
        self.occ_root = occ_root or self.data_root
        
        # Set group flag for samplers (required by DistributedGroupSampler)
        if not self.test_mode:
            self._set_group_flag()
        
        # CustomNuScenesOccDataset initialized
    
    def _set_group_flag(self):
        """Set flag according to image aspect ratio.
        
        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0. In 3D datasets, they are all the same, thus are all zeros.
        This is required by DistributedGroupSampler for grouping samples.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        
    def _load_data_list_direct(self):
        """Load data list directly from annotation file without parent filtering."""
        import pickle
        import os.path as osp
        import sys
        
        ann_file_path = osp.join(self.data_root, self.ann_file)
        
        # Check if annotation file exists
        if not osp.exists(ann_file_path):
            raise FileNotFoundError(f"Annotation file not found: {ann_file_path}")
            
        try:
            # Add numpy compatibility patch for different numpy versions
            # pkl files created with old numpy may reference numpy._core
            # but current numpy 2.x uses numpy.core
            import numpy.core as core
            sys.modules['numpy._core'] = core
            sys.modules['numpy._core.numeric'] = core.numeric
            
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
    
    
    # def get_data_info(self, index: int) -> dict:
    #     """Get data info according to the given index.
        
    #     Args:
    #         index (int): Index of the sample data
            
    #     Returns:
    #         dict: Data information of the specified index
    #     """
    #     if index >= len(self.data_list):
    #         raise IndexError(f"Index {index} out of range, dataset has {len(self.data_list)} samples")
        
    #     data_info = self.data_list[index].copy()
        
    #     # Convert data format for LoadMultiViewImageFromFiles
    #     if 'images' not in data_info:
    #         # Handle case where data is in individual keys format
    #         if 'img_filename' in data_info and isinstance(data_info['img_filename'], list):
    #             data_info['images'] = {}
    #             cam_names = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
                
    #             for i, cam_key in enumerate(cam_names):
    #                 if i < len(data_info['img_filename']):
    #                     data_info['images'][cam_key] = {
    #                         'img_path': data_info['img_filename'][i],
    #                         'filename': data_info['img_filename'][i],
    #                         'cam2img': data_info['cam_intrinsic'][i] if 'cam_intrinsic' in data_info and i < len(data_info['cam_intrinsic']) else None,
    #                         'lidar2cam': data_info['lidar2cam'][i] if 'lidar2cam' in data_info and i < len(data_info['lidar2cam']) else None,
    #                         'lidar2img': data_info['lidar2img'][i] if 'lidar2img' in data_info and i < len(data_info['lidar2img']) else None
    #                     }
    #         elif 'cams' in data_info:
    #             data_info['images'] = {}
    #             for cam_key, cam_info in data_info['cams'].items():
    #                 data_info['images'][cam_key] = {
    #                     'img_path': cam_info['data_path'],
    #                     'filename': cam_info['data_path'],
    #                     'cam2img': cam_info['cam_intrinsic'],
    #                     'lidar2cam': cam_info.get('sensor2lidar_rotation', None),
    #                     'lidar2img': cam_info.get('lidar2img', None)
    #                 }
        
    #     # Add occ_path if missing - construct from sample information
    #     if 'occ_path' not in data_info:
    #         # Try to construct occ_path from available information
    #         if 'sample_idx' in data_info:
    #             sample_idx = data_info['sample_idx']
    #             data_info['occ_path'] = f'./data/nuscenes_occ/samples/{sample_idx}.npy'
    #         elif 'pts_filename' in data_info:
    #             # Extract sample info from pts_filename if available
    #             pts_file = data_info['pts_filename']
    #             if isinstance(pts_file, str) and 'LIDAR_TOP' in pts_file:
    #                 import os
    #                 base_name = os.path.basename(pts_file).replace('.pcd.bin', '')
    #                 data_info['occ_path'] = f'./data/nuscenes_occ/samples/{base_name}.pcd.bin.npy'
        
    #     # Add required fields for mmdet3d compatibility
    #     if 'ann_info' not in data_info:
    #         data_info['ann_info'] = {
    #             'gt_bboxes_3d': [],
    #             'gt_labels_3d': [],
    #             'gt_names': []
    #         }
        
    #     return data_info
    
    
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
            # CRITICAL: Use get_data_info to properly calculate lidar2img matrices
            # This is essential for transformer to work correctly
            data_info = self.get_data_info(idx)
            
            if data_info is None:
                raise RuntimeError(f"get_data_info returned None for index {idx}")
            
            # Process data through pipeline
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
                            
                            # If no images found, raise error with debug info
                            if imgs is None:
                                error_msg = f"ERROR: Pipeline did not load images properly for sample {idx}\n"
                                error_msg += f"  Result keys: {list(result.keys())}\n"
                                if 'img' in result:
                                    error_msg += f"  'img' type: {type(result['img'])}\n"
                                    if isinstance(result['img'], list):
                                        error_msg += f"  'img' length: {len(result['img'])}\n"
                                        if len(result['img']) > 0:
                                            error_msg += f"  'img[0]' type: {type(result['img'][0])}\n"
                                error_msg += f"  Pipeline transforms: {[type(t).__name__ for t in self.pipeline.transforms] if hasattr(self.pipeline, 'transforms') else 'N/A'}\n"
                                raise ValueError(error_msg)
                            
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
                            
                            # Add metadata from result or img_metas (CustomCollect3D output)
                            metainfo = {}
                            
                            # Check if CustomCollect3D has wrapped metadata in img_metas
                            img_metas_dict = None
                            if 'img_metas' in result:
                                img_metas = result['img_metas']
                                # Handle DataContainer wrapper
                                if hasattr(img_metas, 'data'):
                                    img_metas_dict = img_metas.data
                                elif isinstance(img_metas, dict):
                                    img_metas_dict = img_metas
                            
                            # Extract metadata from img_metas or result
                            for key in ['sample_idx', 'timestamp', 'scene_token', 'pc_range', 'occ_size', 'occ_path']:
                                if img_metas_dict and key in img_metas_dict:
                                    metainfo[key] = img_metas_dict[key]
                                elif key in result:
                                    metainfo[key] = result[key]
                            
                            # Add camera matrices - CRITICAL: lidar2img from data_info
                            for matrix_key in ['lidar2img', 'cam2img', 'lidar2cam']:
                                if img_metas_dict and matrix_key in img_metas_dict:
                                    metainfo[matrix_key] = img_metas_dict[matrix_key]
                                elif matrix_key in result:
                                    metainfo[matrix_key] = result[matrix_key]
                                elif matrix_key in data_info:
                                    # CRITICAL: Get from data_info which was computed by get_data_info
                                    metainfo[matrix_key] = data_info[matrix_key]
                            
                            # Add required img_shape metadata
                            if 'img_shape' not in metainfo and 'img' in result:
                                if isinstance(result['img'], list) and len(result['img']) > 0:
                                    img_shape = result['img'][0].shape
                                    metainfo['img_shape'] = [img_shape] * len(result['img'])
                            
                            data_sample.set_metainfo(metainfo)
                            
                            # Return in MMEngine format
                            return {
                                'inputs': {'imgs': imgs},
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
        
        # CRITICAL: Add integer index for evaluation metric matching
        input_dict['index'] = index
        
        # Add occ_path if available in original data
        if 'occ_path' in info:
            input_dict['occ_path'] = info['occ_path']
        
        # Add occ3d_gt_path if available (for occ3d GT format)
        if 'occ3d_gt_path' in info:
            input_dict['occ3d_gt_path'] = info['occ3d_gt_path']

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

            # For Occ3D (Ego-frame GT): set lidar2img = ego2img so encoder and depth supervision use the same frame
            if self.use_ego_frame:
                lidar2ego_rot = Quaternion(
                    input_dict['lidar2ego_rotation']
                ).rotation_matrix
                lidar2ego_trans = np.array(input_dict['lidar2ego_translation'])
                lidar2ego_4x4 = np.eye(4)
                lidar2ego_4x4[:3, :3] = lidar2ego_rot
                lidar2ego_4x4[:3, 3] = lidar2ego_trans
                ego2lidar_4x4 = np.linalg.inv(lidar2ego_4x4)
                ego2img_rts = [
                    (lidar2img_rt.astype(np.float64) @ ego2lidar_4x4.astype(np.float64)).astype(np.float32)
                    for lidar2img_rt in lidar2img_rts
                ]
                input_dict.update(
                    dict(
                        img_filename=image_paths,
                        lidar2img=ego2img_rts,
                        ego2lidar=ego2lidar_4x4,
                        cam_intrinsic=cam_intrinsics,
                        lidar2cam=lidar2cam_rts,
                        lidar2cam_dic=lidar2cam_dic,
                    ))
            else:
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