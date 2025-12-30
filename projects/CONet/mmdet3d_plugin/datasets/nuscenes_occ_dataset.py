import numpy as np
from mmdet3d.registry import DATASETS
from mmdet3d.datasets import NuScenesDataset
import os
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from ..utils.formating import cm_to_ious, format_SC_results, format_SSC_results

@DATASETS.register_module(force=True)
class NuscOCCDataset(NuScenesDataset):
    def __init__(self, occ_size, pc_range, occ_root, **kwargs):
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
        
        # Initialize base attributes manually to avoid parent class data filtering
        from mmengine.dataset import BaseDataset
        
        # Initialize basic attributes without calling full_init()
        self.serialize_data = kwargs.get('serialize_data', True) 
        self.test_mode = kwargs.get('test_mode', False)
        self.pipeline = kwargs.get('pipeline', [])
        
        # Build pipeline
        if self.pipeline:
            from mmengine.registry import TRANSFORMS
            from mmcv.transforms import Compose
            pipeline = []
            for transform in self.pipeline:
                if isinstance(transform, dict):
                    pipeline.append(TRANSFORMS.build(transform))
                else:
                    pipeline.append(transform)
            self.pipeline = Compose(pipeline)
        
        # Load data directly without parent class filtering
        self._load_data_list_direct()
        
        self.data_list = list(sorted(self.data_list, key=lambda e: e['timestamp']))
        # Use load_interval if available, otherwise default to 1
        load_interval = getattr(self, 'load_interval', 1)
        self.data_list = self.data_list[::load_interval]
        self.occ_size = occ_size
        self.pc_range = pc_range
        self.occ_root = occ_root
        
        # Set group flag for DistributedGroupSampler compatibility
        self._set_group_flag()      

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.
        
        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0. In 3D datasets, they are all the same, thus are all
        zeros.
        """
        import numpy as np
        # Use len(self.data_list) instead of len(self) to avoid initialization issues
        if hasattr(self, 'data_list') and self.data_list:
            self.flag = np.zeros(len(self.data_list), dtype=np.uint8)
        else:
            # Fallback: use 0 if data_list is not available
            self.flag = np.zeros(0, dtype=np.uint8)

    def _load_data_list_direct(self):
        """Load data list directly from annotation file without parent filtering."""
        import pickle
        import os.path as osp
        
        ann_file_path = osp.join(self.data_root, self.ann_file)
        with open(ann_file_path, 'rb') as f:
            data = pickle.load(f)
        
        # Use the infos from the annotation file as data_list
        if 'data_list' in data:
            self.data_list = data['data_list']
        elif 'infos' in data:
            self.data_list = data['infos']
        else:
            raise KeyError(f"Neither 'data_list' nor 'infos' found in {ann_file_path}")
        
        # Store metainfo if available (avoid setting as attribute due to property conflicts)
        if 'metainfo' in data:
            self._metainfo_data = data['metainfo']
        elif 'metadata' in data:
            self._metainfo_data = data['metadata']
            
        print(f"Direct load: {len(self.data_list)} samples from {ann_file_path}")
    
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
        if self.test_mode:
            return self.prepare_test_data(idx)
            
        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            
            return data

    def prepare_train_data(self, index):
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None

        # No pre_pipeline needed - data is already prepared
        example = self.pipeline(input_dict)
        return example

    def prepare_test_data(self, index):
        input_dict = self.get_data_info(index)

        if input_dict is None:
            return None

        # No pre_pipeline needed - data is already prepared
        example = self.pipeline(input_dict)
        return example

    def get_data_info(self, index):
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
            lidar2ego_translation=info['lidar2ego_translation'],
            lidar2ego_rotation=info['lidar2ego_rotation'],
            ego2global_translation=info['ego2global_translation'],
            ego2global_rotation=info['ego2global_rotation'],
            prev_idx=info['prev'],
            next_idx=info['next'],
            scene_token=info['scene_token'],
            can_bus=info['can_bus'],
            # frame_idx=info['frame_idx'],
            timestamp=info['timestamp'] / 1e6,
            occ_size = np.array(self.occ_size),
            pc_range = np.array(self.pc_range),
            lidar_token=info['lidar_token'],
            lidarseg=info['lidarseg'],
            curr=info,
        )
        
        # occ3d 지원: pkl 파일의 occ_path 키 전달
        if 'occ_path' in info:
            input_dict['occ_path'] = info['occ_path']

        if self.modality['use_camera']:
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
        if self.modality['use_lidar']:
            # FIXME alter lidar path
            input_dict['pts_filename'] = input_dict['pts_filename'].replace('./data/nuscenes/', self.data_root)
            for sw in input_dict['sweeps']:
                sw['data_path'] = sw['data_path'].replace('./data/nuscenes/', self.data_root)

        return input_dict


    def evaluate(self, results, logger=None, **kawrgs):
        eval_results = {}
        
        ''' evaluate SC '''
        evaluation_semantic = sum(results['SC_metric'])
        ious = cm_to_ious(evaluation_semantic)
        res_table, res_dic = format_SC_results(ious[1:], return_dic=True)
        for key, val in res_dic.items():
            eval_results['SC_{}'.format(key)] = val
        if logger is not None:
            logger.info('SC Evaluation')
            logger.info(res_table)
        
        ''' evaluate SSC '''
        evaluation_semantic = sum(results['SSC_metric'])
        ious = cm_to_ious(evaluation_semantic)
        res_table, res_dic = format_SSC_results(ious, return_dic=True)
        for key, val in res_dic.items():
            eval_results['SSC_{}'.format(key)] = val
        if logger is not None:
            logger.info('SSC Evaluation')
            logger.info(res_table)
        
        ''' evaluate SSC_fine '''
        if 'SSC_metric_fine' in results.keys():
            evaluation_semantic = sum(results['SSC_metric_fine'])
            ious = cm_to_ious(evaluation_semantic)
            res_table, res_dic = format_SSC_results(ious, return_dic=True)
            for key, val in res_dic.items():
                eval_results['SSC_fine_{}'.format(key)] = val
            if logger is not None:
                logger.info('SSC fine Evaluation')
                logger.info(res_table)
            
        return eval_results
