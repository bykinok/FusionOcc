import numpy as np
from mmdet3d.registry import DATASETS
from mmdet3d.datasets import NuScenesDataset
import os
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from ..utils.formating import cm_to_ious, format_SC_results, format_SSC_results
import torch

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


def conet_collate_fn(batch):
    """Custom collate function for CONet that properly handles batch stacking.
    
    CONet's img_inputs is a list/tuple of tensors that need to be stacked properly
    across the batch dimension. This collate function ensures correct batching for:
    - img_inputs: list of tensors [imgs, rots, trans, intrins, post_rots, post_trans, ...]
    - points: list of point clouds (variable length per sample)
    - gt_occ: list of ground truth occupancy grids
    - img_metas: list of metadata dicts
    
    Args:
        batch: List of samples from dataset, each sample contains:
               - 'inputs': dict with 'points', 'img_inputs'
               - 'data_samples': OccDataSample with metadata and gt_occ
    
    Returns:
        dict: Collated batch data with properly stacked tensors
    """
    if not batch:
        return {}
    
    # Extract inputs and data_samples from each sample
    batch_inputs = [sample['inputs'] for sample in batch]
    batch_data_samples = [sample['data_samples'] for sample in batch]
    
    # Collate img_inputs (list of tensors)
    # img_inputs structure: [imgs, rots, trans, intrins, post_rots, post_trans, bda_rot, img_shape, gt_depths, sensor2sensors]
    img_inputs_list = [inp.get('img_inputs') for inp in batch_inputs if inp.get('img_inputs') is not None]
    
    if img_inputs_list:
        # img_inputs is a list/tuple of tensors
        # Each element in the list needs to be stacked across batch dimension
        collated_img_inputs = []
        num_elements = len(img_inputs_list[0])
        
        for i in range(num_elements):
            elements = [img_inp[i] for img_inp in img_inputs_list]
            
            # Stack tensors if possible
            if isinstance(elements[0], torch.Tensor):
                try:
                    # Stack along batch dimension
                    stacked = torch.stack(elements, dim=0)
                    collated_img_inputs.append(stacked)
                except:
                    # If stacking fails (e.g., different shapes), keep as list
                    collated_img_inputs.append(elements)
            else:
                # Non-tensor elements (e.g., torch.Size, list), keep as list
                collated_img_inputs.append(elements)
    else:
        collated_img_inputs = None
    
    # Collate points (keep as list since each sample has different number of points)
    points_list = [inp.get('points') for inp in batch_inputs if inp.get('points') is not None]
    
    # Collate gt_occ
    gt_occ_list = []
    for data_sample in batch_data_samples:
        if hasattr(data_sample, 'gt_occ'):
            gt_occ = data_sample.gt_occ
            if isinstance(gt_occ, torch.Tensor):
                gt_occ_list.append(gt_occ)
            elif isinstance(gt_occ, np.ndarray):
                gt_occ_list.append(torch.from_numpy(gt_occ))
    
    # Stack gt_occ if available
    if gt_occ_list:
        try:
            gt_occ = torch.stack(gt_occ_list, dim=0)
        except:
            # If shapes don't match, keep as list
            gt_occ = gt_occ_list
    else:
        gt_occ = None
    
    # Collate img_metas (keep as list of dicts)
    img_metas = []
    for data_sample in batch_data_samples:
        if hasattr(data_sample, 'metainfo'):
            img_metas.append(data_sample.metainfo)
        else:
            img_metas.append({})
    
    # Return in MMEngine-compatible format
    # Put everything in inputs to prevent filtering
    result = {
        'inputs': {
            'points': points_list,
            'img_inputs': collated_img_inputs,
            'gt_occ': gt_occ,
        },
        'data_samples': img_metas,
    }
    
    return result