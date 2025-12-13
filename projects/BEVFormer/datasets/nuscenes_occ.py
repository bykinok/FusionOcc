import copy
import os
import numpy as np
from tqdm import tqdm
try:
    from mmdet.datasets import DATASETS
except ImportError:
    from mmdet3d.registry import DATASETS
from mmdet3d.datasets import NuScenesDataset
import mmcv
from os import path as osp
try:
    from mmdet.datasets import DATASETS
except ImportError:
    from mmdet3d.registry import DATASETS
import torch
import numpy as np
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from .nuscnes_eval import NuScenesEval_custom
from projects.BEVFormer.utils.visual import save_tensor
try:
    from mmcv.parallel import DataContainer as DC
except ImportError:
    # DataContainer is deprecated in newer versions, create a simple wrapper
    class DC:
        def __init__(self, data, **kwargs):
            self.data = data
            self._kwargs = kwargs
        def __repr__(self):
            return f'DC({self.data})'
import random
from nuscenes.utils.geometry_utils import transform_matrix
from .occ_metrics import Metric_mIoU, Metric_FScore


@DATASETS.register_module()
class NuSceneOcc(NuScenesDataset):
    r"""NuScenes Dataset.

    This datset only add camera intrinsics and extrinsics to the results.
    """

    def __init__(self, queue_length=4, bev_size=(200, 200), overlap_test=False, eval_fscore=False, load_interval=1, *args, **kwargs):
        # Set attributes BEFORE calling parent __init__ (which calls full_init -> load_data_list)
        self.eval_fscore = eval_fscore
        self.queue_length = queue_length
        self.overlap_test = overlap_test
        self.bev_size = bev_size
        self.load_interval = load_interval
        
        # Extract deprecated parameters before calling parent __init__ (new mmengine doesn't accept them)
        classes = kwargs.pop('classes', None)
        kwargs.pop('samples_per_gpu', None)  # Remove old-style parameter
        
        # CRITICAL: Force serialize_data=False to use load_data_list() instead of pickle cache
        kwargs['serialize_data'] = False
        
        super().__init__(*args, **kwargs)
        if classes is not None:
            self.CLASSES = classes
        
        # After parent init, data_list should be populated. Store it as data_infos for compatibility
        self.data_infos = self.data_list
        
        # CRITICAL: Set group flag for DistributedGroupSampler
        # This is required by DistributedGroupSampler to group samples
        # Set flag for ALL modes (train/val/test) since DistributedGroupSampler may be used
        # Ensure data_list is loaded before setting flag
        if not hasattr(self, 'data_list') or not self.data_list:
            self.data_list = self.load_data_list()
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

    def load_data_list(self):
        """Load data list from annotation file for mmengine compatibility."""
        # In mmengine, BaseDataset calls load_data_list() in full_init()
        # which happens in __init__
        try:
            from mmengine.fileio import load
            data = load(self.ann_file)
        except ImportError:
            import mmcv
            data = mmcv.load(self.ann_file)
        
        # Sort and filter data
        data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))
        data_infos = data_infos[::self.load_interval]
        
        # Store metadata
        self.metadata = data.get('metadata', {})
        self.version = self.metadata.get('version', 'v1.0')
        
        # IMPORTANT: Return the data_infos so BaseDataset can store it as self.data_list
        return data_infos

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.
        
        Note: This method is kept for compatibility but is deprecated in mmengine.
        The actual data loading happens in load_data_list().

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations (same as data_list).
        """
        # This should only be called after full_init, so data_list should exist
        return getattr(self, 'data_list', [])
    
    def pre_pipeline(self, results):
        """Prepare results dict for pipeline processing.
        
        Args:
            results (dict): Result dict from get_data_info().
        """
        # Add required fields
        results['img_prefix'] = self.data_root
        results['seg_prefix'] = None
        results['proposal_file'] = None
        results['bbox3d_fields'] = []
        results['pts_mask_fields'] = []
        results['pts_seg_fields'] = []
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []

    def prepare_train_data(self, index):
        """
        Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
        """
        # breakpoint()
        queue = []
        index_list = list(range(index - self.queue_length, index))
        random.shuffle(index_list)
        index_list = sorted(index_list[1:])
        index_list.append(index)
        for i in index_list:
            # Clip index to valid range [0, len(data_list)-1]
            i = max(0, min(i, len(self.data_list) - 1))
            input_dict = self.get_data_info(i)
            if input_dict is None:
                return None
            self.pre_pipeline(input_dict)
            example = self.pipeline(input_dict)
            queue.append(example)
        return self.union2one(queue)
    
    def prepare_test_data(self, index):
        """
        Testing data preparation - simplified for faster loading.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Testing data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example

    def union2one(self, queue):
        # CRITICAL: Follow original BEVFormer exactly
        imgs_list = [each['img'].data for each in queue]
        metas_map = {}
        prev_scene_token = None
        prev_pos = None
        prev_angle = None
        for i, each in enumerate(queue):
            metas_map[i] = each['img_metas'].data
            if metas_map[i]['scene_token'] != prev_scene_token:
                metas_map[i]['prev_bev_exists'] = False
                prev_scene_token = metas_map[i]['scene_token']
                prev_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                prev_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                metas_map[i]['can_bus'][:3] = 0
                metas_map[i]['can_bus'][-1] = 0
            else:
                metas_map[i]['prev_bev_exists'] = True
                tmp_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                tmp_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                metas_map[i]['can_bus'][:3] -= prev_pos
                metas_map[i]['can_bus'][-1] -= prev_angle
                prev_pos = copy.deepcopy(tmp_pos)
                prev_angle = copy.deepcopy(tmp_angle)
        
        # CRITICAL: Stack images AFTER the for loop
        # Handle both tensor and list cases
        if isinstance(imgs_list[0], list):
            # imgs_list is a list of lists (multi-view images)
            # Convert each list of multi-view images to a stacked tensor
            # imgs_list: [[img1_view1, img1_view2, ...], [img2_view1, img2_view2, ...], ...]
            # Result: [num_queue, num_views, C, H, W]
            stacked_imgs = []
            for img_views in imgs_list:
                if isinstance(img_views[0], torch.Tensor):
                    stacked_imgs.append(torch.stack(img_views))
                else:
                    # Already stacked
                    stacked_imgs.append(img_views)
            # CRITICAL: Follow original BEVFormer - use cpu_only=False for img
            # This allows automatic GPU transfer by DataLoader
            queue[-1]['img'] = DC(torch.stack(stacked_imgs), cpu_only=False, stack=True)
        else:
            # imgs_list is a list of tensors
            queue[-1]['img'] = DC(torch.stack(imgs_list), cpu_only=False, stack=True)

        # CRITICAL: Follow original BEVFormer format
        # Original: queue[-1]['img_metas'] = DC(metas_map, cpu_only=True)
        # Do NOT wrap in list for train mode compatibility
        queue[-1]['img_metas'] = DC(metas_map, cpu_only=True)
        queue = queue[-1]
        return queue

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        # Use data_list (mmengine) instead of data_infos (old mmdet3d)
        info = self.data_list[index]
        # standard protocal modified from SECOND.Pytorch
        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            ego2global_translation=info['ego2global_translation'],
            ego2global_rotation=info['ego2global_rotation'],
            prev_idx=info['prev'],
            next_idx=info['next'],
            scene_token=info['scene_token'],
            can_bus=info['can_bus'],
            frame_idx=info['frame_idx'],
            timestamp=info['timestamp'] / 1e6,
        )
        if 'occ_gt_path' in info:
            input_dict['occ_gt_path'] = info['occ_gt_path']
        lidar2ego_rotation = info['lidar2ego_rotation']
        lidar2ego_translation = info['lidar2ego_translation']
        ego2lidar = transform_matrix(translation=lidar2ego_translation, rotation=Quaternion(lidar2ego_rotation),
                                     inverse=True)
        input_dict['ego2lidar'] = ego2lidar
        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []
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
            # Build images dict for mmdet3d compatibility
            images_dict = {}
            idx = 0
            for cam_type, cam_info in info['cams'].items():
                cam_dict = dict(
                    img_path=cam_info['data_path'],
                    cam2img=cam_intrinsics[idx],
                    lidar2cam=lidar2cam_rts[idx]
                )
                images_dict[cam_type] = cam_dict
                idx += 1
            
            input_dict.update(
                dict(
                    img_filename=image_paths,
                    images=images_dict,
                    lidar2img=lidar2img_rts,
                    cam_intrinsic=cam_intrinsics,
                    lidar2cam=lidar2cam_rts,
                ))

        if not self.test_mode:
            # CRITICAL: Don't call self.get_ann_info() to avoid infinite recursion with mmdet3d's base class
            # Instead, directly construct annos from info dict
            try:
                from mmdet3d.structures.bbox_3d import LiDARInstance3DBoxes
            except ImportError:
                from mmdet3d.core.bbox import LiDARInstance3DBoxes
            
            gt_boxes = info.get('gt_boxes', np.zeros((0, 7), dtype=np.float32))
            if len(gt_boxes) > 0:
                gt_bboxes_3d = LiDARInstance3DBoxes(gt_boxes, box_dim=gt_boxes.shape[-1], origin=(0.5, 0.5, 0))
            else:
                gt_bboxes_3d = LiDARInstance3DBoxes(np.zeros((0, 7), dtype=np.float32), box_dim=7, origin=(0.5, 0.5, 0))
            
            annos = dict(
                gt_bboxes_3d=gt_bboxes_3d,
                gt_labels_3d=np.array([self.CLASSES.index(name) if name in self.CLASSES else -1 for name in info.get('gt_names', [])], dtype=np.int64),
                gt_names=info.get('gt_names', [])
            )
            input_dict['ann_info'] = annos

        rotation = Quaternion(input_dict['ego2global_rotation'])
        translation = input_dict['ego2global_translation']
        can_bus = input_dict['can_bus']
        can_bus[:3] = translation
        can_bus[3:7] = rotation
        patch_angle = quaternion_yaw(rotation) / np.pi * 180
        if patch_angle < 0:
            patch_angle += 360
        can_bus[-2] = patch_angle / 180 * np.pi
        can_bus[-1] = patch_angle

        return input_dict

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

    def evaluate_miou(self, occ_results, runner=None, show_dir=None, **eval_kwargs):
        if show_dir is not None:
            if not os.path.exists(show_dir):
                os.mkdir(show_dir)
            print('\nSaving output and gt in {} for visualization.'.format(show_dir))
            begin=eval_kwargs.get('begin',None)
            end=eval_kwargs.get('end',None)
        self.occ_eval_metrics = Metric_mIoU(
            num_classes=18,
            use_lidar_mask=False,
            use_image_mask=True)
        if self.eval_fscore:
            self.fscore_eval_metrics = Metric_FScore(
                leaf_size=10,
                threshold_acc=0.4,
                threshold_complete=0.4,
                voxel_size=[0.4, 0.4, 0.4],
                range=[-40, -40, -1, 40, 40, 5.4],
                void=[17, 255],
                use_lidar_mask=False,
                use_image_mask=True,
            )
        print('\nStarting Evaluation...')
        for index, occ_pred in enumerate(tqdm(occ_results)):
            # Use data_list (mmengine) instead of data_infos (old mmdet3d)
            info = self.data_list[index]

            occ_gt = np.load(os.path.join(self.data_root, info['occ_gt_path']))
            if show_dir is not None:
                if begin is not None and end is not None:
                    if index>= begin and index<end:
                        sample_token = info['token']
                        save_path = os.path.join(show_dir,str(index).zfill(4))
                        np.savez_compressed(save_path, pred=occ_pred, gt=occ_gt, sample_token=sample_token)
                else:
                    sample_token=info['token']
                    save_path=os.path.join(show_dir,str(index).zfill(4))
                    np.savez_compressed(save_path,pred=occ_pred,gt=occ_gt,sample_token=sample_token)


            gt_semantics = occ_gt['semantics']
            mask_lidar = occ_gt['mask_lidar'].astype(bool)
            mask_camera = occ_gt['mask_camera'].astype(bool)
            # occ_pred = occ_pred
            self.occ_eval_metrics.add_batch(occ_pred, gt_semantics, mask_lidar, mask_camera)
            if self.eval_fscore:
                self.fscore_eval_metrics.add_batch(occ_pred, gt_semantics, mask_lidar, mask_camera)

        self.occ_eval_metrics.count_miou()
        if self.eval_fscore:
            self.fscore_eval_metrics.count_fscore()

    def format_results(self, occ_results,submission_prefix,**kwargs):
        if submission_prefix is not None:
            mmcv.mkdir_or_exist(submission_prefix)

        for index, occ_pred in enumerate(tqdm(occ_results)):
            # Use data_list (mmengine) instead of data_infos (old mmdet3d)
            info = self.data_list[index]
            sample_token = info['token']
            save_path=os.path.join(submission_prefix,'{}.npz'.format(sample_token))
            np.savez_compressed(save_path,occ_pred.astype(np.uint8))
        print('\nFinished.')


def bevformer_collate_fn(batch):
    """Optimized collate function for BEVFormer that unwraps DataContainer and converts to tensors.
    
    This function performs all DataContainer unwrapping and tensor conversion at collate time
    to avoid repeated processing in forward_train, significantly improving training speed.
    
    Args:
        batch: List of samples from dataset, each sample is a dict with keys like:
               'img', 'voxel_semantics', 'mask_camera', 'img_metas', etc.
        
    Returns:
        dict: Collated batch data with unwrapped tensors ready for training
    """
    if not batch:
        return {}
    
    import numpy as np
    import torch
    
    keys = batch[0].keys()
    collated = {}
    
    for key in keys:
        samples = [sample[key] for sample in batch]
        
        # Handle different data types
        if key in ['img', 'voxel_semantics', 'mask_camera']:
            # These need DataContainer unwrapping and tensor conversion
            tensors = []
            for sample in samples:
                # Unwrap DataContainer
                if hasattr(sample, 'data'):
                    data = sample.data
                else:
                    data = sample
                
                # Convert to torch.Tensor if needed
                if isinstance(data, np.ndarray):
                    data = torch.from_numpy(data)
                elif not isinstance(data, torch.Tensor):
                    # Handle memoryview or other types
                    data = torch.from_numpy(np.asarray(data))
                
                tensors.append(data)
            
            # Stack tensors into batch
            if key == 'img':
                # img: [queue_length, N_cams, C, H, W] -> [B, queue_length, N_cams, C, H, W]
                collated[key] = torch.stack(tensors, dim=0)
            else:
                # voxel_semantics, mask_camera: [H, W, D] -> [B, H, W, D]
                try:
                    collated[key] = torch.stack(tensors, dim=0)
                except RuntimeError:
                    # If shapes don't match, keep as list (shouldn't happen normally)
                    collated[key] = tensors
        
        elif key == 'img_metas':
            # Unwrap img_metas from DataContainer but keep as list
            unwrapped = []
            for sample in samples:
                if hasattr(sample, 'data'):
                    unwrapped.append(sample.data)
                else:
                    unwrapped.append(sample)
            collated[key] = unwrapped
        
        else:
            # Other keys - use default collate behavior
            if isinstance(samples[0], torch.Tensor):
                try:
                    collated[key] = torch.stack(samples, dim=0)
                except:
                    collated[key] = samples
            else:
                collated[key] = samples
    
    return collated

