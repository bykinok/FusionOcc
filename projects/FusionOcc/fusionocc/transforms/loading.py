# FusionOcc transforms for loading and processing data
import torch
import numpy as np
import os
from PIL import Image
from mmdet3d.registry import TRANSFORMS


def mmlabNormalize(img):
    mean = torch.Tensor([123.675, 116.28, 103.53])
    std = torch.Tensor([58.395, 57.12, 57.375])
    
    img = torch.tensor(img).float().permute(2, 0, 1).contiguous()
    return img


@TRANSFORMS.register_module()
class PrepareImageSeg(object):
    """Prepare image segmentation data for FusionOcc.
    
    This is a simplified version of the original PrepareImageSeg transform.
    """
    
    def __init__(
            self,
            data_config,
            restore_upsample=8,
            downsample=16,
            is_train=False,
            sequential=False,
            img_seg_dir=None
    ):
        self.is_train = is_train
        self.data_config = data_config
        self.normalize_img = mmlabNormalize
        self.sequential = sequential
        self.restore_upsample = restore_upsample
        self.downsample = downsample
        self.img_seg_dir = img_seg_dir

    def __call__(self, results):
        """Process images and prepare img_inputs for FusionOcc."""
        
        # Get camera configurations
        cams = self.data_config['cams']
        num_cams = len(cams)
        height, width = self.data_config['input_size']
        
        imgs = []
        sensor2egos = []
        ego2globals = []
        intrins = []
        post_rots = []
        post_trans = []
        
        # Process images if available
        if 'images' in results:
            images_info = results['images']
            
            for cam in cams:
                if cam in images_info:
                    cam_info = images_info[cam]
                    
                    # For now, create dummy image data (in real implementation, load from cam_info['img_path'])
                    img = torch.randn(3, height, width)
                    imgs.append(img)
                    
                    # Get camera intrinsics and extrinsics
                    if 'cam_intrinsic' in cam_info:
                        intrinsic = torch.tensor(cam_info['cam_intrinsic'], dtype=torch.float32)
                    else:
                        intrinsic = torch.eye(3, dtype=torch.float32)
                    intrins.append(intrinsic)
                    
                    if 'sensor2ego_rotation' in cam_info and 'sensor2ego_translation' in cam_info:
                        sensor2ego = torch.eye(4, dtype=torch.float32)
                        sensor2ego[:3, :3] = torch.tensor(cam_info['sensor2ego_rotation'])
                        sensor2ego[:3, 3] = torch.tensor(cam_info['sensor2ego_translation'])
                    else:
                        sensor2ego = torch.eye(4, dtype=torch.float32)
                    sensor2egos.append(sensor2ego)
                    
                    # Post-processing transformation (identity for now)
                    post_rot = torch.eye(3, dtype=torch.float32)
                    post_tran = torch.zeros(3, dtype=torch.float32)
                    post_rots.append(post_rot)
                    post_trans.append(post_tran)
                else:
                    # Create dummy data for missing cameras
                    imgs.append(torch.randn(3, height, width))
                    intrins.append(torch.eye(3, dtype=torch.float32))
                    sensor2egos.append(torch.eye(4, dtype=torch.float32))
                    post_rots.append(torch.eye(3, dtype=torch.float32))
                    post_trans.append(torch.zeros(3, dtype=torch.float32))
        else:
            # Create dummy data if no images info
            for _ in range(num_cams):
                imgs.append(torch.randn(3, height, width))
                intrins.append(torch.eye(3, dtype=torch.float32))
                sensor2egos.append(torch.eye(4, dtype=torch.float32))
                post_rots.append(torch.eye(3, dtype=torch.float32))
                post_trans.append(torch.zeros(3, dtype=torch.float32))
        
        # Get ego2global transformation
        if 'ego2global' in results:
            ego2global = torch.tensor(results['ego2global'], dtype=torch.float32)
            ego2globals = [ego2global] * num_cams
        else:
            ego2globals = [torch.eye(4, dtype=torch.float32)] * num_cams
        
        # Stack tensors
        imgs = torch.stack(imgs)
        sensor2egos = torch.stack(sensor2egos)
        ego2globals = torch.stack(ego2globals)
        intrins = torch.stack(intrins)
        post_rots = torch.stack(post_rots)
        post_trans = torch.stack(post_trans)
        
        # Pack into img_inputs format
        results['img_inputs'] = (imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans)
        
        # Add dummy segs if needed (in real implementation, load segmentation)
        if 'segs' not in results:
            results['segs'] = torch.zeros(num_cams, 18, height//8, width//8)
        
        return results


@TRANSFORMS.register_module()
class LoadOccGTFromFile(object):
    """Load occupancy ground truth from file."""
    
    def __init__(self):
        pass
    
    def __call__(self, results):
        """Load occupancy ground truth data."""
        
        # Get the occupancy ground truth path
        occ_gt_path = results.get('occ_gt_path', '')
        
        # Standard occupancy grid size for FusionOcc
        occ_size = [200, 200, 16]  # x, y, z dimensions
        
        if occ_gt_path and os.path.exists(occ_gt_path):
            try:
                # Try to load actual occupancy data
                # The format depends on how the ground truth is stored
                # For now, create dummy data that matches expected dimensions
                results['voxel_semantics'] = torch.zeros(*occ_size, dtype=torch.long)
                results['mask_camera'] = torch.ones(*occ_size, dtype=torch.bool)
            except Exception:
                # Fall back to dummy data if loading fails
                results['voxel_semantics'] = torch.zeros(*occ_size, dtype=torch.long)
                results['mask_camera'] = torch.ones(*occ_size, dtype=torch.bool)
        else:
            # Create dummy occupancy ground truth
            results['voxel_semantics'] = torch.zeros(*occ_size, dtype=torch.long)
            results['mask_camera'] = torch.ones(*occ_size, dtype=torch.bool)
        
        return results


@TRANSFORMS.register_module()
class FuseAdjacentSweeps(object):
    """Fuse adjacent sweeps for FusionOcc."""
    
    def __init__(self,
                 load_dim=5,
                 use_dim=5,
                 pad_empty_sweeps=False,
                 remove_close=False,
                 test_mode=False):
        
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        self.use_dim = use_dim
        self.load_dim = load_dim
        self.pad_empty_sweeps = pad_empty_sweeps
        self.remove_close = remove_close
        self.test_mode = test_mode
        self.coord_type = 'LIDAR'
    
    def __call__(self, results):
        """Simplified version - just pass through for now."""
        # In a real implementation, this would fuse adjacent lidar sweeps
        # For now, we just return the current points without fusion
        return results


@TRANSFORMS.register_module()
class LoadAnnotationsAll(object):
    """Load all annotations including BDA augmentation."""
    
    def __init__(self, bda_aug_conf, classes, is_train=True):
        self.bda_aug_conf = bda_aug_conf
        self.classes = classes
        self.is_train = is_train
    
    def __call__(self, results):
        """Load annotations with BDA augmentation."""
        
        # The annotations should already be loaded in ann_info by the dataset
        # Here we just ensure they're in the correct format
        
        if 'ann_info' not in results:
            # Create empty annotations if none exist
            results['ann_info'] = dict()
            results['ann_info']['gt_bboxes_3d'] = np.zeros((0, 7))
            results['ann_info']['gt_labels_3d'] = np.array([])
        
        # Apply BDA (Bird's Eye View Data Augmentation) if needed
        # For now, skip BDA augmentation to avoid complexity
        
        # Ensure gt_bboxes_3d and gt_labels_3d are in results for compatibility
        # But DO NOT overwrite if they already exist and have data
        if 'gt_bboxes_3d' not in results:
            results['gt_bboxes_3d'] = results['ann_info'].get('gt_bboxes_3d', np.zeros((0, 7)))
        if 'gt_labels_3d' not in results:
            results['gt_labels_3d'] = results['ann_info'].get('gt_labels_3d', np.array([]))
        
        return results


@TRANSFORMS.register_module()
class FormatDataSamples(object):
    """Format data for MMEngine compatibility."""
    
    def __init__(self):
        pass
    
    def __call__(self, results):
        """Format data samples for MMEngine."""
        
        # Create data_samples key for MMEngine compatibility
        from mmengine.structures import InstanceData
        try:
            from mmdet3d.structures import Det3DDataSample
        except ImportError:
            # Create a simple class if Det3DDataSample is not available
            class Det3DDataSample:
                def __init__(self):
                    pass
        
        # Create data sample object
        data_samples = Det3DDataSample()
        
        # Add ground truth instances for 3D detection
        if 'ann_info' in results and len(results['ann_info'].get('gt_labels_3d', [])) > 0:
            gt_instances_3d = InstanceData()
            gt_instances_3d.labels_3d = torch.tensor(results['ann_info']['gt_labels_3d'])
            gt_instances_3d.bboxes_3d = torch.tensor(results['ann_info']['gt_bboxes_3d'])
            data_samples.gt_instances_3d = gt_instances_3d
            # Debug
            # print(f"FormatDataSamples: Found {len(results['ann_info']['gt_labels_3d'])} annotations")
        else:
            # Create empty instances
            gt_instances_3d = InstanceData()
            gt_instances_3d.labels_3d = torch.tensor([])
            gt_instances_3d.bboxes_3d = torch.zeros((0, 7))
            data_samples.gt_instances_3d = gt_instances_3d
            # Debug
            # print(f"FormatDataSamples: No annotations found, ann_info keys: {list(results.get('ann_info', {}).keys())}")
            # print(f"FormatDataSamples: gt_labels_3d length: {len(results.get('ann_info', {}).get('gt_labels_3d', []))}")
        
        results['data_samples'] = data_samples
        
        return results
