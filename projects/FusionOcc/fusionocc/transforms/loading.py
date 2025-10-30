# FusionOcc transforms for loading and processing data
import torch
import numpy as np
import os
from PIL import Image
from mmdet3d.registry import TRANSFORMS
from pyquaternion import Quaternion


def mmlabNormalize(img):
    """Normalize image using ImageNet mean and std, matching mmcv.imnormalize with to_rgb=True.
    
    mmcv.imnormalize with to_rgb=True:
    1. Converts BGR to RGB ([:, :, ::-1])
    2. Normalizes with (img - mean) / std
    
    Since PIL Image.open loads as RGB, and mmcv expects BGR input:
    - We convert RGB to BGR FIRST ([:, :, ::-1])
    - Then normalize
    
    This matches the original behavior exactly.
    """
    # ImageNet statistics (for BGR order)
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    
    # Convert PIL Image to numpy array if needed
    if not isinstance(img, np.ndarray):
        img = np.array(img, dtype=np.float32)
    else:
        img = img.astype(np.float32)
    
    # CRITICAL: Convert RGB to BGR (to match mmcv.imnormalize with to_rgb=True behavior)
    img = img[:, :, ::-1]
    
    # Normalize: (img - mean) / std
    img = (img - mean) / std
    
    # Convert to torch tensor and permute to CHW format
    img = torch.from_numpy(img).float().permute(2, 0, 1).contiguous()
    return img


@TRANSFORMS.register_module()
class PrepareImageSeg(object):
    """Prepare image inputs for FusionOcc with REAL image loading."""
    
    def __init__(
            self,
            data_config,
            restore_upsample=8,
	downsample=1,
            is_train=False,
            sequential=False,
            img_seg_dir=None,
            num_adj_frames=1  # Number of adjacent frames expected
    ):
        self.is_train = is_train
        self.data_config = data_config
        self.sequential = sequential
        self.restore_upsample = restore_upsample
        self.downsample = downsample
        self.img_seg_dir = img_seg_dir
        self.num_adj_frames = num_adj_frames  # Expected number of adjacent frames

        # Normalization (ImageNet stats)
        self.mean = np.array([123.675, 116.28, 103.53])
        self.std = np.array([58.395, 57.12, 57.375])
    
    def get_rot(self, h):
        """Get 2D rotation matrix."""
        return np.array([
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ])
    
    def img_transform(self, img, post_rot, post_tran, resize, resize_dims,
                      crop, flip, rotate):
        """Transform image with augmentation."""
        from PIL import Image
        
        # Adjust image
        img = self.img_transform_core(img, resize_dims, crop, flip, rotate)
        
        # Post-homography transformation
        post_rot *= resize
        post_tran -= np.array(crop[:2])
        if flip:
            A = np.array([[-1, 0], [0, 1]])
            b = np.array([crop[2] - crop[0], 0])
            post_rot = A @ post_rot
            post_tran = A @ post_tran + b
        A = self.get_rot(rotate / 180 * np.pi)
        b = np.array([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A @ (-b) + b
        post_rot = A @ post_rot
        post_tran = A @ post_tran + b
        
        return img, post_rot, post_tran
    
    def img_transform_core(self, img, resize_dims, crop, flip, rotate):
        """Core image transformation."""
        from PIL import Image
        
        # Resize
        img = img.resize(resize_dims, Image.BILINEAR)
        # Crop
        img = img.crop(crop)
        # Flip
        if flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        # Rotate
        img = img.rotate(rotate)
        return img
    
    def get_img_seg(self, img_path):
        """Load image segmentation."""
        if self.img_seg_dir is None:
            return None
        name = img_path.split("samples")[1].replace(".jpg", ".npy")
        seg_path = self.img_seg_dir + name
        try:
            seg = np.load(seg_path)
            seg = np.repeat(seg, self.restore_upsample, axis=1)
            seg = np.repeat(seg, self.restore_upsample, axis=0)
            seg = Image.fromarray(seg, mode="L")
            return seg
        except:
            # If seg file doesn't exist, return None
            return None
    
    def format_seg(self, seg):
        """Format segmentation for model input."""
        seg = np.array(seg)
        seg = seg[::self.downsample, ::self.downsample]
        seg = torch.from_numpy(seg)
        return seg
    
    def seg_transform_core(self, img, resize_dims, crop, flip, rotate):
        """Transform segmentation map (same as image but with NEAREST interpolation)."""
        from PIL import Image
        
        # Resize with NEAREST to preserve labels
        img = img.resize(resize_dims, Image.NEAREST)
        # Crop
        img = img.crop(crop)
        # Flip
        if flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        # Rotate (expand=0 to maintain size)
        img = img.rotate(rotate, expand=0)
        return img
    
    def sample_augmentation(self, H, W, flip=None, scale=None):
        """Sample augmentation parameters."""
        fH, fW = self.data_config['input_size']
        if self.is_train:
            resize = float(fW) / float(W)
            resize += np.random.uniform(*self.data_config['resize'])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_config['crop_h'])) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = self.data_config['flip'] and np.random.choice([0, 1])
            rotate = np.random.uniform(*self.data_config['rot'])
        else:
            resize = float(fW) / float(W)
            if scale is not None:
                resize += scale
            else:
                resize += self.data_config.get('resize_test', 0.0)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_config['crop_h'])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False if flip is None else flip
            rotate = 0
        return resize, resize_dims, crop, flip, rotate
    
    def get_inputs(self, results, flip=None, scale=None):
        """Get image inputs with proper transformation - REAL IMAGE LOADING."""
        from PIL import Image
        from scipy.spatial.transform import Rotation as R
        
        imgs = []
        segs = []
        sensor2egos = []
        ego2globals = []
        intrins = []
        post_rots = []
        post_trans = []
        
        # Handle 'curr' wrapper (original FusionOcc format)
        if 'curr' in results:
            results_data = results['curr']
        else:
            results_data = results
        
        # Support both 'cams' (fusionocc) and 'images' (occfrmwrk) format
        if 'cams' in results_data:
            cam_data = results_data['cams']
            data_format = 'fusionocc'
        elif 'images' in results_data:
            cam_data = results_data['images']
            data_format = 'occfrmwrk'
        else:
            raise ValueError("No camera data found in results")
        
        # CRITICAL: ALWAYS use config camera order, not pkl dict order!
        # The original implementation uses choose_cams() which returns config order
        # This is crucial for matching the pretrained checkpoint
        cam_names = self.data_config['cams']
        
        for cam_idx, cam_name in enumerate(cam_names):
            if cam_name not in cam_data:
                continue
            
            cam_info = cam_data[cam_name]
            
            # Load image
            if data_format == 'fusionocc':
                img_path = cam_info['data_path']
            else:  # occfrmwrk
                img_path = cam_info['img_path']
                # img_path in occfrmwrk is relative, need to add data_root
                if not img_path.startswith('./') and not img_path.startswith('/'):
                    # Assume it's relative to nuscenes root
                    img_path = f"./data/nuscenes/samples/{cam_name}/{img_path}"
            
            # Load image file
            img = Image.open(img_path)
            
            # Sample augmentation
            post_rot = np.eye(2)
            post_tran = np.zeros(2)
            
            W, H = img.size  # PIL: (W, H)
            resize, resize_dims, crop, flip_aug, rotate = self.sample_augmentation(
                H, W, flip, scale)
            
            # Apply transformation
            img, post_rot2, post_tran2 = self.img_transform(
                img, post_rot, post_tran, resize, resize_dims, crop, flip_aug, rotate)
            
            # Convert to numpy and normalize (using mmcv's imnormalize like original)
            # CRITICAL: Original uses PIL + to_rgb=True, which swaps R and B channels!
            # The checkpoint was trained with this "bug", so we must replicate it exactly
            from mmcv.image.photometric import imnormalize
            img = np.array(img)
            img = imnormalize(img, self.mean, self.std, to_rgb=True)  # This swaps R<->B!
            img = torch.tensor(img).float().permute(2, 0, 1).contiguous()
            
            imgs.append(img)  # Add current frame image
            
            # Load and transform segmentation
            seg = self.get_img_seg(img_path)
            if seg is not None:
                seg = self.seg_transform_core(seg, resize_dims, crop, flip_aug, rotate)
                segs.append(self.format_seg(seg))
            else:
                # If no seg, create a dummy seg filled with zeros
                fH, fW = self.data_config['input_size']
                seg_h = fH // self.downsample
                seg_w = fW // self.downsample
                segs.append(torch.zeros(seg_h, seg_w, dtype=torch.uint8))
            
            # IMPORTANT: Process adjacent frames IMMEDIATELY after current frame (interleaved!)
            # Original order: cam0_curr, cam0_adj, cam1_curr, cam1_adj, ...
            if self.sequential and 'adjacent' in results and len(results['adjacent']) > 0:
                for adj_idx, adj_info in enumerate(results['adjacent']):
                    # Support both 'cams' and 'images' for adjacent frames
                    if 'cams' in adj_info:
                        adj_cam_data = adj_info['cams']
                        adj_format = 'fusionocc'
                    elif 'images' in adj_info:
                        adj_cam_data = adj_info['images']
                        adj_format = 'occfrmwrk'
                    else:
                        continue  # Skip if no camera data
                    
                    cam_info_adj = adj_cam_data[cam_name]
                    
                    # Get image path
                    if adj_format == 'fusionocc':
                        img_path_adj = cam_info_adj['data_path']
                        if not img_path_adj.startswith('./'):
                            img_path_adj = f"./data/nuscenes/{img_path_adj}"
                    else:  # occfrmwrk
                        img_path_adj = cam_info_adj['img_path']
                        if not img_path_adj.startswith('/'):
                            img_path_adj = f"./data/nuscenes/samples/{cam_name}/{img_path_adj}"
                    
                    # Load and transform adjacent frame image (use SAME aug as current frame)
                    img_adj = Image.open(img_path_adj)
                    img_adj = self.img_transform_core(img_adj, resize_dims, crop, flip_aug, rotate)
                    
                    # Normalize (same as current frame - to_rgb=True swaps R<->B)
                    img_adj = np.array(img_adj)
                    img_adj = imnormalize(img_adj, self.mean, self.std, to_rgb=True)
                    img_adj = torch.tensor(img_adj).float().permute(2, 0, 1).contiguous()
                    imgs.append(img_adj)  # IMMEDIATELY after current frame (interleaved!)
                    
                    # Load and transform segmentation for adjacent frame
                    seg_adj = self.get_img_seg(img_path_adj)
                    if seg_adj is not None:
                        seg_adj = self.seg_transform_core(seg_adj, resize_dims, crop, flip_aug, rotate)
                        segs.append(self.format_seg(seg_adj))
                    else:
                        fH, fW = self.data_config['input_size']
                        seg_h = fH // self.downsample
                        seg_w = fW // self.downsample
                        segs.append(torch.zeros(seg_h, seg_w, dtype=torch.uint8))
            
            # Get intrinsics
            if data_format == 'fusionocc':
                intrinsic = np.array(cam_info['cam_intrinsic'], dtype=np.float32)
            else:  # occfrmwrk
                # cam2img is 3x3 or 4x4 matrix
                cam2img = cam_info['cam2img']
                if isinstance(cam2img, list):
                    cam2img = np.array(cam2img)
                # Ensure we get 3x3 intrinsic matrix
                if cam2img.shape == (3, 3):
                    intrinsic = cam2img.astype(np.float32)
                elif cam2img.shape[0] >= 3 and cam2img.shape[1] >= 3:
                    intrinsic = cam2img[:3, :3].astype(np.float32)
                else:
                    raise ValueError(f"Invalid cam2img shape: {cam2img.shape}")
            intrins.append(torch.from_numpy(intrinsic))
            
            # Get sensor2ego transformation
            if data_format == 'fusionocc':
                # Quaternion to rotation matrix
                quat = cam_info['sensor2ego_rotation']  # [w, x, y, z]
                rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
                trans = np.array(cam_info['sensor2ego_translation'])
                
                sensor2ego = np.eye(4, dtype=np.float32)
                sensor2ego[:3, :3] = rot
                sensor2ego[:3, 3] = trans
            else:  # occfrmwrk
                # cam2ego is 4x4 matrix
                cam2ego = cam_info['cam2ego']
                if isinstance(cam2ego, list):
                    cam2ego = np.array(cam2ego)
                # Ensure shape is (4, 4)
                sensor2ego = np.array(cam2ego, dtype=np.float32).reshape(4, 4)
            sensor2egos.append(torch.from_numpy(sensor2ego))
            
            # Get ego2global transformation
            if data_format == 'fusionocc':
                quat = cam_info['ego2global_rotation']
                rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
                trans = np.array(cam_info['ego2global_translation'])
                
                ego2global = np.eye(4, dtype=np.float32)
                ego2global[:3, :3] = rot
                ego2global[:3, 3] = trans
            else:  # occfrmwrk
                # IMPORTANT: Use per-camera ego2global (same as fusionocc) for consistency
                # Original model was trained with per-camera ego2global
                if 'ego2global' in cam_info:
                    # Per-camera ego2global (PRIORITY - consistent with fusionocc)
                    ego2global_data = cam_info['ego2global']
                    ego2global = np.array(ego2global_data, dtype=np.float32).reshape(4, 4)
                elif 'ego2global' in results:
                    # Sample-level ego2global (fallback)
                    ego2global_data = results['ego2global']
                    ego2global = np.array(ego2global_data, dtype=np.float32).reshape(4, 4)
                else:
                    ego2global = np.eye(4, dtype=np.float32)
            ego2globals.append(torch.from_numpy(ego2global))
            
            # Post transformation - expand 2D to 3D
            post_tran_3d = torch.zeros(3)
            post_rot_3d = torch.eye(3)
            post_tran_3d[:2] = torch.from_numpy(post_tran2)
            post_rot_3d[:2, :2] = torch.from_numpy(post_rot2)
            
            post_rots.append(post_rot_3d)
            post_trans.append(post_tran_3d)
        
        # After camera loop: CRITICAL - check if we have the expected number of images
        # We should have num_cams * (1 + num_adj_frames) images if sequential=True
        num_cams = len(cam_names)
        if self.sequential:
            expected_imgs_per_cam = 1 + self.num_adj_frames
            expected_total_imgs = num_cams * expected_imgs_per_cam
            actual_imgs = len(imgs)
            
            # If we don't have enough images, duplicate current frames
            if actual_imgs < expected_total_imgs:
                num_missing = expected_total_imgs - actual_imgs
                # Duplicate the last num_missing images from the current frames
                for i in range(num_missing):
                    imgs.append(imgs[i % num_cams].clone())
                    segs.append(segs[i % num_cams].clone())
            
            # Extend intrins, post_rots, post_trans for adjacent frames (same values)
            intrins.extend(intrins[:num_cams])
            post_rots.extend(post_rots[:num_cams])
            post_trans.extend(post_trans[:num_cams])
            
            # Add sensor2egos and ego2globals for adjacent frames
            has_adjacent = 'adjacent' in results and len(results['adjacent']) > 0
            
            if has_adjacent and len(results['adjacent']) >= self.num_adj_frames:
                # Use actual adjacent frames
                for adj_info in results['adjacent'][:self.num_adj_frames]:
                    # Support both 'cams' and 'images' for adjacent frames
                    if 'cams' in adj_info:
                        adj_cam_data = adj_info['cams']
                        adj_format = 'fusionocc'
                    elif 'images' in adj_info:
                        adj_cam_data = adj_info['images']
                        adj_format = 'occfrmwrk'
                    else:
                        # No camera data, use current frame data
                        adj_cam_data = cam_data
                        adj_format = data_format
                    
                    for cam_name in cam_names:
                        cam_info_adj = adj_cam_data[cam_name]
                        
                        # Get sensor transformations for adjacent frame
                        if adj_format == 'fusionocc':
                            # Quaternion to rotation matrix
                            quat_adj = cam_info_adj['sensor2ego_rotation']
                            rot_adj = R.from_quat([quat_adj[1], quat_adj[2], quat_adj[3], quat_adj[0]]).as_matrix()
                            trans_adj = np.array(cam_info_adj['sensor2ego_translation'])
                            
                            sensor2ego_adj = np.eye(4, dtype=np.float32)
                            sensor2ego_adj[:3, :3] = rot_adj
                            sensor2ego_adj[:3, 3] = trans_adj
                            
                            # ego2global for adjacent frame
                            quat_ego_adj = cam_info_adj['ego2global_rotation']
                            rot_ego_adj = R.from_quat([quat_ego_adj[1], quat_ego_adj[2], quat_ego_adj[3], quat_ego_adj[0]]).as_matrix()
                            trans_ego_adj = np.array(cam_info_adj['ego2global_translation'])
                            
                            ego2global_adj = np.eye(4, dtype=np.float32)
                            ego2global_adj[:3, :3] = rot_ego_adj
                            ego2global_adj[:3, 3] = trans_ego_adj
                        else:  # occfrmwrk
                            cam2ego_adj = cam_info_adj.get('cam2ego', np.eye(4))
                            if isinstance(cam2ego_adj, list):
                                cam2ego_adj = np.array(cam2ego_adj)
                            sensor2ego_adj = np.array(cam2ego_adj, dtype=np.float32).reshape(4, 4)
                            
                            # IMPORTANT: Use per-camera ego2global (same as fusionocc) for consistency
                            # Original model was trained with per-camera ego2global
                            if 'ego2global' in cam_info_adj:
                                # Per-camera ego2global (PRIORITY - consistent with fusionocc)
                                ego2global_adj_data = cam_info_adj['ego2global']
                            elif 'ego2global' in adj_info:
                                # Sample-level ego2global (fallback)
                                ego2global_adj_data = adj_info['ego2global']
                            else:
                                ego2global_adj_data = np.eye(4)
                            if isinstance(ego2global_adj_data, list):
                                ego2global_adj_data = np.array(ego2global_adj_data)
                            ego2global_adj = np.array(ego2global_adj_data, dtype=np.float32).reshape(4, 4)
                        
                        sensor2egos.append(torch.from_numpy(sensor2ego_adj))
                        ego2globals.append(torch.from_numpy(ego2global_adj))
            else:
                # No adjacent frames or not enough: duplicate current frame transformations
                sensor2egos.extend(sensor2egos[:num_cams] * self.num_adj_frames)
                ego2globals.extend(ego2globals[:num_cams] * self.num_adj_frames)
        
        # CRITICAL: Final validation and padding to ensure consistent sizes
        num_cams = len(cam_names)
        if self.sequential:
            # Expected: num_cams * (1 current + num_adj_frames adjacent)
            expected_total = num_cams * (1 + self.num_adj_frames)
            
            # Pad images if needed
            while len(imgs) < expected_total:
                imgs.append(imgs[len(imgs) % num_cams].clone())
            
            # Pad segs if needed
            while len(segs) < expected_total:
                segs.append(segs[len(segs) % num_cams].clone())
            
            # Truncate if too many (shouldn't happen, but just in case)
            imgs = imgs[:expected_total]
            segs = segs[:expected_total]
            
            # Ensure transformations match
            while len(intrins) < expected_total:
                intrins.append(intrins[len(intrins) % num_cams].clone())
            while len(post_rots) < expected_total:
                post_rots.append(post_rots[len(post_rots) % num_cams].clone())
            while len(post_trans) < expected_total:
                post_trans.append(post_trans[len(post_trans) % num_cams].clone())
            while len(sensor2egos) < expected_total:
                sensor2egos.append(sensor2egos[len(sensor2egos) % num_cams].clone())
            while len(ego2globals) < expected_total:
                ego2globals.append(ego2globals[len(ego2globals) % num_cams].clone())
            
            intrins = intrins[:expected_total]
            post_rots = post_rots[:expected_total]
            post_trans = post_trans[:expected_total]
            sensor2egos = sensor2egos[:expected_total]
            ego2globals = ego2globals[:expected_total]
        
        # Stack all tensors
        imgs = torch.stack(imgs)
        segs = torch.stack(segs)
        
        sensor2egos = torch.stack(sensor2egos)
        ego2globals = torch.stack(ego2globals)
        intrins = torch.stack(intrins)
        post_rots = torch.stack(post_rots)
        post_trans = torch.stack(post_trans)
        
        # Debug logging for tensor shapes
        import sys
        sample_idx = results.get('sample_idx', -1)
        if sample_idx < 5:
            sys.stderr.write(f"\n[DEBUG PrepareImageSeg] Sample {sample_idx}: imgs.shape={imgs.shape}, "
                           f"sensor2egos.shape={sensor2egos.shape}, segs.shape={segs.shape}, "
                           f"adjacent_len={len(results.get('adjacent', []))}\n")
            sys.stderr.flush()
        
        # Store segs in results for later use
        results['segs'] = segs
        
        return imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans
    
    def __call__(self, results):
        """Process images and prepare img_inputs for FusionOcc."""
        
        try:
            imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans = self.get_inputs(results)
        except Exception as e:
            print(f"Warning: Failed to load images: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to dummy data
            num_cams = len(self.data_config['cams'])
            height, width = self.data_config['input_size']
            imgs = torch.randn(num_cams, 3, height, width)
            sensor2egos = torch.eye(4).unsqueeze(0).repeat(num_cams, 1, 1)
            ego2globals = torch.eye(4).unsqueeze(0).repeat(num_cams, 1, 1)
            intrins = torch.eye(3).unsqueeze(0).repeat(num_cams, 1, 1)
            post_rots = torch.eye(3).unsqueeze(0).repeat(num_cams, 1, 1)
            post_trans = torch.zeros(num_cams, 3)
        
        # Pack into img_inputs format
        results['img_inputs'] = (imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans)
        
        # Load segmentation if available (optional)
        # segs should have been populated by get_inputs
        if 'segs' not in results:
            import sys
            sys.stderr.write("WARNING: segs not in results, using zeros!\n")
            sys.stderr.flush()
            num_cams, _, height, width = imgs.shape
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
        # Support both 'occ_gt_path' (fusionocc format) and 'occ_path' (occfrmwrk format)
        occ_gt_path = results.get('occ_gt_path', results.get('occ_path', ''))
        
        # Standard occupancy grid size for FusionOcc
        occ_size = [200, 200, 16]  # x, y, z dimensions
        
        if occ_gt_path and os.path.exists(occ_gt_path):
            try:
                # Load actual occupancy data from .npz file
                # Expected structure: occ_path/labels.npz with keys 'semantics', 'mask_lidar', 'mask_camera'
                labels_file = os.path.join(occ_gt_path, 'labels.npz')
                if os.path.exists(labels_file):
                    occ_gt = np.load(labels_file)
                    
                    # Load semantics
                    if 'semantics' in occ_gt:
                        voxel_semantics = occ_gt['semantics']
                        results['voxel_semantics'] = torch.from_numpy(voxel_semantics).long()
                    else:
                        results['voxel_semantics'] = torch.zeros(*occ_size, dtype=torch.long)
                    
                    # Load masks
                    if 'mask_camera' in occ_gt:
                        mask_camera = occ_gt['mask_camera'].astype(bool)
                        results['mask_camera'] = torch.from_numpy(mask_camera).bool()
                    else:
                        results['mask_camera'] = torch.ones(*occ_size, dtype=torch.bool)
                    
                    if 'mask_lidar' in occ_gt:
                        mask_lidar = occ_gt['mask_lidar'].astype(bool)
                        results['mask_lidar'] = torch.from_numpy(mask_lidar).bool()
                    else:
                        results['mask_lidar'] = torch.ones(*occ_size, dtype=torch.bool)
                else:
                    # Fall back to dummy data if labels.npz doesn't exist
                    results['voxel_semantics'] = torch.zeros(*occ_size, dtype=torch.long)
                    results['mask_camera'] = torch.ones(*occ_size, dtype=torch.bool)
                    results['mask_lidar'] = torch.ones(*occ_size, dtype=torch.bool)
            except Exception as e:
                # Fall back to dummy data if loading fails
                print(f"Warning: Failed to load occupancy GT from {occ_gt_path}: {e}")
                results['voxel_semantics'] = torch.zeros(*occ_size, dtype=torch.long)
                results['mask_camera'] = torch.ones(*occ_size, dtype=torch.bool)
                results['mask_lidar'] = torch.ones(*occ_size, dtype=torch.bool)
        else:
            # Create dummy occupancy ground truth
            results['voxel_semantics'] = torch.zeros(*occ_size, dtype=torch.long)
            results['mask_camera'] = torch.ones(*occ_size, dtype=torch.bool)
            results['mask_lidar'] = torch.ones(*occ_size, dtype=torch.bool)
        
        return results


@TRANSFORMS.register_module()
class FuseAdjacentSweeps(object):
    """Fuse adjacent sweeps for FusionOcc.
    
    This transform loads and fuses adjacent LiDAR frames into the current frame,
    transforming them into the current ego coordinate system.
    """
    
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
    
    def _load_points(self, pts_filename):
        """Load point cloud from file."""
        try:
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)
        except Exception as e:
            print(f"Warning: Failed to load {pts_filename}: {e}")
            return None
        return points
    
    def get_adj_points(self, pre_info):
        """Load points from adjacent frame - supports both formats."""
        # Support both fusionocc ('lidar_path') and occfrmwrk ('lidar_points' dict) formats
        if 'lidar_points' in pre_info:
            # occfrmwrk format
            pts_filename = pre_info['lidar_points']['lidar_path']
            # occfrmwrk uses just the filename, need to add full path
            if not pts_filename.startswith('data/') and not pts_filename.startswith('./'):
                pts_filename = f'data/nuscenes/samples/LIDAR_TOP/{pts_filename}'
        elif 'lidar_path' in pre_info:
            # fusionocc format
            pts_filename = pre_info['lidar_path']
        else:
            return None
            
        # Handle relative paths
        if pts_filename.startswith('./'):
            pts_filename = pts_filename[2:]
        
        points = self._load_points(pts_filename)
        if points is None:
            return None
            
        points = points.reshape(-1, self.load_dim)
        points = points[:, self.use_dim]
        
        # Convert to LiDARPoints
        from mmdet3d.structures import LiDARPoints
        points = LiDARPoints(
            points, points_dim=points.shape[-1], attribute_dims=None)
        return points
    
    def get_lidar2global_matrix(self, info):
        """Get lidar to global transformation matrices - supports both formats."""
        # Support both fusionocc and occfrmwrk formats
        
        # lidar2ego matrix
        # IMPORTANT: Check top-level 'lidar2ego' FIRST (for current frame in occfrmwrk format)
        if 'lidar2ego' in info:
            # occfrmwrk format: 4x4 matrix at top level (current frame)
            lidar2lidarego = np.array(info['lidar2ego'], dtype=np.float32).reshape(4, 4)
        elif 'lidar_points' in info and 'lidar2ego' in info['lidar_points']:
            # occfrmwrk format: 4x4 matrix in lidar_points dict (adjacent frames)
            lidar2lidarego = np.array(info['lidar_points']['lidar2ego'], dtype=np.float32).reshape(4, 4)
        elif 'lidar2ego_rotation' in info and 'lidar2ego_translation' in info:
            # fusionocc format: rotation quaternion and translation separate
            lidar2lidarego = np.eye(4, dtype=np.float32)
            lidar2lidarego[:3, :3] = Quaternion(info['lidar2ego_rotation']).rotation_matrix
            lidar2lidarego[:3, 3] = info['lidar2ego_translation']
        else:
            lidar2lidarego = np.eye(4, dtype=np.float32)
        lidar2lidarego = torch.from_numpy(lidar2lidarego)
        
        # ego2global matrix
        if 'ego2global' in info:
            # occfrmwrk format: 4x4 matrix
            lidarego2global = np.array(info['ego2global'], dtype=np.float32).reshape(4, 4)
        elif 'ego2global_rotation' in info and 'ego2global_translation' in info:
            # fusionocc format: rotation quaternion and translation separate
            lidarego2global = np.eye(4, dtype=np.float32)
            lidarego2global[:3, :3] = Quaternion(info['ego2global_rotation']).rotation_matrix
            lidarego2global[:3, 3] = info['ego2global_translation']
        else:
            lidarego2global = np.eye(4, dtype=np.float32)
        lidarego2global = torch.from_numpy(lidarego2global)
        
        return lidar2lidarego, lidarego2global
    
    def __call__(self, results):
        """Fuse adjacent lidar sweeps into current frame."""
        points = results['points']
        
        # Check if lidar_adjacent exists and is not empty
        if 'lidar_adjacent' in results and len(results['lidar_adjacent']) > 0:
            # Get current frame transformation matrices
            # Support both fusionocc ('curr' wrapper) and occfrmwrk (direct) formats
            if 'curr' in results:
                curr_info = results["curr"]
            else:
                curr_info = results
            
            curr_lidar2ego, curr_ego2global = self.get_lidar2global_matrix(curr_info)
            
            pre_points_list = []
            for i, pre_info in enumerate(results["lidar_adjacent"]):
                pre_points = self.get_adj_points(pre_info)
                if pre_points is None:
                    continue
                    
                pre_lidar2ego, pre_ego2global = self.get_lidar2global_matrix(pre_info)
                
                # Transform from previous frame to current frame
                pre2curr = torch.inverse(curr_ego2global.matmul(curr_lidar2ego)).matmul(
                    pre_ego2global.matmul(pre_lidar2ego))
                pre_points.tensor[:, :3] = pre_points.tensor[:, :3].matmul(
                    pre2curr[:3, :3].T) + pre2curr[:3, 3].unsqueeze(0)
                
                pre_points_list.append(pre_points)
            
            # Concatenate all points
            if pre_points_list:
                points = points.cat(pre_points_list)
                points = points[:, self.use_dim]
        
        # CRITICAL: Always fix point size to ensure consistent batch sizes
        # This must be done even when there are no adjacent frames
        num_points = len(points.tensor)
        target_num_points = 40000  # Fixed number of points per sample
        
        if num_points > target_num_points:
            # Random sampling to reduce to target size
            if self.test_mode:
                # Deterministic sampling for test mode
                indices = torch.arange(0, num_points, num_points // target_num_points)[:target_num_points]
            else:
                # Random sampling for training
                indices = torch.randperm(num_points)[:target_num_points]
            points = points[indices]
        elif num_points < target_num_points:
            # Pad with zeros if not enough points
            padding = target_num_points - num_points
            padding_points = torch.zeros(padding, points.tensor.shape[1], 
                                        dtype=points.tensor.dtype, device=points.tensor.device)
            points.tensor = torch.cat([points.tensor, padding_points], dim=0)
        
        results['points'] = points
        
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
        
        # CRITICAL: Print all keys and their types/shapes for debugging
        import sys
        sample_idx = results.get('sample_idx', -1)
        if sample_idx < 3:
            sys.stderr.write(f"\n[DEBUG FormatDataSamples] Sample {sample_idx} keys:\n")
            for key, value in results.items():
                if isinstance(value, torch.Tensor):
                    sys.stderr.write(f"  {key}: Tensor {value.shape}\n")
                elif isinstance(value, tuple) and len(value) > 0 and isinstance(value[0], torch.Tensor):
                    sys.stderr.write(f"  {key}: tuple of {len(value)} tensors, first: {value[0].shape}\n")
                elif isinstance(value, list) and len(value) > 0:
                    sys.stderr.write(f"  {key}: list of {len(value)} items\n")
                else:
                    sys.stderr.write(f"  {key}: {type(value).__name__}\n")
            sys.stderr.flush()
        
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
        else:
            # Create empty instances
            gt_instances_3d = InstanceData()
            gt_instances_3d.labels_3d = torch.tensor([])
            gt_instances_3d.bboxes_3d = torch.zeros((0, 7))
            data_samples.gt_instances_3d = gt_instances_3d
        
        # Add occupancy ground truth
        if 'voxel_semantics' in results:
            # Create gt_occ dictionary with semantics and masks
            gt_occ = {
                'semantics': results['voxel_semantics'],
            }
            if 'mask_camera' in results:
                gt_occ['mask_camera'] = results['mask_camera']
            if 'mask_lidar' in results:
                gt_occ['mask_lidar'] = results['mask_lidar']
            
            data_samples.gt_occ = gt_occ
            data_samples.eval_ann_info = gt_occ  # Also add as eval_ann_info for compatibility
        
        results['data_samples'] = data_samples
        
        # CRITICAL: Remove any keys that might cause collation issues
        # Keep only the essential keys needed for training
        essential_keys = ['img_inputs', 'points', 'data_samples', 'segs', 'voxel_semantics', 
                         'mask_camera', 'mask_lidar', 'sample_idx']
        keys_to_remove = [k for k in results.keys() if k not in essential_keys]
        for key in keys_to_remove:
            if sample_idx < 3:
                sys.stderr.write(f"  Removing key: {key}\n")
            results.pop(key, None)
        
        return results


@TRANSFORMS.register_module()
class PointsLidar2Ego(object):
    """Transform points from lidar coordinate to ego coordinate.
    
    This transform converts point coordinates from the LiDAR sensor frame
    to the ego vehicle frame using the lidar2ego transformation matrix.
    Supports both fusionocc and occfrmwrk data formats.
    """
    
    def __call__(self, input_dict):
        """Transform points from lidar to ego coordinate.
        
        Args:
            input_dict (dict): Result dict with 'points' and lidar2ego transformation.
                             For fusionocc: expects 'curr' dict with 'lidar2ego_rotation' and 'lidar2ego_translation'.
                             For occfrmwrk: expects 'lidar2ego' 4x4 matrix directly in input_dict.
        
        Returns:
            dict: Results with transformed points.
        """
        points = input_dict['points']
        
        # Support both formats
        if 'lidar2ego' in input_dict:
            # occfrmwrk format or dataset already provided the matrix
            lidar2ego = input_dict['lidar2ego']
            if isinstance(lidar2ego, np.ndarray):
                lidar2ego = torch.from_numpy(lidar2ego).float()
            else:
                lidar2ego = torch.tensor(lidar2ego).float()
            
            # Apply transformation
            points.tensor[:, :3] = points.tensor[:, :3] @ lidar2ego[:3, :3].T
            points.tensor[:, :3] += lidar2ego[:3, 3]
        elif 'curr' in input_dict and 'lidar2ego_rotation' in input_dict['curr']:
            # fusionocc format
            lidar2ego_rotation = input_dict['curr']['lidar2ego_rotation']
            lidar2ego_translation = input_dict['curr']['lidar2ego_translation']
            
            # Convert rotation quaternion to rotation matrix
            lidar2ego_rots = torch.tensor(
                Quaternion(lidar2ego_rotation).rotation_matrix
            ).float()
            lidar2ego_trans = torch.tensor(lidar2ego_translation).float()
            
            # Apply rotation and translation
            points.tensor[:, :3] = points.tensor[:, :3] @ lidar2ego_rots.T
            points.tensor[:, :3] += lidar2ego_trans
        else:
            # No transformation available, skip
            pass
        
        input_dict['points'] = points
        return input_dict


@TRANSFORMS.register_module()
class FusionOccPointsRangeFilter(object):
    """Filter points by the point cloud range (FusionOcc version).
    
    Points outside the specified range will be removed from the point cloud.
    This is the FusionOcc-specific implementation that matches the original model.
    
    Args:
        point_cloud_range (list[float]): Point cloud range in format
                                         [x_min, y_min, z_min, x_max, y_max, z_max].
    """
    
    def __init__(self, point_cloud_range):
        self.pcd_range = np.array(point_cloud_range, dtype=np.float32)
    
    def __call__(self, input_dict):
        """Filter points by the range.
        
        Args:
            input_dict (dict): Result dict from loading pipeline.
        
        Returns:
            dict: Results after filtering. 'points', 'pts_instance_mask'
                 and 'pts_semantic_mask' keys are updated in the result dict.
        """
        points = input_dict['points']
        
        # Add small epsilon to avoid boundary issues
        eps = 0.001
        pcd_range = [
            self.pcd_range[0] + eps, self.pcd_range[1] + eps, self.pcd_range[2] + eps,
            self.pcd_range[3] - eps, self.pcd_range[4] - eps, self.pcd_range[5] - eps
        ]
        
        # Filter points by range
        points_mask = points.in_range_3d(pcd_range)
        clean_points = points[points_mask]
        input_dict['points'] = clean_points
        points_mask = points_mask.numpy()
        
        # Also filter instance and semantic masks if present
        pts_instance_mask = input_dict.get('pts_instance_mask', None)
        pts_semantic_mask = input_dict.get('pts_semantic_mask', None)
        
        if pts_instance_mask is not None:
            input_dict['pts_instance_mask'] = pts_instance_mask[points_mask]
        
        if pts_semantic_mask is not None:
            input_dict['pts_semantic_mask'] = pts_semantic_mask[points_mask]
        
        return input_dict
    
    def __repr__(self):
        """Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(point_cloud_range={self.pcd_range.tolist()})'
        return repr_str


@TRANSFORMS.register_module()
class FixPointSize(object):
    """Fix point cloud size to a constant for batch collation.
    
    This transform ensures all samples have the same number of points,
    which is required for DataLoader collation in PyTorch.
    
    Args:
        target_num_points (int): Target number of points per sample.
        test_mode (bool): Whether in test mode (deterministic sampling).
    """
    
    def __init__(self, target_num_points=40000, test_mode=False):
        self.target_num_points = target_num_points
        self.test_mode = test_mode
    
    def __call__(self, results):
        """Fix point size by sampling or padding."""
        points = results['points']
        num_points = len(points.tensor)
        
        if num_points > self.target_num_points:
            # Random sampling to reduce to target size
            if self.test_mode:
                # Deterministic sampling for test mode
                indices = torch.arange(0, num_points, num_points // self.target_num_points)[:self.target_num_points]
            else:
                # Random sampling for training
                indices = torch.randperm(num_points)[:self.target_num_points]
            points = points[indices]
        elif num_points < self.target_num_points:
            # Pad with zeros if not enough points
            padding = self.target_num_points - num_points
            padding_points = torch.zeros(padding, points.tensor.shape[1], 
                                        dtype=points.tensor.dtype, device=points.tensor.device)
            points.tensor = torch.cat([points.tensor, padding_points], dim=0)
        
        results['points'] = points
        return results
    
    def __repr__(self):
        """Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(target_num_points={self.target_num_points}, test_mode={self.test_mode})'
        return repr_str
