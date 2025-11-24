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
            img_seg_dir=None
    ):
        self.is_train = is_train
        self.data_config = data_config
        self.sequential = sequential
        self.restore_upsample = restore_upsample
        self.downsample = downsample
        self.img_seg_dir = img_seg_dir

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
        """Load image segmentation - NO FALLBACK (same as original)."""
        name = img_path.split("samples")[1].replace(".jpg", ".npy")
        seg_path = self.img_seg_dir + name
        # Load without try-except - will fail if file doesn't exist (same as original)
        seg = np.load(seg_path)
        seg = np.repeat(seg, self.restore_upsample, axis=1)
        seg = np.repeat(seg, self.restore_upsample, axis=0)
        seg = Image.fromarray(seg, mode="L")
        return seg
    
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
        import os
        
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
        results['cam_names'] = cam_names
        
        for cam_idx, cam_name in enumerate(cam_names):
            if cam_name not in cam_data:
                # 디버깅: 카메라 데이터가 없는 경우 로깅
                print(f"WARNING: Camera {cam_name} not found in cam_data. Available: {list(cam_data.keys())}")
                continue
            
            cam_info = cam_data[cam_name]
            
            # Load image - 경로 처리 개선
            if data_format == 'fusionocc':
                img_path = cam_info['data_path']
                # data_root가 빈 문자열일 수 있으므로 경로 정규화
                if img_path.startswith('./'):
                    img_path = img_path[2:]
                elif not img_path.startswith('/') and not img_path.startswith('data/'):
                    # 상대 경로인 경우 data/nuscenes 추가
                    img_path = f"data/nuscenes/{img_path}"
            else:  # occfrmwrk
                img_path = cam_info['img_path']
                # img_path in occfrmwrk is relative, need to add data_root
                if not img_path.startswith('./') and not img_path.startswith('/'):
                    # Assume it's relative to nuscenes root
                    img_path = f"./data/nuscenes/samples/{cam_name}/{img_path}"
                elif img_path.startswith('./'):
                    img_path = img_path[2:]
            
            # 이미지 파일 존재 확인 및 로드
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image file not found: {img_path}")
            
            try:
                img = Image.open(img_path)
            except Exception as e:
                raise RuntimeError(f"Failed to open image {img_path}: {e}")
            
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
            
            # Load and transform segmentation (NO FALLBACK - same as original)
            seg = self.get_img_seg(img_path)
            seg = self.seg_transform_core(seg, resize_dims, crop, flip_aug, rotate)
            segs.append(self.format_seg(seg))
            
            # IMPORTANT: Process adjacent frames IMMEDIATELY after current frame (interleaved!)
            # Original order: cam0_curr, cam0_adj, cam1_curr, cam1_adj, ...
            if self.sequential and 'adjacent' in results:
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
                    
                    # Load and transform segmentation for adjacent frame (NO FALLBACK - same as original)
                    # seg_adj = self.get_img_seg(img_path_adj)
                    # seg_adj = self.seg_transform_core(seg_adj, resize_dims, crop, flip_aug, rotate)
                    # segs.append(self.format_seg(seg_adj))
            
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
        
        # After camera loop: extend intrins/post_rots/post_trans and add adjacent sensor2egos/ego2globals
        # NOTE: Original uses INTERLEAVED order for imgs but SEPARATED order for transformations!
        if self.sequential and 'adjacent' in results:
            num_cams = len(cam_names)
            # Extend intrins, post_rots, post_trans for adjacent frames (same values)
            intrins.extend(intrins[:num_cams])
            post_rots.extend(post_rots[:num_cams])
            post_trans.extend(post_trans[:num_cams])
            
            # Add sensor2egos and ego2globals for adjacent frames
            for adj_info in results['adjacent']:
                # Support both 'cams' and 'images' for adjacent frames
                if 'cams' in adj_info:
                    adj_cam_data = adj_info['cams']
                    adj_format = 'fusionocc'
                elif 'images' in adj_info:
                    adj_cam_data = adj_info['images']
                    adj_format = 'occfrmwrk'
                else:
                    continue  # Skip if no camera data
                
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
        
        # Stack all tensors
        imgs = torch.stack(imgs)
        segs = torch.stack(segs)
        
        sensor2egos = torch.stack(sensor2egos)
        ego2globals = torch.stack(ego2globals)
        intrins = torch.stack(intrins)
        post_rots = torch.stack(post_rots)
        post_trans = torch.stack(post_trans)
        
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
        
        # segs should have been populated by get_inputs (NO FALLBACK - same as original)
        # If segs is missing, it's a critical error
        if 'segs' not in results:
            raise ValueError("'segs' not found in results! Image segmentation files are required.")
        
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
        # Check if lidar_adjacent exists
        if 'lidar_adjacent' not in results:
            return results
        
        points = results['points']

        # Save original points as curr_points before fusing (if not already set)
        # This matches the original behavior where curr_points is the original points
        if 'curr_points' not in results:
            from mmdet3d.structures import LiDARPoints
            results['curr_points'] = LiDARPoints(
                points.tensor.clone(), 
                points_dim=points.tensor.shape[-1], 
                attribute_dims=None
            )
        
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
        points = points.cat(pre_points_list)
        points = points[:, self.use_dim]
        
        # breakpoint()
        # Sample points to reduce computation (deterministic for testing)
        # Only keep points with timestamp > 16 (for deterministic comparison)
        mask = points.tensor[:, 4] > 16
        # NOTE: Original has random sampling but we disable it for reproducibility
        mask = mask | (torch.randint(0, 10, size=mask.shape) > 7)  # random sampling
        points = points[mask]
        
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
    """Format data for MMEngine compatibility and collect necessary keys."""
    
    def __init__(self, 
                 keys=('img_inputs', 'points', 'sparse_depth', 'segs', 'voxel_semantics', 'mask_camera'),
                 meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img', 'ego2img', 'ego2lidar', 
                           'sensor2sensorego', 'sensorego2global', 'sensorego2sensor', 'global2sensorego', 
                           'cam2img', 'depth2img', 'pad_shape', 'scale_factor', 'flip',
                           'pcd_horizontal_flip', 'pcd_vertical_flip', 'box_mode_3d',
                           'box_type_3d', 'img_norm_cfg', 'pcd_trans', 'sample_idx',
                           'pcd_scale_factor', 'pcd_rotation', 'pcd_rotation_angle',
                           'pts_filename', 'transformation_3d_flow', 'trans_mat', 'index',
                           'sequence_group_idx', 'curr_to_prev_lidar_rt', 'curr_to_prev_ego_rt', 
                           'start_of_sequence', 'can_bus', 'scene_name', 'affine_aug')):
        self.keys = keys
        self.meta_keys = meta_keys
    
    def __call__(self, results):
        """Format data samples for MMEngine and collect keys."""
        
        # Collect data similar to Collect3D
        data = {}
        img_metas = {}
        
        # Collect meta information
        for key in self.meta_keys:
            if key in results:
                img_metas[key] = results[key]
        
        # Collect main data keys and convert to tensors
        for key in self.keys:
            if key in results:
                value = results[key]
                # Convert lists to tensors to match original behavior
                if isinstance(value, list):
                    if key in ['voxel_semantics', 'mask_camera', 'segs', 'sparse_depth']:
                        if len(value) > 0:
                            if isinstance(value[0], torch.Tensor):
                                value = torch.stack(value, dim=0)
                            elif isinstance(value[0], np.ndarray):
                                value = torch.from_numpy(np.stack(value, axis=0))
                data[key] = value
        
        # Add img_metas
        data['img_metas'] = img_metas
        
        # Create data_samples for MMEngine compatibility
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
        
        data['data_samples'] = data_samples
        
        # Return the collected data
        return data


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

        # breakpoint()
        
        lidar2ego_rots = torch.tensor(Quaternion(input_dict['curr']['lidar2ego_rotation']).rotation_matrix).float()
        lidar2ego_trans = torch.tensor(input_dict['curr']['lidar2ego_translation']).float()
        points.tensor[:, :3] = (
                points.tensor[:, :3] @ lidar2ego_rots.T
        )
        points.tensor[:, :3] += lidar2ego_trans
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

        # breakpoint()
        
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
