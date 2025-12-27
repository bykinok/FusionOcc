#import open3d as o3d
import trimesh
import mmcv
import numpy as np
import numba as nb

from mmdet3d.registry import TRANSFORMS as PIPELINES
import yaml, os
import torch
from scipy import stats
from scipy.ndimage import zoom
from skimage import transform
import pdb
import torch.nn.functional as F
import copy

@PIPELINES.register_module(force=True)
class LoadOccupancy(object):

    def __init__(self, to_float32=True, use_semantic=False, occ_path=None, grid_size=[512, 512, 40], unoccupied=0,
            pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0], gt_resize_ratio=1, cal_visible=False, use_vel=False, 
            use_occ3d=False):
        self.to_float32 = to_float32
        self.use_semantic = use_semantic
        self.occ_path = occ_path
        self.cal_visible = cal_visible
        self.use_occ3d = use_occ3d

        self.grid_size = np.array(grid_size)
        self.unoccupied = unoccupied
        self.pc_range = np.array(pc_range)
        self.voxel_size = (self.pc_range[3:] - self.pc_range[:3]) / self.grid_size
        self.gt_resize_ratio = gt_resize_ratio
        self.use_vel = use_vel
    
    def __call__(self, results):
        # occ3d 형식 지원: pkl 파일의 occ_path 키 사용
        if self.use_occ3d and 'occ_path' in results:
            return self._load_occ3d(results)
        
        # 기존 nuScenes-Occupancy 형식
        rel_path = 'scene_{0}/occupancy/{1}.npy'.format(results['scene_token'], results['lidar_token'])
        if self.occ_path is None:
            raise ValueError("occ_path must be provided")
        full_path = os.path.join(self.occ_path, rel_path)
        #  [z y x cls] or [z y x vx vy vz cls]
        pcd = np.load(full_path)
        pcd_label = pcd[..., -1:]
        pcd_label[pcd_label==0] = 255
        pcd_np_cor = self.voxel2world(pcd[..., [2,1,0]] + 0.5)  # x y z
        untransformed_occ = copy.deepcopy(pcd_np_cor)  # N 4
        # bevdet augmentation
        pcd_np_cor = (results['bda_mat'] @ torch.from_numpy(pcd_np_cor).unsqueeze(-1).float()).squeeze(-1).numpy()
        pcd_np_cor = self.world2voxel(pcd_np_cor)

        # make sure the point is in the grid
        pcd_np_cor = np.clip(pcd_np_cor, np.array([0,0,0]), self.grid_size - 1)
        transformed_occ = copy.deepcopy(pcd_np_cor)
        pcd_np = np.concatenate([pcd_np_cor, pcd_label], axis=-1)

        # velocity
        if self.use_vel:
            pcd_vel = pcd[..., [3,4,5]]  # x y z
            pcd_vel = (results['bda_mat'] @ torch.from_numpy(pcd_vel).unsqueeze(-1).float()).squeeze(-1).numpy()
            pcd_vel = np.concatenate([pcd_np, pcd_vel], axis=-1)  # [x y z cls vx vy vz]
            results['gt_vel'] = pcd_vel

        # 255: noise, 1-16 normal classes, 0 unoccupied
        pcd_np = pcd_np[np.lexsort((pcd_np_cor[:, 0], pcd_np_cor[:, 1], pcd_np_cor[:, 2])), :]
        pcd_np = pcd_np.astype(np.int64)
        processed_label = np.ones(self.grid_size, dtype=np.uint8) * self.unoccupied
        processed_label = nb_process_label(processed_label, pcd_np)
        results['gt_occ'] = processed_label


        if self.cal_visible:
            visible_mask = np.zeros(self.grid_size, dtype=np.uint8)
            # camera branch
            if 'img_inputs' in results.keys():
                _, rots, trans, intrins, post_rots, post_trans = results['img_inputs'][:6]
                occ_uvds = self.project_points(torch.Tensor(untransformed_occ), 
                                                rots, trans, intrins, post_rots, post_trans)  # N 6 3
                N, n_cam, _ = occ_uvds.shape
                img_visible_mask = np.zeros((N, n_cam))
                img_h, img_w = results['img_inputs'][0].shape[-2:]
                for cam_idx in range(n_cam):
                    basic_mask = (occ_uvds[:, cam_idx, 0] >= 0) & (occ_uvds[:, cam_idx, 0] < img_w) & \
                                (occ_uvds[:, cam_idx, 1] >= 0) & (occ_uvds[:, cam_idx, 1] < img_h) & \
                                (occ_uvds[:, cam_idx, 2] >= 0)

                    basic_valid_occ = occ_uvds[basic_mask, cam_idx]  # M 3
                    M = basic_valid_occ.shape[0]  # TODO M~=?
                    basic_valid_occ[:, 2] = basic_valid_occ[:, 2] * 10
                    basic_valid_occ = basic_valid_occ.cpu().numpy()
                    basic_valid_occ = basic_valid_occ.astype(np.int16)  # TODO first round then int?
                    depth_canva = np.ones((img_h, img_w), dtype=np.uint16) * 2048
                    nb_valid_mask = np.zeros((M), dtype=bool)
                    nb_valid_mask = nb_process_img_points(basic_valid_occ, depth_canva, nb_valid_mask)  # M
                    img_visible_mask[basic_mask, cam_idx] = nb_valid_mask

                img_visible_mask = img_visible_mask.sum(1) > 0  # N  1:occupied  0: free
                img_visible_mask = img_visible_mask.reshape(-1, 1).astype(pcd_label.dtype) 

                img_pcd_np = np.concatenate([transformed_occ, img_visible_mask], axis=-1)
                img_pcd_np = img_pcd_np[np.lexsort((transformed_occ[:, 0], transformed_occ[:, 1], transformed_occ[:, 2])), :]
                img_pcd_np = img_pcd_np.astype(np.int64)
                img_occ_label = np.zeros(self.grid_size, dtype=np.uint8)
                voxel_img = nb_process_label(img_occ_label, img_pcd_np) 
                visible_mask = visible_mask | voxel_img
                results['img_visible_mask'] = voxel_img


            # lidar branch
            if 'points' in results.keys():
                pts = results['points'].tensor.cpu().numpy()[:, :3]
                pts_in_range = ((pts>=self.pc_range[:3]) & (pts<self.pc_range[3:])).sum(1)==3
                pts = pts[pts_in_range]
                pts = (pts - self.pc_range[:3])/self.voxel_size
                pts = np.concatenate([pts, np.ones((pts.shape[0], 1)).astype(pts.dtype)], axis=1) 
                pts = pts[np.lexsort((pts[:, 0], pts[:, 1], pts[:, 2])), :].astype(np.int64)
                pts_occ_label = np.zeros(self.grid_size, dtype=np.uint8)
                voxel_pts = nb_process_label(pts_occ_label, pts)  # W H D 1:occupied 0:free
                visible_mask = visible_mask | voxel_pts
                results['lidar_visible_mask'] = voxel_pts

            results['visible_mask'] = visible_mask

        return results

    def voxel2world(self, voxel):
        """
        voxel: [N, 3]
        """
        return voxel * self.voxel_size[None, :] + self.pc_range[:3][None, :]


    def world2voxel(self, wolrd):
        """
        wolrd: [N, 3]
        """
        return (wolrd - self.pc_range[:3][None, :]) / self.voxel_size[None, :]


    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}'
        return repr_str

    def project_points(self, points, rots, trans, intrins, post_rots, post_trans):
        
        # from lidar to camera
        points = points.reshape(-1, 1, 3)
        points = points - trans.reshape(1, -1, 3)
        inv_rots = rots.inverse().unsqueeze(0)
        points = (inv_rots @ points.unsqueeze(-1))
        
        # from camera to raw pixel
        points = (intrins.unsqueeze(0) @ points).squeeze(-1)
        points_d = points[..., 2:3]
        points_uv = points[..., :2] / points_d
        
        # from raw pixel to transformed pixel
        points_uv = post_rots[:, :2, :2].unsqueeze(0) @ points_uv.unsqueeze(-1)
        points_uv = points_uv.squeeze(-1) + post_trans[..., :2].unsqueeze(0)
        points_uvd = torch.cat((points_uv, points_d), dim=2)
        
        return points_uvd
    
    def _load_occ3d(self, results):
        """occ3d 형식 GT 로드 (pkl의 occ_path 키 사용)
        
        Occ3D 원본 클래스 정의:
        - 0: others (occupied but not classified)
        - 1-16: semantic classes (barrier, bicycle, bus, car, ...)
        - 17: free (empty space)
        
        변환 후 (SurroundOcc 방식):
        - 0: free (empty space)
        - 1-16: semantic classes (동일)
        - 17: others (occupied but not classified)
        """
        # breakpoint()
        occ_path = results['occ_path']
        
        # occ3d 데이터 로드
        occ_gt_path = os.path.join(occ_path, 'labels.npz')
            
        # labels.npz 로드
        occ_labels = np.load(occ_gt_path)
        occ_semantics = occ_labels['semantics']  # [200, 200, 16]
        
        # Camera mask 로드 (npz 파일에서 직접 확인)
        occ_cam_mask = None
        if 'mask_camera' in occ_labels.files:
            occ_cam_mask = occ_labels['mask_camera'].astype(bool)  # Explicit boolean conversion
        
        # Label 변환 (SurroundOcc 방식): 0→18→17 (others), 17→0 (free), 18→17 (others)
        # 이렇게 하면 free=0, semantic=1-16, others=17로 통일됨
        gt_occ = occ_semantics.copy().astype(np.int32)  # int32 for intermediate computation
        gt_occ[occ_semantics == 0] = 18   # others: 0 → 18 (temporary)
        gt_occ[occ_semantics == 17] = 0   # free: 17 → 0
        gt_occ[gt_occ == 18] = 17         # others: 18 → 17 (final)
        gt_occ = gt_occ.astype(np.uint8)
        
        # Store dense GT for evaluation (STCOcc metric compatible)
        gt_occ_dense = gt_occ.copy()
        
        # Apply camera mask for evaluation (invisible voxels → 255)
        if occ_cam_mask is not None:
            gt_occ_masked = gt_occ_dense.copy()
            # Use numpy.where for safer indexing (SurroundOcc style)
            gt_occ_masked = np.where(occ_cam_mask, gt_occ_dense, 255).astype(np.uint8)
            results['occ_3d_masked'] = gt_occ_masked  # Dense format with mask
        
        results['occ_3d'] = gt_occ_dense  # Full dense GT
        
        # BEVDet augmentation 적용 (with vectorization) - Training에서만 활성화
        if 'bda_mat' in results:
            # Vectorized sparse conversion (100-200x faster than Python loop)
            # Convert to torch for faster processing
            gt_occ_torch = torch.from_numpy(gt_occ)
            
            # Find all occupied voxels (non-free: class 1-17)
            # free=0을 제외한 모든 voxel
            occupied_coords = torch.nonzero(gt_occ_torch > 0, as_tuple=False)  # (N, 3)
            
            if len(occupied_coords) > 0:
                # Vectorized label extraction (all at once, no loop!)
                labels = gt_occ_torch[occupied_coords[:, 0], occupied_coords[:, 1], occupied_coords[:, 2]]  # (N,)
                
                # voxel to world coordinates (vectorized)
                pcd_np_cor = occupied_coords.numpy().astype(np.float32) + 0.5
                pcd_np_cor = self.voxel2world(pcd_np_cor)
                
                # Apply BDA augmentation (vectorized)
                pcd_np_cor = (results['bda_mat'] @ torch.from_numpy(pcd_np_cor).unsqueeze(-1).float()).squeeze(-1).numpy()
                pcd_np_cor = self.world2voxel(pcd_np_cor)
                
                # Clip to grid boundaries (ensure all coordinates are within valid range)
                pcd_np_cor = np.clip(pcd_np_cor, np.array([0,0,0]), self.grid_size - 1)
                
                # Combine coordinates and labels (vectorized)
                pcd_np = np.concatenate([pcd_np_cor, labels.numpy().reshape(-1, 1)], axis=-1)
                
                # Velocity processing (if needed, similar to nuScenes-Occupancy)
                # Note: occ3d doesn't provide velocity data, but keep structure for compatibility
                if self.use_vel:
                    # occ3d doesn't have velocity, so we skip this
                    # For future compatibility, keep the structure
                    pass
                
                # Sort by coordinates (required for nb_process_label)
                pcd_np = pcd_np[np.lexsort((pcd_np_cor[:, 0], pcd_np_cor[:, 1], pcd_np_cor[:, 2])), :]
                pcd_np = pcd_np.astype(np.int64)
                
                # Create dense grid (255: noise/ignore, 0-16: semantic classes, 17: others)
                processed_label = np.zeros(self.grid_size, dtype=np.uint8)  # free=0
                processed_label = nb_process_label(processed_label, pcd_np)
                gt_occ = processed_label
        else:
            # No BDA augmentation (test mode)
            # Sanity check: ensure gt_occ shape matches grid_size
            if tuple(gt_occ.shape) != tuple(self.grid_size):
                print(f"Warning: gt_occ shape {gt_occ.shape} != grid_size {self.grid_size}")
        
        results['gt_occ'] = gt_occ
        
        # visible mask 계산 (선택적)
        if self.cal_visible and occ_cam_mask is not None:
            results['visible_mask'] = occ_cam_mask.astype(np.uint8)
        
        return results
    
# b1:boolean, u1: uint8, i2: int16, u2: uint16
@nb.jit('b1[:](i2[:,:],u2[:,:],b1[:])', nopython=True, cache=True, parallel=False)
def nb_process_img_points(basic_valid_occ, depth_canva, nb_valid_mask):
    # basic_valid_occ M 3
    # depth_canva H W
    # label_size = M   # for original occ, small: 2w mid: ~8w base: ~30w
    canva_idx = -1 * np.ones_like(depth_canva, dtype=np.int16)
    for i in range(basic_valid_occ.shape[0]):
        occ = basic_valid_occ[i]
        if occ[2] < depth_canva[occ[1], occ[0]]:
            if canva_idx[occ[1], occ[0]] != -1:
                nb_valid_mask[canva_idx[occ[1], occ[0]]] = False

            canva_idx[occ[1], occ[0]] = i
            depth_canva[occ[1], occ[0]] = occ[2]
            nb_valid_mask[i] = True
    return nb_valid_mask

# u1: uint8, u8: uint16, i8: int64
@nb.jit('u1[:,:,:](u1[:,:,:],i8[:,:])', nopython=True, cache=True, parallel=False)
def nb_process_label_withvel(processed_label, sorted_label_voxel_pair):
    label_size = 256
    counter = np.zeros((label_size,), dtype=np.uint16)
    counter[sorted_label_voxel_pair[0, 3]] = 1
    cur_sear_ind = sorted_label_voxel_pair[0, :3]
    for i in range(1, sorted_label_voxel_pair.shape[0]):
        cur_ind = sorted_label_voxel_pair[i, :3]
        if not np.all(np.equal(cur_ind, cur_sear_ind)):
            processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
            counter = np.zeros((label_size,), dtype=np.uint16)
            cur_sear_ind = cur_ind
        counter[sorted_label_voxel_pair[i, 3]] += 1
    processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
    
    return processed_label


# u1: uint8, u8: uint16, i8: int64
@nb.jit('u1[:,:,:](u1[:,:,:],i8[:,:])', nopython=True, cache=True, parallel=False)
def nb_process_label(processed_label, sorted_label_voxel_pair):
    label_size = 256
    counter = np.zeros((label_size,), dtype=np.uint16)
    counter[sorted_label_voxel_pair[0, 3]] = 1
    cur_sear_ind = sorted_label_voxel_pair[0, :3]
    for i in range(1, sorted_label_voxel_pair.shape[0]):
        cur_ind = sorted_label_voxel_pair[i, :3]
        if not np.all(np.equal(cur_ind, cur_sear_ind)):
            processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
            counter = np.zeros((label_size,), dtype=np.uint16)
            cur_sear_ind = cur_ind
        counter[sorted_label_voxel_pair[i, 3]] += 1
    processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
    
    return processed_label