# Acknowledgments: https://github.com/tarashakhurana/4d-occ-forecasting
# Modified by Haisong Liu

import math
import copy
import numpy as np
import torch
from torch.utils.cpp_extension import load
from tqdm import tqdm
from prettytable import PrettyTable
import os

# Simplified print_log for compatibility
def print_log(msg, logger=None):
    if logger is None:
        print(msg)
    else:
        logger.info(msg)

# Load DVR extension
dvr_cpp_path = os.path.join(os.path.dirname(__file__), '..', 'libs', 'dvr', 'dvr.cpp')
dvr_cu_path = os.path.join(os.path.dirname(__file__), '..', 'libs', 'dvr', 'dvr.cu')
dvr = load("dvr", sources=[dvr_cpp_path, dvr_cu_path], verbose=True,
           extra_cuda_cflags=['-allow-unsupported-compiler'])

_pc_range = [-40, -40, -1.0, 40, 40, 5.4]
_voxel_size = 0.4
_occ_size = [200, 200, 16]

occ_class_names = [
    'others','barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk',
    'terrain', 'manmade', 'vegetation','free'
]

flow_class_names = [
    'bicycle', 'bus', 'car', 'construction_vehicle', 'motorcycle', 'pedestrian', 'trailer', 'truck'
]


# https://github.com/tarashakhurana/4d-occ-forecasting/blob/ff986082cd6ea10e67ab7839bf0e654736b3f4e2/test_fgbg.py#L29C1-L46C16
def get_rendered_pcds(origin, points, tindex, pred_dist):
    pcds = []

    for t in range(len(origin)):
        mask = (tindex == t)
        # skip the ones with no data
        if not mask.any():
            continue
        _pts = points[mask, :3]
        # use ground truth lidar points for the raycasting direction
        v = _pts - origin[t][None, :]
        d = v / np.sqrt((v ** 2).sum(axis=1, keepdims=True))
        pred_pts = origin[t][None, :] + d * pred_dist[mask][:, None]
        pcds.append(torch.from_numpy(pred_pts))

    return pcds


def meshgrid3d(occ_size, pc_range):
    W, H, D = occ_size

    xs = torch.linspace(0.5, W - 0.5, W).view(W, 1, 1).expand(W, H, D) / W
    ys = torch.linspace(0.5, H - 0.5, H).view(1, H, 1).expand(W, H, D) / H
    zs = torch.linspace(0.5, D - 0.5, D).view(1, 1, D).expand(W, H, D) / D
    xs = xs * (pc_range[3] - pc_range[0]) + pc_range[0]
    ys = ys * (pc_range[4] - pc_range[1]) + pc_range[1]
    zs = zs * (pc_range[5] - pc_range[2]) + pc_range[2]
    xyz = torch.stack((xs, ys, zs), -1)

    return xyz


def generate_lidar_rays():
    # prepare lidar ray angles
    pitch_angles = []
    for k in range(10):
        angle = math.pi / 2 - math.atan(k + 1)
        pitch_angles.append(-angle)

    # nuscenes lidar fov: [0.2107773983152201, -0.5439104895672159] (rad)
    while pitch_angles[-1] < 0.21:
        delta = pitch_angles[-1] - pitch_angles[-2]
        pitch_angles.append(pitch_angles[-1] + delta)

    lidar_rays = []
    for pitch_angle in pitch_angles:
        for azimuth_angle in np.arange(0, 360, 1):
            azimuth_angle = np.deg2rad(azimuth_angle)

            x = np.cos(pitch_angle) * np.cos(azimuth_angle)
            y = np.cos(pitch_angle) * np.sin(azimuth_angle)
            z = np.sin(pitch_angle)

            lidar_rays.append((x, y, z))

    return np.array(lidar_rays, dtype=np.float32)


def process_one_sample(sem_pred, lidar_rays, output_origin, flow_pred):
    # lidar origin in ego coordinate
    # lidar_origin = torch.tensor([[[0.9858, 0.0000, 1.8402]]])
    T = output_origin.shape[1]
    pred_pcds_t = []

    free_id = len(occ_class_names) - 1
    occ_pred = copy.deepcopy(sem_pred)
    occ_pred[sem_pred < free_id] = 1
    occ_pred[sem_pred == free_id] = 0
    occ_pred = torch.from_numpy(occ_pred).permute(2, 1, 0)
    occ_pred = occ_pred[None, None, :].contiguous().float()

    offset = torch.Tensor(_pc_range[:3])[None, None, :]
    scaler = torch.Tensor([_voxel_size] * 3)[None, None, :]

    lidar_tindex = torch.zeros([1, lidar_rays.shape[0]])

    for t in range(T):
        lidar_origin = output_origin[:, t:t + 1, :]  # [1, 1, 3]
        lidar_endpts = lidar_rays[None] + lidar_origin  # [1, 15840, 3]

        output_origin_render = ((lidar_origin - offset) / scaler).float()  # [1, 1, 3]
        output_points_render = ((lidar_endpts - offset) / scaler).float()  # [1, N, 3]
        output_tindex_render = lidar_tindex  # [1, N], all zeros

        with torch.no_grad():
            pred_dist, _, coord_index = dvr.render_forward(
                occ_pred.cuda(),
                output_origin_render.cuda(),
                output_points_render.cuda(),
                output_tindex_render.cuda(),
                [1, 16, 200, 200],
                "test"
            )
            pred_dist *= _voxel_size
            
            # CPU로 복사 후 CUDA 텐서 명시적 삭제
            pred_dist_full = pred_dist[0, :].cpu()  # [N] 형태 - get_rendered_pcds용
            pred_dist_cpu = pred_dist[0, :, None].cpu()  # [N, 1] 형태 - torch.cat용
            coord_index_cpu = coord_index[0, :, :].int().cpu()
            
            # CUDA 텐서 삭제
            del pred_dist, coord_index

        pred_pcds = get_rendered_pcds(
            lidar_origin[0].cpu().numpy(),
            lidar_endpts[0].cpu().numpy(),
            lidar_tindex[0].cpu().numpy(),
            pred_dist_full.numpy()  # [N] 형태로 전달
        )

        pred_flow = torch.from_numpy(flow_pred[coord_index_cpu[:, 0], coord_index_cpu[:, 1], coord_index_cpu[:, 2]])
        pred_label = torch.from_numpy(sem_pred[coord_index_cpu[:, 0], coord_index_cpu[:, 1], coord_index_cpu[:, 2]])[:, None]
        
        # pred_pcds[0]의 길이에 맞춰서 pred_dist_cpu를 필터링
        # coord_index_cpu는 전체 레이에 대한 것이므로, pred_pcds[0]과 길이가 다를 수 있음
        # 하지만 실제로는 coord_index_cpu의 모든 포인트가 pred_pcds[0]에 포함되어야 함
        # 원본 코드를 보면 pred_dist[0, :, None].cpu()를 그대로 사용하므로,
        # pred_pcds[0]의 길이가 coord_index_cpu의 길이와 같아야 함
        
        # 원본 코드와 동일하게: pred_dist_cpu는 [N, 1] 형태이고, 
        # pred_pcds[0]은 [N_t, 3] 형태인데, N_t == N이어야 함
        # 만약 다르다면, coord_index_cpu로 인덱싱된 것만 사용해야 함
        pred_pcds = torch.cat([pred_pcds[0], pred_label, pred_dist_cpu, pred_flow], dim=-1)

        pred_pcds_t.append(pred_pcds)

    pred_pcds_t = torch.cat(pred_pcds_t, dim=0)
    
    # occ_pred CUDA 텐서 삭제
    del occ_pred
    torch.cuda.synchronize()  # CUDA 작업 완료 대기

    return pred_pcds_t.numpy()


def calc_metrics(pcd_pred_list, pcd_gt_list):
    thresholds = [1, 2, 4]

    gt_cnt = np.zeros([len(occ_class_names)])
    pred_cnt = np.zeros([len(occ_class_names)])
    tp_cnt = np.zeros([len(thresholds), len(occ_class_names)])

    ave = np.zeros([len(thresholds), len(occ_class_names)])
    for i, cls in enumerate(occ_class_names):
        if cls not in flow_class_names:
            ave[:, i] = np.nan

    ave_count = np.zeros([len(thresholds), len(occ_class_names)])

    for pcd_pred, pcd_gt in zip(pcd_pred_list, pcd_gt_list):
        for j, threshold in enumerate(thresholds):
            # L1
            depth_pred = pcd_pred[:, 4]
            depth_gt = pcd_gt[:, 4]
            l1_error = np.abs(depth_pred - depth_gt)
            tp_dist_mask = (l1_error < threshold)

            for i, cls in enumerate(occ_class_names):
                cls_id = occ_class_names.index(cls)
                cls_mask_pred = (pcd_pred[:, 3] == cls_id)
                cls_mask_gt = (pcd_gt[:, 3] == cls_id)

                gt_cnt_i = cls_mask_gt.sum()
                pred_cnt_i = cls_mask_pred.sum()
                if j == 0:
                    gt_cnt[i] += gt_cnt_i
                    pred_cnt[i] += pred_cnt_i

                tp_cls = cls_mask_gt & cls_mask_pred  # [N]
                tp_mask = np.logical_and(tp_cls, tp_dist_mask)
                tp_cnt[j][i] += tp_mask.sum()

                # flow L2 error
                if cls in flow_class_names and tp_mask.sum() > 0:
                    gt_flow_i = pcd_gt[tp_mask, 5:7]
                    pred_flow_i = pcd_pred[tp_mask, 5:7]
                    flow_error = np.linalg.norm(gt_flow_i - pred_flow_i, axis=1)
                    ave[j][i] += np.sum(flow_error)
                    ave_count[j][i] += flow_error.shape[0]

    iou_list = []
    for j, threshold in enumerate(thresholds):
        iou_list.append((tp_cnt[j] / (gt_cnt + pred_cnt - tp_cnt[j]))[:-1])

    ave_list = ave[1][:-1] / ave_count[1][:-1]  # use threshold = 2

    return iou_list, ave_list


def main(sem_pred_list, sem_gt_list, flow_pred_list, flow_gt_list, lidar_origin_list, logger):
    torch.cuda.empty_cache()

    # generate lidar rays
    lidar_rays = generate_lidar_rays()
    lidar_rays = torch.from_numpy(lidar_rays)

    # 메트릭을 누적할 변수들 (calc_metrics의 로직을 여기로 이동)
    thresholds = [1, 2, 4]
    gt_cnt = np.zeros([len(occ_class_names)])
    pred_cnt = np.zeros([len(occ_class_names)])
    tp_cnt = np.zeros([len(thresholds), len(occ_class_names)])
    ave = np.zeros([len(thresholds), len(occ_class_names)])
    for i, cls in enumerate(occ_class_names):
        if cls not in flow_class_names:
            ave[:, i] = np.nan
    ave_count = np.zeros([len(thresholds), len(occ_class_names)])

    # 배치 단위로 처리 (메모리 절약)
    batch_size = 500  # 500개씩 처리 후 메모리 정리
    pcd_pred_batch, pcd_gt_batch = [], []
    
    for idx, (sem_pred, sem_gt, flow_pred, flow_gt, lidar_origins) in enumerate(tqdm(
            zip(sem_pred_list, sem_gt_list, flow_pred_list, flow_gt_list, lidar_origin_list), ncols=50)):
        sem_pred = np.reshape(sem_pred, [200, 200, 16])
        sem_gt = np.reshape(sem_gt, [200, 200, 16])
        flow_pred = np.reshape(flow_pred, [200, 200, 16, 2])
        flow_gt = np.reshape(flow_gt, [200, 200, 16, 2])

        pcd_pred = process_one_sample(sem_pred, lidar_rays, lidar_origins, flow_pred)
        pcd_gt = process_one_sample(sem_gt, lidar_rays, lidar_origins, flow_gt)

        # evalute on non-free rays
        valid_mask = (pcd_gt[:, 3] != len(occ_class_names) - 1)
        pcd_pred = pcd_pred[valid_mask]
        pcd_gt = pcd_gt[valid_mask]

        assert pcd_pred.shape == pcd_gt.shape
        pcd_pred_batch.append(pcd_pred)
        pcd_gt_batch.append(pcd_gt)
        
        # 배치 단위로 메트릭 계산 및 누적
        if (idx + 1) % batch_size == 0 or (idx + 1) == len(sem_pred_list):
            # 배치에 대한 메트릭 계산
            for pcd_pred_item, pcd_gt_item in zip(pcd_pred_batch, pcd_gt_batch):
                for j, threshold in enumerate(thresholds):
                    depth_pred = pcd_pred_item[:, 4]
                    depth_gt = pcd_gt_item[:, 4]
                    l1_error = np.abs(depth_pred - depth_gt)
                    tp_dist_mask = (l1_error < threshold)

                    for i, cls in enumerate(occ_class_names):
                        cls_id = occ_class_names.index(cls)
                        cls_mask_pred = (pcd_pred_item[:, 3] == cls_id)
                        cls_mask_gt = (pcd_gt_item[:, 3] == cls_id)

                        gt_cnt_i = cls_mask_gt.sum()
                        pred_cnt_i = cls_mask_pred.sum()
                        if j == 0:
                            gt_cnt[i] += gt_cnt_i
                            pred_cnt[i] += pred_cnt_i

                        tp_cls = cls_mask_gt & cls_mask_pred
                        tp_mask = np.logical_and(tp_cls, tp_dist_mask)
                        tp_cnt[j][i] += tp_mask.sum()

                        # flow L2 error
                        if cls in flow_class_names and tp_mask.sum() > 0:
                            gt_flow_i = pcd_gt_item[tp_mask, 5:7]
                            pred_flow_i = pcd_pred_item[tp_mask, 5:7]
                            flow_error = np.linalg.norm(gt_flow_i - pred_flow_i, axis=1)
                            ave[j][i] += np.sum(flow_error)
                            ave_count[j][i] += flow_error.shape[0]
            
            # 배치 처리 후 메모리 정리
            del pcd_pred_batch, pcd_gt_batch
            pcd_pred_batch, pcd_gt_batch = [], []
            torch.cuda.empty_cache()
            import gc
            gc.collect()  # Python garbage collection
        
        # 주기적으로 CUDA 메모리 정리 (매 100개 샘플마다)
        elif (idx + 1) % 100 == 0:
            torch.cuda.empty_cache()

    # 최종 메트릭 계산
    iou_list = []
    for j, threshold in enumerate(thresholds):
        iou_list.append((tp_cnt[j] / (gt_cnt + pred_cnt - tp_cnt[j]))[:-1])

    ave_list = ave[1][:-1] / ave_count[1][:-1]  # use threshold = 2

    table = PrettyTable([
        'Class Names',
        'IoU@1', 'IoU@2', 'IoU@4', 'AVE'
    ])
    table.float_format = '.3'

    for i in range(len(occ_class_names) - 1):
        table.add_row([
            occ_class_names[i],
            iou_list[0][i], iou_list[1][i], iou_list[2][i], ave_list[i]
        ], divider=(i == len(occ_class_names) - 2))

    table.add_row([
        'MEAN',
        np.nanmean(iou_list[0]),
        np.nanmean(iou_list[1]),
        np.nanmean(iou_list[2]),
        np.nanmean(ave_list)
    ])

    print_log(table, logger=logger)

    miou = np.nanmean(iou_list)
    mave = np.nanmean(ave_list)

    occ_score = miou * 0.9 + max(1 - mave, 0.0) * 0.1
    print_log('MIOU: {}'.format(miou), logger=logger)
    print_log('MAVE: {}'.format(mave), logger=logger)
    print_log('Occ score: {}'.format(occ_score), logger=logger)

    torch.cuda.empty_cache()

    return miou, mave, occ_score
