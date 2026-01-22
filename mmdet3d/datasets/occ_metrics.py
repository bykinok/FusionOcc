import os
from functools import reduce

import numpy as np
from sklearn.neighbors import KDTree
from termcolor import colored

np.seterr(divide='ignore', invalid='ignore')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def pcolor(string, color, on_color=None, attrs=None):
    """
    Produces a colored string for printing

    Parameters
    ----------
    string : str
        String that will be colored
    color : str
        Color to use
    on_color : str
        Background color to use
    attrs : list of str
        Different attributes for the string

    Returns
    -------
    string: str
        Colored string
    """
    return colored(string, color, on_color, attrs)


def getCellCoordinates(points, voxelSize):
    return (points / voxelSize).astype(np.int)


def getNumUniqueCells(cells):
    M = cells.max() + 1
    return np.unique(cells[:, 0] + M * cells[:, 1] + M ** 2 * cells[:, 2]).shape[0]


class Metric_mIoU():
    def __init__(self,
                 save_dir='.',
                 num_classes=18,
                 use_lidar_mask=False,
                 use_image_mask=False,
                 ):
        self.class_names = ['others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
                            'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
                            'driveable_surface', 'other_flat', 'sidewalk',
                            'terrain', 'manmade', 'vegetation', 'free']
        self.save_dir = save_dir
        self.use_lidar_mask = use_lidar_mask
        self.use_image_mask = use_image_mask
        self.num_classes = num_classes

        self.point_cloud_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
        self.occupancy_size = [0.4, 0.4, 0.4]
        self.voxel_size = 0.4
        self.occ_xdim = int((self.point_cloud_range[3] - self.point_cloud_range[0]) / self.occupancy_size[0])
        self.occ_ydim = int((self.point_cloud_range[4] - self.point_cloud_range[1]) / self.occupancy_size[1])
        self.occ_zdim = int((self.point_cloud_range[5] - self.point_cloud_range[2]) / self.occupancy_size[2])
        self.voxel_num = self.occ_xdim * self.occ_ydim * self.occ_zdim
        self.hist = np.zeros((self.num_classes, self.num_classes))
        self.cnt = 0
        
        # Radius-based statistics
        self.radius_bins = [0, 20, 25, 30, 35, 40, 45, 50]
        # Calculate actual max radius from point cloud range
        self.max_radius = np.sqrt((self.point_cloud_range[3] - self.point_cloud_range[0])**2 / 4 + 
                                   (self.point_cloud_range[4] - self.point_cloud_range[1])**2 / 4)
        self.hist_by_radius = {f'{self.radius_bins[i]}-{self.radius_bins[i+1]}m': 
                               np.zeros((self.num_classes, self.num_classes)) 
                               for i in range(len(self.radius_bins) - 1)}
        
        # Height-based statistics
        # Use relative height bins based on actual z range
        # z_min = -1.0m, so we need to account for negative heights
        self.height_bins_relative = [0, 2, 4, 6]  # Relative height from ground (0m)
        # Adjust bins to actual z coordinates
        z_min = self.point_cloud_range[2]  # -1.0m
        self.height_bins = [z_min + h for h in self.height_bins_relative]  # [-1, 1, 3, 5]
        
        # Create labels for display (using relative heights)
        self.hist_by_height = {}
        for i in range(len(self.height_bins_relative) - 1):
            h_low = self.height_bins_relative[i]
            h_high = self.height_bins_relative[i + 1]
            label = f'{h_low}-{h_high}m'
            self.hist_by_height[label] = np.zeros((self.num_classes, self.num_classes))
    
    def reset(self):
        """Reset confusion matrix and count for new epoch."""
        self.hist = np.zeros((self.num_classes, self.num_classes))
        self.cnt = 0
        for key in self.hist_by_radius:
            self.hist_by_radius[key] = np.zeros((self.num_classes, self.num_classes))
        for key in self.hist_by_height:
            self.hist_by_height[key] = np.zeros((self.num_classes, self.num_classes))

    def hist_info(self, n_cl, pred, gt):
        """
        build confusion matrix
        # empty classes:0
        non-empty class: 0-16
        free voxel class: 17

        Args:
            n_cl (int): num_classes_occupancy
            pred (1-d array): pred_occupancy_label
            gt (1-d array): gt_occupancu_label

        Returns:
            tuple:(hist, correctly number_predicted_labels, num_labelled_sample)
        """
        assert pred.shape == gt.shape
        k = (gt >= 0) & (gt < n_cl)  # exclude 255
        labeled = np.sum(k)
        correct = np.sum((pred[k] == gt[k]))

        return (
            np.bincount(
                n_cl * gt[k].astype(int) + pred[k].astype(int), minlength=n_cl ** 2
            ).reshape(n_cl, n_cl),
            correct,
            labeled,
        )

    def per_class_iu(self, hist):

        return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

    def compute_mIoU(self, pred, label, n_classes):
        hist = np.zeros((n_classes, n_classes))
        new_hist, correct, labeled = self.hist_info(n_classes, pred.flatten(), label.flatten())
        hist += new_hist
        mIoUs = self.per_class_iu(hist)
        # for ind_class in range(n_classes):
        #     print(str(round(mIoUs[ind_class] * 100, 2)))
        # print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)))
        return round(np.nanmean(mIoUs) * 100, 2), hist

    def add_batch(self, semantics_pred, semantics_gt, mask_lidar, mask_camera):
        self.cnt += 1
        if self.use_image_mask:
            masked_semantics_gt = semantics_gt[mask_camera]
            masked_semantics_pred = semantics_pred[mask_camera]
            mask_used = mask_camera
        elif self.use_lidar_mask:
            masked_semantics_gt = semantics_gt[mask_lidar]
            masked_semantics_pred = semantics_pred[mask_lidar]
            mask_used = mask_lidar
        else:
            masked_semantics_gt = semantics_gt
            masked_semantics_pred = semantics_pred
            mask_used = None

        # # pred = np.random.randint(low=0, high=17, size=masked_semantics.shape)
        _, _hist = self.compute_mIoU(masked_semantics_pred, masked_semantics_gt, self.num_classes)
        self.hist += _hist
        
        # Calculate radius-based histograms
        # Create coordinate grids for voxels
        x_coords = np.arange(self.occ_xdim) * self.occupancy_size[0] + self.point_cloud_range[0] + self.occupancy_size[0] / 2
        y_coords = np.arange(self.occ_ydim) * self.occupancy_size[1] + self.point_cloud_range[1] + self.occupancy_size[1] / 2
        
        # Create meshgrid for x, y coordinates
        x_grid = np.repeat(x_coords[:, np.newaxis, np.newaxis], self.occ_ydim, axis=1)
        x_grid = np.repeat(x_grid, self.occ_zdim, axis=2)
        y_grid = np.repeat(y_coords[np.newaxis, :, np.newaxis], self.occ_xdim, axis=0)
        y_grid = np.repeat(y_grid, self.occ_zdim, axis=2)
        
        # Calculate radius for each voxel
        radius = np.sqrt(x_grid**2 + y_grid**2)
        
        # Apply mask if used
        if mask_used is not None:
            radius_masked = radius[mask_used]
            gt_masked = semantics_gt[mask_used]
            pred_masked = semantics_pred[mask_used]
        else:
            radius_masked = radius.flatten()
            gt_masked = semantics_gt.flatten()
            pred_masked = semantics_pred.flatten()
        
        # Accumulate histogram for each radius bin
        for i in range(len(self.radius_bins) - 1):
            r_min = self.radius_bins[i]
            r_max = self.radius_bins[i + 1]
            radius_key = f'{r_min}-{r_max}m'
            
            # Find voxels in this radius range
            # For the last bin, include all voxels >= r_min to avoid missing any
            if i == len(self.radius_bins) - 2:  # Last bin
                in_range = (radius_masked >= r_min)
            else:
                in_range = (radius_masked >= r_min) & (radius_masked < r_max)
            
            if np.sum(in_range) > 0:
                gt_in_range = gt_masked[in_range]
                pred_in_range = pred_masked[in_range]
                
                # Compute histogram for this radius range
                _, hist_radius = self.compute_mIoU(pred_in_range, gt_in_range, self.num_classes)
                self.hist_by_radius[radius_key] += hist_radius
        
        # Calculate height-based histograms
        # Create z coordinate grid for voxels
        z_coords = np.arange(self.occ_zdim) * self.occupancy_size[2] + self.point_cloud_range[2] + self.occupancy_size[2] / 2
        
        # Create meshgrid for z coordinates
        z_grid = np.repeat(z_coords[np.newaxis, np.newaxis, :], self.occ_xdim, axis=0)
        z_grid = np.repeat(z_grid, self.occ_ydim, axis=1)
        
        # Apply mask if used
        if mask_used is not None:
            height_masked = z_grid[mask_used]
            gt_masked_h = semantics_gt[mask_used]
            pred_masked_h = semantics_pred[mask_used]
        else:
            height_masked = z_grid.flatten()
            gt_masked_h = semantics_gt.flatten()
            pred_masked_h = semantics_pred.flatten()
        
        # Accumulate histogram for each height bin
        for i in range(len(self.height_bins) - 1):
            h_min = self.height_bins[i]  # Actual z coordinate
            h_max = self.height_bins[i + 1]  # Actual z coordinate
            
            # Use relative height labels for display
            h_min_rel = self.height_bins_relative[i]
            h_max_rel = self.height_bins_relative[i + 1]
            height_key = f'{h_min_rel}-{h_max_rel}m'
            
            # Find voxels in this height range (using actual z coordinates)
            # For the last bin, include all voxels >= h_min to avoid missing any
            if i == len(self.height_bins) - 2:  # Last bin
                in_range = (height_masked >= h_min)
            else:
                in_range = (height_masked >= h_min) & (height_masked < h_max)
            
            if np.sum(in_range) > 0:
                gt_in_range = gt_masked_h[in_range]
                pred_in_range = pred_masked_h[in_range]
                
                # Compute histogram for this height range
                _, hist_height = self.compute_mIoU(pred_in_range, gt_in_range, self.num_classes)
                self.hist_by_height[height_key] += hist_height

    def count_miou(self):
        # print("hist: ", self.hist)
        mIoU = self.per_class_iu(self.hist)
        
        # Calculate TP, FP, FN for each class
        TP = np.diag(self.hist)  # True Positives
        FP = self.hist.sum(0) - TP  # False Positives (column sum - TP)
        FN = self.hist.sum(1) - TP  # False Negatives (row sum - TP)
        
        # print(f'===> per class IoU of {self.cnt} samples:')
        # for ind_class in range(self.num_classes):
        #     print(f'===> {self.class_names[ind_class]} - IoU = ' + str(round(mIoU[ind_class] * 100, 2)))
        # assert cnt == num_samples, 'some samples are not included in the miou calculation'
        print(f'===> per class IoU of {self.cnt} samples:')
        for ind_class in range(self.num_classes - 1):
            print(f'===> {self.class_names[ind_class]} - IoU = {round(mIoU[ind_class] * 100, 2)}, '
                  f'TP = {int(TP[ind_class])}, FP = {int(FP[ind_class])}, FN = {int(FN[ind_class])}')

        print(f'===> mIoU of {self.cnt} samples: ' + str(round(np.nanmean(mIoU[:self.num_classes - 1]) * 100, 2)))
        
        # Verify that radius-based sums match total
        print('\n===> Verification: Radius-based sum vs Total')
        TP_radius_sum = np.zeros(self.num_classes)
        FP_radius_sum = np.zeros(self.num_classes)
        FN_radius_sum = np.zeros(self.num_classes)
        
        for radius_key in self.hist_by_radius.keys():
            hist_r = self.hist_by_radius[radius_key]
            TP_radius_sum += np.diag(hist_r)
            FP_radius_sum += hist_r.sum(0) - np.diag(hist_r)
            FN_radius_sum += hist_r.sum(1) - np.diag(hist_r)
        
        all_match = True
        for ind_class in range(self.num_classes - 1):
            if TP[ind_class] != TP_radius_sum[ind_class] or \
               FP[ind_class] != FP_radius_sum[ind_class] or \
               FN[ind_class] != FN_radius_sum[ind_class]:
                print(f'Class {self.class_names[ind_class]}: TP_diff={int(TP[ind_class]-TP_radius_sum[ind_class])}, '
                      f'FP_diff={int(FP[ind_class]-FP_radius_sum[ind_class])}, '
                      f'FN_diff={int(FN[ind_class]-FN_radius_sum[ind_class])}')
                all_match = False
        
        if all_match:
            print('✓ All radius-based sums match total (verification passed)')
        
        # Print radius-based statistics summary
        print('\n===> Radius-based mIoU Summary:')
        print(f'{"Radius Range":>15s} | {"mIoU":>8s}')
        print('-' * 27)
        for radius_key in sorted(self.hist_by_radius.keys(), key=lambda x: float(x.split('-')[0])):
            hist_r = self.hist_by_radius[radius_key]
            mIoU_r = self.per_class_iu(hist_r)
            miou_value = np.nanmean(mIoU_r[:self.num_classes - 1]) * 100
            print(f'{radius_key:>15s} | {miou_value:>7.2f}%')
        
        # Print detailed radius-based statistics
        print('\n===> Radius-based TP/FP/FN statistics:')
        for radius_key in sorted(self.hist_by_radius.keys(), key=lambda x: float(x.split('-')[0])):
            hist_r = self.hist_by_radius[radius_key]
            TP_r = np.diag(hist_r)
            FP_r = hist_r.sum(0) - TP_r
            FN_r = hist_r.sum(1) - TP_r
            
            # Calculate mIoU for this radius range
            mIoU_r = self.per_class_iu(hist_r)
            miou_value = np.nanmean(mIoU_r[:self.num_classes - 1]) * 100
            
            print(f'\n===> Radius range: {radius_key} - mIoU: {miou_value:.2f}%')
            for ind_class in range(self.num_classes - 1):
                if TP_r[ind_class] > 0 or FP_r[ind_class] > 0 or FN_r[ind_class] > 0:
                    iou_r = TP_r[ind_class] / (TP_r[ind_class] + FP_r[ind_class] + FN_r[ind_class]) if (TP_r[ind_class] + FP_r[ind_class] + FN_r[ind_class]) > 0 else 0
                    print(f'     {self.class_names[ind_class]:20s} - IoU = {round(iou_r * 100, 2):6.2f}, '
                          f'TP = {int(TP_r[ind_class]):6d}, FP = {int(FP_r[ind_class]):6d}, FN = {int(FN_r[ind_class]):6d}')
        
        # print(f'===> sample-wise averaged mIoU of {cnt} samples: ' + str(round(np.nanmean(mIoU_avg), 2)))
        
        # Verify height-based statistics
        print('\n===> Verification: Height-based sum vs Total')
        print(f'Height bins (actual z coords): {self.height_bins}')
        print(f'Height bins (relative to ground): {self.height_bins_relative}')
        
        TP_height_sum = np.zeros(self.num_classes)
        FP_height_sum = np.zeros(self.num_classes)
        FN_height_sum = np.zeros(self.num_classes)
        
        for height_key in self.hist_by_height.keys():
            hist_h = self.hist_by_height[height_key]
            TP_height_sum += np.diag(hist_h)
            FP_height_sum += hist_h.sum(0) - np.diag(hist_h)
            FN_height_sum += hist_h.sum(1) - np.diag(hist_h)
        
        all_match_h = True
        for ind_class in range(self.num_classes - 1):
            if TP[ind_class] != TP_height_sum[ind_class] or \
               FP[ind_class] != FP_height_sum[ind_class] or \
               FN[ind_class] != FN_height_sum[ind_class]:
                print(f'Class {self.class_names[ind_class]}: TP_diff={int(TP[ind_class]-TP_height_sum[ind_class])}, '
                      f'FP_diff={int(FP[ind_class]-FP_height_sum[ind_class])}, '
                      f'FN_diff={int(FN[ind_class]-FN_height_sum[ind_class])}')
                all_match_h = False
        
        if all_match_h:
            print('✓ All height-based sums match total (verification passed)')
        
        # Print height-based statistics summary
        print('\n===> Height-based mIoU Summary:')
        print(f'{"Height Range":>20s} | {"Actual Z":>15s} | {"mIoU":>8s}')
        print('-' * 48)
        for i, height_key in enumerate(sorted(self.hist_by_height.keys(), key=lambda x: float(x.split('-')[0]))):
            hist_h = self.hist_by_height[height_key]
            mIoU_h = self.per_class_iu(hist_h)
            miou_value = np.nanmean(mIoU_h[:self.num_classes - 1]) * 100
            
            # Get actual z range for this bin
            z_min = self.height_bins[i]
            if i == len(self.height_bins) - 2:  # Last bin
                z_max_str = f'{z_min:.1f}m+'
            else:
                z_max = self.height_bins[i + 1]
                z_max_str = f'{z_min:.1f}-{z_max:.1f}m'
            
            print(f'{height_key:>20s} | {z_max_str:>15s} | {miou_value:>7.2f}%')
        
        # Print detailed height-based statistics
        print('\n===> Height-based TP/FP/FN statistics:')
        for i, height_key in enumerate(sorted(self.hist_by_height.keys(), key=lambda x: float(x.split('-')[0]))):
            hist_h = self.hist_by_height[height_key]
            TP_h = np.diag(hist_h)
            FP_h = hist_h.sum(0) - TP_h
            FN_h = hist_h.sum(1) - TP_h
            
            # Calculate mIoU for this height range
            mIoU_h = self.per_class_iu(hist_h)
            miou_value = np.nanmean(mIoU_h[:self.num_classes - 1]) * 100
            
            # Get actual z range for this bin
            z_min = self.height_bins[i]
            if i == len(self.height_bins) - 2:  # Last bin
                z_range_str = f'{z_min:.1f}m+'
            else:
                z_max = self.height_bins[i + 1]
                z_range_str = f'{z_min:.1f}-{z_max:.1f}m'
            
            print(f'\n===> Height range: {height_key} (z: {z_range_str}) - mIoU: {miou_value:.2f}%')
            for ind_class in range(self.num_classes - 1):
                if TP_h[ind_class] > 0 or FP_h[ind_class] > 0 or FN_h[ind_class] > 0:
                    iou_h = TP_h[ind_class] / (TP_h[ind_class] + FP_h[ind_class] + FN_h[ind_class]) if (TP_h[ind_class] + FP_h[ind_class] + FN_h[ind_class]) > 0 else 0
                    print(f'     {self.class_names[ind_class]:20s} - IoU = {round(iou_h * 100, 2):6.2f}, '
                          f'TP = {int(TP_h[ind_class]):6d}, FP = {int(FP_h[ind_class]):6d}, FN = {int(FN_h[ind_class]):6d}')

        print("\nuse mask: ", self.use_image_mask)
        return self.class_names, np.around(mIoU, decimals=4), self.cnt


class Metric_FScore():
    def __init__(self,

                 leaf_size=10,
                 threshold_acc=0.6,
                 threshold_complete=0.6,
                 voxel_size=[0.4, 0.4, 0.4],
                 range=[-40, -40, -1, 40, 40, 5.4],
                 void=[17, 255],
                 use_lidar_mask=False,
                 use_image_mask=False, ) -> None:

        self.leaf_size = leaf_size
        self.threshold_acc = threshold_acc
        self.threshold_complete = threshold_complete
        self.voxel_size = voxel_size
        self.range = range
        self.void = void
        self.use_lidar_mask = use_lidar_mask
        self.use_image_mask = use_image_mask
        self.cnt = 0
        self.tot_acc = 0.
        self.tot_cmpl = 0.
        self.tot_f1_mean = 0.
        self.eps = 1e-8
    
    def reset(self):
        """Reset F-Score metrics for new epoch."""
        self.cnt = 0
        self.tot_acc = 0.
        self.tot_cmpl = 0.
        self.tot_f1_mean = 0.

    def voxel2points(self, voxel):
        # occIdx = torch.where(torch.logical_and(voxel != FREE, voxel != NOT_OBSERVED))
        # if isinstance(voxel, np.ndarray): voxel = torch.from_numpy(voxel)
        mask = np.logical_not(reduce(np.logical_or, [voxel == self.void[i] for i in range(len(self.void))]))
        occIdx = np.where(mask)

        points = np.concatenate((occIdx[0][:, None] * self.voxel_size[0] + self.voxel_size[0] / 2 + self.range[0], \
                                 occIdx[1][:, None] * self.voxel_size[1] + self.voxel_size[1] / 2 + self.range[1], \
                                 occIdx[2][:, None] * self.voxel_size[2] + self.voxel_size[2] / 2 + self.range[2]),
                                axis=1)
        return points

    def add_batch(self, semantics_pred, semantics_gt, mask_lidar, mask_camera):

        # for scene_token in tqdm(preds_dict.keys()):
        self.cnt += 1

        if self.use_image_mask:

            semantics_gt[mask_camera == False] = 255
            semantics_pred[mask_camera == False] = 255
        elif self.use_lidar_mask:
            semantics_gt[mask_lidar == False] = 255
            semantics_pred[mask_lidar == False] = 255
        else:
            pass

        ground_truth = self.voxel2points(semantics_gt)
        prediction = self.voxel2points(semantics_pred)
        if prediction.shape[0] == 0:
            accuracy = 0
            completeness = 0
            fmean = 0

        else:
            prediction_tree = KDTree(prediction, leaf_size=self.leaf_size)
            ground_truth_tree = KDTree(ground_truth, leaf_size=self.leaf_size)
            complete_distance, _ = prediction_tree.query(ground_truth)
            complete_distance = complete_distance.flatten()

            accuracy_distance, _ = ground_truth_tree.query(prediction)
            accuracy_distance = accuracy_distance.flatten()

            # evaluate completeness
            complete_mask = complete_distance < self.threshold_complete
            completeness = complete_mask.mean()

            # evalute accuracy
            accuracy_mask = accuracy_distance < self.threshold_acc
            accuracy = accuracy_mask.mean()

            fmean = 2.0 / (1 / (accuracy + self.eps) + 1 / (completeness + self.eps))

        self.tot_acc += accuracy
        self.tot_cmpl += completeness
        self.tot_f1_mean += fmean

    def count_fscore(self, ):
        base_color, attrs = 'red', ['bold', 'dark']
        print(pcolor('\n######## F score: {} #######'.format(self.tot_f1_mean / self.cnt), base_color, attrs=attrs))
