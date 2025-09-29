# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch

from mmdet3d.structures.points import BasePoints


class RadarPoints(BasePoints):
    """Points of instances in RADAR coordinates.

    Args:
        tensor (torch.Tensor | np.ndarray | list): a N x points_dim matrix.
        points_dim (int): Number of the dimension of a point.
            Each row is (x, y, z, ...). Defaults to 3.
        attribute_dims (dict): Dictionary to indicate the meaning of extra
            dimension. Defaults to None.

    Attributes:
        tensor (torch.Tensor): Float matrix of N x points_dim.
        points_dim (int): Integer indicating the dimension of a point.
            Each row is (x, y, z, ...).
        attribute_dims (bool): Dictionary to indicate the meaning of extra
            dimension. Defaults to None.
        rotation_axis (int): Default rotation axis for points rotation.
    """

    def __init__(self,
                 tensor,
                 points_dim=3,
                 attribute_dims=None):
        super(RadarPoints, self).__init__(
            tensor, points_dim, attribute_dims)
        self.rotation_axis = 2

    def flip(self, bev_direction='horizontal', points=None):
        """Flip the points in BEV along given BEV direction.

        In RADAR coordinates, x and y axis are horizontal and vertical axis respectively.

        Args:
            bev_direction (str): Direction by which to flip. Can be chosen from
                'horizontal' and 'vertical'. Defaults to 'horizontal'.
            points (torch.Tensor, numpy.ndarray, :obj:`BasePoints`, optional):
                Points to flip. Defaults to None.

        Returns:
            torch.Tensor, numpy.ndarray or :obj:`BasePoints`: Flipped points.

        Note:
            The actual flip operation is defined in flip_func in base_points.py
        """
        assert bev_direction in ('horizontal', 'vertical')
        if bev_direction == 'horizontal':
            flip_axis = 1
        elif bev_direction == 'vertical':
            flip_axis = 0
        else:
            raise NotImplementedError
        if points is None:
            points = self.tensor

        assert points.shape[-1] >= 3, \
            f'Points size should be larger than 2, got {points.shape}'

        points = self.flip_func(points, flip_axis)

        return points

    def rotate(self, rotation, points=None):
        """Rotate points with the given rotation matrix or angle.

        Args:
            rotation (float, np.ndarray, torch.Tensor): Rotation matrix or angle.
            points (torch.Tensor, numpy.ndarray, :obj:`BasePoints`, optional):
                Points to rotate. Defaults to None.

        Returns:
            torch.Tensor, numpy.ndarray or :obj:`BasePoints`: Rotated points.
        """
        if points is None:
            points = self.tensor

        # Handle rotation as angle (float/scalar)
        if isinstance(rotation, (int, float)):
            # Create rotation matrix for 2D rotation around z-axis
            cos_r, sin_r = np.cos(rotation), np.sin(rotation)
            rotation_matrix = points.new_tensor([
                [cos_r, -sin_r, 0],
                [sin_r, cos_r, 0],
                [0, 0, 1]
            ])
            # Rotate XYZ coordinates
            rotated_points = torch.matmul(points[:, :3], rotation_matrix.T)
            # Keep other dimensions unchanged
            if points.shape[1] > 3:
                rotated_points = torch.cat([rotated_points, points[:, 3:]], dim=1)
            self.tensor = rotated_points
        else:
            # Handle rotation matrix
            if not isinstance(rotation, torch.Tensor):
                rotation = points.new_tensor(rotation)
            rotated_points = torch.matmul(points[:, :3], rotation.T)
            if points.shape[1] > 3:
                rotated_points = torch.cat([rotated_points, points[:, 3:]], dim=1)
            self.tensor = rotated_points

    def in_range_3d(self, point_range):
        """Check whether the points are in the given range.

        Args:
            point_range (list | torch.Tensor): The range of point
                (x_min, y_min, z_min, x_max, y_max, z_max)

        Note:
            In the original implementation of SECOND, checking whether
            a box in the range checks whether the points are in a convex
            polygon, we try to reduce the burden for simpler cases.

        Returns:
            torch.Tensor: A binary vector with the same length as the points.
        """
        in_range_flags = ((self.tensor[:, 0] > point_range[0])
                          & (self.tensor[:, 1] > point_range[1])
                          & (self.tensor[:, 2] > point_range[2])
                          & (self.tensor[:, 0] < point_range[3])
                          & (self.tensor[:, 1] < point_range[4])
                          & (self.tensor[:, 2] < point_range[5]))
        return in_range_flags

    @property
    def bev(self):
        """torch.Tensor: BEV of the points in shape (N, 2)."""
        return self.tensor[:, [0, 1]]

    @property  
    def height(self):
        """torch.Tensor: Height of the points in shape (N, )."""
        if self.tensor.shape[-1] > 2:
            return self.tensor[:, 2]
        else:
            return None
