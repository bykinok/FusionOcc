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

    def rotate(self, rotation, axis=None):
        """Rotate points with the given rotation matrix or angle.

        Args:
            rotation (float, np.ndarray, torch.Tensor): Rotation matrix
                or angle.
            axis (int): Axis to rotate at. Defaults to None.
        """
        if not isinstance(rotation, torch.Tensor):
            rotation = self.tensor.new_tensor(rotation)
        assert (
            rotation.shape == torch.Size([3, 3]) or rotation.numel() == 1
        ), f"invalid rotation shape {rotation.shape}"

        if axis is None:
            axis = self.rotation_axis

        if rotation.numel() == 1:
            rot_sin = torch.sin(rotation)
            rot_cos = torch.cos(rotation)
            if axis == 1:
                rot_mat_T = rotation.new_tensor(
                    [[rot_cos, 0, -rot_sin], [0, 1, 0], [rot_sin, 0, rot_cos]]
                )
            elif axis == 2 or axis == -1:
                rot_mat_T = rotation.new_tensor(
                    [[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]]
                )
            elif axis == 0:
                rot_mat_T = rotation.new_tensor(
                    [[0, rot_cos, -rot_sin], [0, rot_sin, rot_cos], [1, 0, 0]]
                )
            else:
                raise ValueError("axis should in range")
            rot_mat_T = rot_mat_T.T
        elif rotation.numel() == 9:
            rot_mat_T = rotation
        else:
            raise NotImplementedError
        self.tensor[:, :3] = self.tensor[:, :3] @ rot_mat_T

        self.tensor[:, 3:5] = self.tensor[:, 3:5] @ rot_mat_T[:2, :2]

        return rot_mat_T

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
