# Copyright (c) OpenMMLab. All rights reserved.
import torch


def normalize_bbox(bboxes, pc_range):
    """Normalize bbox coordinates to [0, 1]."""
    cx = bboxes[..., 0:1]
    cy = bboxes[..., 1:2]
    cz = bboxes[..., 2:3]
    w = bboxes[..., 3:4].log()
    l = bboxes[..., 4:5].log()
    h = bboxes[..., 5:6].log()

    rot = bboxes[..., 6:7]
    if bboxes.size(-1) > 7:
        vx = bboxes[..., 7:8] 
        vy = bboxes[..., 8:9]
        normalized_bboxes = torch.cat(
            (cx, cy, cz, w, l, h, rot.sin(), rot.cos(), vx, vy), dim=-1
        )
    else:
        normalized_bboxes = torch.cat(
            (cx, cy, cz, w, l, h, rot.sin(), rot.cos()), dim=-1
        )
    return normalized_bboxes
