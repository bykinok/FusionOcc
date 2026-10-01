# Copyright (c) OpenMMLab. All rights reserved.
"""Single shared GT-directory resolver.

STCOcc's Occ3D and OpenOcc GT releases share the same per-sample info pkl
(`occ_path` always points at the Occ3D 'gts' tree); OpenOcc GT lives in a
sibling 'openocc_v2' tree addressed by string substitution. Before this
module existed, this substitution was copy-pasted independently in the
training loader and in three places inside OccupancyMetric -- this is the
single place all of them should call, so GT identity cannot silently
diverge between training, inline evaluation, and file-based evaluation.
"""

GTS_DIR_NAME = 'gts'
OPENOCC_DIR_NAME = 'openocc_v2'


def resolve_occ_gt_dir(occ_path: str, dataset_name: str) -> str:
    """Resolve the GT *directory* for one sample given its dataset identity.

    Args:
        occ_path: the `occ_path` field from the shared info pkl, e.g.
            './data/nuscenes/gts/scene-0001/<token>' (always an Occ3D-style
            path regardless of which GT release is actually being used).
        dataset_name: 'occ3d' (no-op) or 'openocc' ('gts' -> 'openocc_v2').

    Returns:
        The directory to read `labels.npz` / `labels_1_{2,4,8}.npz` from.
    """
    if dataset_name == 'openocc':
        return occ_path.replace(GTS_DIR_NAME, OPENOCC_DIR_NAME)
    return occ_path
