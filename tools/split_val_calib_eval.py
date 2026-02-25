# Copyright (c) OpenMMLab. All rights reserved.
"""
Split validation pkl into val_calib and val_eval with temporal continuity.

Temporal models use sequential context, so we split by scene (not by individual frames).
Whole scenes are assigned to either calib or eval, so each split keeps consecutive
timestamps within each scene.

Input pkl: dict with 'infos' or 'data_list' (list of dicts with 'timestamp', 'scene_token').
Output: two pkl files with the same structure; infos/data_list are replaced with the split.

Usage:
  python tools/split_val_calib_eval.py data/nuscenes/occ_infos_temporal_val.pkl \\
    --out-calib data/nuscenes/occ_infos_temporal_val_calib.pkl \\
    --out-eval data/nuscenes/occ_infos_temporal_val_eval.pkl \\
    --ratio 0.5
"""

import argparse
import os
from collections import defaultdict


def load_pkl(path):
    try:
        from mmengine.fileio import load
        return load(path)
    except ImportError:
        import mmcv
        return mmcv.load(path)


def save_pkl(obj, path):
    import pickle
    os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def get_infos(data):
    """Return list of infos from pkl (supports 'infos' or 'data_list')."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if 'data_list' in data:
            return data['data_list']
        if 'infos' in data:
            return data['infos']
        raise KeyError(f"Pkl dict has no 'infos' or 'data_list'. Keys: {list(data.keys())}")
    raise TypeError(f"Expected dict or list, got {type(data)}")


def get_timestamp(info):
    """Return sort key for temporal order."""
    return info.get('timestamp', info.get('ts', 0))


def get_scene(info):
    """Return scene identifier (scene_token or scene_id)."""
    return info.get('scene_token', info.get('scene_id', id(info)))


def split_val_calib_eval(infos, ratio=0.5):
    """
    Split infos into calib and eval with temporal continuity.

    - Group samples by scene; sort each scene by timestamp.
    - Sort scenes by first timestamp (global temporal order).
    - Assign whole scenes to calib or eval: first ratio*N_scenes -> calib, rest -> eval,
      so each split has contiguous timestamps (no interleaving).
    - If there is only one scene: split that scene's frames by time (first ratio -> calib,
      rest -> eval) so both splits remain temporally consecutive.
    - ratio: fraction for calib (by scene count, or by frame count in single-scene case).

    Returns:
        calib_infos, eval_infos (each sorted by timestamp).
    """
    if not infos:
        return [], []

    # Group by scene
    scene_to_infos = defaultdict(list)
    for i in infos:
        scene_to_infos[get_scene(i)].append(i)

    # Sort samples within each scene by timestamp
    for scene in scene_to_infos:
        scene_to_infos[scene].sort(key=get_timestamp)

    # Sort scenes by their first sample's timestamp (global temporal order)
    scenes_ordered = sorted(
        scene_to_infos.keys(),
        key=lambda s: get_timestamp(scene_to_infos[s][0])
    )
    n_scenes = len(scenes_ordered)

    calib_infos = []
    eval_infos = []

    if n_scenes == 1:
        # Single scene: split by time (first ratio of frames -> calib, rest -> eval)
        lst = scene_to_infos[scenes_ordered[0]]
        n = len(lst)
        n_calib = max(1, min(int(n * ratio), n - 1))
        calib_infos = lst[:n_calib]
        eval_infos = lst[n_calib:]
    else:
        # Multiple scenes: assign whole scenes to calib / eval
        n_calib = max(1, int(n_scenes * ratio))
        n_calib = min(n_calib, n_scenes - 1)  # ensure eval gets at least one scene
        calib_scenes = set(scenes_ordered[:n_calib])
        for scene in scenes_ordered:
            lst = scene_to_infos[scene]
            if scene in calib_scenes:
                calib_infos.extend(lst)
            else:
                eval_infos.extend(lst)

    return calib_infos, eval_infos


def main():
    parser = argparse.ArgumentParser(
        description='Split val pkl into val_calib and val_eval by scene (temporal continuity)'
    )
    parser.add_argument('pkl', help='Input validation pkl path (has infos or data_list)')
    parser.add_argument('--out-calib', type=str, default=None,
                        help='Output path for val_calib pkl (default: <dir>/<base>_val_calib.pkl)')
    parser.add_argument('--out-eval', type=str, default=None,
                        help='Output path for val_eval pkl (default: <dir>/<base>_val_eval.pkl)')
    parser.add_argument('--ratio', type=float, default=0.5,
                        help='Fraction of scenes for calib (default 0.5); rest for eval')
    args = parser.parse_args()

    data = load_pkl(args.pkl)
    infos = get_infos(data)

    if not infos:
        raise SystemExit('No infos in pkl.')

    # Preserve top-level structure (metadata, etc.)
    is_list = isinstance(data, list)
    template = data if isinstance(data, dict) else {'infos': infos}

    calib_infos, eval_infos = split_val_calib_eval(infos, ratio=args.ratio)

    base = os.path.splitext(os.path.basename(args.pkl))[0]
    out_dir = os.path.dirname(os.path.abspath(args.pkl))
    out_calib = args.out_calib or os.path.join(out_dir, f'{base}_val_calib.pkl')
    out_eval = args.out_eval or os.path.join(out_dir, f'{base}_val_eval.pkl')

    if is_list:
        save_pkl(calib_infos, out_calib)
        save_pkl(eval_infos, out_eval)
    else:
        key = 'data_list' if 'data_list' in template else 'infos'
        out_data_calib = {k: v for k, v in template.items()}
        out_data_calib[key] = calib_infos
        out_data_eval = {k: v for k, v in template.items()}
        out_data_eval[key] = eval_infos
        save_pkl(out_data_calib, out_calib)
        save_pkl(out_data_eval, out_eval)

    print(f'Samples: calib={len(calib_infos)}, eval={len(eval_infos)}')
    print(f'Saved: {out_calib}')
    print(f'Saved: {out_eval}')


if __name__ == '__main__':
    main()
