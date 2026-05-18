# Copyright (c) OpenMMLab. All rights reserved.
"""
Split validation pkl into val_calib and val_eval with temporal continuity.

Temporal models use sequential context, so we split by scene (not by individual frames).
Whole scenes are assigned to either calib or eval, so each split keeps consecutive
timestamps within each scene.

Input pkl: dict with 'infos' or 'data_list' (list of dicts with 'timestamp', 'scene_token').
Output: two pkl files with the same structure; infos/data_list are replaced with the split.

Scene detection priority:
  1. 'scene_token' or 'scene_id' field in info dict (standard mmdet3d format)
  2. sweeps==0 heuristic: NuScenes 첫 번째 keyframe은 이전 sweep이 없음 → scene 경계 자동 감지
  3. timestamp gap: 연속 샘플 간 간격이 임계값(기본 2 초) 이상이면 새 scene 으로 처리

Usage:
  python tools/split_val_calib_eval.py data/nuscenes/nuscenes_infos_val_sweep.pkl \\
    --out-calib data/nuscenes/nuscenes_infos_val_sweep_calib.pkl \\
    --out-eval  data/nuscenes/nuscenes_infos_val_sweep_eval.pkl \\
    --ratio 0.5
"""

import argparse
import os
from collections import defaultdict


def _patch_numpy_compat():
    """numpy 2.x pkl을 numpy 1.x 환경에서 로드할 수 있도록 sys.modules 패치."""
    import sys
    import numpy as np
    np_ver = tuple(int(x) for x in np.__version__.split('.')[:2])
    if np_ver < (2, 0):
        import numpy.core
        import numpy.core.numeric
        # numpy 2.x pkl이 참조하는 경로를 numpy 1.x 경로로 매핑
        aliases = {
            'numpy._core': numpy.core,
            'numpy._core.numeric': numpy.core.numeric,
            'numpy._core.multiarray': numpy.core.multiarray,
            'numpy._core.umath': numpy.core.umath,
            'numpy._core.fromnumeric': numpy.core.fromnumeric,
            'numpy._core.numerictypes': numpy.core.numerictypes,
            'numpy._core._methods': numpy.core._methods,
        }
        for key, mod in aliases.items():
            if key not in sys.modules:
                sys.modules[key] = mod


def load_pkl(path):
    import pickle
    _patch_numpy_compat()
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception:
        try:
            from mmengine.fileio import load
            return load(path)
        except Exception:
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
    """Return scene identifier (scene_token or scene_id), or None if absent."""
    token = info.get('scene_token') or info.get('scene_id')
    return token  # None when both are absent


def _assign_scene_ids_by_sweeps(infos):
    """
    sweeps==0 을 scene 시작점으로 간주해 연속 scene_id 부여.

    NuScenes 에서 각 scene 의 첫 keyframe 은 이전 sweep 이 없으므로
    sweeps 리스트가 비어 있다.  이 특성을 이용해 scene 경계를 감지한다.

    Returns list of int scene_ids, same length as infos.
    """
    scene_ids = []
    curr_id = 0
    for i, info in enumerate(infos):
        if i > 0 and len(info.get('sweeps', [])) == 0:
            curr_id += 1
        scene_ids.append(curr_id)
    return scene_ids


def _assign_scene_ids_by_gap(infos, gap_threshold_s=2.0):
    """
    연속 샘플 간 timestamp 차이가 gap_threshold_s 이상이면 새 scene 으로 처리.
    timestamp 는 마이크로초(us) 단위라고 가정.

    Returns list of int scene_ids, same length as infos.
    """
    scene_ids = []
    curr_id = 0
    prev_ts = None
    for info in infos:
        ts = get_timestamp(info)
        if prev_ts is not None and abs(ts - prev_ts) / 1e6 > gap_threshold_s:
            curr_id += 1
        scene_ids.append(curr_id)
        prev_ts = ts
    return scene_ids


def split_val_calib_eval(infos, ratio=0.5, gap_threshold_s=2.0):
    """
    Split infos into calib and eval with temporal continuity.

    Scene detection order:
      1. scene_token / scene_id 필드가 있으면 사용 (표준 mmdet3d 포맷)
      2. sweeps==0 휴리스틱: 첫 번째 keyframe 은 sweep 이 없음 → scene 경계
      3. timestamp gap: 연속 샘플 간 gap > gap_threshold_s 이면 새 scene

    - Sort scenes by first timestamp (global temporal order).
    - Assign whole scenes to calib (first ratio fraction) and eval (rest).
    - If only one scene detected: split that scene's frames by time.
    - ratio: fraction for calib by scene count (or frame count in single-scene case).

    Returns:
        (calib_infos, eval_infos, method_used)  each list sorted by timestamp.
    """
    if not infos:
        return [], [], 'empty'

    # ── 1. scene_token / scene_id 가 있는지 확인 ──────────────────────────────
    has_scene_field = any(get_scene(i) is not None for i in infos)

    if has_scene_field:
        method = 'scene_token'
        scene_to_infos = defaultdict(list)
        for i in infos:
            scene_to_infos[get_scene(i)].append(i)
    else:
        # ── 2. sweeps==0 휴리스틱 시도 ────────────────────────────────────────
        sweep_ids = _assign_scene_ids_by_sweeps(infos)
        n_scenes_sweep = sweep_ids[-1] + 1 if sweep_ids else 1

        if n_scenes_sweep > 1:
            method = 'sweeps==0'
            scene_to_infos = defaultdict(list)
            for sid, info in zip(sweep_ids, infos):
                scene_to_infos[sid].append(info)
        else:
            # ── 3. timestamp gap 폴백 ──────────────────────────────────────────
            sorted_infos = sorted(infos, key=get_timestamp)
            gap_ids = _assign_scene_ids_by_gap(sorted_infos, gap_threshold_s)
            n_scenes_gap = gap_ids[-1] + 1 if gap_ids else 1

            if n_scenes_gap > 1:
                method = f'timestamp_gap>{gap_threshold_s}s'
                scene_to_infos = defaultdict(list)
                for sid, info in zip(gap_ids, sorted_infos):
                    scene_to_infos[sid].append(info)
            else:
                # 마지막 수단: 전체를 단일 scene 으로 처리
                method = 'single_scene'
                scene_to_infos = {0: list(infos)}

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
        lst = scene_to_infos[scenes_ordered[0]]
        n = len(lst)
        n_calib = max(1, min(int(n * ratio), n - 1))
        calib_infos = lst[:n_calib]
        eval_infos = lst[n_calib:]
    else:
        n_calib = max(1, int(n_scenes * ratio))
        n_calib = min(n_calib, n_scenes - 1)
        calib_scenes = set(scenes_ordered[:n_calib])
        for scene in scenes_ordered:
            lst = scene_to_infos[scene]
            if scene in calib_scenes:
                calib_infos.extend(lst)
            else:
                eval_infos.extend(lst)

    return calib_infos, eval_infos, method


def main():
    parser = argparse.ArgumentParser(
        description='Split val pkl into val_calib and val_eval by scene (temporal continuity)'
    )
    parser.add_argument('pkl', help='Input validation pkl path (has infos or data_list)')
    parser.add_argument('--out-calib', type=str, default=None,
                        help='Output path for val_calib pkl (default: <dir>/<base>_calib.pkl)')
    parser.add_argument('--out-eval', type=str, default=None,
                        help='Output path for val_eval pkl (default: <dir>/<base>_eval.pkl)')
    parser.add_argument('--ratio', type=float, default=0.5,
                        help='Fraction of scenes for calib (default 0.5); rest for eval')
    parser.add_argument('--gap-threshold', type=float, default=2.0,
                        help='Timestamp gap (seconds) to treat as scene boundary (default 2.0)')
    args = parser.parse_args()

    data = load_pkl(args.pkl)
    infos = get_infos(data)

    if not infos:
        raise SystemExit('No infos in pkl.')

    # Preserve top-level structure (metadata, etc.)
    is_list = isinstance(data, list)
    template = data if isinstance(data, dict) else {'infos': infos}

    calib_infos, eval_infos, method = split_val_calib_eval(
        infos, ratio=args.ratio, gap_threshold_s=args.gap_threshold)
    print(f'Scene detection method: {method}')

    base = os.path.splitext(os.path.basename(args.pkl))[0]
    out_dir = os.path.dirname(os.path.abspath(args.pkl))
    out_calib = args.out_calib or os.path.join(out_dir, f'{base}_calib.pkl')
    out_eval = args.out_eval or os.path.join(out_dir, f'{base}_eval.pkl')

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
