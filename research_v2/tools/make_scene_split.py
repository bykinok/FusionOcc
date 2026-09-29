"""Generate the calibration/development/confirmation scene-level split for research_v2.

Reuses tools/split_val_calib_eval.py's scene_token grouping (does not duplicate its
heuristics for scene detection -- this script requires scene_token to be present,
which it is for the STCOcc val pkl; see research_v2/reports/E1_execution_plan.md
investigation). Whole scenes are randomly shuffled (seeded) and assigned to
calibration (20%) / development (40%) / confirmation (40%) by scene count, per
research_v2/split_manifest.json's declared policy.

Usage:
  python research_v2/tools/make_scene_split.py \
    --pkl data/nuscenes/stcocc-nuscenes_infos_val.pkl \
    --seed 20260927 \
    --out research_v2/splits/
"""
import argparse
import hashlib
import json
import os
import pickle
import random
from collections import defaultdict


def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_pkl(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def get_infos(data):
    if isinstance(data, list):
        return data
    if 'data_list' in data:
        return data['data_list']
    return data['infos']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkl', default='data/nuscenes/stcocc-nuscenes_infos_val.pkl')
    ap.add_argument('--seed', type=int, default=20260927)
    ap.add_argument('--calibration-fraction', type=float, default=0.20)
    ap.add_argument('--development-fraction', type=float, default=0.40)
    ap.add_argument('--out', default='research_v2/splits/')
    args = ap.parse_args()

    data = load_pkl(args.pkl)
    infos = get_infos(data)
    is_list = isinstance(data, list)
    template = data if isinstance(data, dict) else None

    scene_to_infos = defaultdict(list)
    for info in infos:
        token = info.get('scene_token')
        if token is None:
            raise SystemExit(f"info missing scene_token -- this script requires it (found in {args.pkl})")
        scene_to_infos[token].append(info)

    for scene in scene_to_infos:
        scene_to_infos[scene].sort(key=lambda i: i.get('timestamp', 0))

    scene_tokens = sorted(scene_to_infos.keys())  # sort first for determinism pre-shuffle
    rng = random.Random(args.seed)
    rng.shuffle(scene_tokens)

    n_scenes = len(scene_tokens)
    n_calib = round(n_scenes * args.calibration_fraction)
    n_dev = round(n_scenes * args.development_fraction)
    calib_scenes = scene_tokens[:n_calib]
    dev_scenes = scene_tokens[n_calib:n_calib + n_dev]
    conf_scenes = scene_tokens[n_calib + n_dev:]

    splits = {
        'calibration': calib_scenes,
        'development': dev_scenes,
        'confirmation': conf_scenes,
    }

    os.makedirs(args.out, exist_ok=True)
    for name, scenes in splits.items():
        scene_set = set(scenes)
        split_infos = []
        for scene in scenes:
            split_infos.extend(scene_to_infos[scene])
        split_infos.sort(key=lambda i: (i.get('scene_token'), i.get('timestamp', 0)))

        if is_list:
            out_obj = split_infos
        else:
            key = 'data_list' if 'data_list' in template else 'infos'
            out_obj = {k: v for k, v in template.items()}
            out_obj[key] = split_infos

        out_path = os.path.join(args.out, f'stcocc-nuscenes_infos_val_{name}.pkl')
        save_pkl(out_obj, out_path)
        print(f'{name}: {len(scenes)} scenes, {len(split_infos)} samples -> {out_path}')

    manifest = {
        'source_pkl': os.path.abspath(args.pkl),
        'seed': args.seed,
        'partition_unit': 'scene',
        'total_scenes': n_scenes,
        'total_samples': len(infos),
        'fractions_requested': {
            'calibration': args.calibration_fraction,
            'development': args.development_fraction,
            'confirmation': 1.0 - args.calibration_fraction - args.development_fraction,
        },
        'actual_scene_counts': {k: len(v) for k, v in splits.items()},
        'actual_sample_counts': {
            name: sum(len(scene_to_infos[s]) for s in scenes)
            for name, scenes in splits.items()
        },
        'scene_tokens': {name: scenes for name, scenes in splits.items()},
        'split_hash_sha256': hashlib.sha256(
            json.dumps({k: sorted(v) for k, v in splits.items()}, sort_keys=True).encode()
        ).hexdigest(),
        'label_confirmation_as_pristine_test': False,
        'confirmation_caveat': (
            "retrospective: all 6 legacy runs (and the temperature=1.6136 config) already saw the "
            "full val set including whatever falls in this 'confirmation' partition when prior "
            "lambda selection happened. Never call this a fresh independent test set."
        ),
    }
    manifest_path = os.path.join(args.out, 'split_manifest_materialized.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f'manifest -> {manifest_path}')
    print(f'split_hash_sha256: {manifest["split_hash_sha256"]}')


if __name__ == '__main__':
    main()
