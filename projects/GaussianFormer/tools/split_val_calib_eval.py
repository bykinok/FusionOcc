"""Split GaussianFormer's val pkl into a calibration-fitting split and an
eval split, for the `_calib_train`/`_calib_eval_before`/`_calib_eval` config
axis (see projects/GaussianFormer/README.md).

Mirrors tools/split_val_calib_eval.py's methodology (whole scenes assigned
to one split or the other, so temporal continuity within a scene is
preserved) but operates on GaussianFormer's own pkl schema --
`{'infos': {scene_token: [frame_dict, ...]}, 'metadata': [(scene_token, idx), ...]}`
-- which the shared tools/split_val_calib_eval.py's flat-list assumption
(`data['infos']`/`data['data_list']` as a list) does not support.

Usage:
  python projects/GaussianFormer/tools/split_val_calib_eval.py \\
    projects/GaussianFormer/data/nuscenes_cam/nuscenes_infos_val_sweeps_occ.pkl \\
    --out-calib projects/GaussianFormer/data/nuscenes_cam/nuscenes_infos_val_sweeps_occ_calib.pkl \\
    --out-eval  projects/GaussianFormer/data/nuscenes_cam/nuscenes_infos_val_sweeps_occ_eval.pkl \\
    --ratio 0.5
"""
import argparse
import mmengine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pkl')
    parser.add_argument('--out-calib', required=True)
    parser.add_argument('--out-eval', required=True)
    parser.add_argument('--ratio', type=float, default=0.5,
                         help='Fraction of scenes assigned to the calib split.')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    data = mmengine.load(args.pkl)
    scene_tokens = sorted(data['infos'].keys())

    import random
    rng = random.Random(args.seed)
    shuffled = scene_tokens[:]
    rng.shuffle(shuffled)
    n_calib = round(len(shuffled) * args.ratio)
    calib_scenes = set(shuffled[:n_calib])
    eval_scenes = set(shuffled[n_calib:])

    def subset(scenes):
        return {
            'infos': {t: data['infos'][t] for t in scenes},
            'metadata': [(t, i) for (t, i) in data['metadata'] if t in scenes],
        }

    calib_data = subset(calib_scenes)
    eval_data = subset(eval_scenes)

    mmengine.dump(calib_data, args.out_calib)
    mmengine.dump(eval_data, args.out_eval)
    print(f'{len(scene_tokens)} scenes -> calib: {len(calib_scenes)} scenes / '
          f'{len(calib_data["metadata"])} frames, eval: {len(eval_scenes)} scenes / '
          f'{len(eval_data["metadata"])} frames')


if __name__ == '__main__':
    main()
