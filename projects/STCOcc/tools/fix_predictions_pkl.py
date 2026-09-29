#!/usr/bin/env python3
"""One-off repair for predictions .pkl files written by the (now-fixed) buggy
SavePredictionsEvaluator: every array field carried a spurious leading batch-dim
of size 1 (e.g. occ_results shaped (1, 200, 200, 16) instead of (200, 200, 16)),
which made compute_metrics_from_file.py's GT-vs-prediction boolean indexing raise
"boolean index did not match indexed array along dimension 0" for ~every sample.

Streams the input file batch-by-batch (never holds the whole file in memory) and
writes a repaired copy with the leading dim squeezed off every array field.

Usage:
  python projects/STCOcc/tools/fix_predictions_pkl.py \
    --in /NAS/work_dirs/.../predictions_rank0.pkl \
    --out /NAS/work_dirs/.../predictions_rank0_fixed.pkl
"""
import argparse
import pickle
import time


def _squeeze(x):
    if hasattr(x, 'shape') and hasattr(x, '__getitem__') and len(getattr(x, 'shape', ())) > 0 and x.shape[0] == 1:
        return x[0]
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', required=True)
    ap.add_argument('--out', dest='out', required=True)
    args = ap.parse_args()

    n_batches = 0
    n_samples = 0
    t0 = time.time()
    with open(args.inp, 'rb') as fin, open(args.out, 'wb') as fout:
        while True:
            try:
                batch_list = pickle.load(fin)
            except EOFError:
                break
            for pred_dict in batch_list:
                for key in ('occ_results', 'flow_results', 'uncertainty_msp',
                            'uncertainty_entropy', 'softmax_probs'):
                    val = pred_dict.get(key)
                    if val is None:
                        continue
                    if isinstance(val, (list, tuple)):
                        pred_dict[key] = [_squeeze(v) for v in val]
                    else:
                        pred_dict[key] = _squeeze(val)
                n_samples += 1
            pickle.dump(batch_list, fout, protocol=pickle.HIGHEST_PROTOCOL)
            n_batches += 1
            if n_batches % 500 == 0:
                print(f"  ...{n_batches} batches / {n_samples} samples "
                      f"({time.time()-t0:.0f}s)", flush=True)

    print(f"Done: {n_batches} batches, {n_samples} samples, "
          f"{time.time()-t0:.0f}s -> {args.out}")


if __name__ == '__main__':
    main()
