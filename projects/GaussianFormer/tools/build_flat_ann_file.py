"""Build a flat, index-addressable companion pkl from GaussianFormer's own
nested pkl (`{'infos': {scene_token: [frame_dict, ...]}, 'metadata': [...]}`),
for tools that reload GT by integer index from a flat `ann_file`
(`projects/STCOcc/stcocc/evaluation/occupancy_metric.py::OccupancyMetric`,
used directly by `tools/compute_metrics_from_file.py`, and transitively by
`tools/test.py --save-predictions`'s SavePredictionsEvaluator + that same
metric class). That class only supports `data['data_list']`/`data['infos']`
as a flat LIST (`self.data_infos[index]`), which GaussianFormer's own nested
schema isn't.

Unlike the RayIoU metric's lidar-origin lookup (which needed nuscenes-devkit's
own val-sample ordering and was deliberately NOT built this way -- see
gaussianformer/evaluation/rayiou_metric.py's docstring), GT-reload-by-index
here only needs to line up with GaussianFormer's OWN dataset iteration
order, since `predict()`'s `index` field (dataset/nuscenes_dataset.py::
__getitem__) *is* that same position. So this script simply replicates
NuScenesDataset.__init__'s own sort
(`sorted(data['metadata'], key=lambda x: x[0] + zero-padded str(x[1]))`) and
writes one flat list in that exact order -- no external ordering dependency.

Usage:
  python projects/GaussianFormer/tools/build_flat_ann_file.py \\
    projects/GaussianFormer/data/nuscenes_cam/nuscenes_infos_val_sweeps_occ.pkl \\
    --out projects/GaussianFormer/data/nuscenes_cam/nuscenes_infos_val_sweeps_occ_flat.pkl
"""
import argparse
import mmengine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pkl')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    data = mmengine.load(args.pkl)
    keyframes = sorted(data['metadata'], key=lambda x: x[0] + "{:0>3}".format(str(x[1])))

    data_list = [data['infos'][scene_token][idx] for scene_token, idx in keyframes]
    mmengine.dump({'data_list': data_list}, args.out)
    print(f'{len(data_list)} frames -> {args.out}')


if __name__ == '__main__':
    main()
