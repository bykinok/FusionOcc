# 실행 명령 모음 (research_waymo)

모든 명령은 로컬 호스트(2x RTX 3090) 기준이다 — Waymo 원본 데이터가 로컬에만 있기 때문(`smoke_test_results.md` 참고). conda env 활성화를 꼭 먼저 할 것(`source ~/miniconda3/etc/profile.d/conda.sh && conda activate occfrmwrk`) — 활성화 안 하면 `base` env로 떨어져 `ModuleNotFoundError: torch`/`mmengine` 등으로 바로 죽는다(이번 세션에 실제로 여러 번 겪음).

## 1. Waymo 전용 추가 데이터 준비 (이번 세션에서 실행 완료)

```bash
# coarse(0.4m) occupancy GT를 tar에서 추출 (training은 _04.npz만 선별 추출, validation은 이미 coarse-only라 전체 추출)
mkdir -p /home/h00323/DATA/Occ3D-Waymo/voxel04/training
cd /home/h00323/DATA/Occ3D-Waymo/voxel04/training
for f in /home/h00323/DATA/Occ3D-Waymo/voxel01/training/*.tar; do
  tar xf "$f" --wildcards '*_04.npz'
done
cd /home/h00323/DATA/Occ3D-Waymo/voxel04/validation
for f in /home/h00323/DATA/Occ3D-Waymo/voxel04/validation/*.tar; do tar xf "$f"; done

# data/waymo 심볼릭 링크 (data/nuscenes 관례와 동일)
mkdir -p data/waymo
ln -s /home/h00323/DATA/mmDataset/waymo/kitti_format data/waymo/kitti_format
ln -s /home/h00323/DATA/Occ3D-Waymo data/waymo/occ3d
```

## 2. STCOcc 호환 info pkl 생성 (실행 완료)

```bash
python3 tools/create_data_waymo_occ.py
# -> data/waymo/stcocc-waymo_infos_{train,val}.pkl
#    train: 47737 samples / 241 scenes (798 중 로컬에 있는 만큼만)
#    val:   39987 samples / 202 scenes (202 전체)
```

## 3. 멀티스케일 occupancy GT 생성 (실행 완료, ~45분 소요)

```bash
python3 tools/generate_ms_occ_waymo_parallel.py --split training --workers 28
python3 tools/generate_ms_occ_waymo_parallel.py --split validation --workers 28
```
로그: `research_waymo/logs/generate_ms_occ_waymo.log`. OpenOcc 때와 달리 mask 필드(`origin_voxel_state`/`final_voxel_state`)는 멀티스케일로 만들지 않았다 — baseline-none 프로필이 쓰지 않기 때문(속도상 약 2배 이득, `generate_ms_occ_waymo_parallel.py` 주석 참고). 필요해지면 그 코드에 다시 추가하면 된다.

## 4. Unit/config 검증 (GPU 불필요)

```bash
python3 -c "
from mmengine.config import Config
cfg = Config.fromfile('projects/STCOcc/configs/stcocc_r50_704x256_16f_occ3d_waymo_baseline.py')
print(cfg.num_classes, cfg.data_config['cams'])
"
```

## 5. GPU smoke test (실행 완료, 로컬 1x RTX 3090)

```bash
# 20-iteration 학습 (단일 scene, 1 GPU -- 2 GPU는 scene이 >=2개 필요, OpenOcc 때와 동일한 이유)
python3 -c "
import mmengine
d = mmengine.load('data/waymo/stcocc-waymo_infos_train.pkl')
small = dict(d); small['infos'] = [e for e in d['infos'] if e['waymo_scene_idx']==549]
mmengine.dump(small, 'data/waymo/stcocc-waymo_infos_train_smoke549.pkl')
"
bash tools/dist_train.sh projects/STCOcc/configs/stcocc_r50_704x256_16f_occ3d_waymo_baseline.py 1 \
  --cfg-options train_cfg.max_iters=20 default_hooks.checkpoint.interval=20 default_hooks.logger.interval=1 \
    train_dataloader.dataset.ann_file=data/waymo/stcocc-waymo_infos_train_smoke549.pkl \
    train_dataloader.batch_sampler.batch_size=2 \
  --work-dir work_dirs/waymo_baseline_smoke

# 24-sample mIoU 평가 (OccupancyMetric 우회, 신규 standalone 스크립트)
python3 -c "
import mmengine
d = mmengine.load('data/waymo/stcocc-waymo_infos_val.pkl')
small = dict(d); small['infos'] = [e for e in d['infos'] if e['waymo_scene_idx']==0][:24]
mmengine.dump(small, 'data/waymo/stcocc-waymo_infos_val_smoke24.pkl')
"
python3 tools/eval_waymo_smoke.py \
  projects/STCOcc/configs/stcocc_r50_704x256_16f_occ3d_waymo_baseline.py \
  work_dirs/waymo_baseline_smoke/iter_20.pth \
  --ann-file data/waymo/stcocc-waymo_infos_val_smoke24.pkl --limit 24
```

결과: `research_waymo/smoke_test_results.md`.

## 5b. OccupancyMetric 정식 배선 smoke test (실행 완료, mIoU + RayIoU 둘 다)

`OccupancyMetric`이 Waymo를 지원하게 됐으므로(§research_waymo/implementation_audit.md #6), 이제 `tools/eval_waymo_smoke.py` 없이 `tools/test.py`로 직접 평가 가능하다.

```bash
# mIoU (기본값) -- eval_waymo_smoke.py와 수치 완전 일치(0.0233) 확인됨
python3 tools/test.py \
  projects/STCOcc/configs/stcocc_r50_704x256_16f_occ3d_waymo_baseline.py \
  work_dirs/waymo_baseline_smoke/iter_20.pth \
  --cfg-options \
    test_dataloader.dataset.ann_file=data/waymo/stcocc-waymo_infos_val_smoke24.pkl \
    val_dataloader.dataset.ann_file=data/waymo/stcocc-waymo_infos_val_smoke24.pkl \
  --work-dir work_dirs/waymo_baseline_smoke_eval

# RayIoU (연구용 근사 프로토콜 -- research_waymo/remaining_issues.md #13 참고)
python3 tools/test.py \
  projects/STCOcc/configs/stcocc_r50_704x256_16f_occ3d_waymo_baseline.py \
  work_dirs/waymo_baseline_smoke/iter_20.pth \
  --cfg-options \
    test_dataloader.dataset.ann_file=data/waymo/stcocc-waymo_infos_val_smoke24.pkl \
    val_dataloader.dataset.ann_file=data/waymo/stcocc-waymo_infos_val_smoke24.pkl \
    val_evaluator.eval_metric=rayiou test_evaluator.eval_metric=rayiou \
  --work-dir work_dirs/waymo_baseline_smoke_rayiou
```

## 6. 전체 학습 (별도 승인 필요, 아직 실행 안 함)

```bash
# 로컬 2 GPU 사용 시 (원격 사이트에 데이터 동기화 안 했으므로)
bash tools/dist_train.sh projects/STCOcc/configs/stcocc_r50_704x256_16f_occ3d_waymo_baseline.py 2
```
`max_iters`는 config 기준 35796(47737 샘플 기준, **241/798 시퀀스만 포함된 부분 데이터셋**). 전체 798 시퀀스를 다 받아 학습할지는 `remaining_issues.md` #8 참고, 별도 승인 필요.

## 7. 전체 평가 (별도 승인 필요, 아직 실행 안 함)

`OccupancyMetric`이 이제 Waymo를 지원하므로(§5b), §5b의 명령에서 `ann_file`을 `data/waymo/stcocc-waymo_infos_val.pkl`(전체 39987 샘플)로 바꾸고 `--cfg-options`의 smoke 관련 override를 제거하면 된다. 다만 전체 평가는 비용이 크므로 실행 전 별도 승인 필요.
