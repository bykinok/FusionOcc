# 실행 명령 모음 (research_openocc)

## 1. 멀티스케일 OpenOcc GT 생성 (이번 세션에서 실행 완료)

```bash
python3 tools/generate_ms_occ_parallel.py --dataset openocc \
    --pkl_path data/nuscenes/stcocc-nuscenes_infos_train.pkl --workers 28
python3 tools/generate_ms_occ_parallel.py --dataset openocc \
    --pkl_path data/nuscenes/stcocc-nuscenes_infos_val.pkl --workers 28
```
로그: `research_openocc/logs/generate_ms_occ_openocc.log`. 멱등적(idempotent) -- 이미
존재하는 `labels_1_2/1_4/1_8.npz`는 건너뛰므로 재실행해도 안전하다.

## 2. Unit test (CPU-only, GPU 불필요)

```bash
python3 research_openocc/tests/test_openocc_baseline.py
```
결과: `research_openocc/tests/results.csv`.

## 3. 기존 Occ3D 회귀 테스트 (이번 세션 코드 변경 후 재확인)

```bash
python3 research_v2/tests/test_adaptive_lambda_radius.py
python3 research_v2/tests/test_rayiou_decomposition_regression.py
# test_loss_gradient_properties_v2.py는 REPO_ROOT가 /workspace/FusionOcc로 하드코딩돼
# 있어 호스트에서 직접 실행하려면 경로 치환이 필요함 (컨테이너 안에서는 그대로 실행 가능):
python3 -c "
src = open('research_v2/tests/test_loss_gradient_properties_v2.py').read()
src = src.replace('/workspace/FusionOcc', '$(pwd)')
exec(compile(src, 'test_loss_gradient_properties_v2.py', 'exec'))
"
```

## 4. config 로드 검증 (GPU 불필요)

```bash
python3 -c "
from mmengine.config import Config
cfg = Config.fromfile('projects/STCOcc/configs/stcocc_r50_704x256_16f_openocc_occ_only_screen.py')
print(cfg.num_classes, cfg.train_cfg.max_iters)
"
```

## 5. 승인된 GPU에서 1-iteration backward + 소규모 평가 smoke test (아직 미실행, 승인 후 실행)

컨테이너 안에서 (ssh mando-h100 / occfrmwrk_h100_new, 또는 ssh mando-h100_2 /
occfrmwrk_h100_2_new), repository_root=/workspace/FusionOcc 기준:

```bash
# 2-GPU 분산 학습, 아주 짧게 (예: max_iters를 config override로 10~20으로 제한)
bash tools/dist_train.sh projects/STCOcc/configs/stcocc_r50_704x256_16f_openocc_occ_only_screen.py 2 \
    --cfg-options train_cfg.max_iters=20 default_hooks.checkpoint.interval=20

# 소규모 평가 (val 전체가 아니라 일부 샘플만 -- InfiniteGroupEachSampleInBatchSamplerEval은
# 전체 val을 순회하므로, 최초 승인 범위에서는 짧은 체크포인트로 한정된 iteration만 평가하거나
# 별도 작은 ann_file을 만들어 사용할 것. 장시간 전체 val 평가는 별도 승인 필요.)
bash tools/dist_test.sh projects/STCOcc/configs/stcocc_r50_704x256_16f_openocc_occ_only_screen.py \
    work_dirs/stcocc_r50_704x256_16f_openocc_occ_only_screen/iter_20.pth 2
```

결과는 `research_openocc/evaluation_smoke.json`에 기록한다 (실행 후 작성 예정, 아직 생성 안 함).

## 6. 전체 학습 (별도 승인 필요, 아직 실행 안 함)

```bash
bash tools/dist_train.sh projects/STCOcc/configs/stcocc_r50_704x256_16f_openocc_occ_only_screen.py 2
```
21096 iteration, effective batch 16, ~Occ3D screen 프로필과 동일 비용 규모로 예상
(실측 GPU-시간은 첫 수백 iteration 후 산정해 승인 요청 시 포함할 것).
