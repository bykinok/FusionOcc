# P1 실행 계획 (계획만 — 미실행)

이 문서는 spec section 5 / template `stages.P1`을 이 repo에 맞게 정리한 것이다. **아래 어떤 job도 실행되지 않았다.** GPU 번호와 사용 한도가 지정되면 그때 실행한다.

---

## 0. P1 시작 전 선행 작업 (훈련 아님, 별도로 승인 가능)

| # | 작업 | 필요 자원 | 예상 소요 | 비고 |
|---|---|---|---|---|
| 0a | scene 기반 calibration(20%)/development(40%)/confirmation(40%) split 생성, seed=20260927 | CPU만, GPU 불필요 | 수분 | `research/splits/`에 token 목록+hash 저장. 학습 job이 아니므로 GPU 미지정 상태에서도 승인만 있으면 지금 실행 가능 |
| 0b | A0(nomask)/A1(cameramask) EMA 체크포인트 재평가 | GPU 1개, 각 ~2.5h (mIoU+RayIoU+AUROC+ECE+NLL 전체) | ~5h | F1(EMA vs raw) 실질 영향 확인 — **가장 우선순위 높음** |
| 0c | α_match 계산 (C2용, A2의 train-set 통계에서 도출) | CPU만 | 수분 | 학습 전 필요, GT 통계만 사용(spec 5.2 C2 공식) |

---

## 1. Anchor (재학습 불필요 — 이미 존재, P0에서 reusable로 분류됨)

| ID | 실험 | 상태 | run_id |
|---|---|---|---|
| A0 | w/o mask | REUSE | legacy_e12_nomask_95c07e1d_seed0_20260924 |
| A1 | w/ mask | REUSE | legacy_e12_cameramask_01ca9852_seed0_20260924 |
| A2 | oracle λ=.25 | REUSE | legacy_e12_invfree_l025_123aa016_seed0_20260926 |

세 anchor 모두 새로 학습하지 않는다. (0b의 EMA 재평가만 추가로 수행)

---

## 2. 신규 학습이 필요한 job (mandatory simple controls)

| ID | 실험 | 신규 학습 필요 | 비고 |
|---|---|---|---|
| C1 | global-free α=.25 (camera mask 전혀 미사용, 모든 valid free voxel에 weight=.25) | **예** | 새 config 필요 (`lambda_inv_free` 대신 global free-weight, camera_mask 의존성 완전 제거) |
| C2 | oracle mean-budget matched global-free (α_match, 0c에서 계산) | **예** | α_match 계산 후 진행 |
| C3 | shuffled-oracle free weighting (A2의 weight multiset을 free 위치 안에서만 permutation) | **예** | sample token+epoch+seed로 결정적 셔플 필요 |

조건부:
| ID | 조건 | 
|---|---|
| A4 (allterm_harddrop) | Lovász residual이 해석에 중요하다고 판단될 때만 1회 |
| free_logit_bias (offline) | A0 체크포인트에 대해 학습 없이 delta grid 평가, calibration split만 사용 |

**신규 학습 job 수: 3개 필수 (C1/C2/C3) + 조건부 최대 1개 (A4) + optional α 2개 = template의 `max_new_training_jobs: 7` 이내.**

---

## 3. 실행 명령 (현재 repo의 실제 entrypoint 기준 — `tools/research_pipeline.py`는 아직 구현되지 않았으므로 기존 `dist_train.sh`/`eval_all_metrics.py`를 직접 사용)

```bash
# (0a) split 생성 — 구현 필요한 스크립트, 아직 작성 안 함 (P1 첫 작업)
python research/tools/make_scene_split.py --seed 20260927 \
  --calibration 0.20 --development 0.40 --confirmation 0.40 \
  --out research/splits/

# (0b) EMA 재평가 (기존 eval_all_metrics.py 그대로 재사용, 체크포인트만 EMA로 교체)
python projects/STCOcc/tools/eval_all_metrics.py \
  --name stage1_nomask_EMA --miou-config projects/STCOcc/configs/stcocc_r50_704x256_16f_occ3d_e12_stage1_nomask.py \
  --rayiou-config projects/STCOcc/configs/stcocc_r50_704x256_16f_occ3d_e12_stage1_nomask_rayiou.py \
  --checkpoint /NAS/work_dirs/stcocc_r50_704x256_16f_occ3d_e12_stage1_nomask/iter_21096_ema.pth \
  --out-dir research/runs/stage1_nomask_ema_eval --gpu <GPU_ID> \
  --csv research/analysis/ema_vs_raw_summary.csv
# (동일하게 stage1_cameramask_EMA)

# (C1/C2/C3) 새 config 작성 필요 (아직 없음) 후:
bash tools/dist_train.sh projects/STCOcc/configs/stcocc_r50_704x256_16f_occ3d_p1_c1_global_alpha025.py 2 \
  --work-dir /NAS/work_dirs/stcocc_p1_c1_global_alpha025
# C2, C3도 동일 패턴, config만 다름
```

`tools/research_pipeline.py`의 `audit/test/plan/run/collect/analyze/report` 서브커맨드 계약(spec 2.2)은 **아직 구현되지 않았다** — 지금은 기존 스크립트를 직접 호출하는 방식으로 대체했다. 이 wrapper 자체를 만드는 것도 P1 초반 작업 중 하나로 잡을 수 있다 (사용자 확인 필요).

---

## 4. 필요 자원 정리

| 항목 | 값 | 출처 |
|---|---|---|
| 12-epoch, 2-GPU 학습 1건 실측 소요 시간 | 약 10.5~11시간 (wall-clock), GPU-hour로는 약 21~22 GPU-hour | 이번 세션에서 실제로 6회 측정한 평균 |
| 평가(mIoU+RayIoU+AUROC+ECE+NLL) 1건 소요 시간 | 약 2.5시간 | 이번 세션 실측 |
| 신규 학습 3건(C1/C2/C3) 총 GPU-hour | 약 63~66 GPU-hour | 3 × 21~22 |
| 로컬 사이트 가용 GPU | 2개 (idle 확인) | 이 감사에서 확인 |
| 원격 mando-h100_2/occfrmwrk_h100_2_new 가용 GPU | 2개 (H100 NVL, idle 확인, 별도 NAS) | 이 감사에서 확인 |
| 두 사이트 병행 시 예상 wall-clock | C1/C3를 로컬, C2를 원격(또는 임의 배분)에서 동시 진행 → 약 21~22시간 안에 3건 모두 완료 가능 | 사용자가 이번에 원격 사용을 요청함 |
| 디스크 (체크포인트, run당) | 약 0.68GB (raw 700MB + EMA 240MB, 중간 epoch 체크포인트 제외 시) | 실측 |
| 디스크 여유 (로컬/원격) | 30TB / 11TB | df -h 실측 |

---

## 5. 미설정 항목 (사용자 확인 필요 — 이게 없으면 실행하지 않음)

1. **GPU 번호** — 로컬 2개 중 어느 것을 P1 학습에 쓸지 (둘 다 idle이라 임의 선택 가능하지만 명시적 지정 필요), 원격 mando-h100_2도 마찬가지.
2. **max_gpu_hours** — 이번 P1 승인의 총 GPU-hour 상한.
3. **max_disk_gb** — 신규 run들이 쓸 수 있는 디스크 상한 (로컬/원격 각각, 또는 합산).
4. **`tools/research_pipeline.py` CLI wrapper를 지금 만들지, 아니면 기존 스크립트 직접 호출로 계속 갈지** — spec 2.2는 이 wrapper 구현을 요구하지만 P0 범위는 아니었음.
5. **0a(split 생성)를 지금 바로 실행해도 되는지** — GPU 불필요, 훈련도 아니라서 "학습을 시작하지 말라"는 제약에 안 걸린다고 판단되지만, 확인 후 진행하는 게 안전.
6. **C1/C2/C3용 새 config 3개를 지금 작성해둘지** — config 작성 자체는 파일 생성이라 학습이 아니지만, 사용자가 "P0만 수행"이라고 했으므로 이번 턴에서는 만들지 않았다.

이 6가지가 해결되면 바로 `--execute` 없이 dry-run부터 시작할 수 있다.
