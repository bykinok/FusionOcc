# Stage P0 보고

## 1. 이번에 답하려던 질문
기존 STCOcc invisible-free λ 스윕 구현이 CLAUDE_CODE_RESEARCH_SPEC.md의 엄격한 기준(loss 적용 범위, gradient 정확성, λ=1 동치성, 평가 코드 신뢰성, 기존 결과 재사용 가능성)을 통과하는가? 통과하지 못하면 무엇을 먼저 고쳐야 P1(단순 대안 검증)을 시작할 수 있는가?

## 2. 구현/설정과 기존 결과 대비 변경점
코드는 **변경하지 않았다** (감사만 수행). 새로 만든 것: `research/` 디렉터리 전체(plan.yaml, resolved_paths.json, protocol_manifest.json, loss_inventory.json, gradient_coverage.csv, experiment_registry.jsonl, decisions.jsonl, sources.md, audit_report.md, 그리고 CPU 전용 unit test 스크립트 1개). 기존 config/checkpoint/summary.csv는 전혀 덮어쓰지 않았다.

## 3. 핵심 표

| 항목 | 결과 |
|---|---|
| λ=1 equivalence (CE/sem_scal/geo_scal) | **PASS** (value·gradient bit-identical, 합성 데이터) |
| λ=0 partial gradient (같은 3개 loss) | **PASS** (invisible-free voxel의 직접 gradient 정확히 0) |
| Edge case 6종 (all-ignore 등) | 4 PASS / **2 FAIL** — `sem_scal_loss`가 all-ignore·no-occupied 배치에서 0-division (기존부터 있던 잠재 버그, 실제 배치에서는 미발생) |
| Lovász λ 적용 | 미적용 확인 (설계상 의도, 이미 사용자에게 공지됨) |
| live vs. file-based evaluator 일치 | **실증됨** (이번 세션 중 실제로 버그 발견→수정→mIoU 30.05 재현) |
| EMA vs raw checkpoint | **raw만 평가됨, EMA는 한 번도 평가 안 함** — 6개 run 전부 동일 |
| 기존 결과 6개 재사용 분류 | 전부 `reusable`(재학습 불필요), 단 EMA·split 문제로 `reevaluation_required` 태그 동반 |

## 4. 확인된 사실
- 기존 λ 스윕 구현의 gradient 성질(λ=1 동치, λ=0 부분 gradient)은 코드 읽기 수준이 아니라 **실제 forward/backward를 돌려서** 검증했고 통과했다.
- `sem_scal_loss`에 실제 존재하는 latent bug 1건을 새로 발견했다(실사용 영향 없음, 극단적 배치에서만 발생).
- 평가 파이프라인의 신뢰성(live vs. file-based)은 이번 세션에서 이미 실제로 검증된 바 있다(가정이 아니라 사실).
- **모든 기존 결과가 EMA가 아닌 raw checkpoint로 평가됐다** — 이건 이번 감사에서 새로 발견한, 절대 수치 해석에 영향을 줄 수 있는 사실이다.

## 5. 아직 검증되지 않은 해석
- EMA 대신 raw를 평가한 것이 λ 간 **상대적** 순위/트렌드에도 영향을 주는지는 미확인 (baseline 2개에서만이라도 EMA 재평가가 필요).
- `native_full_v1`(36 epoch) 후보 config는 찾았으나 `screen_e12_v1`과 다른 모든 axis(EMA, split, mask policy)에서 정확히 동일한지는 미확인.
- λ=0.25가 "가장 좋다"는 결론은 여전히 **retrospective, full-val, dev-exposed, seed 1회** 결과 위에 서 있다 — 새 split에서 재확인 전까지는 confirmation 근거가 아니다.

## 6. 다음 단계 판단
**P0 gate: PASS.** 방법론적 결함(core loss, mask 정렬, evaluator 정확성)은 발견되지 않았다. 다만 P1 시작 전에 우선순위로 처리할 두 가지를 권고한다: (1) EMA 체크포인트 최소 2개(baseline) 재평가로 F1의 실질적 영향 확인, (2) scene 기반 calibration/development/confirmation split 생성. **GPU 번호와 max_gpu_hours/max_disk_gb가 아직 지정되지 않아 P1의 실제 학습(anchor 재현, C1/C2/C3)은 실행하지 않았다** — 아래 P1 실행 계획은 계획서일 뿐 실행 결과가 아니다.
