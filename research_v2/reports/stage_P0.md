# Stage P0-v2 보고

## 1. 이번에 답하려던 질문
v1 감사가 v2 스펙(더 엄격한 annotation·origin·evaluator provenance 요구)을 통과하는가? 특히 RayIoU origin 실제 정책, label validity와 visibility 의존성 분리가 새로 확인되어야 했다.

## 2. 변경점
코드는 여전히 변경하지 않았다(감사만). `research_v2/`를 새로 만들었고 `research/`(v1)는 전혀 건드리지 않았다. v1 CLAUDE_CODE_RESEARCH_SPEC.md 원문은 커밋 전에 덮어써져 복구 불가 — `research_v2/archive_v1/README.md`에 이 사실과 v1의 실제 산출물(=`research/`) 위치를 기록해뒀다.

## 3. 핵심 표

| 항목 | 결과 |
|---|---|
| RayIoU origin 실제 정책 | **신규 확인**: 같은 scene 내 과거+미래 pose를 모두 후보로 삼아 8개로 subsample, 모델의 실제 16-frame 입력창과 무관 |
| label validity vs. camera mask | 현재 w/mask baseline은 둘을 사실상 같은 메커니즘으로 구현(255 주입) — 새로 명시적으로 기록 |
| mask_lidar 사용 여부 | 원본 GT에 존재하지만 코드 어디에서도 사용 안 함 (grep으로 확인) |
| Lovász 잔존 비중 (λ=1 대비) | 합성 데이터에서 ~0.4% (방향은 확인됐으나 실제 배치 크기로 재확인 필요) |
| 신규 latent bug | `lovasz_softmax.py`의 `flatten_probas`에서 `UnboundLocalError` (실사용 미발생) |
| 기존 6개 run 재분류 | 전부 `reusable`, EMA·split 이슈는 v1과 동일하게 미해결 |

## 4. 확인된 사실
- RayIoU 평가가 실제로 미래 시점 pose를 origin으로 쓸 수 있다는 것이 **코드 확인**됐다(가정이 아님).
- `mask_lidar` 미사용, `ignore_index=255`가 데이터셋이 아니라 로컬 코드가 주입하는 값이라는 것도 확인됐다.
- Lovász UnboundLocalError는 재현 가능한 실제 버그이나 지금까지 학습에는 영향 없었다.

## 5. 아직 검증되지 않은 해석
- RayIoU origin이 실제로 λ-스윕 결과에 얼마나 영향을 주는지는 E2가 답할 질문이지 P0가 답한 게 아니다.
- Lovász 잔존 비중의 실제 학습 배치 크기는 미확인(합성 데이터만).

## 6. 다음 단계 판단
**P0-v2 gate: PASS.** E1/E2는 계획만 준비했다(`reports/E1_execution_plan.md`, `reports/E2_execution_plan.md`) — GPU 번호, 학습 budget뿐 아니라 v2에서 새로 요구하는 **평가 budget**(`max_evaluation_gpu_hours`, `max_evaluation_jobs`)도 미설정이라 어떤 expensive 평가도 실행하지 않았다.
