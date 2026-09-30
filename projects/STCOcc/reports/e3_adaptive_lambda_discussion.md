# STCOcc Invisible-Free Supervision — 전체 진행 상황 및 E3 결과 (2026-09-30)

**작성일**: 2026-09-30
**목적**: 지금까지의 연구 흐름(v1 baseline → v2 진단 → E3 adaptive λ 실험)을 교수님과 논의하기 위한 정리
**관련 문서**: `invisible_free_sweep_report.md`(v1 원본), `research_v2/reports/consolidated_facts_2026-09-29.md`(v2 사실 종합), `research_v2/reports/E3_adaptive_lambda_results.md`(이번 결과의 원본)

---

## 1. 출발점 요약 (v1)

STCOcc는 학습 시 camera visibility mask로 invisible voxel의 loss를 제외할지 선택 가능하다.

| Setting | mIoU (visible voxel만) | RayIoU mean |
|---|---|---|
| w/o camera mask | 30.05 | 0.370 |
| w/ camera mask | 37.03 | 0.302 |

w/mask는 mIoU가 높고 RayIoU가 낮다(그 반대도 성립) — 뚜렷한 trade-off. Invisible & free voxel에만 적용하는 loss weight `λ`를 0~1 사이로 스윕한 결과:

| λ | mIoU | RayIoU mean |
|---|---|---|
| 0.00 | 36.10 | 0.303 |
| **0.25** | **35.52** | **0.368** |
| 0.50 | 32.15 | 0.371 |
| 0.75 | 31.74 | 0.369 |
| 1.00 | 30.05 | 0.370 |

**λ=0.25가 지금까지 발견된 최선점**: RayIoU는 w/o-mask 수준(0.368 vs 0.370)까지 거의 회복하면서, mIoU는 35.52로 w/mask(37.03)에 가깝게 유지된다. naive interpolation(두 baseline을 잇는 직선)보다도 명확히 우위에 있다.

## 2. v2 진단 단계에서 확인한 사실 (요약)

전체 내용은 `research_v2/reports/consolidated_facts_2026-09-29.md` 참고. 핵심만:

- RayIoU origin(라이다 ray 시작점)은 모델의 실제 16-frame 입력창과 무관하게 과거+미래 pose에서 뽑힌다 — 이게 RayIoU 수치를 왜곡할 수 있다는 우려가 있었다.
- **실측 결과, future-origin은 오히려 RayIoU가 가장 낮은(어려운) 조건이었다** (3개 checkpoint 전체 val 기준 reference > past > future 일관) — 즉 RayIoU가 future-origin 덕에 부풀려졌다는 근거는 없다.
- cameramask(w/mask) 체크포인트는 invisible 영역의 free 여부를 native decode에서 거의 못 맞춘다(invisible_free_iou 0.327, nomask/λ=0.25는 0.92~0.96) — post-hoc bias 보정으로도 이 격차는 못 메운다. 이는 학습 시 invisible 영역에 loss를 전혀 안 준 것과 정확히 일치하는 결과다.

## 3. E3 — 거리 기반 adaptive λ 실험 (이번 라운드, 2026-09-29~30)

### 동기
flat(전역 상수) λ의 최선점은 0.25다. "근거리/원거리에 다른 λ를 주면 이 최선점을 넘어설 수 있는가?"를 확인하기 위해, 20m(RayIoU의 거리 구간 경계와 동일)를 기준으로 근거리/원거리에 별도의 λ를 주는 방식을 구현하고 2가지 설계를 병렬 학습(로컬/원격 GPU 각각)했다.

### 결과

| | mIoU | RayIoU mean |
|---|---|---|
| flat λ=0.25 (기존 최선) | 35.52 | 0.368 |
| flat λ=0.75 | 31.74 | 0.369 |
| flat λ=1.00 | 30.05 | 0.370 |
| adaptive 근거리0.10/원거리0.50 | 31.12 | **0.348** |
| adaptive 근거리0.10/원거리1.00 | 31.41 | **0.351** |

**두 adaptive 설계 모두 실패했다.** mIoU가 비슷한 flat λ=0.75/1.00과 비교해도 RayIoU가 더 낮다 — 즉 새로운 trade-off 지점이 아니라 flat 값에 의해 완전히 열세(dominate)인 나쁜 점이다.

### 실패 원인 추정 (미검증)
근거리 λ=0.10을 지금까지 테스트한 어떤 flat 값(최저 0.25)보다 낮게 잡았다. 근거리는 실제 물체(차량·보행자)가 많아 mIoU 기여가 큰 영역인데, 여기 supervision을 약화시킨 게 오히려 손해였을 가능성이 높다. 20m 경계의 급격한 전환(하드 discontinuity) 자체가 나쁜 신호였을 가능성도 배제 못 한다.

### 다음 단계 (진행 승인됨, 실행 중)
반대 방향 설계 1회 추가 테스트: **근거리 λ=0.25(flat 최선점 유지) / 원거리 λ=0.10(낮춤)**. 근거리 supervision을 flat 최선점 수준으로 보존하면서 원거리만 줄였을 때, RayIoU 손실 없이 mIoU를 더 회복할 수 있는지 확인하는 것이 목적이다.

---

## 4. 논의하고 싶은 지점

1. flat λ=0.25가 이미 상당히 좋은 절충점인데, adaptive 방향이 이걸 넘어설 가능성이 있다고 보시는지 — 아니면 단순 스칼라 λ 자체를 최종 결론으로 볼지.
2. E3에서 실패한 것이 "근거리 supervision을 낮춘 것" 때문인지, 아니면 "하드 경계" 자체 때문인지 구분할 필요가 있어 보이는데, 이 구분이 연구 방향에 얼마나 중요한지.
3. novelty 측면에서 — adaptive weighting이 성공하더라도 여전히 상대적으로 단순한 확장인데, 이걸 넘어서는 기여(예: RayIoU origin-provenance 진단 자체를 방법론적 기여로 포지셔닝하는 것 등)를 병행할지.
