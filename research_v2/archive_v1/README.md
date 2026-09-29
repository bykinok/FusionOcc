# archive_v1 — 실제로는 파일이 아니라 포인터

v2 template(`deferred_candidate_v1.spec_location: archive_v1/CLAUDE_CODE_RESEARCH_SPEC.md`)은
v1 스펙 원문이 여기 보존돼 있다고 가정하지만, 확인 결과 **v1 스펙 원문은 git에 커밋된 적이 없고
(untracked 상태에서) 사용자가 직접 파일을 v2 내용으로 덮어썼기 때문에 이 세션에서는 복구할 수 없다.**
텍스트를 재구성해서 채워 넣지 않는다 — 사실이 아닌 것을 사실처럼 두는 것이기 때문이다.

**v1의 실제 산출물(스펙 원문 대신 신뢰할 수 있는 근거)은 `research/` 디렉터리에 그대로 남아 있다:**
- `research/audits/loss_inventory.json`, `gradient_coverage.csv`, `audit_report.md`
- `research/protocol_manifest.json`, `resolved_paths.json`
- `research/decisions.jsonl`, `research/experiment_registry.jsonl`
- `research/reports/stage_P0.md`, `research/reports/P1_execution_plan.md`

v2의 s/g/h 후보(`deferred_candidate_v1`)에 대한 상세 수식·가정은 이 README가 아니라
**이번 대화(세션) 기록** 및 `research/reports/P1_execution_plan.md`에 남아 있는 설명이 유일한
1차 자료다. s/g/h를 재개할 경우, 그 세션 기록을 근거로 다시 정리해야 한다.
