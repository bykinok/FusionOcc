# Architecture Adaptation Blockers

작성일: 2026-10-01

## 결론: 이번 감사 범위에서 architecture 변경이 필요하다고 판단된 지점은 없다.

`implementation_audit.md` §3, `camera_temporal_audit.md` §2에서 상세히 다룬 대로:

- camera 수(6→5)에 실질적으로 얽힌 학습 가능 weight는 `bevformer.py:37`의 `cams_embeds` 하나뿐이며, 모든 기존 config가 `use_cams_embeds=False`로 이를 0-곱 처리해 이미 비활성 상태다. `num_cams` 값만 config에서 바꾸면 되고, pretrained checkpoint 로딩 시 이 한 키만 `strict=False`로 스킵하면 된다 — **architecture 변경이 아니라 config 값 조정 수준.**
- forward projection(`BEVDetStereoForwardProjection`), view transformer(`LSSVStereoForwardPorjection`), BEVFormer encoder(spatial/temporal cross-attention 포함)는 전부 camera 차원을 입력 텐서 shape에서 매 forward마다 동적으로 읽는다. 이미 camera-count-agnostic하게 설계돼 있다.
- temporal fusion(`SparseFusion`)은 frame-count 기반으로만 동작하고 실제 시간 간격에 의존하는 계산이 없다 — frame 수를 그대로 쓰든(Protocol A) 늘리든(Protocol B) architecture 수정 없이 가능하다.

## 형식상 기록 (blocker는 아니지만 adapter 설계에 영향을 주는 사실)

아래는 "architecture 변경 필요"가 아니라 **adapter/config 레벨에서 반드시 처리해야 할 데이터 포맷 차이**다 (요구사항 문서 2.1절의 "허용되는 변경" 범주):

| 항목 | 파일/함수 | 가정 | 이유 | 최소 필요 변경 | architecture-neutral 대안 |
|---|---|---|---|---|---|
| calibration 필드 읽기 | `loading.py` 830~1050행대 (`STCOccPrepareImageInputs` 계열) | nuScenes의 translation+quaternion 분리 포맷(`sensor2ego_translation`, `sensor2ego_rotation` 등) | Waymo `cam_infos.pkl`은 4x4 행렬(`sensor2ego`, `ego2global`) 포맷 | Waymo 전용 info-pkl 로더(또는 기존 클래스에 분기)가 4x4 행렬을 읽어 기존 코드가 기대하는 중간 표현(동차변환행렬)으로 바로 변환 — 이미 내부적으로 쓰는 중간 표현과 동일하면 변환 코드만 필요 | 가능 (adapter만으로 해결) |
| camera 이름/순서 | config의 `data_config['cams']` | nuScenes 6개 고정 리스트 | Waymo는 5개, 이름도 다름(`CAM_SIDE_LEFT/RIGHT` vs `CAM_BACK_LEFT/RIGHT`) | config의 `cams` 리스트만 교체 | 가능 (config만으로 해결, 코드는 이미 `len(cam_names)`로 동적 순회) |

## 2026-10-01 추가 업데이트: 실제 구현/smoke test로 발견된 추가 사실 (architecture 변경 아님, 그러나 config 필수 항목)

실제로 config를 작성하고 로컬 GPU에서 smoke test를 돌려본 결과, "camera 수는 전부 동적이라 문제없다"는 이전 판단에 **중요한 예외가 2곳 있었다** — 둘 다 **수정 안 하면 RuntimeError로 즉시 크래시**하지만, **architecture 변경이 아니라 config에 `num_cams=5`를 명시하는 것만으로 해결됐다**:

1. `BEVFormer`(`bevformer.py:18`)의 `num_cams=6` 생성자 기본값: `use_cams_embeds=False`일 때 `cams_embeds`가 "0을 곱해 무효화"되는 것은 맞지만(이전 판단대로), **그 덧셈 연산 자체가 broadcasting을 위해 shape이 맞아야 한다** — 5-camera 입력에 6-camera shape의 zero-tensor를 더하려다 `RuntimeError: size of tensor a (5) must match size of tensor b (6)`로 즉시 실패함을 실측 확인. "이미 비활성화돼 있어 문제없다"는 이전 결론은 **수치적으로는 맞지만 shape 호환성까지는 보장 못했다** — 정정한다.
2. `OA_SpatialCrossAttention`(`spatial_cross_attention.py:54`)에도 **별도의, 독립적인** `num_cams=6` 기본값이 있었다(이전 조사에서 놓침 — 이전 조사는 `bevformer.py`만 찾았고 같은 패턴이 다른 attention 클래스에도 있는지 전수 검색하지 않았음). `RuntimeError: shape '[12, 704, 96]' is invalid for input of size 675840`(6-camera 가정의 reshape을 5-camera 데이터에 시도)로 확인.

둘 다 `stcocc_r50_704x256_16f_occ3d_waymo_baseline.py`에서 `num_cams=5`를 명시적으로 넘기는 것으로 해결했다(각 20줄 미만 config 추가, **architecture 코드 자체는 전혀 수정하지 않음** — 두 클래스 모두 이미 `num_cams`를 생성자 인자로 받고 있었고, 단지 기본값이 6이라 Waymo 설정에서 명시가 빠지면 안 됐을 뿐). 이 발견은 "config만으로 해결 가능"이라는 §2.1 표의 결론 자체를 바꾸지 않지만, "이미 비활성화돼 있어 **아무것도 안 해도 된다**"는 더 강한 (그리고 틀린) 주장은 정정한다 — 최소 **두 곳에 `num_cams=5` 명시가 필수**다.

## 이번 단계에서 architecture 변경을 보류한 항목은 없음

데이터 자체의 불확실성(5-camera 이미지 파일 위치 미확인, class 수 미확인, 정밀 range 미확인 — `remaining_issues.md` 참고)으로 어댑터 코드 자체를 아직 작성하지 않았지만, 이는 architecture blocker가 아니라 **데이터 준비(info-generation) blocker**다.
