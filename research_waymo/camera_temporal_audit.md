# Camera / Temporal 감사

## 1. Camera 구성 비교

| | Occ3D-nuScenes / OpenOcc | Occ3D-Waymo |
|---|---|---|
| camera 수 | 6 | 5 (실측 확인: `mmDataset/waymo/kitti_format/waymo_infos_train.pkl`의 `images` 키) |
| camera 이름 | `CAM_FRONT_LEFT, CAM_FRONT, CAM_FRONT_RIGHT, CAM_BACK_LEFT, CAM_BACK, CAM_BACK_RIGHT` | `CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT, CAM_SIDE_LEFT, CAM_SIDE_RIGHT` (실측 확인) — 후방(BACK) 카메라가 없고 측면(SIDE) 카메라로 구성, 즉 전방위가 아니라 전방+측면 위주 |
| calibration 필드 포맷 | `sensor2ego_translation`(3-vector) + `sensor2ego_rotation`(quaternion) + `ego2global_*` 분리 필드 | `cam_infos.pkl`: `ego2global`(4,4 행렬), `sensor2ego`(4,4 행렬로 추정, 분리된 translation/quaternion 아님), `intrinsics` — **포맷 자체가 다름(행렬 vs translation+quaternion), 1:1 필드 매핑 불가, adapter에서 변환 필요** |
| intrinsic 포맷 | `cam_intrinsic`(3x3) | `intrinsics`(실측 shape 미기록, 3x3 또는 4x4 가능성 — 추가 확인 필요, BLOCKED) |

## 2. STCOcc 코드의 camera-count 결합 지점 (상세는 implementation_audit.md §3 표 참고)

핵심 결론: **학습 가능 weight 중 camera 수에 shape이 실제로 묶인 것은 `bevformer.py:37`의 `cams_embeds` 파라미터 하나뿐이고, 모든 기존 config가 `use_cams_embeds=False`로 이를 0-곱 처리해 현재도 비활성 상태다.** 나머지(forward projection, view transformer, BEVFormer encoder)는 전부 `B, N, C, H, W = x.shape`처럼 camera 차원 `N`을 텐서 shape에서 동적으로 읽는다(`view_transformer.py:186,339,396,442`, `bevdet_stereo_projection.py:82,145,176`, `bevformer_encoder.py:209,334` 등 10곳 이상 확인). → **5-camera로 바꿔도 pretrained weight 재사용 가능, architecture 변경 불필요.**

calibration 필드명(`sensor2ego_translation` 등)은 `loading.py` 830~1050행대에서 nuScenes 포맷으로 고정 읽기 때문에, Waymo의 4x4 행렬 포맷을 그대로 넣을 수 없다 — **adapter(신규 로더 클래스 또는 기존 클래스의 Waymo 분기)가 반드시 필요**하지만 이는 데이터 어댑테이션이지 architecture 변경이 아니다.

## 3. Temporal — 16-frame history의 실제 시간 폭

STCOcc의 `history_frame_num=[16,8,4]`와 `multi_adj_frame_id_cfg`는 **순수 frame-count 기반**이다. `temporal_fusion.py:29`의 `self.history_cam_sweep_freq = 0.5  # seconds between each frame`(nuScenes 2Hz 키프레임 가정)는 대입 후 전혀 읽히지 않는 dead 상수임을 확인했다 — 즉 temporal fusion 모듈 자체는 "16프레임이 몇 초인지" 모르고 동작한다. 이는 코드가 깨진다는 뜻이 아니라, **frame 수만 같으면 architecture는 그대로 동작하지만, nuScenes와 Waymo에서 "16프레임"이 가리키는 실제 시간 폭이 전혀 다를 수 있다**는 뜻이다.

```text
nuScenes:
- history frames: 16
- key frame 주기: 2Hz(0.5초/프레임)
- effective temporal window: 16 x 0.5s = 8.0초

Waymo:
- history frames: 16(코드상 그대로 적용 가능)
- 센서 캡처 주기: 10Hz(0.1초/프레임)
- effective temporal window: 16 x 0.1s = 1.6초
```

**2026-10-01 업데이트 — 해소됨(공식 1차 출처로 확인)**: CVT-Occ `docs/dataset.md`(Tsinghua-MARS-Lab 자체 공식 후속 repo, `https://github.com/Tsinghua-MARS-Lab/CVT-Occ/blob/main/docs/dataset.md`)가 두 dataset을 나란히 명시한다:

| | Time Span | Frame/scene | Time Interval |
|---|---|---|---|
| Occ3D-nuScenes | 20s | 40 | **0.5s (2Hz)** |
| Occ3D-Waymo | 20s | 200 | **0.1s (10Hz)** |

10Hz 가정이 그대로 확인됐다 — **동일한 16-frame history가 nuScenes에서는 8.0초, Waymo에서는 1.6초를 가리켜 정확히 5배 차이난다.** 이는 WAYMO_EXTENSION_REQUIREMENTS_KO.md 6절이 정확히 경고한 상황이며, 이제 추정이 아니라 확인된 사실이다. 흥미롭게도 두 dataset 모두 "20초 분량 scene"이라는 점은 동일해, scene 하나가 담는 실제 주행 시간 자체는 같다 — 차이는 오직 frame 샘플링 밀도(2Hz vs 10Hz)다.

### Protocol 비교 (연구자가 선택해야 함, 자동 결정하지 않음)

| | Protocol A: frame-count matched | Protocol B: temporal-duration matched |
|---|---|---|
| 방법 | 양쪽 다 16 frame 그대로 사용 | Waymo 10Hz 기준 8.0초를 커버하려면 80 frame 필요(확정 계산, 가정 아님) — `history_frame_num`을 [80,40,20] 등으로 조정하거나 adjacent frame 간격 자체를 넓혀야 함 |
| 장점 | 코드/config 변경 최소, 구현 간단 | "같은 시간 폭의 temporal context"라는 공정한 비교 |
| 단점 | 두 dataset의 temporal context 폭이 5배 달라 모델이 실질적으로 다른 task를 푸는 셈이 될 수 있음 | history 길이가 5배로 늘어나면 메모리/연산량 증가, 80프레임이 한 시퀀스(198프레임) 내에서 항상 확보되는지 확인 필요, STCOcc의 `intermediate_pred_loss_weight`/cascade 구조가 이 정도 history 길이로 설계·검증된 적이 없어 별도 안정성 확인 필요 |

**권고**: screening 단계는 Protocol A(frame-count matched, 구현 간단)로 시작하고, 의미 있는 결과가 나오면 Protocol B를 보조 실험으로 고려하는 것을 제안한다 — 단, 이 선택은 자동으로 하지 않았고 연구자 승인이 필요하다.
