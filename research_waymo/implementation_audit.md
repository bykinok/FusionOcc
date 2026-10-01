# Waymo 지원 구현 감사

작성일: 2026-10-01
범위: 읽기 전용 감사(코드/데이터 모두 수정 없음). `WAYMO_EXTENSION_REQUIREMENTS_KO.md`의 "데이터가 있는 경우" 트랙을 따른다 — `/home/h00323/DATA/Occ3D-Waymo/`에 실제 occupancy GT(voxel01/voxel04)가 존재함을 확인했기 때문이다.

## 0. 결론 요약

**architecture는 바꿀 필요가 없다.** STCOcc 코드베이스를 전수 조사한 결과 camera 수(6)에 실질적으로 결합된 학습 가능 weight는 `cams_embeds` 하나뿐이고, 이마저 모든 기존 config에서 `use_cams_embeds=False`로 0-곱 처리돼 **현재도 사실상 비활성**이다. forward projection/view transformer/BEVFormer encoder는 전부 camera 수를 텐서 shape에서 동적으로 읽는다.

**진짜 난관은 architecture가 아니라 데이터 정합성이다.** Occ3D-Waymo의 (a) occupancy voxel GT(`voxel01`/`voxel04`), (b) `Occ3D-Waymo/waymo_infos_{train,val}.pkl`, (c) `cam_infos.pkl`, (d) `mmDataset/waymo/kitti_format/`의 원본 센서 데이터 — 이 4개가 **서로 다른 변환 파이프라인의 산출물**로, 지금 상태로는 "한 샘플의 이미지+calib+pose+occupancy GT"를 하나로 엮어주는 info pkl이 존재하지 않는다. STCOcc가 학습/평가에 쓸 수 있으려면 이 4개를 잇는 **신규 info-generation 스크립트**가 먼저 필요하다 — 이는 코드 architecture 문제가 아니라 데이터 준비(data preparation) 문제다.

## 1. 기존 Waymo 지원 여부 — 확인됨: 없음

`projects/STCOcc/` 전체에서 "waymo" 문자열 grep 결과 0건. STCOcc는 지금까지 nuScenes(Occ3D-nuScenes, OpenOcc-nuScenes)로만 사용돼 왔다. `mmdet3d`(이 저장소의 범용 3D 라이브러리) 쪽에는 일반적인 Waymo 3D object detection 지원(`tools/dataset_converters/waymo_converter.py`, `configs/_base_/datasets/waymoD5-*.py`)이 있지만, 이는 **3D detection용(car/pedestrian/cyclist 3-class)**이고 occupancy GT와는 무관함을 `configs/_base_/datasets/waymoD5-3d-3class.py` 등을 직접 열어 확인했다.

## 2. 수정한 파일 — 없음 (이번 패스는 감사만)

이번 세션에서는 Waymo 관련 코드/config를 작성하지 않았다. 이유: §4(adaptation_blockers 없음)에서 보듯 architecture 변경은 불필요하지만, 데이터 쪽 BLOCKED 항목(§3, `remaining_issues.md`)이 여러 개 남아 있어 — 특히 **5-camera 실제 이미지 파일의 정확한 디스크 위치 미확인**, **공식 class 수/이름 미확인** — 이 상태에서 로더/config를 작성하면 검증되지 않은 가정을 코드에 그대로 박아넣게 된다. `CLAUDE_CODE_OPENOCC_REQUIREMENTS_KO.md`와 동일한 원칙("문서나 기억만으로 값을 하드코딩하지 말고")을 Waymo에도 동일하게 적용해, 확인 안 된 값을 코드에 넣는 대신 `run_commands.md`에 "무엇을 먼저 확인/생성해야 하는지"를 명시하는 쪽을 선택했다.

## 3. nuScenes 하드코딩 발견 사항 (architecture-neutral 여부 판단)

전체 내용은 `camera_temporal_audit.md` 참고. 요약:

| 항목 | 위치 | 판정 |
|---|---|---|
| `PointToMultiViewDepth.num_cam=6` | `loading.py:639` | dead code, 읽는 곳 없음 — 무해 |
| forward projection 전체 | `bevdet_stereo_projection.py`, `view_transformer.py` | 전부 `B,N,C,H,W=x.shape`로 동적 — **architecture-neutral** |
| `cams_embeds` 파라미터 | `bevformer.py:37` | shape이 `num_cams`에 결합되지만 `use_cams_embeds=False`로 전 config에서 비활성(0-곱) — **config 값만 조정하면 됨** |
| BEVFormer encoder `num_cam` | `bevformer_encoder.py:209,334` | 생성자 기본값 무시하고 매 forward마다 실제 입력에서 재계산 — **architecture-neutral** |
| calibration 필드명(`sensor2ego_translation` 등) | `loading.py` 830~1050행대 | nuScenes 전용 포맷 — **adapter 필요(architecture 아님)** |
| `history_cam_sweep_freq=0.5` | `temporal_fusion.py:29` | dead code(대입 후 안 읽힘), 실제 history는 frame-count 기반 — **architecture-neutral이지만 "16 frame"의 실제 시간 의미는 adapter/연구 설계 결정 필요(§ camera_temporal_audit.md)** |

**architecture 변경이 필요하다고 판단된 지점: 없음.** `adaptation_blockers.md`는 형식상 존재하되 내용은 "해당 없음"으로 채운다.

## 4. 외부 참고 구현 (2026-10-01, 사용자 요청으로 조사)

mmdetection3d 기반이면서 Occ3D-Waymo GT를 실제로 쓰는 공개 구현을 웹에서 찾아 데이터 파이프라인(필드 의미, class remap, 카메라 경로 매핑)을 대조했다. **코드를 이 저장소에 가져다 쓰지는 않았다** — architecture가 BEVFormer 계열로 STCOcc와 다르고, 라이선스/정확성 재검증이 별도로 필요하다. 아래는 "참고용으로 대조해 사실관계를 확인한" 출처 목록이다.

| 출처 | 용도 |
|---|---|
| `github.com/Tsinghua-MARS-Lab/Occ3D` (공식 Occ3D GT 배포 저장소, README) | class 목록, voxel size/range, mask 필드 의미, scene 수의 1차 출처 |
| `github.com/Tsinghua-MARS-Lab/CVT-Occ` (ECCV'24, Occ3D 저자 그룹의 후속 mmdet3d 기반 구현) — `docs/dataset.md` | train/val/test 수, frame rate(10Hz), 디렉터리 구조 1차 출처 |
| 〃 `projects/mmdet3d_plugin/datasets/pipelines/loading.py::LoadOccGTFromFileWaymo` | occupancy GT 로더 실코드 — mask 3종 조합, `FREE_LABEL=23→15` remap 로직 |
| 〃 `projects/mmdet3d_plugin/datasets/waymo_temporal_zlt.py::CustomWaymoDataset_T.get_data_info` | 5-camera 이미지 경로 생성, `image_idx` 디코딩(`scene=idx%1000000//1000` 등), camera index 2/3 스왑 버그 우회 |
| 〃 `projects/configs/cvtocc/bevformer_waymo.py` | 실제 config에 박힌 `point_cloud_range`/`voxel_size`/`num_classes`/`CLASS_NAMES`/`FREE_LABEL` 리터럴 — 공식 README와 교차검증 |

이 대조로 `dataset_schema.md`/`camera_temporal_audit.md`/`remaining_issues.md`의 상당수 BLOCKED 항목이 해소됐다(상세는 각 문서 참고). 참고 구현의 **설계 패턴**(GT 로더의 mask 조합·remap, 이미지 경로 보정 로직)을 참고해 STCOcc의 기존 인터페이스에 맞게 새로 작성했다(§5).

## 5. 실제 구현 (2026-10-01, 같은 세션에서 이어서 진행 — "진행해줘" 승인 후)

조사에 이어 실제 코드/config를 작성하고 로컬 GPU(RTX 3090)에서 smoke test까지 완료했다. 전체 결과는 `smoke_test_results.md`, 명령은 `run_commands.md` 참고. 신규/수정 파일:

| 파일 | 내용 |
|---|---|
| `tools/create_data_waymo_occ.py` (신규) | Occ3D-Waymo의 3개 서로 다른 pkl(`waymo_infos_*.pkl`, `cam_infos*.pkl`)을 조인해 nuScenes info 스키마를 그대로 흉내 낸 `stcocc-waymo_infos_{train,val}.pkl` 생성. camera 2/3 스왑 보정, 4x4 calibration -> nuScenes translation+quaternion+K 분해(이번 세션에 유도하고 실측 LiDAR 투영으로 검증한 수학) 포함 |
| `tools/generate_ms_occ_waymo_parallel.py` (신규) | OpenOcc 때와 같은 패턴의 멀티스케일(1/2,1/4,1/8) GT 병렬 생성기, Waymo 디렉터리 구조/raw free=23에 맞춤 |
| `tools/eval_waymo_smoke.py` (신규) | `OccupancyMetric` 우회, standalone 소규모 mIoU 평가 스크립트 |
| `projects/STCOcc/stcocc/transforms/pipelines/loading.py` (수정) | `STCOccLoadOccGTFromFileWaymo` 신규 클래스 추가(23->15 remap, FOV ignore-masking) |
| `mmdet3d/datasets/occ_metrics.py` (수정) | `Metric_mIoU`에 `num_classes==16`(Waymo) 분기 추가 |
| `projects/STCOcc/configs/stcocc_r50_704x256_16f_occ3d_waymo_baseline.py` (신규) | baseline-none occupancy-only Waymo config |
| `data/waymo/` (신규 심볼릭 링크) | `data/nuscenes` 관례와 동일 |

**architecture 코드(backbone/view transformer/temporal fusion/occupancy head 자체)는 한 줄도 수정하지 않았다** — 연구의 핵심 제약을 끝까지 지켰다. smoke test 중 발견한 버그 4개(lidar_path 누락, load_dim 불일치, BEVFormer/SpatialCrossAttention의 num_cams=6 기본값 2곳)는 전부 config 또는 신규 도구 스크립트 레벨에서 해결했다 — 상세는 `smoke_test_results.md`.

## 6. `OccupancyMetric` 정식 배선 + RayIoU 커널 (2026-10-01, 같은 세션 이어서 — "7,8,9를 진행해줘" 승인 후)

이전 패스에서 "의도적으로 범위 밖"으로 남겨뒀던 두 항목(`remaining_issues.md` #12, #13)을 실제로 구현했다.

| 파일 | 내용 |
|---|---|
| `projects/STCOcc/stcocc/evaluation/occupancy_metric_utils.py` (수정) | `load_occ_gt_npz()` 신규 — Occ3D/OpenOcc(`semantics`+`mask_camera`, `labels.npz`)와 occ3d_waymo(`voxel_label`+`final_voxel_state`, `<prefix>_04.npz`, free=23→num_classes-1 remap)를 하나의 진입점으로 통합한 단일 GT 로딩 primitive |
| `projects/STCOcc/stcocc/evaluation/occupancy_metric.py` (수정) | 3개 내부 GT 재로딩 지점(`_process_one_chunk`/`_compute_miou`/`_compute_rayiou`) 전부에 `load_occ_gt_npz` 적용 + `occ3d_waymo` 분기 추가. `_compute_rayiou`는 추가로 nuScenes-devkit(`NuScenes`/`nuScenesDataset`) 인스턴스화를 Waymo일 때 완전히 건너뛰고, origin을 `info['lidar2ego_translation']` 단일값(T=1)으로 대체. Occ3D/OpenOcc 경로는 `if self.dataset_name == 'occ3d_waymo': ... continue`로 감싸기만 해서 기존 코드는 1바이트도 바꾸지 않음 |
| `projects/STCOcc/stcocc/datasets/ray_metrics_waymo.py` (신규) | `ray_metrics_openocc.py`를 템플릿으로 한 Waymo RayIoU 커널. class 목록을 Waymo 16-class로, `flow_class_names=[]`(occupancy-only baseline이라 flow 없음 → mAVE는 항상 NaN/N-A로 올바르게 처리). **연구용 근사 프로토콜**: 공식 Occ3D-Waymo RayIoU 벤치마크는 존재하지 않음(참고 구현 CVT-Occ도 mIoU만) — `generate_lidar_rays()`의 LiDAR 수직 FOV는 nuScenes 센서값을 그대로 재사용(공식 Waymo 값 미확인), origin은 단일값 단순화. 두 근사 모두 파일 상단 docstring에 명시 |

**검증**: (a) 24-sample Waymo smoke 평가를 `tools/test.py`로 mIoU/RayIoU 둘 다 end-to-end 실행 — mIoU는 기존 standalone 스크립트(`tools/eval_waymo_smoke.py`)와 수치 완전 일치(`0.0233`), RayIoU는 크래시 없이 테이블 출력(저학습 체크포인트라 수치 자체는 낮음, 구조적 정합성만 확인). (b) Occ3D/OpenOcc 양쪽에서 실제 GT로 perfect self-comparison 회귀 테스트 — mIoU 경로 100.0, RayIoU 경로도 semantic-only self-comparison에서 mIoU=1.0으로 정상, 기존 동작과 완전히 동일함을 재확인(둘 다 nuScenes-devkit을 실제로 인스턴스화하는 경로까지 포함). architecture 코드는 이번에도 전혀 건드리지 않았다.
