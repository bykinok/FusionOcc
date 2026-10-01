# 남은 이슈 (BLOCKED 항목 총정리)

작성일: 2026-10-01, **업데이트: 2026-10-01 (같은 날, 사용자 요청으로 mmdetection3d 기반 Occ3D-Waymo 참고 구현을 웹에서 찾아 대조한 뒤)**.

업데이트 출처:
- 공식 Occ3D repo README: `https://github.com/Tsinghua-MARS-Lab/Occ3D`
- CVT-Occ(Tsinghua-MARS-Lab 자체 후속작, mmdetection3d 기반, ECCV'24): `https://github.com/Tsinghua-MARS-Lab/CVT-Occ` — `docs/dataset.md`, `projects/mmdet3d_plugin/datasets/pipelines/loading.py::LoadOccGTFromFileWaymo`, `projects/mmdet3d_plugin/datasets/waymo_temporal_zlt.py::CustomWaymoDataset_T.get_data_info`, `projects/configs/cvtocc/bevformer_waymo.py`

## 해소됨 (더 이상 BLOCKED 아님)

| # | 이슈 | 해소 내용 | 근거 |
|---|---|---|---|
| 2 | Occ3D-Waymo 공식 class 수/이름/free index | **num_classes=16** (0-14: GO,VEHICLE,PEDESTRIAN,SIGN,CYCLIST,TRAFFIC_LIGHT,POLE,CONSTRUCTION_CONE,BICYCLE,MOTORCYCLE,BUILDING,VEGETATION,TREE_TRUNK,ROAD,WALKABLE / 15: FREE). raw npz의 `voxel_label`은 free를 **23**으로 저장하고 로더가 `semantics==23 → 15`로 remap한다. 이전 세션 실측값(`{0,1,2,3,4,5,6,7,9,10,11,12,13,14,23}`)이 이 스킴과 완전히 정합함을 재확인. | README 원문 + `LoadOccGTFromFileWaymo` 코드(`semantics[semantics==self.FREE_LABEL]=self.num_classes-1`) + `bevformer_waymo.py`의 `FREE_LABEL=23`, `CLASS_NAMES`(16개) 리터럴 |
| 3 | fine(voxel01) GT의 voxel size/point_cloud_range | fine=[0.1,0.1,0.2]m, range=[-80,-80,-5,80,80,7.8]; coarse=[0.4,0.4,0.4]m, range=[-40,-40,-1,40,40,5.4] (coarse는 Occ3D-nuScenes/OpenOcc와 수치까지 완전 동일). 이전 세션이 지적한 "모순"은 전제가 틀렸음 — 두 grid는 애초에 다른 물리 범위(160m vs 80m)를 커버하도록 설계된 별개 release. | 공식 README 표 + `bevformer_waymo.py`의 `point_cloud_range`/`voxel_size` 리터럴, 검산 완료(1600×0.1=160m=80-(-80) 등) |
| 4 | Waymo 센서 캡처 주기 | **10Hz(0.1s/frame) 확정.** nuScenes는 0.5s(2Hz). 둘 다 "20초/scene"은 동일, frame 밀도만 다름. 16-frame history 실제 시간폭: nuScenes 8.0s vs Waymo 1.6s (5배 차이, 확정). | CVT-Occ `docs/dataset.md`의 Time Interval/Frame/Time Span 표 |
| 5 | voxel01/validation(42)과 voxel04/validation(202) 불일치 | 데이터 손상이 아니라 **부분 다운로드**. 공식 전체는 train 798 / val 202 / test 150 scene — 로컬 voxel04/validation(202)은 val 전체가 이미 있고, voxel01/validation(42)·voxel01/training(241/798)은 일부만 받아둔 상태. | README + dataset.md의 공식 scene 수, scene 폴더 명명 규칙(000~797/000~201)이 로컬 디렉터리명과 일치 |
| 6(일부) | 전체 프레임 수 | 공식: 200 frame/scene(train 798×200≈159,600, val 202×200≈40,400). 로컬 coarse 보유분: train 241×200≈48,200(부분), val 202×200≈40,400(전체). | CVT-Occ dataset.md |
| 7 | `origin_voxel_state`/`final_voxel_state`/`infov`의 정확한 의미 | `origin_voxel_state`=mask_lidar, `final_voxel_state`=mask_camera(평가에 사용), `infov`=mask_fov(Waymo는 5-camera라 360도가 아니므로 추가 제공). **중요 정정**: Waymo도 Occ3D-nuScenes와 동일하게 camera mask가 "있다" — 이전 세션이 "Waymo=OpenOcc처럼 mask 없음"이라 암묵 가정했던 것은 틀렸다. `legacy_occ3d_camera` oracle 정책을 Waymo에도 그대로 적용 가능. | README 원문 + `LoadOccGTFromFileWaymo.__call__`의 3개 mask 변수 할당 코드 |
| 1(상당부분) | 5-camera 이미지 파일 위치/인덱싱 | 로컬 `mmDataset/waymo/kitti_format/training/image_{0..4}/`에 전부 존재(각 198,068 파일, 수 일치 실측 확인). `image_idx` 디코딩 공식(`scene=idx%1000000//1000`, `frame=idx%1000000%1000`) 직접 검증 완료(예: idx=549000 → scene=549,frame=0). pkl의 `image_path` 필드는 `.png`로 적혀 있으나 실제 파일은 `.jpg`(확장자만 다름, 실측 확인). camera index 2/3이 pose 데이터와 실제 이미지 폴더 사이에서 바뀌어 있다는 공식적으로 알려진 버그와 그 우회 코드(`get_data_info`)까지 확보. | 로컬 파일시스템 직접 확인 + `waymo_temporal_zlt.py::get_data_info` |

## 아직 남은 것 (범위가 크게 좁혀짐)

| # | 이슈 | 다음 필요 작업 |
|---|---|---|
| 1-잔여 | ~~camera index 2/3 스왑 버그가 **우리 로컬 복사본에도** 실제로 적용되는지~~ | **해소됨(2026-10-01)** — LiDAR 포인트를 실제 이미지에 투영해 시각적으로 확인. 보정된 매핑(pose2↔image_3, pose3↔image_2)은 차량/도로/표지판 윤곽과 픽셀 단위로 정확히 일치하고, 스왑 안 한 naive 매핑(pose2↔image_2, pose3↔image_3)은 장면 geometry를 완전히 무시한 뭉개진 덩어리로 투영됨 — 4장 모두 `research_waymo/figs/camera_swap_verified_*.png`에 저장. 수치 검증(§dataset_schema.md, in-bounds rate)과 이미지 내용 검증이 서로 독립적으로 같은 결론에 도달. |
| 8(신규) | training 557개 시퀀스(798-241) 추가 다운로드 필요 여부/용량/시간 | 전체 학습을 할지, 부분(241 시퀀스)만으로 screening할지 연구자 결정 필요 — 이번 세션엔 다운로드 안 함 |

### 해소됨(추가, 같은 세션 내 직접 검증)

| # | 이슈 | 해소 내용 |
|---|---|---|
| 9 | `sensor2ego`/`intrinsics`/`ego2global` 행렬 포맷 | 셋 다 `(4,4) float64` 동차행렬 실측 확인(`cam_infos.pkl[549][0][0]`). `intrinsics`는 단순 3x3 K-matrix가 아니라 **"sensor2img" 합성 투영행렬**(참고 코드 주석과 일치) — `lidar2img = intrinsics @ np.linalg.inv(sensor2ego)` 식이 그대로 성립함을 shape으로 재확인. |

## 이슈가 아닌 것 (명확히 해소됨, 재확인 불필요)

- STCOcc architecture의 camera-count 결합: 해소됨(`adaptation_blockers.md`) — 5-camera 적용에 architecture 변경 불필요.
- coarse(voxel04) GT의 shape/point_cloud_range: 해소됨.
- `cam_infos.pkl`이 voxel GT와 시퀀스/프레임 id를 공유하는지: 해소됨.
- class 수/이름, mask 필드 의미, 10Hz, 데이터 부분다운로드 이유, 이미지 파일 위치/인덱싱: 전부 위 표에서 해소됨.

## 2026-10-01 최종 업데이트 — 실제 구현 + smoke test 완료 후

위에서 "다음 필요 작업"으로 남겨뒀던 항목 중 #1-잔여를 제외한 핵심 경로(이미지 경로, class remap, calibration 포맷)는 실제 구현(`tools/create_data_waymo_occ.py`)과 로컬 GPU smoke test로 **전부 검증 완료**됐다(`smoke_test_results.md`). 추가로 발견된 새 항목:

| # | 이슈 | 상태 |
|---|---|---|
| 10 | `BEVFormer`/`OA_SpatialCrossAttention`의 `num_cams=6` 기본값 | **해소됨** — `adaptation_blockers.md` 참고, config에 `num_cams=5` 명시로 해결(architecture 아님) |
| 11 | Waymo velodyne `.bin`이 nuScenes 기본(`load_dim=5`)과 다른 6-float/point 레이아웃 | **해소됨** — config에서 `load_dim=6`으로 수정, 파일 크기로 실측 확인 |
| 12 | `OccupancyMetric`이 Waymo GT 포맷(`_04.npz`, `voxel_label` 키)을 모름 | **해소됨(2026-10-01)** — `occupancy_metric_utils.py::load_occ_gt_npz` 단일 GT 로더를 신설해 `OccupancyMetric`의 3개 내부 GT 재로딩 지점(`_process_one_chunk`/`_compute_miou`/`_compute_rayiou`) 전부에 `occ3d_waymo` 분기를 추가. `tools/test.py`로 직접 mIoU/RayIoU 평가 가능(24-sample smoke 평가로 `tools/eval_waymo_smoke.py` standalone 결과와 수치 완전 일치 확인, `mIoU=0.0233`). Occ3D/OpenOcc 경로는 손대지 않음 — perfect self-comparison 회귀 테스트(mIoU=100.0, RayIoU mIoU=1.0) 통과. |
| 13 | RayIoU는 Waymo용으로 구현되지 않음 | **해소됨(2026-10-01, 연구용 근사 프로토콜로)** — 신규 `ray_metrics_waymo.py` 작성(`occ_class_names`=Waymo 16-class, `flow_class_names=[]`이라 mAVE는 N/A). **주의**: 공식 Occ3D-Waymo RayIoU 벤치마크는 존재하지 않음(참고 구현 CVT-Occ도 mIoU만 보고) — 이 구현은 (a) `generate_lidar_rays()`의 pitch 각도가 nuScenes LiDAR 수직 FOV를 그대로 재사용(공식 Waymo top-LiDAR FOV 미확인), (b) origin을 단일값(T=1, `info['lidar2ego_translation']`, 실측상 거의 0)으로 단순화(nuScenes의 multi-origin/time-offset 방식 미사용) — 두 지점 모두 각 파일의 docstring/주석에 명시. 24-sample smoke 평가로 end-to-end 동작 확인(크래시 없음, mAVE=NaN으로 올바르게 "해당없음" 처리). |
| 8 | training 557개 시퀀스(798-241) 추가 다운로드 | 변경 없음 — 여전히 연구자 승인 필요, 이번 세션엔 241개로만 작업 |

## 다음 세션/작업자를 위한 권고 순서 (업데이트)

1. **BLOCKED #9(calibration 행렬 포맷)** 먼저 — `cam_infos.pkl` 샘플 하나의 `intrinsics`/`sensor2ego` shape을 직접 출력해보면 몇 분 내 해소 가능.
2. 그 다음 **STCOcc용 Waymo GT 로더**(`STCOccLoadOccGTFromFileWaymo` 가칭) 작성 — OpenOcc 때 만든 `load_flow`/`resolve_occ_gt_dir` 패턴과 CVT-Occ의 `LoadOccGTFromFileWaymo`(free-label remap, mask 3종 조합)를 참고해 구현. architecture 변경 전혀 필요 없음(이미 확인됨).
3. 이미지 경로/calib 매핑은 `get_data_info`를 참고하되, **그대로 복사하지 말고** STCOcc의 기존 `STCOccPrepareImageInputs` 인터페이스(nuScenes의 sensor2ego_translation+quaternion 포맷을 기대)에 맞게 4x4 행렬 → 그 중간표현으로 변환하는 어댑터로 재작성.
4. BLOCKED #8(training 데이터 커버리지)은 연구자 승인 필요 — 241/798로 screening할지 결정.
5. 미세 검증(#1-잔여, camera index 2/3 스왑 재현)은 낮은 우선순위 — 틀려도 "카메라가 서로 바뀐 채 학습"이라는 명확한 증상으로 빨리 드러날 것이라 risk가 낮음.
