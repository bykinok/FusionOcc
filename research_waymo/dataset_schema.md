# Occ3D-Waymo Dataset Schema 감사

모든 수치는 실제 파일을 열어 확인했다(`/home/h00323/DATA/Occ3D-Waymo/voxel01/training/549.tar`를 `/tmp`에 압축 해제해 직접 검증, 원본은 건드리지 않음). 추측한 값은 명시적으로 "추정"이라고 표시했다.

**2026-10-01 업데이트**: 사용자 요청으로 mmdetection3d 기반 Occ3D-Waymo occupancy 코드를 웹에서 찾아 대조한 결과, 아래 §2·§3·§4·§5·§6의 대부분의 BLOCKED 항목이 **공식 1차 출처로 해소**됐다. 출처:
- 공식 Occ3D repo README: `https://github.com/Tsinghua-MARS-Lab/Occ3D` (raw: `raw.githubusercontent.com/Tsinghua-MARS-Lab/Occ3D/master/README.md`)
- mmdetection3d 기반 다운스트림 구현 CVT-Occ(ECCV'24, Tsinghua-MARS-Lab 자체 후속작): `https://github.com/Tsinghua-MARS-Lab/CVT-Occ`
  - `docs/dataset.md` — Occ3D-Waymo 준비 가이드(시간 간격, 카메라, 클래스, 디렉터리 구조)
  - `projects/mmdet3d_plugin/datasets/pipelines/loading.py::LoadOccGTFromFileWaymo` — occupancy GT 로더 실코드
  - `projects/mmdet3d_plugin/datasets/waymo_temporal_zlt.py::CustomWaymoDataset_T.get_data_info` — 5-camera 이미지 경로/calib 매핑 실코드(인덱스 불일치 버그 우회 포함)
  - `projects/configs/cvtocc/bevformer_waymo.py` — 실제 학습 config(경로/class/grid 전부 하드코딩된 값으로 확인 가능)

이 저장소 코드를 그대로 가져다 쓰지는 않았다(라이선스/검증 별도 필요, architecture도 STCOcc와 다름 — BEVFormer 계열). **값과 데이터 계약(필드 의미, 경로 규칙, 클래스 매핑)만 1차 출처로 확인하는 데 썼다.**

## 1. GT 파일 구조 — 확인됨

`voxel01/{training,validation}/<seq_id>.tar`, `voxel04/validation/<seq_id>.tar` (training의 0.4m 버전은 별도 디렉터리가 없고 `voxel01` tar 안에 `_04` 접미사 파일로 번들돼 있음 — 아래 참고). 하나의 tar는 시퀀스 하나, 내부에 프레임마다 2개 파일:

| 파일 | shape | 설명 |
|---|---|---|
| `<frame>.npz` | 추정 0.05m/voxel (아래 §2 참고), (1600,1600,64) | fine-resolution GT |
| `<frame>_04.npz` | 0.4m/voxel, (200,200,16) | coarse-resolution GT, STCOcc가 쓸 대상 |

두 파일 모두 동일한 5개 키: `voxel_label`(uint8), `origin_voxel_state`(uint8), `final_voxel_state`(uint8), `infov`(fine=uint8, coarse=bool), `ego2global`(4,4 float64).

**해소됨(공식 README + CVT-Occ 코드로 확인)**: 필드 의미가 Occ3D-nuScenes의 `mask_camera`/`mask_lidar`와 정확히 대응한다. 공식 README의 코드 스니펫과 CVT-Occ의 `LoadOccGTFromFileWaymo`(`loading.py:133-136`)가 동일하게 확인해준다:

| npz 키 | 의미 | Occ3D-nuScenes 대응 |
|---|---|---|
| `voxel_label` | semantics (raw, free=23) | `semantics`(free=17) |
| `origin_voxel_state` | `mask_lidar` | `mask_lidar` |
| `final_voxel_state` | `mask_camera` (vision-centric 평가에 사용) | `mask_camera` |
| `infov` | `mask_fov` — Waymo는 5-camera라 360도 surround가 아니므로 추가 제공 | 없음(nuScenes는 6-camera로 거의 360도라 불필요) |

CVT-Occ 기본 config(`bevformer_waymo.py`)는 `use_infov_mask=True, use_lidar_mask=False, use_camera_mask=True`로 `valid_mask = infov_mask & camera_mask`를 쓴다 — 즉 **Occ3D-Waymo도 Occ3D-nuScenes와 동일하게 "camera mask가 있는" oracle 정책(`legacy_occ3d_camera`)을 그대로 적용할 수 있다.** OpenOcc처럼 "camera mask가 아예 없는 GT"가 아니다 — `baseline_plan.yaml`의 정책 집합 중 `none`뿐 아니라 `legacy_occ3d_camera`(oracle 대조군)도 Waymo에서 구현 가능하다는 뜻으로, 이전 세션의 암묵적 가정(Waymo=OpenOcc처럼 mask 없음)은 틀렸다 — 정정한다.

## 2. voxel 해상도/물리 범위 — 해소됨(공식 README로 확인, 이전 세션의 "모순" 추론은 틀렸음을 정정)

공식 Occ3D README의 Occ3D-Waymo 표(직접 인용):

| | fine | coarse |
|---|---|---|
| voxel size | **[0.1m, 0.1m, 0.2m]** (비등방, z축만 0.2m) | [0.4m, 0.4m, 0.4m] |
| range | **[-80, -80, -5, 80, 80, 7.8]** | [-40, -40, -1, 40, 40, 5.4] |
| volume size | [1600, 1600, 64] | [200, 200, 16] |

검산: fine 1600×0.1=160m(=80-(-80)) ✓, 64×0.2=12.8m(=7.8-(-5)) ✓. coarse 200×0.4=80m(=40-(-40)) ✓, 16×0.4=6.4m(=5.4-(-1)) ✓ — **CVT-Occ의 실제 config 파일(`bevformer_waymo.py`)도 coarse를 `point_cloud_range=[-40,-40,-1.0,40,40,5.4]`, `voxel_size=[0.4,0.4,0.4]`로 명시해 동일 수치를 재확인시켜준다.**

**이전 세션에서 "fine(1600×0.1=160m)과 coarse(200×0.4=80m)가 같은 물리 공간을 나타낸다면 모순"이라고 지적했던 것은 전제 자체가 틀렸다 — 두 grid는애초에 서로 다른(160m vs 80m) 물리 범위를 커버하도록 설계된 별개 release이지, 같은 범위를 다른 해상도로 나타낸 것이 아니다.** 디렉터리명 "voxel01"="0.1m", "voxel04"="0.4m"가 정확히 맞다(공식 README 원문 "We provide two types of voxel size data, with voxel size of 0.1m and 0.4m respectively").

**중요**: coarse(0.4m, 80m×80m×6.4m)의 point_cloud_range·voxel_size가 **Occ3D-nuScenes/OpenOcc와 숫자까지 완전히 동일**하다(`[-40,-40,-1,40,40,5.4]`, 0.4m) — STCOcc의 backward_projection/BEV grid 설정을 Waymo coarse profile에 그대로 재사용할 수 있다는 뜻이다(새 수치를 만들 필요가 없다).

## 3. Class 수 / class names — 해소됨(공식 README + CVT-Occ 코드로 완전히 확인, 이전 세션의 관측치와도 정합함을 재확인)

**공식 정의 (Occ3D README 원문 + CVT-Occ `bevformer_waymo.py`의 `CLASS_NAMES` 리터럴, 두 출처 일치):**

```text
num_classes = 16
index  name
0      TYPE_GENERALOBJECT   (GO)
1      TYPE_VEHICLE
2      TYPE_PEDESTRIAN
3      TYPE_SIGN
4      TYPE_CYCLIST (= Bicyclist)
5      TYPE_TRAFFIC_LIGHT
6      TYPE_POLE
7      TYPE_CONSTRUCTION_CONE
8      TYPE_BICYCLE
9      TYPE_MOTORCYCLE
10     TYPE_BUILDING
11     TYPE_VEGETATION
12     TYPE_TREE_TRUNK
13     TYPE_ROAD
14     TYPE_WALKABLE
15     TYPE_FREE (free_index)
```

**핵심 발견 — 이전 세션의 "BLOCKED" 원인이 바로 이것이었다**: raw npz 파일의 `voxel_label`은 free를 **23**이라는 값으로 저장하고, 0-14는 위 표와 동일하게 쓴다. 공식 README 원문: *"Please note that there is a slight difference between the Occ classes and the classes used in the Waymo LiDAR segmentation... Indeed `free` label is `23` in ground truth file. It is converted to `15` in dataloader."* CVT-Occ의 `LoadOccGTFromFileWaymo.__call__`(`loading.py:157-159`)이 정확히 이 remap을 코드로 수행한다:
```python
if self.FREE_LABEL is not None:
    semantics[semantics == self.FREE_LABEL] = self.num_classes - 1   # FREE_LABEL=23 -> 15
```

이전 세션에서 실측한 raw 값 `{0,1,2,3,4,5,6,7,9,10,11,12,13,14,23}`(23이 ~96%)은 **이 공식 스킴과 완전히 정합한다** — 0-14 범위 내 관측값들은 전부 유효 class이고(8=TYPE_CONSTRUCTION_CONE은 두 번째 fork의 더 넓은 샘플링에서 관측됨), 23은 remap 전 raw free sentinel이었던 것이다. "15~22가 관측되지 않음"도 당연하다 — 그 구간은애초에 쓰이지 않는 값이다(raw 스킴이 0-14 다음 바로 23으로 건너뛴다).

**STCOcc 적용 시 필요 작업**: `STCOccLoadOccGTFromFileOpenOcc`를 OpenOcc에 맞춰 만들 때와 동일한 패턴으로, Waymo 로더에서 `semantics[semantics == 23] = 15`(또는 `num_classes - 1`) remap을 한 줄 추가하면 된다 — 이미 OpenOcc 작업에서 구축한 `load_flow` 플래그·`resolve_occ_gt_dir` 패턴과 구조적으로 동일한 규모의 변경이다. `num_classes`/`class_weights`/`free_index=15`는 config-driven이므로 코드 변경 없이 값만 채우면 된다.

## 4. Train/Val 실제 수 — 해소됨(공식 수치로 설명됨)

**공식 전체 규모(Occ3D README + CVT-Occ dataset.md, 두 출처 일치)**: train 798 scenes, val 202 scenes, test 150 scenes, scene 폴더명은 `000`~`797`(train)/`000`~`201`(val) 식의 3자리 숫자 — **로컬 `voxel01/training/549.tar`, `voxel04/validation/` 등의 디렉터리명 관례와 정확히 일치한다(동일한 공식 시퀀스 id 체계).**

| | 로컬 실측 | 공식 전체 | 해석 |
|---|---|---|---|
| training 시퀀스 수 | 241 | 798 | **로컬은 공식 798개 중 241개만 부분 다운로드됨** |
| voxel01/validation 시퀀스 수 | 42 | 202 | **부분 다운로드**(0.1m 버전은 42/202만 받음) |
| voxel04/validation 시퀀스 수 | 202 | 202 | **전체 다운로드 완료**(0.4m coarse는 val 전체가 로컬에 있음) |
| voxel04/training | 디렉터리 없음, tar 내부 `_04.npz`로 번들 | — | 문제 없음(§1) |

**이전 세션이 "원인 불명 불일치"로 BLOCKED 처리했던 voxel01(42)과 voxel04(202)의 차이는 데이터 손상이 아니라 단순히 "0.1m 버전은 일부만, 0.4m 버전은 validation 전체를 받아뒀다"는 부분 다운로드 상태였다.** STCOcc가 쓸 대상은 coarse(0.4m)이므로 **val 전체(202 scene)가 이미 로컬에 준비돼 있다** — 이는 긍정적인 발견이다. 반면 training은 241/798만 있어, 전체 학습을 하려면 나머지 557개 시퀀스를 추가로 받아야 한다(용량/시간 추산은 §`remaining_issues.md` 참고).

**프레임 수**: CVT-Occ `dataset.md`가 명시 — **"Frame: 200 per scene"** (nuScenes는 40/scene). 시퀀스 549 실측(198 프레임)과 거의 일치(공식 수치는 반올림/평균치로 보임, 시퀀스마다 ±소수 프레임 편차는 있을 수 있음). 이를 적용하면:
- 공식 전체: train ≈ 798×200=159,600 프레임, val ≈ 202×200=40,400 프레임
- 로컬 보유분(coarse 기준, 학습에 쓸 대상): train ≈ 241×200=48,200 프레임(부분), val ≈ 202×200=40,400 프레임(전체)

## 5. Info pkl 현황 — 확인됨, 중요한 문제 발견

세 가지 서로 다른 pkl이 있고, **어느 것도 occupancy GT와 바로 연결되지 않는다**:

1. **`Occ3D-Waymo/waymo_infos_{train,val}.pkl`**: 구버전 KITTI 스타일 리스트(val 길이 39987). `image` 필드가 단일 카메라(`image_0`)만 가리키고, 멀티뷰 이미지 경로 필드가 없다. `image_path`가 가리키는 실제 파일(`training/image_0/1000000.png`)은 `mmDataset/waymo/kitti_format/`에 **다른 인덱싱(7자리 0-base) · 다른 확장자(.jpg)**로만 존재해 그대로 못 쓴다. occ_path에 해당하는 필드 자체가 없다.
2. **`mmDataset/waymo/kitti_format/waymo_infos_train.pkl`**: mmengine v1.4 포맷(`data_list` 길이 158081), `images` 키가 **`CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT, CAM_SIDE_LEFT, CAM_SIDE_RIGHT`(5-camera, 실측 확인)**로 nuScenes와 다른 Waymo 고유 camera 구성을 보여준다. 이쪽도 occupancy GT 경로 필드는 없다.
3. **`Occ3D-Waymo/cam_infos.pkl`**: dict(정수 key 0~797) → 시퀀스별 → 프레임별(dict, 0부터) → camera별(0~4) `{ego2global, sensor2ego, intrinsics}`. **timestamp 필드 없음, 이미지 경로 필드 없음.**

**긍정적 발견(실측 확인)**: `cam_infos[549]`의 프레임 수(198)가 `voxel01/training/549.tar`의 프레임 수(198)와 정확히 일치한다 — 즉 `cam_infos.pkl`의 정수 key가 voxel GT의 시퀀스 id 체계를 공유한다. 이것이 신규 occupancy-specific info pkl을 만들 때 calib/pose 소스로 쓸 수 있는 다리가 된다.

**결론(업데이트)**: §6에서 보듯 참고 구현(CVT-Occ)이 바로 이 "서로 다른 pkl을 엮는" 문제를 이미 풀어놓은 실제 코드가 있다 — STCOcc용으로 새 info pkl을 만들거나, 혹은 CVT-Occ 방식처럼 **별도 통합 pkl을 만들지 않고 두 기존 pkl(`waymo_infos_*.pkl` + `cam_infos.pkl`)을 로더 안에서 join**하는 방식도 가능하다는 것이 확인됐다(아래 §6). 어느 쪽을 택할지는 STCOcc의 기존 `ann_file` 단일 pkl 관례(OpenOcc 때도 하나의 pkl로 처리)와 비교해 결정할 문제로, 이번 세션에서는 설계만 하고 실제 스크립트/코드는 작성하지 않았다 — `research_waymo/run_commands.md` 참고.

## 6. 5-camera 실제 이미지 파일 위치 — 상당 부분 해소됨(참고 구현 코드로 확인)

CVT-Occ `docs/dataset.md` + `waymo_temporal_zlt.py::get_data_info`(둘 다 1차 코드 출처)로 아래가 확인됐다:

- **카메라-폴더 매핑(공식 README 원문)**: "Front(image_0), front left(image_1), front right(image_2), side left(image_3), side right(image_4)".
- **그런데 pose 정보(= `cam_infos.pkl`)의 카메라 index와 실제 이미지 폴더 번호가 어긋나는 공식적으로 알려진 버그가 있다** — CVT-Occ 코드 주석 원문: *"I write the coresponding data file folder in the brackets. But the pose info idx dismatch the image data file."* 실제 우회 코드(`get_data_info`, `waymo_temporal_zlt.py` 157-163행 상당):
  ```python
  if idx_img == 2:
      image_paths.append(img_filename.replace('image_0', 'image_3'))
  elif idx_img == 3:
      image_paths.append(img_filename.replace('image_0', 'image_2'))
  else:
      image_paths.append(img_filename.replace('image_0', f'image_{idx_img}'))
  ```
  즉 **pose 배열의 index 2(front-right)와 3(side-left)이 실제 이미지 폴더 2/3과 서로 바뀌어 있고, 0/1/4는 정상**이다. 우리 코드를 작성할 때도 동일한 우회가 필요할 가능성이 높다 — 단, 이 버그가 우리 로컬 데이터(`mmDataset/waymo/kitti_format`)에도 동일하게 적용되는지는 **아직 직접 검증 안 함(아래 참고, 여전히 일부 BLOCKED)**.
- **경로 생성 방식**: `info['image']['image_path']`(기존 KITTI-style pkl의 필드, 예: `training/image_0/1000000.png`류)를 가져와 `image_0`을 `image_{0..4}`로 문자열 치환 — 즉 **5개 카메라 이미지가 전부 `image_0`과 동일한 디렉터리 트리 구조(동일 파일명, 디렉터리만 다름)로 존재한다고 가정**한다.
- **scene/frame id 디코딩**: `sample_idx = info['image']['image_idx']`; `scene_idx = sample_idx % 1000000 // 1000`; `frame_idx = sample_idx % 1000000 % 1000` — 즉 `image_idx`가 `(split_prefix)×1000000 + scene×1000 + frame` 형태의 합성 정수임을 코드로 확인했다. 이게 우리 로컬 `Occ3D-Waymo/waymo_infos_train.pkl`의 `image_idx`(또는 동급) 필드와 같은 인코딩인지는 **직접 대조하지 않았다 — 다음 세션에서 로컬 pkl 샘플 하나를 열어 이 공식과 맞춰보면 바로 확인 가능하다(낮은 노력으로 해소 가능한 잔여 BLOCKED).**
- **이미지 크기**: img0-2(front, front-left, front-right) 1280×1920; img3-4(side-left, side-right) 886×1920, 전부 640×960으로 resize+pad.

**위 두 항목도 이번 세션에 직접 검증 완료 — BLOCKED #1 사실상 해소됨:**
- `image_1`~`image_4` 디렉터리 전부 로컬에 존재하고(`mmDataset/waymo/kitti_format/training/image_{0..4}/`), 각각 198,068개 파일로 파일 수가 정확히 동일하다(실측 확인).
- `Occ3D-Waymo/waymo_infos_train.pkl`에서 `image_idx=549000`인 엔트리를 직접 찾아 `scene_idx=549, frame_idx=0`으로 정확히 디코딩됨을 확인(위 공식과 일치).
- 단, `image_path` 필드값은 `training/image_0/0549000.png`(확장자 `.png`)인데 **실제 파일은 `.jpg`**였다(`0549000.jpg` 실존 확인) — 사소하지만 실제 코드 작성 시 확장자를 치환해야 함을 기억할 것.
- camera index 2/3이 pose 데이터와 실제 이미지 폴더 사이에서 바뀌어 있다는 CVT-Occ의 버그 리포트는, 우리 로컬 복사본에서 **이미지 내용까지 비교해 직접 재현하지는 않았다**(파일 존재 여부만 확인) — 실제 코드 작성 시 이 우회 로직을 그대로 가져오되, 한 프레임이라도 시각화해 확인하는 것을 권장한다.
