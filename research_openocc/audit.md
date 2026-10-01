# OpenOcc 지원 감사 (Audit) — stcocc-selective-free-supervision

작성일: 2026-10-01
범위: 코드 수정 없음, 읽기 전용 감사 + 실제 파일시스템/데이터 확인.
방법: `START_OPENOCC_KO.md` / `CLAUDE_CODE_OPENOCC_REQUIREMENTS_KO.md` / `DATASET_RECOMMENDATIONS_20261001_KO.md` 3개 문서를 읽은 뒤, 4개 영역(① config/loader/dataset/sampler, ② detector/loss, ③ evaluation, ④ GT resolver 일관성+실데이터)을 병렬로 정밀 감사하고, 그 결과 중 가장 중요한 주장들(val_evaluator 내용, selective λ no-op 분기, IterBasedTrainLoop의 StepLR 타이밍)은 코드/라이브러리 소스를 직접 열어 재검증했다. 추가로 멀티스케일 GT 파일 실존 여부를 직접 확인해, 어떤 fork도 찾지 못한 치명적 BLOCKED 항목 하나를 새로 발견했다(§3.5).

## 0. 결론 요약

**현재 `stcocc_r50_704x256_16f_openocc_12e.py`는 "OpenOcc occupancy-only 학습이 가능한 config"가 아니다. 다음 이유로 실행 자체가 불가능하거나(①), 실행되더라도 결과가 조용히 틀리거나 0점이 된다(②~⑤).**

| # | 심각도 | 요약 |
|---|---|---|
| ① | **치명적 / 실행 불가** | config가 `scale_1_2=True, scale_1_4=True, scale_1_8=True`를 요구하는데, OpenOcc GT 디렉터리에는 멀티스케일 파일(`labels_1_2/1_4/1_8.npz`)이 **단 하나도 존재하지 않는다**. STCOcc의 3-stage cascade(`num_stage=3`)는 이 파일들을 무조건 요구하므로, 이 config로 학습을 시작하면 첫 배치에서 바로 `FileNotFoundError`가 난다. |
| ② | **치명적 / 평가 조용히 0점** | `val_evaluator = dict(type='OccupancyMetric')`에 `dataset_name`/`ann_file`/`num_classes`/`free_index` 등이 전혀 명시되지 않았다. 기본값 `dataset_name='occ3d'`, `ann_file=None`이 적용되어 평가를 돌리면 **GT 로딩조차 없이 `{'mIoU':0.0,'count':0}`이 반환**되거나(ann_file 없음) `mask_camera` KeyError 후 같은 0점 fallback으로 떨어진다(occ3d 분기 오적용). |
| ③ | **치명적 / 연구 전제 위반** | selective λ(invisible-free supervision, 이 연구의 핵심 방법)가 camera mask 부재 시 **조용히 no-op**이 된다(`stcocc.py:347-352`). OpenOcc에는 camera mask가 없으므로, λ를 0.25로 설정해도 실제로는 λ=1.0(no-op)과 동일하게 학습된다 — 요구사항 T6("mask 없이 실행되면 명확한 오류")에 정면으로 위배. |
| ④ | **높음 / flow 미분리** | config는 flow-enabled(occupancy+flow) 상태이며 occupancy-only profile이 아직 존재하지 않는다. `flow_head`가 생성자에 넘어가는 한 `train_flow` 플래그는 죽은 코드이고 flow loss/추론이 계속 실행된다. |
| ⑤ | **높음 / temporal 무결성 불확실** | train sampler가 Occ3D의 grouped/recurrent sampler(`InfiniteGroupEachSampleInBatchSampler`) 대신 `DefaultSampler(shuffle=True)`로 바뀌어 있어, 16-frame recurrent temporal fusion이 요구하는 scene 연속성이 깨질 위험이 있다. |
| ⑥ | **중간 / 미세하지만 실재하는 버그** | `IterBasedTrainLoop` + `by_epoch=True StepLR` 조합에서, `after_train_epoch` 훅(= LR decay 트리거)이 전체 96060 iteration이 다 끝난 뒤 단 한 번만 호출됨을 mmengine 소스에서 직접 확인했다. **즉 이 config로 학습하면 LR decay가 사실상 전혀 적용되지 않는다** — "12epoch-equivalent screening"이라는 의도와 실제 동작이 어긋난다. |
| ⑦ | **중간 / GT resolver 분산** | `'gts'→'openocc_v2'` 문자열 치환이 하나의 resolver가 아니라 최소 5곳(로더 1곳, inline evaluator 3곳, file evaluator v2의 도달 불가능한 1곳)에 중복 구현돼 있다. 지금은 값이 같지만 요구사항 3.1 위반이며 향후 분기 위험이 있다. |
| ⑧ | **긍정적 사실** | OpenOcc GT 자체는 실제로 존재하고 품질도 확인됐다: train 28130 / val 6019 전수(100%) 커버리지, scene=850(700 train+150 val), schema(`semantics` int32, `instances` uint8, `flow` float32, free=16, car=0, mask 키 없음)가 요구사항 문서의 설명과 정확히 일치한다. |

아래 절들은 요구사항 문서의 섹션 순서(2~6절)에 맞춰 근거를 상세히 기록한다.

---

## 1. 실제 데이터 현황 (요구사항 3.1~3.4절)

- `data/nuscenes/openocc_v2` → symlink → `/home/h00323/DATA/mmDataset/nuscenes_stcocc/openocc_v2.1/openocc_v2`. **850개 scene 디렉터리**(nuScenes trainval 전체 700 train + 150 val과 일치).
- `/home/h00323/Downloads/openocc_v2`(852 dir, 아마 중복 다운로드본)와 scene 목록·임의 샘플 md5sum이 완전히 동일 — 버전 불일치 없음.
- `data/nuscenes/stcocc-nuscenes_infos_{train,val}.pkl`은 **Occ3D config와 완전히 동일한 info 파일**(train 28130건 / val 6019건, keys: `occ_path`, `token`, `scene_token`, `timestamp` 등). OpenOcc GT는 이 동일 info의 `occ_path`를 `'gts'→'openocc_v2'`로 치환해서 얻는다.
- `occ_path.replace('gts','openocc_v2') + '/labels.npz'` 기준 **train 28130건 + val 6019건 전수 조사 결과 누락 0건** — OpenOcc 쪽 "GT가 실제로 있는가"라는 요구사항 핵심 질문에 **명확히 "있다, 100% 커버"**로 답할 수 있다.
- `labels.npz` 키: `semantics`(int32, `(200,200,16)`), `instances`(uint8), `flow`(float32, `(200,200,16,2)`). **mask_lidar/mask_camera 키 없음** — camera visibility mask가 원천적으로 없다는 요구사항 문서의 전제가 실측으로 확인됨.
- semantics 값 분포 실측(scene-0001 한 샘플): `{0:280, 1:1879, 7:78, 8:32, 10:12106, 11:16, 12:5282, 13:13745, 14:4827, 15:11000, 16:590755}` — **값 0(=car, 요구사항이 "car=0"이라고 못박은 것과 일치)이 실제로 존재**하고, 16(=free)이 압도적 다수(~93%)로 free-space가 맞다.
- Occ3D `data/nuscenes/gts` 대조: `semantics`(uint8, free=17) + `mask_lidar` + `mask_camera`, 34149 sample. **스키마가 명확히 다르다** (17채널/free=16 vs 18채널/free=17, mask 유무).
- **§3.5에서 다루는 멀티스케일 GT는 OpenOcc 쪽에 전혀 생성되어 있지 않음** (Occ3D `gts/`에는 `labels_1_2/1_4/1_8.npz`가 존재, 날짜상 이번 연구 세션 중 생성된 것으로 보임).

## 2. Config 감사 — `stcocc_r50_704x256_16f_openocc_12e.py`

### 2.1 class/free index — 확인됨, 일치
- `:43-45` `occ_class_names` 17개(car…vegetation, free) — 요구사항 3.3절 순서와 정확히 일치.
- `:113` `num_classes = len(occ_class_names)` = 17, `:130` `empty_idx = occ_class_names.index('free')` = 16.
- `occupancy_head`(`:248`), `flow_head`(`:256`) 모두 같은 `num_classes` 변수를 참조 — config 내부 일관성 OK.

### 2.2 Geometry — 확인됨, Occ3D와 동일
- `:87-92` grid_config x/y=[-40,40,0.8], z=[-1,5.4,0.8], depth=[1.0,45.0,0.5]; `:100` point_cloud_range=[-40,-40,-1.0,40,40,5.4]. Occ3D 참조 config와 리터럴 동일. 실제 npz shape `(200,200,16)`과도 일치.

### 2.3 Flow — 확인됨: occupancy-only 아님
- `flow_head` 모듈이 config에서 실제로 생성됨(`:251-257`). `STCOccLoadOccGTFromFileOpenOcc`가 `flow`를 무조건 읽어 `voxel_flows`로 반환(로더 쪽, §3 참고). `train_pipeline`의 `Collect3D` keys(`:273-275`)에 `voxel_flows`가 **필수** 포함.
- 즉 현재 config는 요구사항 4.1절이 말하는 "occupancy-only profile"이 아니라 **flow-enabled native reference 시도**다. occupancy-only profile은 아직 별도로 만들어진 적이 없다.

### 2.4 Temporal sampler — 확인됨, 핵심 리스크
- Occ3D 참조 config는 `batch_size=1` + `batch_sampler=InfiniteGroupEachSampleInBatchSampler`로 `self.flag`(scene-sequence group) 순서를 강제한다.
- `openocc_12e`(`:305-309, 331-336`)는 `batch_size=samples_per_gpu` 직접 지정 + `sampler=dict(type='DefaultSampler', shuffle=True)`만 사용, **batch_sampler 없음**.
- `use_sequence_group_flag=True`(`:321,348`)라서 `nuscenes_dataset_occ.py`의 `self.flag`/`sequence_group_idx` 계산(`:418-517`) 자체는 여전히 수행되지만, `DefaultSampler`는 이 메타데이터를 전혀 쓰지 않는다. 16-frame recurrent temporal fusion(`history_frame_num=[16,8,4]`)이 요구하는 scene 연속성·순서 보장이 OpenOcc 학습 경로에서 깨질 수 있다 — 요구사항 4.3절이 정확히 우려한 상황.

### 2.5 `num_iters_per_epoch`의 4.554 배수 — 확인됨(수치), 출처는 BLOCKED
- `:84`: `num_iters_per_epoch = int(28130 // (8*2) * 4.554)`. `28130//16=1758`, `1758*4.554=8005.93→8005`. `max_iters = 12*8005 = 96060`.
- `num_gpus=8, samples_per_gpu=2`는 **참조용 리터럴**이며 현재 가용 GPU(로컬 2, 원격 mando-h100/h100_2 각 2)와 다르다. world_size가 바뀌면 동일 effective batch를 유지하려면 `samples_per_gpu`를 조정해야 하는데, 이 식은 world_size를 반영하지 않는 고정 수식이다.
- 4.554라는 계수의 유래(원 논문/저자 repo의 몇 epoch-equivalent 환산인지)는 코드 어디에도 주석/설명이 없다 — **BLOCKED, 출처 불명**. 요구사항 4.2절이 요구하는 "실제 sample/update 수로 확인"이 아직 안 됐다.

### 2.6 IterBasedTrainLoop + by_epoch=True StepLR — 확인됨, 실질적 버그 (직접 재검증 완료)
- `:378-381` `train_cfg=IterBasedTrainLoop(max_iters=total_epoch*num_iters_per_epoch, val_interval=num_iters_per_epoch)`, `:363-375` `StepLR(by_epoch=True, step_size=total_epoch, gamma=0.1)`.
- 설치된 mmengine 0.10.3 소스(`mmengine/runner/loops.py::IterBasedTrainLoop.run`)를 직접 열어 확인: `after_train_epoch` 훅은 `while self._iter < self._max_iters` 루프가 **전부 끝난 뒤 단 한 번만** 호출된다(본 세션에서 직접 `inspect.getsource`로 확인).
- `ParamSchedulerHook.after_train_epoch`가 `by_epoch=True` 스케줄러의 `.step()`을 호출하는 지점이 바로 이것이므로, **StepLR은 96060 iteration 내내 한 번도 step되지 않고 학습이 다 끝난 뒤에야 1번 step된다** — 설정 의도(12epoch마다 ×0.1 감쇠)와 실제 동작(사실상 감쇠 없음)이 어긋난다. 이는 추측이 아니라 라이브러리 소스를 직접 읽어 확인한 사실이다.

### 2.7 val_evaluator — 확인됨 (직접 재검증 완료), 치명적
```python
val_evaluator = dict(
    type='OccupancyMetric')
test_evaluator = val_evaluator
```
(`:386-389`, 직접 Read로 재확인) — `dataset_name`/`ann_file`/`num_classes`/`free_index`/`point_cloud_range` 등 **아무것도 명시하지 않는다.** §4(평가)에서 이게 어떤 결과를 낳는지 상세히 다룬다.

### 2.8 Class order 실데이터 대조 — 확인됨, 일치
- config의 0 car … 16 free 순서가 §1에서 실측한 npz의 값 분포(0 존재=car, 16 최빈값=free)와 상충하지 않는다. (공식 OpenOcc 출처 문서와의 완전한 역대조는 범위 밖.)

---

## 3. GT Loading Pipeline — `projects/STCOcc/stcocc/transforms/pipelines/loading.py`

클래스명: `STCOccLoadOccGTFromFileOpenOcc` (등록명, `:327 @TRANSFORMS.register_module(name='STCOccLoadOccGTFromFileOpenOcc')`; 클래스 본체는 `LoadOccGTFromFileOpenOcc`, `:328`).

### 3.1 GT 경로 생성 — 확인됨, 하드코딩된 string replace
- `:356` 부근에서 `'gts'` 문자열을 `'openocc_v2'`로 치환해서 경로를 만든다. Occ3D 전용 클래스(`LoadOccGTFromFileCVPR2023`, `:144`)와 거의 동일한 구조를 복붙해 만든 형태.

### 3.2 Flow — 확인됨, 무조건 로드 (occupancy-only 미지원)
- `load_flow` 파라미터 자체가 클래스에 없다. `flow = occ_labels['flow']`를 **항상** 읽어 `results['voxel_flows']`에 넣는다. 끌 수 있는 스위치가 없음.

### 3.3 ray_mask2 — 확인됨, 현재 비활성이지만 생성 파일 자체가 없음
- `:352` `'gts'→'openocc_v2_ray_mask'`로 별도 경로를 만들고 `:459` `ray_mask['ray_mask2']`를 읽는 코드가 있다. `load_ray_mask` 기본값은 `False`(`:329`)이고 `openocc_12e` config는 이 인자를 넘기지 않으므로 **현재 config에서는 비활성** — 추가로 확인한 결과 `openocc_v2_ray_mask`라는 이름의 디렉터리 자체가 로컬/NAS 어디에도 존재하지 않는다(파일시스템 검색 결과 0건). 즉 이 경로는 지금 켜면 바로 터진다.
- 이름/생성 방식에 대한 주석이 코드에 없어, "camera visibility mask와 같은 정보"라고 가정할 근거도 "다르다"는 명시적 근거도 코드만으로는 확정할 수 없다 — **BLOCKED, 불확실**. 요구사항 3.4절이 경고한 대로 `ray_mask2`를 camera mask와 섞어 쓰면 안 된다는 원칙을 지키려면, 이 필드의 실제 생성 방식(원 OpenOcc/STCOcc 공식 repo 문서)을 먼저 확인해야 한다.

### 3.4 Selective λ 관련 로딩 — 확인됨, camera mask 키 자체가 없음
- `LoadOccGTFromFileOpenOcc` 전체에 `mask_camera`/`voxel_mask_camera` 관련 코드가 전혀 없다(Occ3D용 클래스에는 `:287-290`에 `mask_camera` 처리가 있는 것과 대조적). 즉 OpenOcc 경로에서는 애초에 camera mask 키가 `results`/`kwargs`에 들어갈 방법이 없다 → §5.2(detector)의 no-op 분기로 직결.

### 3.5 멀티스케일 GT — **신규 발견, 치명적 BLOCKED (이 감사에서 직접 확인, 어느 fork도 다루지 않음)**
- `openocc_12e` config의 로더 호출: `dict(type='STCOccLoadOccGTFromFileOpenOcc', scale_1_2=True, scale_1_4=True, scale_1_8=True)` (직접 grep/Read로 확인).
- 클래스 코드(`:374-428`)는 `scale_1_2/1_4/1_8=True`일 때 각각 `{occ_gt_path}/labels_1_2.npz`, `labels_1_4.npz`, `labels_1_8.npz`가 없으면 **명시적으로 `FileNotFoundError`를 raise**하도록 이미 방어적으로 짜여 있다("This file is required for multi-scale supervision" 메시지 포함) — 이 자체는 요구사항 6.5절("실패를 0점으로 숨기지 않음")의 정신과 일치하는 좋은 설계다.
- 그런데 **직접 파일시스템을 확인한 결과, `data/nuscenes/openocc_v2/scene-*/*/` 밑에는 `labels.npz` 단 하나뿐이고 `labels_1_2/1_4/1_8.npz`는 단 하나도 존재하지 않는다**(Occ3D `gts/` 쪽은 반대로 4개 파일이 모두 존재 — 아마 이번 연구 세션 중 Occ3D용으로만 생성됨).
- STCOcc의 occupancy_head는 `num_stage=3`(config `:110`) cascade 구조로, detector의 `forward_train`이 `kwargs['voxel_semantics_1_{2,4,8}']`를 **무조건** 읽는다(§5.7, 다른 fork가 확인) — 즉 멀티스케일 GT는 끌 수 있는 옵션이 아니라 아키텍처가 구조적으로 요구하는 필수 입력이다.
- **결론: 이 config로 학습을 시작하면 첫 iteration의 데이터 로딩 단계에서 즉시 `FileNotFoundError`로 죽는다.** 지금까지 git log상 이 config가 한 번도 실행된 적이 없다는 §6 발견과 정합적이다.
- 다행히 `tools/generate_ms_occ.py`가 이미 **OpenOcc를 지원하도록 작성돼 있다**(`--dataset occ3d|openocc` 인자, `empty_idx=16` 디폴트가 OpenOcc free index와 일치, `occ_path.replace('gts','openocc_v2')` 분기 존재) — 다만 OpenOcc에 대해 실제로 실행된 적은 없다(출력 파일 부재로 확인). 이 도구로 멀티스케일 GT를 생성하는 것 자체는 코드 변경이 필요 없는 구현 작업이지만, (a) 28130+6019개 샘플 × 3개 scale 파일 생성이라는 **대규모 파일 쓰기 작업**이고 (b) 쓰기 대상이 `/home/h00323/DATA/mmDataset/nuscenes_stcocc/openocc_v2.1/openocc_v2`(원본 GT 보관 위치)라서, 요구사항 4.4절의 "변환이 필요하면 별도 출력 경로·version을 쓰라"는 원칙과 상충할 소지가 있다 — **실행 전 사용자 승인이 필요한 항목으로 분류**(§8의 "다음 단계"에 포함).

---

## 4. Detector / Loss — `stcocc.py`, `focal_loss.py`, `semkitti.py`

### 4.1 flow_head 분리 함정 — 확인됨
- `train_flow` 생성자 인자는 `self.train_flow`에 저장되는 것 외에 **코드 전체에서 다시 읽히지 않는다**(2곳뿐). 실제 게이트는 전부 `self.with_specific_component('flow_head')`(= `flow_head is not None`)이다 — forward_train의 flow loss(`:1070,1100`), simple_test의 flow 추론(`:798-803`).
- `flow_head`는 `__init__`에서 `config.flow_head`가 주어지면 바로 모듈로 빌드된다(`:116`). **`train_flow=False`로 바꿔도 `flow_head`를 config에서 제거/`None`으로 하지 않으면 flow loss·추론이 그대로 실행된다** — 요구사항 4.1절이 미리 경고한 함정이 코드에 실재함.
- `flow_head` 없을 때 `simple_test`는 `return_pred_voxel_flows = torch.zeros(...)`(`:803`)로 0-flow를 반환 — 이 0-flow가 그대로 mAVE 계산에 들어가면 "완벽한 정지 예측"처럼 보일 위험(요구사항 4.1절이 명시적으로 경고한 상황과 동일).

### 4.2 Selective λ no-op — 확인됨 (직접 재검증 완료), **이 연구에서 가장 심각한 항목**
- `forward_train`: `camera_mask=kwargs.get(camera_mask_key, None)`(`:1096`) — `.get()` 기본값 `None`이라 키가 없어도 죽지 않는다.
- `get_voxel_loss`(`:347-352`, 직접 Read로 재확인):
  ```python
  voxel_weight = None
  _lambda_is_noop = (not isinstance(self.lambda_inv_free, dict)) and self.lambda_inv_free == 1.0
  if camera_mask is not None and not _lambda_is_noop:
      voxel_weight = self.build_inv_free_voxel_weight(...)
  ```
  코드 주석은 "Stage 2 invisible-free reweighting: only active when a camera mask is available for this scale AND lambda_inv_free != 1.0" — 즉 **이 설계는 의도적**이다(Occ3D λ=1.0 설정이 w/o-mask baseline과 bit-identical이 되도록). 하지만 그 부작용으로, **OpenOcc처럼 camera mask가 원천적으로 없는 환경에서는 `lambda_inv_free` 값이 무엇이든(0.25든 뭐든) 항상 `voxel_weight=None`(=균등 가중치)으로 조용히 빠진다.** 에러도 경고도 없다.
- 요구사항 T6("legacy selective λ가 필요한 mask 없이 실행되면 명확한 오류")가 **현재 구현되지 않았다** — 수정이 필요한 항목. (예: `camera_mask is None and not _lambda_is_noop`일 때 명시적 `assert`/`raise`를 추가해야 함.)

### 4.3 free_index/num_classes 하드코딩 — 확인됨, 대부분 config-driven, 함정 1곳
- `stcocc.py:43` 생성자 기본값 `empty_idx=17`(Occ3D 전용 리터럴). 현재 두 config(Occ3D `invfree_l025`, `openocc_12e`) 모두 `empty_idx=occ_class_names.index('free')`로 명시 전달하므로 **지금 당장은 쓰이지 않지만**, 향후 새 config가 이 인자를 빠뜨리면 조용히 17(Occ3D 값)이 적용되는 함정으로 남는다.
- `semkitti.py:172` `sem_scal_loss`의 `begin = 1 if n_classes == 19 else 0`은 SemanticKITTI(19클래스) 유산 하드코딩 — Occ3D(18)·OpenOcc(17) 둘 다 19가 아니므로 현재는 `begin=0`으로 동일하게 동작(죽은 분기, 당장 해는 없음).
- `semkitti.py:135` `sem_scal_loss`는 free index를 파라미터로 받지 않고 **"free=마지막 채널(n_classes-1)"을 암묵 가정**(`:173` `for i in range(begin, n_classes-1)`). Occ3D(17=마지막)·OpenOcc(16=마지막) 둘 다 이 관례를 따르므로 지금은 문제없지만, 일반화(Waymo 등) 시 free가 마지막이 아니면 깨지는 암묵 의존성으로 기록.
- `focal_loss.py`의 class 수는 `pred.size(1)`(텐서 shape)로 동적 추론 — 하드코딩 없음. `class_weights`는 전부 config에서 주입(`stcocc.py:102`), focal_loss.py 자체에는 리스트 없음.
- occupancy_head 출력 채널은 config로 빌드된 모듈이 결정 — detector/loss 코드 자체에는 18/17 리터럴이 없음.

### 4.4 멀티스케일 GT 소비 — 확인됨 (§3.5와 교차 확인 완료)
- `forward_train`(`:1076-1080`)은 `kwargs['voxel_semantics_1_{2,4,8}']`를 **그대로** 꺼내 쓸 뿐, detector/loss 안에 average-pool/nearest-downsample 코드가 전혀 없다. 멀티스케일 생성 책임은 전적으로 §3.5에서 다룬 로딩 파이프라인(결국 사전 생성된 `labels_1_2/1_4/1_8.npz`)에 있다 — 그 파일이 없으면(OpenOcc의 현재 상태) 이 코드에 도달하기도 전에 로더에서 죽는다(§3.5).

---

## 5. Evaluation — `occupancy_metric.py`, `ray_metrics_occ3d.py` / `ray_metrics_openocc.py`, `eval_all_metrics.py`

### 5.1 `OccupancyMetric` 기본값 — 확인됨
- `:342-356`: `num_classes=17`(우연히 OpenOcc와 일치, Occ3D는 각 config가 18을 명시해야 정상), `use_lidar_mask=False`, `use_image_mask=False`, `ann_file=None`, `dataset_name='occ3d'`(!), `eval_metric='miou'`.

### 5.2 openocc_12e의 val_evaluator 적용 결과 — 확인됨, 치명적 (§2.7과 연결)
- `ann_file=None` → `compute_metrics()`가 데이터 로딩 없이 즉시 `{'mIoU':0.0,'count':0}` 반환(`:469-474`).
- 설사 ann_file을 나중에 넘기더라도 `dataset_name` 기본값이 `'occ3d'`로 남아있으면, `mask_camera` 키를 읽으려다(`:1109,664`) OpenOcc npz(키 없음)에서 KeyError → 아래 5.5의 "실패를 0점으로 숨기는" try/except로 떨어져 또 0점.

### 5.3 RayIoU 라우팅 — 확인됨, 존재하지만 도달 불가
- `_compute_rayiou`(`:1540-1545`)는 `self.dataset_name`에 따라 `ray_based_miou_openocc` vs `ray_based_miou_occ3d`로 올바르게 분기하는 코드가 **존재**한다. 하지만 §5.2대로 `dataset_name`이 설정되지 않으면 이 분기 자체에 도달하지 못하고 항상 occ3d 경로로 간다(이 경우 `gt_flow=np.zeros(...)`로 강제 대체, `:1522-1523`, OpenOcc 실제 flow GT 무시).

### 5.4 class_names 하드코딩 — 확인됨, Occ3D 18-class 순서 고정
- `Metric_mIoU.class_names`(의존 파일 `mmdet3d/datasets/occ_metrics.py:51-54`)가 Occ3D 18-class 순서로 **하드코딩**돼 있다. `num_classes` 파라미터와 무관하게 이 리스트가 그대로 쓰여, OpenOcc(`num_classes=17`, 순서도 다름)로 평가하면 confusion matrix 숫자 자체는 맞아도 **`IoU_{class_name}` 키와 콘솔 출력의 클래스 이름이 전부 엉뚱한 클래스를 가리킨다**(예: index 4가 Occ3D='car'지만 OpenOcc=`construction_vehicle`). 요구사항 3.3절이 우려한 지점이 실재한다.

### 5.5 실패를 0점으로 숨기는 경로 — 확인됨, 최소 2곳 + 샘플 단위 1곳
- `compute_metrics()`의 포괄적 `try/except Exception`(`:481-496`)이 어떤 예외든(GT 없음, KeyError 등) 잡아 warning 출력 후 `{'mIoU':0.0,'count':0}` 반환.
- `_process_one_chunk`류에도 `except Exception: pass` / `except Exception as e: print(...)`(`:807-810`)로 개별 샘플 실패가 조용히 스킵됨.
- 요구사항 6.5절("실패를 mIoU=0,count=0인 정상 결과로 CSV에 적지 않음")을 현재 코드는 만족하지 못한다 — strict_evaluation 모드가 아직 없음.

### 5.6 `ray_metrics_occ3d.py` vs `ray_metrics_openocc.py` — 확인됨, openocc 버전이 구버전
| 항목 | occ3d | openocc |
|---|---|---|
| class 수/순서 | 18 | 17(요구사항 3.3절 순서와 일치) |
| free index | 17 | 16 |
| origin time-offset 분해(`ORIGIN_TIME_BINS`, research_v2 E2에서 추가) | 있음 | **없음** |
| radius/height breakdown(`RADIUS_BINS`/`HEIGHT_BINS`) | 있음 | **없음** |
| threshold(1/2/4m), DVR renderer, 좌표계 | 동일 | 동일 |

openocc 버전은 구조적으로 occ3d 버전의 구버전이며, 이번 연구(E2, E3)에서 Occ3D 쪽에 추가된 진단 기능이 OpenOcc에 backport되지 않았다.

### 5.7 `eval_all_metrics.py`의 dataset_name 결정 — 코드에 없음(BLOCKED)
- 전체 파일에 `dataset_name`/`openocc`/`occ3d` 문자열이 전혀 등장하지 않는다. `--miou-config`/`--rayiou-config`로 받은 config를 그대로 `tools/test.py`에 넘길 뿐, `dataset_name`은 그 config의 `val_evaluator` dict에 전적으로 위임된다.
- openocc용 `*_rayiou.py` 사본 config(Occ3D에는 `stcocc_r50_704x256_16f_occ3d_e12_stage*_rayiou.py` 등 다수 존재)가 **현재 하나도 없다** — 만들어야 할 신규 산출물.

### 5.8 AUROC/ECE/NLL — 확인됨, 계산 코드는 mask 없이도 동작하지만 population 이름 구분 없음
- `dataset_name=='occ3d' or use_image_mask`일 때만 mask를 읽고, 아니면 `valid=np.ones(...)`(all-valid)로 대체 — mask가 없어도 코드가 죽지는 않는다.
- 다만 이때 평가 population이 "camera-visible voxel"에서 "전체 valid voxel"로 바뀌는데, 결과 딕셔너리 키는 여전히 `mIoU`/`ECE`/`NLL`로 Occ3D와 동일하다 — 요구사항 6.2/6.4절이 요구하는 `mIoU_openocc_all_valid` 같은 population 구분 네이밍이 코드에 없다.

### 5.9 `export_occ_logits.py` — 확인됨, OpenOcc 비명시 지원 + mask 부재를 전부-visible로 처리
- 자체 GT 경로 로직은 없고 config의 pipeline(= `STCOccLoadOccGTFromFileOpenOcc`)에 위임 — 이 자체는 안전.
- `_process_dense_sample`(`:227-278`)이 `mask_camera` 필드가 없으면 `mask=np.ones(...)`(전부 visible)로 간주(`:248-250`). docstring에도 OpenOcc 언급이 없어 애초에 OpenOcc용으로 설계되지 않은 도구로 보인다 — 요구사항 6.5절("missing camera mask를 모두 visible로 기록 금지")과 충돌 소지.

### 5.10 `compute_metrics_from_file.py` / `_v2.py` — 확인됨, v1 미지원·v2는 dead code
- v1: openocc 분기 전혀 없음, `dataset_name='occ3d'` 하드코딩 — **OpenOcc 미지원**.
- v2: `if self.dataset_name=='openocc': occ_path=occ_path.replace(...)` 분기가 추가돼 있으나, `main()`에서 `OccupancyMetricV2` 생성 시 `dataset_name='occ3d'`가 **하드코딩**되고 CLI에 이를 바꿀 옵션이 없어 **도달 불가능한 dead code**다. 설령 수동으로 바꿔도 `use_image_mask=True` 기본값 때문에 `mask_camera` KeyError로 죽는다.

---

## 6. 기존 산출물 — 확인됨: 전혀 없음
- `git log --all`로 `stcocc_r50_704x256_16f_openocc_12e.py`를 건드린 커밋은 최초 STCOcc 스캐폴드 임포트 1건뿐 — 이후 한 번도 수정/실행되지 않았다.
- `work_dirs/`, `/NAS` 전체에서 "openocc" 이름의 디렉터리 없음. `research_v2/`, `research/`, `projects/STCOcc/reports/*.md`, 전체 csv에서 "openocc" 문자열 0건.
- §3.5의 멀티스케일 GT 파일 부재와 정합적 — 이 config는 **단 한 번도 실행된 적이 없다.**

---

## 7. 요구사항 문서 8절 질문에 대한 답

**기존 무엇을 재사용했고 무엇을 고쳤는가?**
아직 코드를 전혀 고치지 않았다 — 이번 턴은 감사만 수행했다(사용자 지시: "먼저 코드를 수정하기 전에 audit 결과부터 보여줘"). 재사용 가능 자산: `STCOccLoadOccGTFromFileOpenOcc`(loader), `ray_metrics_openocc.py`(RayIoU 커널), `OccupancyMetric`의 dataset_name 라우팅 골격, `tools/generate_ms_occ.py`(멀티스케일 생성기, 이미 openocc 지원).

**현재 OpenOcc GT 파일이 실제로 있는가? 어떤 version인가?**
있다. `openocc_v2.1`(디렉터리명 기준), 850 scene, train/val 전수(28130/6019) 100% 커버. 단 멀티스케일(`labels_1_2/1_4/1_8.npz`)은 **없다**(§3.5) — 이것이 생성되지 않으면 학습 자체가 시작조차 안 된다.

**mask/flow 없이 occupancy baseline 학습과 평가가 가능한가?**
**현재는 불가능하다.** 이유: (a) 멀티스케일 GT 부재로 로더에서 즉시 크래시(§3.5), (b) occupancy-only profile 자체가 아직 없음(현재 config는 flow 필수, §2.3/4.1), (c) val_evaluator 미설정으로 평가가 조용히 0점(§2.7/5.2), (d) selective λ가 mask 없이 no-op으로 빠짐(§4.2, baseline `none` 정책에는 영향 없지만 향후 method_plugin 연결 시 문제).

**native reference와 새 screen profile의 차이는 무엇인가?**
아직 `openocc_occ_only_screen`/`openocc_occ_only_full` profile을 만들지 않았으므로 비교 대상이 없다 — 다음 단계에서 생성 필요.

**Occ3D 회귀검사를 통과했는가?**
해당 없음 — 코드를 아직 수정하지 않았으므로 회귀검사 대상 자체가 없다. (단, 멀티스케일 GT 생성기를 돌리거나 GT resolver를 통합하는 등 공유 코드를 건드리게 되면 그 시점에 반드시 Occ3D 회귀검사를 수행해야 한다.)

**final annotation-free method는 구현됐는가, 연결 지점만 있는가?**
연결 지점도 아직 불완전하다. selective λ 메커니즘 자체는 존재하지만(§4.2), OpenOcc 환경에서는 mask가 없어 자동으로 no-op이 되는 것 외에 "명시적으로 실패시키는" 안전장치가 없다(T6 미충족) — method_plugin은 당연히 미구현.

**장시간 실행을 위해 추가로 필요한 자원과 승인 항목은 무엇인가?**
§8(다음 단계)에 정리.

---

## 8. 다음 단계 제안 (각각 별도 승인 필요, 지금 바로 실행하지 않음)

1. **멀티스케일 OpenOcc GT 생성** — `tools/generate_ms_occ.py --dataset openocc`를 실제로 실행해 `labels_1_2/1_4/1_8.npz`를 만들어야 config가 돌아간다. 28130+6019개 샘플 × 3개 파일의 대규모 쓰기 작업이고, 쓰기 대상이 원본 GT 디렉터리(`/home/h00323/DATA/mmDataset/nuscenes_stcocc/openocc_v2.1/openocc_v2`)라서 요구사항 4.4절의 "별도 출력 경로" 원칙과 상충할 수 있음 — 별도 output path로 생성할지, 원본 위치에 생성할지(Occ3D 선례처럼) 결정 필요. class-preservation test(T9 관련)도 함께 준비해야 함.
2. **occupancy-only profile 신규 생성** — `flow_head=None`, `Collect3D`에서 `voxel_flows` 제거, flow 관련 loss/metric을 NOT_APPLICABLE로 명시하는 별도 config(`openocc_occ_only_screen` 등). 기존 `openocc_12e`(native reference 후보)는 보존.
3. **val_evaluator 명시화** — `dataset_name='openocc', num_classes=17, free_index=16, ann_file=<openocc val pkl>, class_names=<17개>` 등을 명시한 평가 config 작성, `eval_all_metrics.py`용 `*_openocc_rayiou.py` 사본 config 추가.
4. **GT resolver 통합** — 5곳에 중복된 `'gts'→'openocc_v2'` string replace를 공통 함수/상수로 통합(T7 대비).
5. **selective λ 안전장치 추가** — `stcocc.py:347` 부근에 "camera mask가 없는데 λ≠1.0이면 명시적으로 실패"하는 assert/raise 추가(T6 충족).
6. **Metric_mIoU.class_names 일반화 또는 별도 경로** — OpenOcc 평가 시 잘못된 클래스 이름이 출력되지 않도록 조치.
7. 위 변경들이 끝난 뒤: **unit test(T1~T15) → 1 iteration backward → 소규모 smoke evaluation** 순서로 승인된 GPU(ssh mando-h100/occfrmwrk_h100_new, ssh mando-h100_2/occfrmwrk_h100_2_new)에서 수행.

이 감사 결과에 대해 어느 항목부터, 어디까지 진행할지 지시해 주시면 그 범위 내에서만 코드를 수정하겠습니다.
