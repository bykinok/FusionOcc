# 공통 DatasetSpec 설계 (문서만, 코드 배선은 하지 않음)

작성일: 2026-10-01. `WAYMO_EXTENSION_REQUIREMENTS_KO.md` 7절 요구사항에 대한 응답. 이번 패스에서는 **설계 문서까지만** 작성했다 — 실제 코드 배선은 Waymo info pkl(§`remaining_issues.md` #1)이 해결되고 실제 로더를 작성할 때 함께 하는 것이 안전하다고 판단했다(아직 쓰이지 않을 추상화를 먼저 만들면 검증 없이 틀린 인터페이스를 고정할 위험이 있다).

## 현재 상태 (이번 세션들에서 실제로 관찰한 것)

Occ3D-nuScenes, OpenOcc-nuScenes, (준비 중인) Occ3D-Waymo 세 GT/dataset 조합 모두에서 이미 공통적으로 config를 통해 전달되는 값들:

```text
dataset_name, eval_metric           # occ3d / openocc / (waymo 예정), miou / rayiou
occ_class_names, class_weights      # 리스트
num_classes, empty_idx(free_index)  # occ_class_names에서 파생
point_cloud_range, grid_config      # voxel 공간 정의
data_config['cams'], Ncams          # camera 이름/수
```

이 값들은 이미 detector/loss 코드에 하드코딩되지 않고 config → model 생성자 인자로 전달된다(OpenOcc 작업에서 `empty_idx` 생성자 기본값 17이 유일한 함정이었고, 그마저 모든 실제 config가 명시적으로 넘겨 당장은 문제가 안 됨을 확인함). **즉 DatasetSpec이 하려는 일의 상당 부분은 이미 "config 딕셔너리"라는 형태로 되어 있다** — 새로 만들어야 할 것은 이 값들을 하나의 이름 있는 객체로 묶어 재사용성을 높이고, GT resolver/evaluator 선택 로직까지 포함시키는 것이다.

## 제안하는 형태

```python
@dataclass
class DatasetSpec:
    dataset_name: str              # 'occ3d' | 'openocc' | 'occ3d_waymo'
    gt_family: str                 # 'occ3d' | 'openocc' | 'occ3d_waymo' (평가 protocol 라우팅 키)
    num_classes: int
    class_names: list[str]
    free_class_idx: int
    ignore_index: int = 255
    point_cloud_range: list[float]
    voxel_shape: tuple[int, int, int]
    camera_names: list[str]
    valid_mask_policy: str         # 'none' | 'camera_mask' | 'fov_mask' | ...
    gt_resolver: Callable[[str, str], str]   # (occ_path, dataset_name) -> 실제 GT 디렉터리
    eval_protocol: str             # 'miou' | 'rayiou'
```

## 기존 자산과의 대응 관계 (재사용 가능한 것)

- `gt_resolver`: 이미 존재한다 — `projects/STCOcc/stcocc/utils/gt_resolver.py::resolve_occ_gt_dir`(OpenOcc 작업에서 신설). Waymo를 추가할 때는 이 함수에 `dataset_name=='occ3d_waymo'` 분기만 추가하면 된다(파일 하나 수정, 4곳 중복 걱정 없음 — 이미 단일 지점이기 때문).
- `eval_protocol` 라우팅: `OccupancyMetric._compute_rayiou`가 이미 `self.dataset_name`으로 `ray_metrics_occ3d`/`ray_metrics_openocc`를 분기한다(OpenOcc 작업에서 검증됨). Waymo 추가 시 `ray_metrics_waymo.py`(신규, 또는 기존 모듈을 일반화)와 분기 한 줄만 추가.
- `valid_mask_policy`: OpenOcc 작업에서 이미 `baseline_plan.yaml`에 `none`/`global_free`/`legacy_occ3d_camera`/`method_plugin`/`optional_gt_raycast_reference` 5가지로 정의해뒀다 — Waymo도 동일한 정책 집합을 그대로 재사용 가능(Waymo의 `infov`/`origin_voxel_state`/`final_voxel_state`가 실제로 camera mask에 대응하는지는 `remaining_issues.md` #7이 풀려야 확정).

## 적용하지 않은 이유 (지금 배선하지 않는 이유)

1. Waymo의 `camera_names`/`point_cloud_range`(voxel01 기준)/`class_names`가 전부 또는 부분적으로 BLOCKED 상태라, 지금 DatasetSpec 인스턴스를 만들면 placeholder 값을 채워야 한다 — 요구사항 문서가 명시적으로 금지한 "추측으로 하드코딩"이 된다.
2. 기존 Occ3D/OpenOcc config들을 DatasetSpec 객체로 리팩터링하는 것은 광범위한 변경이라(이미 검증된 50개 Occ3D config + 2개 OpenOcc config), 이번 Waymo 준비 패스에서 회귀 위험을 키우지 않기 위해 보류했다 — `CLAUDE_CODE_OPENOCC_REQUIREMENTS_KO.md`의 "기존 legacy path를 광범위하게 리팩터링하여 회귀 위험을 만들지 않는다" 원칙과 동일하게 적용.

## 다음 단계 (Waymo info pkl 확정 후)

1. `gt_resolver.py`에 `occ3d_waymo` 분기 추가(소규모).
2. 신규 Waymo config(`stcocc_r50_704x256_16f_occ3d_waymo_baseline.py`)에서 위 DatasetSpec 필드에 해당하는 값들을 전부 config 최상단 변수로 명시 — 기존 OpenOcc screen config와 동일한 패턴.
3. 그 다음에야 실제 `DatasetSpec` dataclass를 코드로 만들지, 아니면 지금처럼 "config 변수 + 한 곳의 resolver/evaluator 분기"로 충분한지 재판단 — 세 번째 dataset(Waymo)까지 들어와도 `if dataset_name==...` 분기가 딱 1~2곳(resolver, evaluator)에만 머문다면 공식 dataclass 없이도 요구사항 7절의 "detector 내부로 확산시키지 않는다"는 목적은 이미 달성된 것으로 볼 수 있다.
