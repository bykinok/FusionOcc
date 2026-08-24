#!/usr/bin/env python3
"""
평가 로그 파일들에서 클래스별 IoU를 추출하고, nuScenes-Occ3D 클래스별 GT 빈도(popularity)와의
스피어만(Spearman) 순위상관계수를 로그(모델)별로 계산합니다.

"mIoU/RayIoU 같은 지표가 빈도 높은(popular) 클래스에 편향되어 있다"는 가설을 검증하는 용도.
클래스 빈도(popularity)는 --freq-source 로 아래 두 가지 중 선택합니다.
  - camera_mask (기본값): projects/GaussianFormer/gaussianformer/losses/occupancy_loss.py 의
    nusc_class_frequencies. camera_mask를 적용한 학습 데이터셋 기준 클래스 분포.
  - no_mask: camera_mask를 적용하지 않은 원본 데이터셋 기준 클래스 분포 (voxel count).
둘 다 모델/실행과 무관한 고정값이라, 포맷이 다른 로그(TP/FN 유무)끼리도 동일한 기준으로 비교할 수 있다.

지원하는 로그 포맷 (자동 감지):
  1) occ3d mIoU 스타일: "===> per class IoU of N samples:" 블록
     (예: "===> car - IoU = 51.78, TP = 2418911, FP = 1312461, FN = 940467")
  2) RayIoU 스타일 표: "| Class Names | IoU@1 | IoU@2 | IoU@4 | AVE |"
     (클래스별 IoU = IoU@1/@2/@4의 평균)

Usage:
  python tools/analyze_class_freq_iou_spearman.py LOG1 [LOG2 ...]
  python tools/analyze_class_freq_iou_spearman.py --freq-source no_mask --verbose LOG1 LOG2
"""
import argparse
import re

from scipy.stats import spearmanr

# occ3d/nuScenes-Occupancy 표준 17개 시맨틱 클래스 순서 (free/empty 제외)
CLASS_ORDER = [
    "others",
    "barrier",
    "bicycle",
    "bus",
    "car",
    "construction_vehicle",
    "motorcycle",
    "pedestrian",
    "traffic_cone",
    "trailer",
    "truck",
    "driveable_surface",
    "other_flat",
    "sidewalk",
    "terrain",
    "manmade",
    "vegetation",
]

# nusc_class_frequencies[:17] (projects/GaussianFormer/gaussianformer/losses/occupancy_loss.py)
# camera_mask를 적용한 학습 데이터셋 기준 클래스별 GT voxel 수.
CLASS_FREQ_CAMERA_MASK = {
    "others": 944004,
    "barrier": 1897170,
    "bicycle": 152386,
    "bus": 2391677,
    "car": 16957802,
    "construction_vehicle": 724139,
    "motorcycle": 189027,
    "pedestrian": 2074468,
    "traffic_cone": 413451,
    "trailer": 2384460,
    "truck": 5916653,
    "driveable_surface": 175883646,
    "other_flat": 4275424,
    "sidewalk": 51393615,
    "terrain": 61411620,
    "manmade": 105975596,
    "vegetation": 116424404,
}

# camera_mask 미적용(원본) 데이터셋 기준 클래스별 voxel count. "free" 클래스는 CLASS_ORDER에
# 포함되지 않으므로 제외 (원본 통계에서는 전체의 95.13%를 차지).
CLASS_FREQ_NO_MASK = {
    "others": 2082349,
    "barrier": 3012970,
    "bicycle": 234046,
    "bus": 5385402,
    "car": 34146494,
    "construction_vehicle": 2044124,
    "motorcycle": 325765,
    "pedestrian": 3330253,
    "traffic_cone": 543815,
    "trailer": 5785079,
    "truck": 13521112,
    "driveable_surface": 198278651,
    "other_flat": 4895895,
    "sidewalk": 56540471,
    "terrain": 66504617,
    "manmade": 227803562,
    "vegetation": 252374615,
}

FREQ_SOURCES = {
    "camera_mask": CLASS_FREQ_CAMERA_MASK,
    "no_mask": CLASS_FREQ_NO_MASK,
}

PER_CLASS_HEADER_RE = re.compile(r"===>\s*per class IoU of \d+ samples:")
MIOU_LINE_RE = re.compile(r"===>\s*mIoU of \d+ samples:")
CLASS_TP_LINE_RE = re.compile(
    r"===>\s*(\w+)\s*-\s*IoU\s*=\s*([\d.]+),\s*TP\s*=\s*(\d+),\s*FP\s*=\s*(\d+),\s*FN\s*=\s*(\d+)"
)

TABLE_HEADER_RE = re.compile(r"\|\s*Class Names\s*\|")
TABLE_ROW_RE = re.compile(r"\|\s*([A-Za-z_]+)\s*\|(.+)\|\s*$")


def parse_occ3d_style(lines):
    """'===> per class IoU of N samples:' 블록에서 {class: IoU(%)} 추출."""
    result = {}
    in_block = False
    for line in lines:
        if PER_CLASS_HEADER_RE.search(line):
            in_block = True
            continue
        if not in_block:
            continue
        if MIOU_LINE_RE.search(line):
            break
        m = CLASS_TP_LINE_RE.search(line)
        if m:
            cls, iou = m.group(1), float(m.group(2))
            if cls in CLASS_ORDER:
                result[cls] = iou
    return result


def parse_rayiou_style(lines):
    """RayIoU 표에서 {class: mean(IoU@1, IoU@2, IoU@4) * 100} 추출 (첫 번째 표만, 이후 Radius/Height 표는 무시)."""
    result = {}
    in_table = False
    for line in lines:
        if not in_table:
            if TABLE_HEADER_RE.search(line):
                in_table = True
            continue
        if line.strip().startswith('+'):
            continue
        m = TABLE_ROW_RE.search(line)
        if not m:
            continue
        name = m.group(1).strip()
        if name == 'MEAN':
            break
        if name not in CLASS_ORDER:
            continue
        cells = [c.strip() for c in m.group(2).split('|')]
        vals = []
        for c in cells[:3]:  # IoU@1, IoU@2, IoU@4 (4번째 컬럼 AVE는 제외)
            try:
                vals.append(float(c))
            except ValueError:
                pass
        if vals:
            result[name] = sum(vals) / len(vals) * 100.0  # 0~1 -> 0~100 스케일로 통일 (순위상관에는 영향 없음)
    return result


def parse_log(log_path):
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    per_class_iou = parse_occ3d_style(lines)
    fmt = 'occ3d'
    if not per_class_iou:
        per_class_iou = parse_rayiou_style(lines)
        fmt = 'rayiou'
    return per_class_iou, fmt


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('logs', nargs='+', help='평가 로그 파일 경로들')
    parser.add_argument(
        '--freq-source', choices=sorted(FREQ_SOURCES), default='camera_mask',
        help="클래스 빈도(popularity) 기준 데이터셋. 'camera_mask'(기본값): camera_mask 적용 학습 데이터셋"
             " 기준 분포. 'no_mask': camera_mask 미적용 원본 데이터셋 기준 분포."
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='로그별로 (클래스, 빈도, IoU) 상세 테이블도 함께 출력'
    )
    args = parser.parse_args()
    class_freq = FREQ_SOURCES[args.freq_source]

    results = []
    for log_path in args.logs:
        per_class_iou, fmt = parse_log(log_path)
        if not per_class_iou:
            print(f"[경고] '{log_path}'에서 클래스별 IoU를 파싱하지 못했습니다.")
            continue

        classes = [c for c in CLASS_ORDER if c in per_class_iou]
        freq = [class_freq[c] for c in classes]
        iou = [per_class_iou[c] for c in classes]
        rho, pval = spearmanr(freq, iou)
        results.append({
            'log': log_path, 'fmt': fmt, 'n': len(classes), 'rho': rho, 'pval': pval,
            'classes': classes, 'freq': freq, 'iou': iou,
        })

        if args.verbose:
            print(f"\n=== {log_path} ({fmt}) ===")
            print(f"  {'Class':<22} {'Freq(GT voxel)':>16} {'IoU':>8}")
            for c, f_, i_ in sorted(zip(classes, freq, iou), key=lambda x: -x[1]):
                print(f"  {c:<22} {f_:>16,} {i_:>7.2f}%")

    if not results:
        raise SystemExit("파싱에 성공한 로그가 없습니다.")

    print(f"\n[빈도 소스: {args.freq_source}]")
    print(f"{'Log file':<70} {'Format':<8} {'#Class':>7} {'Spearman rho':>13} {'p-value':>10}")
    print('-' * 112)
    for r in results:
        print(f"{r['log']:<70} {r['fmt']:<8} {r['n']:>7} {r['rho']:>13.4f} {r['pval']:>10.4g}")

    if len(results) > 1:
        avg_rho = sum(r['rho'] for r in results) / len(results)
        print('-' * 112)
        print(f"모델 {len(results)}개 평균 Spearman rho (클래스 빈도 vs IoU): {avg_rho:.4f}")

    print(
        "\n해석: rho > 0 이고 유의(p < 0.05)하면 '빈도가 높은 클래스일수록 IoU가 높게 나온다'는"
        " 편향 가설을 지지하는 근거입니다."
    )


if __name__ == '__main__':
    main()
