#!/usr/bin/env python3
"""
로그 파일에서 Radius-based mIoU를 파싱하여
0-20m, 20-35m, 35+m 그룹으로 재계산하는 스크립트
"""

import re
import sys
from collections import defaultdict


# 그룹 정의: {그룹명: [포함할 radius range들]}
RADIUS_GROUPS = {
    "0-20m":  ["0-20m"],
    "20-35m": ["20-25m", "25-30m", "30-35m"],
    "35+m":   ["35-40m", "40-45m", "45-50m"],
}

# 출력 클래스 순서
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


def parse_radius_stats(log_path: str) -> dict:
    """
    로그 파일에서 Radius range별 클래스별 TP/FP/FN을 파싱합니다.
    Returns:
        {radius_range: {class_name: {"TP": int, "FP": int, "FN": int}}}
    """
    stats = {}
    current_range = None

    # 패턴: ===> Radius range: 0-20m - mIoU: 47.45%
    range_header = re.compile(
        r"===> Radius range:\s+([\d\-\+]+m)\s+-\s+mIoU:\s+([\d.]+)%"
    )
    # Radius 섹션이 아닌 다른 ===> 헤더 (Height 등) → current_range 초기화 트리거
    other_header = re.compile(r"^===>(?!.*Radius range)")
    # 패턴:      barrier  - IoU =  57.54, TP = 167601, FP =  74699, FN =  48995
    class_line = re.compile(
        r"\s+(\w+)\s+-\s+IoU\s*=\s*([\d.]+),\s*TP\s*=\s*(\d+),\s*FP\s*=\s*(\d+),\s*FN\s*=\s*(\d+)"
    )

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            m = range_header.search(line)
            if m:
                current_range = m.group(1)
                stats[current_range] = {}
                continue

            # Radius range가 아닌 새로운 ===> 섹션이 시작되면 파싱 중단
            if other_header.search(line):
                current_range = None
                continue

            if current_range is not None:
                m = class_line.search(line)
                if m:
                    class_name = m.group(1)
                    tp = int(m.group(3))
                    fp = int(m.group(4))
                    fn = int(m.group(5))
                    stats[current_range][class_name] = {"TP": tp, "FP": fp, "FN": fn}

    return stats


def compute_iou(tp: int, fp: int, fn: int) -> float:
    denom = tp + fp + fn
    if denom == 0:
        return 0.0
    return tp / denom * 100.0


def compute_group_miou(
    stats: dict,
    group_ranges: list,
) -> dict:
    """
    주어진 radius range 목록에서 TP/FP/FN을 합산하여 클래스별 IoU와 mIoU를 계산합니다.
    Returns:
        {"mIoU": float, "classes": {class_name: {"IoU": float, "TP": int, "FP": int, "FN": int}}}
    """
    combined = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0})

    for r in group_ranges:
        if r not in stats:
            print(f"  [경고] '{r}' range가 로그에 없습니다.", file=sys.stderr)
            continue
        for cls, vals in stats[r].items():
            combined[cls]["TP"] += vals["TP"]
            combined[cls]["FP"] += vals["FP"]
            combined[cls]["FN"] += vals["FN"]

    # CLASS_ORDER 기준으로 정렬, 없는 클래스는 뒤에 추가
    ordered_keys = [c for c in CLASS_ORDER if c in combined]
    ordered_keys += [c for c in sorted(combined) if c not in CLASS_ORDER]

    class_stats = {}
    for cls in ordered_keys:
        vals = combined[cls]
        class_stats[cls] = {
            "IoU": compute_iou(vals["TP"], vals["FP"], vals["FN"]),
            "TP":  vals["TP"],
            "FP":  vals["FP"],
            "FN":  vals["FN"],
        }

    miou = sum(v["IoU"] for v in class_stats.values()) / len(class_stats) if class_stats else 0.0
    return {"mIoU": miou, "classes": class_stats}


def print_summary(results: dict) -> None:
    """결과를 보기 좋게 출력합니다."""
    print()
    print("=" * 60)
    print("  Radius-based mIoU Summary (Grouped)")
    print("=" * 60)
    print(f"  {'Radius Range':>12} | {'mIoU':>8}")
    print("-" * 28)
    for group_name, result in results.items():
        print(f"  {group_name:>12} | {result['mIoU']:>7.2f}%")
    print()

    col_w = 25
    num_w = 12
    header = (
        f"  {'Class':<{col_w}} {'IoU':>{num_w}} "
        f"{'TP':>{num_w}} {'FP':>{num_w}} {'FN':>{num_w}}"
    )
    sep = "  " + "-" * (col_w + num_w * 4 + 4)

    for group_name, result in results.items():
        print(f"  === {group_name} (mIoU: {result['mIoU']:.2f}%) ===")
        print(header)
        print(sep)
        for cls, vals in result["classes"].items():
            print(
                f"  {cls:<{col_w}} {vals['IoU']:>{num_w - 1}.2f}% "
                f"{vals['TP']:>{num_w},} {vals['FP']:>{num_w},} {vals['FN']:>{num_w},}"
            )
        print()


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <log_file>")
        sys.exit(1)

    log_path = sys.argv[1]

    print(f"로그 파일 파싱 중: {log_path}")
    stats = parse_radius_stats(log_path)

    if not stats:
        print("오류: Radius-based TP/FP/FN 통계를 찾을 수 없습니다.", file=sys.stderr)
        sys.exit(1)

    print(f"발견된 Radius ranges: {', '.join(sorted(stats.keys()))}")

    results = {}
    for group_name, group_ranges in RADIUS_GROUPS.items():
        results[group_name] = compute_group_miou(stats, group_ranges)

    print_summary(results)


if __name__ == "__main__":
    main()
