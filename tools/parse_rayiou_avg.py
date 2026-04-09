#!/usr/bin/env python3
"""
로그 파일에서 Class별 / Radius별 / Height별 RayIoU 테이블을 파싱하여
각 그룹의 IoU@1, IoU@2, IoU@4 평균(AVG)을 원래 테이블에 추가하여 출력
"""

import re
import sys
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# 테이블 파싱
# ---------------------------------------------------------------------------

def parse_prettytable(lines: List[str], start: int) -> Tuple[List[str], List[List[str]], int]:
    """
    prettytable 형식의 테이블을 파싱합니다.

    Parameters
    ----------
    lines : 로그 전체 라인 목록
    start : 첫 번째 '+---+' 구분선의 인덱스

    Returns
    -------
    (headers, rows, end_idx)
      - headers : 헤더 셀 문자열 목록
      - rows    : 데이터 행(MEAN 포함) 목록. 각 행은 문자열 목록
      - end_idx : 테이블 파싱이 끝난 다음 라인 인덱스
    """
    i = start + 1                        # 첫 번째 separator 건너뜀
    raw_header = lines[i].strip()
    headers = [h.strip() for h in raw_header.strip('|').split('|')]
    i += 2                               # 헤더 + 두 번째 separator 건너뜀

    rows: List[List[str]] = []
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            break
        if line.startswith('+'):
            i += 1
            # separator 다음에 데이터 행이 있으면 계속
            if i < len(lines) and lines[i].strip().startswith('|'):
                continue
            else:
                break
        elif line.startswith('|'):
            cells = [c.strip() for c in line.strip().strip('|').split('|')]
            rows.append(cells)
            i += 1
        else:
            break

    return headers, rows, i


def find_table_start(lines: List[str], from_idx: int = 0) -> int:
    """from_idx 이후에 나오는 첫 번째 '+---+' 구분선의 인덱스를 반환합니다."""
    for i in range(from_idx, len(lines)):
        if lines[i].strip().startswith('+') and '-' in lines[i]:
            return i
    return -1


def find_section(lines: List[str], keyword: str) -> int:
    """keyword를 포함하는 첫 번째 라인의 인덱스를 반환합니다."""
    for i, line in enumerate(lines):
        if keyword in line:
            return i
    return -1


# ---------------------------------------------------------------------------
# 평균 계산 및 테이블 확장
# ---------------------------------------------------------------------------

def safe_float(s: str) -> Optional[float]:
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def avg_iou(cells: List[str], idx1: int, idx2: int, idx4: int) -> str:
    """세 인덱스의 평균을 계산하여 % 문자열로 반환합니다."""
    vals = [safe_float(cells[i]) for i in (idx1, idx2, idx4) if i < len(cells)]
    valid = [v for v in vals if v is not None]
    if not valid:
        return 'nan'
    return f'{sum(valid) / len(valid) * 100:.2f}%'


def add_avg_columns(
    headers: List[str],
    rows: List[List[str]],
    groups: List[Tuple[str, str, str, str]],
) -> Tuple[List[str], List[List[str]]]:
    """
    groups 에 따라 AVG 컬럼을 추가합니다.

    MEAN 행의 AVG는 반올림된 MEAN 값으로 재계산하지 않고,
    각 클래스 행의 AVG 평균으로 계산합니다.

    Parameters
    ----------
    groups : (prefix, col1_name, col2_name, col4_name) 튜플 목록
             prefix    : 추가할 컬럼 이름 앞에 붙을 접두사 (빈 문자열이면 'AVG'만 사용)
             col1_name : IoU@1 헤더 이름
             col2_name : IoU@2 헤더 이름
             col4_name : IoU@4 헤더 이름
    """
    new_headers = headers[:]
    avg_indices: List[Tuple[int, int, int, str]] = []  # (idx1, idx2, idx4, label)

    for prefix, c1, c2, c4 in groups:
        try:
            i1, i2, i4 = headers.index(c1), headers.index(c2), headers.index(c4)
        except ValueError:
            continue
        label = f'{prefix} AVG' if prefix else 'AVG'
        avg_indices.append((i1, i2, i4, label))
        new_headers.append(label)

    data_rows = [r for r in rows if r[0] != 'MEAN']
    mean_rows = [r for r in rows if r[0] == 'MEAN']

    # 클래스 행의 AVG 먼저 계산
    new_data_rows = []
    for row in data_rows:
        new_row = row[:]
        for i1, i2, i4, _ in avg_indices:
            new_row.append(avg_iou(row, i1, i2, i4))
        new_data_rows.append(new_row)

    # MEAN 행: 각 그룹 AVG = 클래스 행 AVG들의 평균 (반올림 오차 방지)
    new_mean_rows = []
    for row in mean_rows:
        new_row = row[:]
        for col_offset, (i1, i2, i4, _) in enumerate(avg_indices):
            # new_data_rows에서 해당 AVG 컬럼 값을 수집
            avg_col_idx = len(headers) + col_offset
            class_avgs = []
            for dr in new_data_rows:
                v = safe_float(dr[avg_col_idx].rstrip('%'))
                if v is not None:
                    class_avgs.append(v)
            if class_avgs:
                new_row.append(f'{sum(class_avgs) / len(class_avgs):.4f}%')
            else:
                new_row.append('nan')
        new_mean_rows.append(new_row)

    # 원래 행 순서를 유지하면서 반환
    result_map = {id(r): nr for r, nr in zip(data_rows, new_data_rows)}
    result_map.update({id(r): nr for r, nr in zip(mean_rows, new_mean_rows)})
    new_rows = [result_map[id(r)] for r in rows]

    return new_headers, new_rows


# ---------------------------------------------------------------------------
# 테이블 출력
# ---------------------------------------------------------------------------

def print_table(headers: List[str], rows: List[List[str]], title: str = '') -> None:
    """prettytable 스타일로 테이블을 출력합니다."""
    # 열 너비 계산
    col_w = [len(h) for h in headers]
    for row in rows:
        for j, cell in enumerate(row):
            if j < len(col_w):
                col_w[j] = max(col_w[j], len(cell))

    sep = '+' + '+'.join('-' * (w + 2) for w in col_w) + '+'

    def fmt_row(cells: List[str]) -> str:
        padded = []
        for j, w in enumerate(col_w):
            c = cells[j] if j < len(cells) else ''
            padded.append(f' {c:^{w}} ')
        return '|' + '|'.join(padded) + '|'

    if title:
        print(f'\n{title}')

    data_rows = [r for r in rows if r[0] != 'MEAN']
    mean_rows = [r for r in rows if r[0] == 'MEAN']

    print(sep)
    print(fmt_row(headers))
    print(sep)
    for row in data_rows:
        print(fmt_row(row))
    print(sep)
    for row in mean_rows:
        print(fmt_row(row))
    print(sep)


# ---------------------------------------------------------------------------
# 메인 처리
# ---------------------------------------------------------------------------

def process_class_table(lines: List[str]) -> None:
    """Class별 RayIoU 테이블 처리"""
    # 첫 번째 prettytable (Class Names 헤더를 포함하고 Radius/Height 이전)
    # "6019it" 또는 MIOU 출력 이후에 나오는 첫 번째 테이블
    idx = -1
    for i, line in enumerate(lines):
        if re.search(r'\d+it \[', line):
            idx = i
            break
    if idx == -1:
        idx = 0

    start = find_table_start(lines, idx)
    if start == -1:
        print('[오류] Class 테이블을 찾을 수 없습니다.', file=sys.stderr)
        return

    headers, rows, _ = parse_prettytable(lines, start)
    # IoU@1 / IoU@2 / IoU@4 컬럼이 있는지 확인
    if 'IoU@1' not in headers:
        print('[오류] Class 테이블 헤더를 인식할 수 없습니다.', file=sys.stderr)
        return

    groups = [('', 'IoU@1', 'IoU@2', 'IoU@4')]
    new_headers, new_rows = add_avg_columns(headers, rows, groups)
    print_table(new_headers, new_rows, title='[Class별 RayIoU + AVG]')


def process_radius_table(lines: List[str]) -> None:
    """Radius별 RayIoU 테이블 처리"""
    sec = find_section(lines, 'RayIoU by Radius')
    if sec == -1:
        print('[오류] Radius 테이블 섹션을 찾을 수 없습니다.', file=sys.stderr)
        return

    start = find_table_start(lines, sec)
    if start == -1:
        print('[오류] Radius 테이블을 찾을 수 없습니다.', file=sys.stderr)
        return

    headers, rows, _ = parse_prettytable(lines, start)

    # 그룹 감지: 헤더에서 "X IoU@1" 패턴 추출
    prefix_pattern = re.compile(r'^(.+?)\s+IoU@1$')
    prefixes = []
    seen = set()
    for h in headers:
        m = prefix_pattern.match(h)
        if m:
            p = m.group(1)
            if p not in seen:
                prefixes.append(p)
                seen.add(p)

    groups = [(p, f'{p} IoU@1', f'{p} IoU@2', f'{p} IoU@4') for p in prefixes]
    new_headers, new_rows = add_avg_columns(headers, rows, groups)
    print_table(new_headers, new_rows, title='[Radius별 RayIoU + AVG]')


def process_height_table(lines: List[str]) -> None:
    """Height별 RayIoU 테이블 처리"""
    sec = find_section(lines, 'RayIoU by Height')
    if sec == -1:
        print('[오류] Height 테이블 섹션을 찾을 수 없습니다.', file=sys.stderr)
        return

    start = find_table_start(lines, sec)
    if start == -1:
        print('[오류] Height 테이블을 찾을 수 없습니다.', file=sys.stderr)
        return

    headers, rows, _ = parse_prettytable(lines, start)

    prefix_pattern = re.compile(r'^(.+?)\s+IoU@1$')
    prefixes = []
    seen = set()
    for h in headers:
        m = prefix_pattern.match(h)
        if m:
            p = m.group(1)
            if p not in seen:
                prefixes.append(p)
                seen.add(p)

    groups = [(p, f'{p} IoU@1', f'{p} IoU@2', f'{p} IoU@4') for p in prefixes]
    new_headers, new_rows = add_avg_columns(headers, rows, groups)
    print_table(new_headers, new_rows, title='[Height별 RayIoU + AVG]')


def main() -> None:
    if len(sys.argv) < 2:
        print(f'Usage: python {sys.argv[0]} <log_file>', file=sys.stderr)
        sys.exit(1)

    log_path = sys.argv[1]
    with open(log_path, encoding='utf-8') as f:
        lines = f.readlines()

    print(f'로그 파일: {log_path}')
    print(f'총 {len(lines)}줄')

    process_class_table(lines)
    process_radius_table(lines)
    process_height_table(lines)


if __name__ == '__main__':
    main()
