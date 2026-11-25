"""데이터 로딩: 객체별 csv → 모든 객체를 세로로 이어 붙인 표 1개 (행 = 객체의 한 시점).

원 코드 대응: utils.py tabularize_data() 의 앞부분 (파일 읽기, 주기 index, heuristic, 라벨 붙이기).
뒷부분 (lag·차분 등 feature) 은 features.py.

라벨 형식: 방향별 열 EW / NS. node 가 있는 시점에만 'IK-EK' 같은 "Node-Type" 문자열, 나머지 시점은 'Nothing'.
           → 시점마다 "이 시점이 어떤 node 인가" 를 맞히는 multi-class 분류 문제가 된다.
"""
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .config import RAW_COLUMNS, TIME_CYCLES
from .heuristic import add_baseline_heuristic

# heuristic 이 시각 ↔ TimeIndex 변환에 쓰는 기준 시각. 실제 관측 시각과 달라도 결과에는 영향이 없다 (다시 TimeIndex 로 되돌림)
_START = datetime.fromisoformat('2023-01-01 00:00:00+00:00')
_END = datetime.fromisoformat('2023-07-01 00:00:00+00:00')


def _list_files(data_dir, max_objects=None):
    """ObjectID 오름차순으로 정렬한 csv 목록.

    정렬이 필요한 이유: features.py 의 envelope·이동 통계는 pandas groupby 결과 (ObjectID 오름차순) 를
    표에 행 순서대로 붙인다. 표가 ObjectID 순이 아니면 값이 다른 위성의 행에 붙는다.
    원 코드는 정렬하지 않아 파일 시스템이 주는 순서에 따라 이 문제가 생긴다.
    """
    files = sorted(Path(data_dir).glob('*.csv'), key=lambda f: int(f.stem))
    if not files:
        raise ValueError(f'csv 파일이 없음: {data_dir}')
    return files[:max_objects]


def load_table(data_dir, labels_path=None, max_objects=None):
    labels = None if labels_path is None else pd.read_csv(labels_path)
    tables = []
    for file in _list_files(data_dir, max_objects):
        object_id = int(file.stem)
        df = pd.read_csv(file)
        df['ObjectID'] = object_id
        df['TimeIndex'] = range(len(df))
        for cycle in TIME_CYCLES:       # 1, 2, ..., cycle, 1, 2, ... : 하루·일주일 중 어디인지
            df[f'time_{cycle}'] = pd.Series(np.tile(np.arange(1, cycle + 1), len(df))[:len(df)])
        df = add_baseline_heuristic(df, object_id, _START, _END, RAW_COLUMNS)

        if labels is not None:
            obj_labels = labels[labels['ObjectID'] == object_id]
            for d in ['EW', 'NS']:
                dir_labels = obj_labels[obj_labels['Direction'] == d].sort_values('TimeIndex')
                dir_labels = pd.DataFrame({'TimeIndex': dir_labels['TimeIndex'], 'ObjectID': object_id,
                                           d: dir_labels['Node'] + '-' + dir_labels['Type']})
                df = pd.merge_ordered(df, dir_labels, on=['TimeIndex', 'ObjectID'], how='left')
        tables.append(df)

    table = pd.concat(tables)           # index 는 객체마다 0 부터 다시 시작한다 (원 코드와 같음)
    if labels is not None:
        table['EW'] = table['EW'].fillna('Nothing')
        table['NS'] = table['NS'].fillna('Nothing')
    return table
