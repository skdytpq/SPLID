"""feature engineering: 시점마다 약 320개의 값을 만든다.

원 코드 대응: utils.py tabularize_data() 의 뒷부분. 열의 순서와 계산식은 원 코드와 같다
(CatBoost 는 열 순서에 영향을 받으므로 재학습 결과를 맞추려면 순서도 지켜야 한다).

만드는 feature (논문 ② 3.3.1)
  원시 값 15 · 주기 index 5 (dataloader 에서 추가)
  envelope 3            위도·경도·고도의 3일 창 (최대 - 최소). station-keeping 의 "상자 크기" 가 바뀌는 곳에서 변함
  lag 180               원시 값 15개 × 앞뒤 6 step
  차분 48               원시 값 9개 + envelope 3개 × 4 간격
  변화율 6              속도 3개 × 2 간격
  이동 표준편차·평균 20  1일 창
  위성 단위 통계 16      궤적 전체 평균 7 + 표준편차 9 — "이 위성이 어떤 위성인가" 를 알려줌
  heuristic one-hot 28  주최 측 baseline 의 예측 (그 시점의 node, 그 시점의 mode)
"""
import pandas as pd

from .config import (DIFF_COLUMNS, DIFF_STEPS, ENVELOPE_COLUMNS, ENVELOPE_WINDOW, HEURISTIC_CLASSES, LAG_STEPS,
                     OBJECT_MEAN_COLUMNS, OBJECT_STD_COLUMNS, PCT_COLUMNS, PCT_STEPS, RAW_COLUMNS, ROLLING_COLUMNS,
                     ROLLING_WINDOW)


def add_features(table):
    """load_table() 의 표에 feature 열을 더해 돌려준다."""
    # 아래 _per_object() 가 값을 행 순서대로 붙이므로 표가 ObjectID 오름차순이어야 한다 (dataloader._list_files 참고)
    assert table['ObjectID'].is_monotonic_increasing, '표가 ObjectID 순으로 정렬되어 있지 않음'

    # ── envelope: 이후 차분의 입력이 되므로 먼저 표에 붙임
    envelope = []
    for col in ENVELOPE_COLUMNS:
        upper = _per_object(table, col, lambda x: x.rolling(ENVELOPE_WINDOW, center=True, min_periods=0).max())
        lower = _per_object(table, col, lambda x: x.rolling(ENVELOPE_WINDOW, center=True, min_periods=0).min())
        envelope.append((upper - lower).rename(f'{col}_enveloppe'))
    table = pd.concat([table] + envelope, axis=1)
    by_object = lambda col: table.groupby('ObjectID')[col]

    # ── lag: i < 0 은 미래 값 (lead), i > 0 은 과거 값
    # (원 코드 그대로) 궤적 시작·끝에서 생기는 빈 값을 표 전체에 대해 ffill·bfill 로 채운다.
    # 객체 경계를 넘어가므로 각 객체의 처음 몇 행에는 "표에서 바로 앞 객체" 의 마지막 값이 들어간다
    new_columns = []
    for col in RAW_COLUMNS:
        for i in [j for j in range(-LAG_STEPS, LAG_STEPS + 1) if j != 0]:
            new_columns.append(by_object(col).shift(i).ffill().bfill().rename(f'{col}_lag_{i}'))

    # ── 차분, 변화율
    for col in DIFF_COLUMNS:
        for i in DIFF_STEPS:
            new_columns.append(by_object(col).diff(i).ffill().bfill().rename(f'{col}_diff_{i}'))
    for col in PCT_COLUMNS:
        for i in PCT_STEPS:
            new_columns.append(by_object(col).pct_change(i).ffill().bfill().rename(f'{col}_pct_change_{i}'))

    # ── 이동 표준편차·평균 (가장자리의 빈 값은 객체 안에서 채움)
    for col in ROLLING_COLUMNS:
        rolling = lambda x: x.rolling(ROLLING_WINDOW, center=True)
        new_columns.append(_per_object(table, col, lambda x: rolling(x).std().ffill().bfill()).rename(f'{col}_rolling_std_{ROLLING_WINDOW}'))
        new_columns.append(_per_object(table, col, lambda x: rolling(x).mean().ffill().bfill()).rename(f'{col}_rolling_mean_{ROLLING_WINDOW}'))

    # ── 위성 단위 통계
    for col in OBJECT_MEAN_COLUMNS:
        new_columns.append(by_object(col).transform('mean').rename(f'{col}_mean_charac'))
    for col in OBJECT_STD_COLUMNS:
        new_columns.append(by_object(col).transform('std').rename(f'{col}_std_charac'))
    table = pd.concat([table] + new_columns, axis=1)

    # ── heuristic 예측의 one-hot. 이 데이터에 없는 class 의 열은 0 으로 만들어 둠
    heuristic_columns = ['EW_baseline_heuristic', 'NS_baseline_heuristic',
                         'EW_baseline_heuristic_ffill', 'NS_baseline_heuristic_ffill']
    table = pd.concat([table] + [pd.get_dummies(table[[col]]) for col in heuristic_columns], axis=1)
    for suffix in ['', '_ffill']:
        for d in ['EW', 'NS']:
            for cls in HEURISTIC_CLASSES[d]:
                name = f'{d}_baseline_heuristic{suffix}_{cls}'
                if name not in table.columns:
                    table[name] = 0
    return table


def _per_object(table, col, func):
    """객체별로 func 을 적용한 결과를 표의 행 순서에 맞춰 돌려준다.

    pandas 2 의 groupby.apply 는 결과를 ObjectID 오름차순으로 이어 붙인다. 표도 같은 순서라는 전제에서 index 만 바꿔 단다.
    """
    result = table.groupby('ObjectID')[col].apply(func)
    result.index = table.index
    return result
