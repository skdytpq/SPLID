"""추론: 궤적 csv → 예측 node 목록.

원 코드 대응: main_submission.py, utils.py 의 dual_threshold_prediction() · convert_classifier_output().

흐름
  ① 표 만들기 + feature          dataloader.load_table → features.add_features
  ② 시점마다 class 확률           model.StackedCatBoost.predict_proba
  ③ 이중 임계값으로 class 결정    dual_threshold
  ④ 시점별 class → node 목록      to_nodes
"""
import numpy as np
import pandas as pd

from .config import THRESHOLDS
from .dataloader import load_table
from .features import add_features
from .model import StackedCatBoost


def dual_threshold(proba, nothing, precision_threshold, recall_threshold):
    """시점마다 class 를 하나 고른다. F2 가 recall 에 4배 가중이라 "node 를 놓치지 않는 쪽" 으로 기울인 규칙.

    proba: (시점 수, class 수), nothing: 'Nothing' class 의 번호
      - 기본은 확률이 가장 높은 class
      - precision 임계값: 최고 확률이 이 값보다 낮으면 'Nothing' (확신이 없는 node 예측을 버림)
      - recall 임계값: 1위가 'Nothing' 이어도 2위 class 의 확률이 이 값을 넘으면 2위를 고름
        ('Nothing' 이 압도적으로 많은 불균형 데이터에서 node 를 살려냄)
    원 코드는 행마다 Python 반복으로 계산한다. 여기서는 같은 규칙을 배열 연산으로 썼다.
    """
    order = np.argsort(proba, axis=1)
    first, second = order[:, -1], order[:, -2]
    second_proba = np.take_along_axis(proba, second[:, None], axis=1)[:, 0]
    pred = first.copy()
    pred[proba.max(axis=1) < precision_threshold] = nothing
    rescue = (first == nothing) & (second_proba > recall_threshold)
    pred[rescue] = second[rescue]
    return pred


def to_nodes(table):
    """시점별 예측 (Predicted_EW, Predicted_NS; node 가 아니면 'Nothing') 을 제출 형식으로 바꾼다.

    'Nothing' 을 직전 예측으로 채워 "그 시점의 mode" 로 만든 뒤, 값이 바뀌는 시점만 node 로 남긴다.
    같은 라벨이 연달아 예측되면 (예: 인접한 두 시점 모두 IK-EK) 첫 시점 하나만 남는 효과가 있다.
    """
    parts = []
    for d in ['EW', 'NS']:
        mode = table[f'Predicted_{d}'].mask(table[f'Predicted_{d}'] == 'Nothing').ffill()
        part = table[['TimeIndex', 'ObjectID']].copy()
        part[['Node', 'Type']] = mode.str.split('-', expand=True)
        part['Direction'] = d
        parts.append(part)
    nodes = pd.concat(parts, ignore_index=True).sort_values(['ObjectID', 'Direction', 'TimeIndex'])

    previous = nodes.groupby(['ObjectID', 'Direction'])[['Node', 'Type']].shift()
    nodes = nodes[(previous != nodes[['Node', 'Type']]).any(axis=1)]         # 객체의 첫 행은 항상 남는다
    nodes = nodes.reset_index(drop=True)[['ObjectID', 'TimeIndex', 'Direction', 'Node', 'Type']]
    nodes = nodes.sort_values(['ObjectID', 'TimeIndex', 'Direction'])
    nodes.loc[nodes['TimeIndex'] == 0, 'Node'] = 'SS'                         # 시작 시점의 node 는 정의상 SS
    return nodes


def run(data_dir, weights_dir, max_objects=None):
    table = add_features(load_table(data_dir, max_objects=max_objects))
    print(f'객체 {table["ObjectID"].nunique()}기, {len(table)}행 × {table.shape[1]}열')
    model = StackedCatBoost.load(weights_dir)
    ew_proba, ns_proba = model.predict_proba(table)
    for d, proba, encoder in [('EW', ew_proba, model.le_ew), ('NS', ns_proba, model.le_ns)]:
        nothing = list(encoder.classes_).index('Nothing')
        pred = dual_threshold(proba.to_numpy(), nothing, *THRESHOLDS[d])
        table[f'Predicted_{d}'] = encoder.inverse_transform(pred)
    return to_nodes(table)
