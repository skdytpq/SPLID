"""평가.

  evaluate()             최종 예측 파일을 공식 지표로 채점 (원 코드: submission.py 끝부분)
  evaluate_localizer()   localizer 단독 평가 — 시점만 채점, 임계값별 (원 코드: base/localizer.py perform_evaluation_pipeline)
  evaluate_classifier()  classifier 단독 평가 — 정답 node 시점에서 type 만 채점 (원 코드: classifier_sweeper.py 끝부분)

뒤의 둘은 학습 중 검증 세트 점검에 쓴다 (train.py).
"""
import numpy as np
import pandas as pd

from common.evaluator import NodeDetectionEvaluator, report

from . import postprocess


def evaluate(pred_path, labels_path):
    return report(pd.read_csv(pred_path), pd.read_csv(labels_path))


def evaluate_localizer(dataset, model, keys, labels_path, thresholds=np.linspace(25.0, 70.0, 10)):
    """keys 객체에서 localizer 의 시점 탐지 성능을 임계값별로 출력하고, F2 가 가장 높은 행을 돌려준다.

    - SS (TimeIndex 0) 는 탐지 대상이 아니므로 뺀다
    - TP·FN 은 cfg.nodes_to_score 의 node 만 센다 (AD·IK localizer 는 ID 를 못 찾아도 감점하지 않음)
    """
    from .predict import localizer_outputs      # tensorflow 가 필요한 부분은 여기서만 불러옴 (채점만 할 때는 불필요)
    cfg = dataset.cfg
    object_ids = [int(k) for k in keys]
    gt = pd.read_csv(labels_path)
    gt = gt[gt['ObjectID'].isin(object_ids) & (gt['TimeIndex'] != 0)]
    outputs = localizer_outputs(dataset, model, keys)

    best = None
    for threshold in thresholds:
        pred = postprocess.detections_from_outputs(outputs, (threshold, threshold), merge_distance=-1)
        pred = pred[pred['TimeIndex'] != 0]
        *_, fp, _, total_df = NodeDetectionEvaluator(gt, pred, ignore_classes=True).score()
        if total_df is None:
            tp = fn = 0
        else:
            scored = total_df[total_df['Node'].isin(cfg.nodes_to_score)]
            tp, fn = (scored['classification'] == 'TP').sum(), (scored['classification'] == 'FN').sum()
        f2 = 5 * tp / max(5 * tp + 4 * fn + fp, 1)
        print(f'  임계값 {threshold:.0f}: Precision {tp / max(tp + fp, 1):.2f} Recall {tp / max(tp + fn, 1):.2f} '
              f'F2 {f2:.3f} | TP {tp} FP {fp} FN {fn}')
        if best is None or f2 > best['F2']:
            best = {'threshold': float(threshold), 'F2': float(f2), 'TP': int(tp), 'FP': int(fp), 'FN': int(fn)}
    return best


def evaluate_classifier(dataset, model, keys, labels_path):
    """keys 객체의 정답 node 시점에서 type 분류 정확도 (채점기의 precision) 를 구한다.

    정답의 시점을 그대로 쓰고 (ID 는 type 까지 확정), 나머지 node 의 type 만 classifier 로 채운다.
    """
    from .predict import classifier_types
    object_ids = [int(k) for k in keys]
    gt = pd.read_csv(labels_path)
    gt = gt[gt['ObjectID'].isin(object_ids)]
    located = gt[gt['Direction'] != 'ES'].copy()
    located.loc[located['Node'] != 'ID', ['Node', 'Type']] = 'UNKNOWN'
    types = classifier_types(dataset, model, keys, only_nodes=True)
    pred = postprocess.assign_nodes(postprocess.assign_types(types, located))
    precision, _, _, _, tp, fp, _, _ = NodeDetectionEvaluator(gt, pred, ignore_nodes=True).score()
    print(f'  type 정확도 {precision:.3f} (TP {tp} FP {fp})')
    return {'Precision': float(precision), 'TP': int(tp), 'FP': int(fp)}
