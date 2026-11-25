"""평가: 예측 파일을 공식 지표로 채점 (원 코드: evaluation.py — 주최 측 채점기의 복사본)."""
import pandas as pd

from common.evaluator import report


def evaluate(pred_path, labels_path):
    return report(pd.read_csv(pred_path), pd.read_csv(labels_path))
