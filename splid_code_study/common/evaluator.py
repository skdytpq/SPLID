"""SPLID 챌린지 공식 채점기 (두 팀 공통).

출처: ARCLab-MIT/splid-devkit 의 NodeDetectionEvaluator ((c) 2023 Peng Mun Siew, MIT License).
Hawaii2024 저장소(base/evaluation.py)가 여기에 ignore_classes / ignore_nodes 옵션을 더한 판을 그대로 옮김.

채점 규칙
  - 정답 node 하나마다, 같은 방향(EW/NS)이고 TimeIndex 가 ±tolerance(기본 6 step = 12시간) 안에 있는
    아직 짝이 없는 예측 중 "첫 번째" 를 짝으로 삼는다.
  - 짝이 있고 Node·Type 이 모두 같으면 TP, 다르면 FP. 짝이 없으면 FN.
  - 어떤 정답과도 짝이 되지 못한 예측은 FP.
  - F2 = 5·TP / (5·TP + 4·FN + FP)  (recall 에 4배 가중)
"""
import numpy as np
import pandas as pd


class NodeDetectionEvaluator:
    def __init__(self, ground_truth, participant, tolerance=6, ignore_classes=False, ignore_nodes=False):
        """ground_truth / participant: 열 [ObjectID, TimeIndex, Direction, Node, Type] 인 DataFrame.

        ignore_classes: 시점만 채점 (Node·Type 무시) — localizer 단독 평가용
        ignore_nodes:   Type 만 채점 — classifier 단독 평가용
        """
        self.ground_truth = ground_truth.copy()
        self.participant = participant.copy()
        self.tolerance = tolerance
        self.ignore_classes = ignore_classes
        self.ignore_nodes = ignore_nodes

    def evaluate(self, object_id):
        """객체 1기를 채점. 정답 행과 예측 행 각각에 classification(TP/FP/FN)·distance 를 붙여 돌려준다."""
        gt_object = self.ground_truth[(self.ground_truth['ObjectID'] == object_id) &
                                      (self.ground_truth['Direction'] != 'ES')].copy()
        p_object = self.participant[(self.participant['ObjectID'] == object_id) &
                                    (self.participant['Direction'] != 'ES')].copy()
        p_object['matched'] = False
        p_object['classification'] = None
        p_object['distance'] = None
        gt_object['classification'] = None
        gt_object['distance'] = None
        tp = fp = fn = 0

        for gt_idx, gt_row in gt_object.iterrows():
            candidates = p_object[(p_object['TimeIndex'] >= gt_row['TimeIndex'] - self.tolerance) &
                                  (p_object['TimeIndex'] <= gt_row['TimeIndex'] + self.tolerance) &
                                  (p_object['Direction'] == gt_row['Direction']) &
                                  (p_object['matched'] == False)]
            if len(candidates) > 0:
                p_idx = candidates.index[0]
                p_row = candidates.iloc[0]
                distance = p_row['TimeIndex'] - gt_row['TimeIndex']
                correct = (self.ignore_classes
                           or (self.ignore_nodes and p_row['Type'] == gt_row['Type'])
                           or (p_row['Node'] == gt_row['Node'] and p_row['Type'] == gt_row['Type']))
                label = 'TP' if correct else 'FP'
                tp += correct
                fp += not correct
                gt_object.loc[gt_idx, ['classification', 'distance']] = [label, distance]
                p_object.loc[p_idx, ['classification', 'distance']] = [label, distance]
                p_object.loc[p_idx, 'matched'] = True
            else:
                fn += 1
                gt_object.loc[gt_idx, 'classification'] = 'FN'

        unmatched = p_object[~p_object['matched']]
        fp += len(unmatched)
        p_object.loc[unmatched.index, 'classification'] = 'FP'
        return tp, fp, fn, gt_object, p_object

    def score(self):
        """전체 객체 채점. (precision, recall, f2, rmse, TP, FP, FN, total_df) 를 돌려준다.

        total_df = 정답 행 전부 + 예측 중 FP 행. node 종류별 TP/FN 집계에 쓴다.
        """
        total_tp = total_fp = total_fn = 0
        distances, gt_objs, p_objs = [], [], []
        for object_id in self.ground_truth['ObjectID'].unique():
            _, _, _, gt_object, p_object = self.evaluate(object_id)
            total_tp += len(p_object[p_object['classification'] == 'TP'])
            total_fp += len(p_object[p_object['classification'] == 'FP'])
            total_fn += len(gt_object[gt_object['classification'] == 'FN'])
            gt_objs.append(gt_object)
            p_objs.append(p_object.loc[p_object['classification'] == 'FP'])
            distances.extend(p_object[p_object['classification'] == 'TP']['distance'].tolist())
        if total_tp + total_fp < 1:
            return 0.0, 0.0, 0.0, 0.0, 0, 0, total_fn, None

        precision = total_tp / (total_tp + total_fp)
        recall = total_tp / max(total_tp + total_fn, 1)        # 정답 node 가 하나도 없는 작은 검증 세트 대비
        f2 = (5 * total_tp) / (5 * total_tp + 4 * total_fn + total_fp)
        rmse = np.sqrt(sum(d ** 2 for d in distances) / len(distances)) if distances else 0.0
        total_df = pd.concat(gt_objs + p_objs).sort_values(['ObjectID', 'TimeIndex']).reset_index(drop=True)
        return precision, recall, f2, rmse, total_tp, total_fp, total_fn, total_df


def report(pred_df, gt_df, tolerance=6):
    """예측 파일 전체를 채점해 출력하고 점수 dict 를 돌려준다 (두 팀의 eval.py 가 함께 쓴다)."""
    gt_df = gt_df[gt_df['ObjectID'].isin(pred_df['ObjectID'].unique())]   # 예측에 들어 있는 객체만 채점
    p, r, f2, rmse, tp, fp, fn, total_df = NodeDetectionEvaluator(gt_df, pred_df, tolerance).score()
    print(f'객체 {gt_df["ObjectID"].nunique()}기')
    print(f'Precision: {p:.2f}')
    print(f'Recall: {r:.2f}')
    print(f'F2: {f2:.3f}')
    print(f'RMSE: {float(rmse):.4f}')
    print(f'TP: {tp} FP: {fp} FN: {fn}')
    scores = {'Precision': p, 'Recall': r, 'F2': f2, 'RMSE': float(rmse), 'TP': tp, 'FP': fp, 'FN': fn}
    if total_df is not None:
        print('node 종류별 TP|FN:')
        for node in ['SS', 'ID', 'IK', 'AD']:
            rows = total_df[total_df['Node'] == node]
            n_tp, n_fn = (rows['classification'] == 'TP').sum(), (rows['classification'] == 'FN').sum()
            scores[f'TP_{node}'], scores[f'FN_{node}'] = int(n_tp), int(n_fn)
            print(f'  {node}: {n_tp}|{n_fn}')
    # 시점만 본 점수 (Node·Type 무시): 탐지 능력과 분류 능력을 나눠 보기 위함
    p, r, f2, _, tp, fp, fn, _ = NodeDetectionEvaluator(gt_df, pred_df, tolerance, ignore_classes=True).score()
    print(f'시점만 채점 (Node·Type 무시): P: {p:.2f} R: {r:.2f} F2: {f2:.3f} | TP: {tp} FP: {fp} FN: {fn}')
    scores['F2_location_only'] = f2
    return scores
