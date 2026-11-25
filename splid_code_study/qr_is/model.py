"""모델: CatBoost 분류기 3개를 차례로 연결.

원 코드 대응: main_train.py (학습 순서), main_submission.py (추론 순서).

    feature ──▶ ① NS 1차 모델 ──NS 확률──▶ ② EW 모델 ──EW 확률──▶ ③ NS 최종 모델
                                            (feature + NS 확률)        (feature + EW 확률)

  - 방향별로 모델을 나눈 이유: 같은 시점에 EW·NS node 가 함께 있을 수 있어 한 분류기로는 둘 다 낼 수 없다
  - 앞 모델의 class 확률을 다음 모델의 입력에 더해 두 방향 사이의 정보를 전달한다 (논문의 recombination)
  - 각 모델은 시점마다 "Nothing / SS-CK / IK-EK / ..." 중 하나를 고르는 multi-class 분류기 (EW 10 class, NS 9 class)

(원 코드 그대로) ②③ 을 학습할 때 넣는 확률은 ① ② 가 자기 학습 데이터에 대해 낸 값이다.
학습 때는 거의 정답에 가까운 확률, 추론 때는 불확실한 확률이 들어가는 차이가 생긴다.
"""
import pickle
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder

from .config import N_ESTIMATORS, SEED

# 저장 파일 이름은 원 저장소와 같다
_FILES = {'ns_first': 'model_NS_preprocess.pkl', 'ew': 'model_EW.pkl', 'ns': 'model_NS.pkl',
          'le_ew': 'le_EW.pkl', 'le_ns': 'le_NS.pkl'}


class StackedCatBoost:
    def fit(self, table, feature_cols):
        """table: features.add_features() 의 결과 (라벨 열 EW, NS 포함)."""
        self.le_ew, self.le_ns = LabelEncoder(), LabelEncoder()
        y_ew = self.le_ew.fit_transform(table['EW'])
        y_ns = self.le_ns.fit_transform(table['NS'])
        # allow_writing_files=False: 학습 로그 폴더 (catboost_info) 를 만들지 않음. 학습 결과에는 영향 없음
        new = lambda: CatBoostClassifier(n_estimators=N_ESTIMATORS, random_state=SEED, allow_writing_files=False)

        self.ns_first = new().fit(table[feature_cols], y_ns)
        ns_proba = _proba(self.ns_first, table, 'proba_feature_NS_')
        self.ew = new().fit(pd.concat([table[feature_cols], ns_proba], axis=1), y_ew)
        ew_proba = _proba(self.ew, pd.concat([table, ns_proba], axis=1), 'proba_feature_EW_')
        self.ns = new().fit(pd.concat([table[feature_cols], ew_proba], axis=1), y_ns)
        return self

    def predict_proba(self, table):
        """시점마다의 class 확률 (EW, NS) 을 DataFrame 2개로 돌려준다. 열 순서 = le_ew.classes_, le_ns.classes_."""
        ns_proba = _proba(self.ns_first, table, 'proba_feature_NS_')
        ew_proba = _proba(self.ew, pd.concat([table, ns_proba], axis=1), 'proba_feature_EW_')
        ns_proba = _proba(self.ns, pd.concat([table, ew_proba], axis=1), 'proba_feature_NS_')
        return ew_proba, ns_proba

    def save(self, out_dir):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for attr, name in _FILES.items():
            pickle.dump(getattr(self, attr), open(out_dir / name, 'wb'))

    @classmethod
    def load(cls, weights_dir):
        self = cls()
        for attr, name in _FILES.items():
            setattr(self, attr, pickle.load(open(Path(weights_dir) / name, 'rb')))
        return self


def _proba(model, table, prefix):
    """model 이 학습 때 쓴 열을 이름으로 골라 class 확률을 구한다 (열 이름: prefix + class 번호)."""
    proba = pd.DataFrame(model.predict_proba(table[model.feature_names_])).add_prefix(prefix)
    proba.index = table.index
    return proba
