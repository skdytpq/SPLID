"""학습: CatBoost 3개를 차례로 학습해 <out>/ 에 저장.

원 코드 대응: main_train.py. 검증 세트는 없다 (원 코드도 최종 학습에서는 1,900기를 전부 쓴다).
"""
from pathlib import Path

from .config import NON_FEATURE_COLUMNS
from .dataloader import load_table
from .features import add_features
from .model import StackedCatBoost


def run(data_dir, labels_path, out_dir, max_objects=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table = add_features(load_table(data_dir, labels_path, max_objects))
    feature_cols = [c for c in table.columns if c not in NON_FEATURE_COLUMNS]
    print(f'객체 {table["ObjectID"].nunique()}기, {len(table)}행, feature {len(feature_cols)}개')
    print('EW class 분포:', table['EW'].value_counts().to_dict())
    print('NS class 분포:', table['NS'].value_counts().to_dict())

    model = StackedCatBoost().fit(table, feature_cols)
    model.save(out_dir)
    print(f'저장: {out_dir}')
