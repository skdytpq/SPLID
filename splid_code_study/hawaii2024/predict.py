"""추론: 궤적 csv → 예측 node 목록.

원 코드 대응: submission.py + base/localizer.py · base/classifier.py 의 perform_submission_pipeline().

전체 흐름 (논문 3.1.1 의 3단계)
  ① AD·IK localizer   모든 시점의 출력 → 임계값 → node 시점
  ② ID localizer      같은 방식, 찾은 시점은 ID-NK 로 확정
  ③ classifier        ①② 의 시점과 시작 시점 (SS) 에서 이후 mode type 을 분류
  ④ 규칙              type 의 앞뒤 순서로 node 종류 결정 → EW 가 NK 인 동안의 NS node 제거 → 다시 결정
"""
import pickle
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import tensorflow as tf

from . import postprocess
from .config import ADIK, CLASSIFIER, ID
from .dataloader import TYPE_ENCODER, SplidDataset, load_objects


def localizer_outputs(dataset, model, keys, objects_per_batch=64):
    """keys 객체들의 모든 시점에 대한 localizer 출력. → [ObjectID, TimeIndex, EW_Loc, NS_Loc]"""
    parts = []
    for i in range(0, len(keys), objects_per_batch):        # 메모리를 아끼려 객체를 나눠서 처리
        X, _, ids = dataset.arrays(keys[i:i + objects_per_batch])
        ew, ns = model.predict(X, batch_size=512, verbose=0)
        parts.append(pd.DataFrame({'ObjectID': ids[:, 0], 'TimeIndex': ids[:, 1], 'EW_Loc': ew[:, 0], 'NS_Loc': ns[:, 0]}))
    return pd.concat(parts).sort_values(['ObjectID', 'TimeIndex']).reset_index(drop=True)


def classifier_types(dataset, model, keys, select=None, only_nodes=False):
    """지정한 시점들의 mode type 예측. → [ObjectID, TimeIndex, EW_Type, NS_Type]

    원 코드는 모든 시점을 분류한 뒤 node 시점만 골라 쓴다. 결과가 같으므로 여기서는 필요한 시점만 분류한다.
    """
    X, _, ids = dataset.arrays(keys, select=select, only_nodes=only_nodes)
    ew, ns = model.predict(X, batch_size=512, verbose=0)
    return pd.DataFrame({'ObjectID': ids[:, 0], 'TimeIndex': ids[:, 1],
                         'EW_Type': TYPE_ENCODER.inverse_transform(np.argmax(ew, axis=1)),
                         'NS_Type': TYPE_ENCODER.inverse_transform(np.argmax(ns, axis=1))})


def _load(cfg, weights_dir):
    weights_dir = Path(weights_dir)
    # 경로에 한글이 있으면 Windows 의 TensorFlow 파일 API 가 실패하므로 h5py 로 연 파일을 넘긴다
    with h5py.File(weights_dir / f'{cfg.name}_model.hdf5', 'r') as file:
        model = tf.keras.models.load_model(file, compile=False)
    scaler = pickle.load(open(weights_dir / f'{cfg.name}_scaler.pkl', 'rb'))
    return model, scaler


def run(data_dir, weights_dir, legacy=False, max_objects=None):
    """data_dir 의 모든 객체를 예측해 제출 형식 DataFrame 을 돌려준다.

    legacy=True: 대회 제출 모델 (weights/submitted) 용. 그 모델이 학습된 입력 방식을 재현한다
                 (각도 차분의 360도 넘김 미보정, classifier 의 시간 위치가 one-hot 이 아닌 선형 값).
    """
    objects = load_objects(data_dir, max_objects=max_objects)
    keys = list(objects.keys())
    print(f'객체 {len(keys)}기 로드: {data_dir}')

    # ①② 시점 탐지
    detections = {}
    for cfg in [ADIK, ID]:
        model, scaler = _load(cfg, weights_dir)
        dataset = SplidDataset(cfg, test=objects, scaler=scaler, with_labels=False, legacy_diff=legacy)
        outputs = localizer_outputs(dataset, model, keys)
        detections[cfg.name] = postprocess.detections_from_outputs(outputs, cfg.thresholds, cfg.merge_distance)
        print(f'{cfg.name}: 탐지 {len(detections[cfg.name])}개')
    located = postprocess.combine_detections(detections[ADIK.name], detections[ID.name], [int(k) for k in keys])

    # ③ type 분류 (탐지된 시점 + SS)
    cfg = replace(CLASSIFIER, time_index_onehot=not legacy)
    model, scaler = _load(cfg, weights_dir)
    dataset = SplidDataset(cfg, test=objects, scaler=scaler, with_labels=False, legacy_diff=legacy)
    select = located.groupby('ObjectID')['TimeIndex'].apply(list)
    types = classifier_types(dataset, model, keys, select={str(k): v for k, v in select.items()})

    # ④ 규칙
    result = postprocess.assign_nodes(postprocess.assign_types(types, located))
    kept = postprocess.drop_ns_outside_ew_sk(result)[['ObjectID', 'TimeIndex', 'Direction']]
    kept = kept.merge(located, how='left', on=['ObjectID', 'TimeIndex', 'Direction'])     # 확정 전 상태로 되돌려
    result = postprocess.assign_nodes(postprocess.assign_types(types, kept))              # node 종류를 다시 정함
    print(f'EW 가 NK 인 동안의 NS node {len(located) - len(kept)}개 제거 → 최종 {len(result)}행')
    return result
