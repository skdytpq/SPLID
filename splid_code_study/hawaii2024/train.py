"""학습: 모델 3종 (adik, id, classifier) 을 하나씩 학습해 <out>/<name>_model.hdf5, <name>_scaler.pkl 로 저장.

원 코드 대응: localizer_sweeper.py, classifier_sweeper.py 의 parameter_sweep() (wandb sweep 의 1회 실행).
wandb 기록과 시험 세트 평가는 뺐다 (시험 평가는 predict → eval 로 따로 한다).
"""
import json
import pickle
import random
from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf
from tensorflow import keras

from .config import DATA_SEED, MODELS, TRAIN_SEED
from .dataloader import SplidDataset, load_objects
from .eval import evaluate_classifier, evaluate_localizer
from .model import build_classifier, build_localizer


def run(model_key, data_dir, labels_path, out_dir, max_objects=None):
    cfg = MODELS[model_key]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(TRAIN_SEED)
    np.random.seed(TRAIN_SEED)
    tf.keras.utils.set_random_seed(TRAIN_SEED)

    objects = load_objects(data_dir, labels_path, max_objects)
    dataset = SplidDataset(cfg, train_val=objects, seed=DATA_SEED)
    n_channels = len(dataset.features) + len(dataset.overview)
    print(f'[{cfg.name}] 학습 {len(dataset.train_keys)}기 / 검증 {len(dataset.val_keys)}기, '
          f'입력 ({cfg.input_width}, {n_channels}): {dataset.features} + overview {dataset.overview}')

    if model_key == 'classifier':
        model, summary = _train_classifier(cfg, dataset, n_channels, labels_path)
    else:
        model, summary = _train_localizer(cfg, dataset, n_channels, labels_path)

    with h5py.File(out_dir / f'{cfg.name}_model.hdf5', 'w') as file:      # h5py 로 여는 이유는 predict._load 참고
        model.save(file)
    pickle.dump(dataset.scaler, open(out_dir / f'{cfg.name}_scaler.pkl', 'wb'))
    (out_dir / f'{cfg.name}_summary.json').write_text(json.dumps(summary, indent=1))
    print(f'[{cfg.name}] 저장: {out_dir}')
    return summary


def _train_localizer(cfg, dataset, n_channels, labels_path):
    """회귀 학습 (MSE). 표본은 "node 주변 시점 전부 + 나머지는 k 번째마다" 로 뽑아 불균형을 줄인다.

    cfg.training_sets 의 단계마다 표본 위치 (offset) 를 바꿔, 같은 계산량으로 궤적의 다른 시점들도 보게 한다.
    """
    model = build_localizer(cfg, n_channels)
    model.summary()
    label_cols = ['EW_Node_Location_nb', 'NS_Node_Location_nb']
    best = None
    for stride, offset, epochs in cfg.training_sets:
        sampling = dict(sample_stride=stride, sample_offset=offset)
        train_ds = dataset.tf_dataset(dataset.train_keys, label_cols, cfg.batch_size, **sampling)
        val_ds = dataset.tf_dataset(dataset.val_keys, label_cols, cfg.batch_size, **sampling)
        print(f'[{cfg.name}] 표본 간격 {stride}, offset {offset}, {epochs} epoch (학습 표본 {train_ds.cardinality() * cfg.batch_size} 이하)')
        model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=2)
        del train_ds, val_ds

        print(f'[{cfg.name}] 검증 세트 시점 탐지:')
        scores = evaluate_localizer(dataset, model, dataset.val_keys, labels_path)
        if best is None or scores['F2'] > best['F2']:
            best = scores
    return model, {'val_best': best}


def _train_classifier(cfg, dataset, n_channels, labels_path):
    """4-class 분류 학습. 표본은 거의 node 시점만 (추론 때도 탐지된 node 시점만 분류하므로 조건을 맞춤)."""
    model = build_classifier(cfg, n_channels)
    model.summary()
    label_cols = ['EW_Type', 'NS_Type']
    sampling = dict(sample_stride=cfg.sample_stride, sample_offset=cfg.sample_offset)
    train_ds = dataset.tf_dataset(dataset.train_keys, label_cols, cfg.batch_size, **sampling)
    val_ds = dataset.tf_dataset(dataset.val_keys, label_cols, cfg.batch_size, **sampling)
    stop = keras.callbacks.EarlyStopping(monitor='val_EW_Type_accuracy', patience=cfg.early_stopping,
                                         restore_best_weights=True, verbose=1)
    hist = model.fit(train_ds, validation_data=val_ds, epochs=cfg.max_epochs, callbacks=[stop], verbose=2)

    print(f'[{cfg.name}] 검증 세트 type 분류:')
    scores = evaluate_classifier(dataset, model, dataset.val_keys, labels_path)
    return model, {'epochs': len(hist.history['loss']), 'val': scores}
