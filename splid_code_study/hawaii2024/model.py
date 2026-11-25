"""모델 구조: changepoint localizer (CNN) 와 classifier (LSTM).

원 코드 대응: base/prediction_models.py 의 Dense_NN (설정에 따라 CNN·LSTM·dense 를 조립하는 범용 클래스).
여기서는 최종 모델 두 가지 구조만 풀어 썼다. 층 이름은 원 코드와 같게 두었다.

두 모델의 공통 뼈대 (논문 Fig. 3)
    입력 (시간, feature) ─┬─ branch L ─┐
                          └─ branch R ─┴─ concatenate → flatten → dense 64 → dense 32 ─┬─ EW 출력
                                                                                        └─ NS 출력
  - branch 는 구조가 같은 두 갈래. 같은 입력을 받지만 가중치는 따로 학습된다
    (방향마다 다른 패턴을 볼 여지를 주되, 뒤의 dense 는 공유해 두 방향에 함께 나타나는 신호를 쓰게 함)
  - localizer 의 branch: Conv1D 3층 (각 층 뒤에 ReLU → BatchNorm → Dropout)
  - classifier 의 branch: LSTM 1층 (모든 시점의 hidden state 출력) → MaxPool
"""
import numpy as np
from tensorflow import keras
from keras import layers, regularizers


def build_localizer(cfg, n_channels):
    """시점마다 "node 와 얼마나 가까운가" (0 ~ 100) 를 방향별로 회귀하는 CNN.

    입력 (cfg.input_width, n_channels) → 출력 2개: EW_Node_Location_nb, NS_Node_Location_nb (각 1 값, linear)
    """
    rng = np.random.default_rng(cfg.model_seed)
    reg = regularizers.L2(cfg.l2)
    inputs = layers.Input(shape=(cfg.input_width, n_channels), name='local_in')

    branches = []
    for side in ['L', 'R']:
        x = inputs
        for i, (filters, kernel, stride) in enumerate(cfg.conv_layers):
            x = layers.Conv1D(filters, kernel, strides=stride, kernel_regularizer=reg,
                              kernel_initializer=_glorot(rng), name=f'conv1d_{i}_{side}_local')(x)
            x = layers.Activation('relu', name=f'conv1d_relu_{i}_{side}_local')(x)
            if cfg.batchnorm:
                x = layers.BatchNormalization(name=f'conv1d_BN_{i}_{side}_local')(x)
            if cfg.dropout_conv > 0:
                x = layers.Dropout(cfg.dropout_conv, seed=_seed(rng), name=f'conv1d_DO_{i}_{side}_local')(x)
        branches.append(x)
    x = layers.Concatenate(axis=-1)(branches)

    x = layers.Flatten(name='dense_flatten_local')(x)
    for i, units in enumerate(cfg.dense_layers):
        x = layers.Dense(units, kernel_regularizer=reg, kernel_initializer=_glorot(rng), name=f'dense_{i}_local')(x)
        x = layers.Activation('relu', name=f'dense_relu_{i}_local')(x)
        if cfg.batchnorm:
            x = layers.BatchNormalization(name=f'dense_BN_{i}_local')(x)
        if cfg.dropout_dense > 0:
            x = layers.Dropout(cfg.dropout_dense, seed=_seed(rng), name=f'dense_DO_{i}_local')(x)

    outputs = [layers.Dense(1, activation='linear', kernel_regularizer=reg, kernel_initializer=_glorot(rng),
                            name=f'{d}_Node_Location_nb')(x) for d in ['EW', 'NS']]
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(_learning_rate(cfg)),
                  loss=[keras.losses.MeanSquaredError(), keras.losses.MeanSquaredError()], metrics=['mse'])
    return model


def build_classifier(cfg, n_channels):
    """node 이후의 mode type (CK / EK / HK / NK) 을 방향별로 분류하는 LSTM.

    입력 (cfg.input_width, n_channels) → 출력 2개: EW_Type, NS_Type (각 4-class softmax)

    LSTM 을 쓰는 이유 (논문 3.1.4): type 은 국소적인 한 지점이 아니라 node 뒤 구간 전체의 거동으로 정해진다.
    모든 시점의 hidden state 가 비슷한 값을 내도록 학습되므로 max pooling 으로 줄여도 정보 손실이 적다.
    """
    rng = np.random.default_rng(cfg.model_seed)
    reg = regularizers.L2(cfg.l2)
    inputs = layers.Input(shape=(cfg.input_width, n_channels), name='local_in')

    branches = []
    for side in ['L', 'R']:
        x = layers.LSTM(cfg.lstm_units, return_sequences=True, kernel_regularizer=reg,
                        kernel_initializer=_glorot(rng),
                        recurrent_initializer=keras.initializers.Orthogonal(seed=_seed(rng)),
                        name=f'lstm_0_{side}_local')(inputs)
        x = layers.MaxPool1D(cfg.lstm_pool, name=f'lstm_MP_0_{side}_local')(x)
        branches.append(x)
    x = layers.Concatenate(axis=-1)(branches)

    x = layers.Flatten(name='dense_flatten_local')(x)
    for i, units in enumerate(cfg.dense_layers):
        x = layers.Dense(units, kernel_regularizer=reg, kernel_initializer=_glorot(rng), name=f'dense_{i}_local')(x)
        x = layers.Activation('relu', name=f'dense_relu_{i}_local')(x)

    outputs = [layers.Dense(4, activation='softmax', kernel_regularizer=reg, kernel_initializer=_glorot(rng),
                            name=f'{d}_Type')(x) for d in ['EW', 'NS']]
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(_learning_rate(cfg)),
                  loss=[keras.losses.SparseCategoricalCrossentropy(), keras.losses.SparseCategoricalCrossentropy()],
                  metrics=['accuracy'])
    return model


def _learning_rate(cfg):
    """(lr,) 이면 고정, (lr, decay_steps, decay_rate) 이면 decay_steps 마다 decay_rate 배로 줄이는 계단식 감소."""
    if len(cfg.learning_rate) == 1:
        return cfg.learning_rate[0]
    lr, steps, rate = cfg.learning_rate
    return keras.optimizers.schedules.ExponentialDecay(lr, decay_steps=steps, decay_rate=rate, staircase=True)


def _seed(rng):
    return int(rng.integers(9999999))


def _glorot(rng):
    """층마다 다른 seed 의 초기값 (model_seed 하나로 전체가 재현됨)."""
    return keras.initializers.GlorotUniform(seed=_seed(rng))
