"""데이터 로딩 → feature 변환 → 창(window) 만들기.

원 코드 대응: base/datahandler.py 의 load_and_prepare_dataframes() 와 DatasetGenerator.
원 코드에서 최종 모델이 쓰지 않는 옵션 (lowpass, daytime·yeartime feature, oneshot, 2-입력 모델 등) 은 뺐다.

흐름
  1. load_objects()          객체별 csv → DataFrame. 라벨이 있으면 시점마다 "현재 mode" 와 "node 여부" 를 붙임
  2. SplidDataset(...)       학습/검증 분할 → feature 변환 (sin, 차분, 시간 위치) → 표준화 → localizer 용 라벨 만들기
  3. SplidDataset.arrays()   시점마다 [과거 history, 미래 future] 창을 잘라 모델 입력 (n, 시간, feature) 로 만듦
"""
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler

RAW_COLUMNS = ['Eccentricity', 'Semimajor Axis (m)', 'Inclination (deg)', 'RAAN (deg)', 'Argument of Periapsis (deg)',
               'True Anomaly (deg)', 'Latitude (deg)', 'Longitude (deg)', 'Altitude (m)',
               'X (m)', 'Y (m)', 'Z (m)', 'Vx (m/s)', 'Vy (m/s)', 'Vz (m/s)']
ANGLE_COLUMNS = ['Inclination (deg)', 'RAAN (deg)', 'Argument of Periapsis (deg)', 'True Anomaly (deg)',
                 'Latitude (deg)', 'Longitude (deg)']
DIRECTIONS = ['EW', 'NS']
NODE_ENCODER = LabelEncoder().fit(['SS', 'ID', 'AD', 'IK'])
TYPE_ENCODER = LabelEncoder().fit(['NK', 'CK', 'EK', 'HK'])     # 정수 0..3 = CK, EK, HK, NK (알파벳 순)


def load_objects(data_dir, labels_path=None, max_objects=None):
    """data_dir 의 <ObjectID>.csv 를 읽어 {str(ObjectID): DataFrame} 으로 돌려준다.

    labels_path 를 주면 방향별로 다음 열을 붙인다 (dir = EW, NS).
      {dir}_Node, {dir}_Type    그 시점에 유효한 node·type (직전 node 의 값을 다음 node 까지 채움)
      {dir}_Node_Location       그 시점이 node 인지 (bool)
    두 방향의 라벨이 모두 있지 않은 객체는 건너뛴다.
    """
    labels = None if labels_path is None else pd.read_csv(labels_path)
    files = sorted(Path(data_dir).glob('*.csv'), key=lambda f: int(f.stem))
    if not files:
        raise ValueError(f'csv 파일이 없음: {data_dir}')
    objects = {}
    for file in files[:max_objects]:
        object_id = int(file.stem)
        df = pd.read_csv(file, dtype={c: np.float32 for c in RAW_COLUMNS})
        df['ObjectID'] = object_id
        df['TimeIndex'] = range(len(df))
        if labels is not None:
            obj_labels = labels[labels['ObjectID'] == object_id]
            if not all(d in obj_labels['Direction'].unique() for d in DIRECTIONS):
                print(f'객체 {object_id}: 두 방향의 라벨이 없어 건너뜀')
                continue
            for d in DIRECTIONS:
                dir_labels = obj_labels[obj_labels['Direction'] == d].sort_values('TimeIndex')
                dir_labels = pd.DataFrame({'ObjectID': object_id, 'TimeIndex': dir_labels['TimeIndex'],
                                           f'{d}_Node': dir_labels['Node'], f'{d}_Type': dir_labels['Type'],
                                           f'{d}_Node_Location': True})
                df = pd.merge_ordered(df, dir_labels, on=['TimeIndex', 'ObjectID'], how='left')
                df[f'{d}_Node'] = df[f'{d}_Node'].ffill()
                df[f'{d}_Type'] = df[f'{d}_Type'].ffill()
                df[f'{d}_Node_Location'] = df[f'{d}_Node_Location'].fillna(False).astype(bool)
        objects[str(object_id)] = df
    return objects


class SplidDataset:
    """feature 변환을 마친 객체별 DataFrame 을 들고, 요청에 따라 창 배열을 만들어 준다.

    cfg:          config.ModelConfig
    train_val:    학습·검증에 쓸 객체 dict (라벨 포함). cfg.train_val_split 으로 객체 단위 분할
    test:         시험·추론 객체 dict
    scaler:       이미 맞춰 둔 StandardScaler (추론 때). None 이면 학습 객체로 새로 맞춤
    legacy_diff:  대회 제출 모델이 학습된 방식 (각도 차분의 360도 넘김을 보정하지 않는 버그) 을 재현
    """

    def __init__(self, cfg, train_val=None, test=None, scaler=None, with_labels=True, seed=42, legacy_diff=False):
        self.cfg = cfg
        self.with_labels = with_labels
        self.seed = seed

        # ── 1. 객체 단위 학습/검증 분할 (키를 정렬한 뒤 seed 로 섞음 → 실행마다 같은 분할)
        keys = sorted(train_val.keys()) if train_val else []
        random.Random(seed).shuffle(keys)
        split = int(len(keys) * cfg.train_val_split)
        excluded = set(cfg.exclude_objects) if train_val else set()
        self.train_keys = [k for k in keys[:split] if int(k) not in excluded]
        self.val_keys = [k for k in keys[split:] if int(k) not in excluded]
        self.test_keys = list(test.keys()) if test else []
        frames = {**(train_val or {}), **(test or {})}
        frames = {k: frames[k].copy() for k in self.train_keys + self.val_keys + self.test_keys}

        # ── 2. feature 변환
        features = list(cfg.raw_features)
        for df in frames.values():
            # 값 범위 통일: True Anomaly 는 [0, 360), Longitude 는 (-180, 180]
            df.loc[df['True Anomaly (deg)'] < 0.0, 'True Anomaly (deg)'] += 360.0
            df.loc[df['Longitude (deg)'] > 180.0, 'Longitude (deg)'] -= 360.0
            # sin: 각도의 0/360 불연속 제거. overview 가 sin 열을 쓸 수 있어 각도 열 전부에 대해 만들어 둠
            for col in ANGLE_COLUMNS:
                df[_sin_name(col)] = np.sin(np.deg2rad(df[col]))
            # 차분: 직전 시점과의 차이 (첫 시점은 0) → 급격한 국소 변화 (maneuver) 를 강조
            for col in cfg.diff_features:
                df[col + ' (diff)'] = _diff(df[col].to_numpy(), col, legacy_diff)
            # 시간 위치: 궤적 시작 -1 → 끝 +1
            if cfg.time_index:
                df['LinearTimeIndex'] = np.linspace(-1.0, 1.0, len(df))
        features += [_sin_name(c) for c in cfg.sin_features] + [c + ' (diff)' for c in cfg.diff_features]
        features += ['LinearTimeIndex'] if cfg.time_index else []
        self.features = sorted(set(features))       # 알파벳 순 = 모델 입력 채널 순서 (scaler 의 열 순서와 같음)
        self.overview = sorted(cfg.overview_std)

        # ── 3. 표준화: 학습 객체 전체로 평균·표준편차를 맞추고 모든 객체에 적용
        if scaler is None:
            scaler = StandardScaler().fit(pd.concat([frames[k] for k in self.train_keys], ignore_index=True)[self.features].values)
        self.scaler = scaler
        for df in frames.values():
            df[self.features] = scaler.transform(df[self.features].values)

        # ── 4. 라벨
        if with_labels:
            for df in frames.values():
                if cfg.label_shape:
                    _add_localizer_labels(df, cfg.label_shape, cfg.label_weight)
                for d in DIRECTIONS:
                    df[f'{d}_Type'] = TYPE_ENCODER.transform(df[f'{d}_Type'])

        input_cols = list(dict.fromkeys(self.features + self.overview))
        for df in frames.values():
            df[input_cols] = df[input_cols].astype(np.float32)
        self.frames = frames

    # ------------------------------------------------------------------
    def arrays(self, keys, label_cols=(), sample_stride=1, sample_offset=0, only_nodes=False, select=None):
        """객체들의 창을 이어 붙여 (X, y, ids) 를 돌려준다.

        X:   (n, cfg.input_width, feature 수 + overview 수) float32
        y:   (n, len(label_cols))
        ids: (n, 2) = [ObjectID, TimeIndex] — 각 창의 "현재 시점"

        어떤 시점의 창을 만들지 (객체마다):
          select={key: [TimeIndex, ...]}  지정한 시점만 (추론에서 탐지된 node 만 분류할 때)
          only_nodes=True                 라벨이 node 인 시점만
          sample_stride=k > 1             k 번째마다 + 라벨이 0 보다 큰 시점 전부 (불균형 완화)
          그 외                            모든 시점
        """
        cfg = self.cfg
        X, y, ids = [], [], []
        for key in sorted(keys, key=int):
            df = self.frames[key]
            n = len(df) - 1                 # 마지막 행 (ES, 관측 끝) 은 예측 대상이 아님
            if select is not None:
                rows = np.asarray(sorted(set(select.get(key, []))), dtype=int)
            elif only_nodes or sample_stride > 1:
                location_cols = ([f'{d}_Node_Location_nb' for d in DIRECTIONS] if cfg.label_shape
                                 else [f'{d}_Node_Location' for d in DIRECTIONS])
                is_node = (df[location_cols].to_numpy(dtype=np.float32)[:n] > 0.0).any(axis=1)
                keep = is_node.copy()
                if sample_stride > 1:
                    keep[sample_offset::sample_stride] = True
                rows = np.flatnonzero(keep)
            else:
                rows = np.arange(n)
            if len(rows) == 0:
                continue
            X.append(self._windows(df, rows, n))
            y.append(df[list(label_cols)].to_numpy()[rows])
            ids.append(df[['ObjectID', 'TimeIndex']].to_numpy(dtype=np.int32)[rows])
        width = cfg.input_width
        if not X:
            return (np.zeros((0, width, len(self.features) + len(self.overview)), np.float32),
                    np.zeros((0, len(label_cols))), np.zeros((0, 2), np.int32))
        return np.concatenate(X), np.concatenate(y), np.concatenate(ids)

    def _windows(self, df, rows, n):
        """한 객체에서 rows 시점들의 입력 창을 만든다."""
        cfg = self.cfg
        values = df[self.features].to_numpy(dtype=np.float32)
        # 궤적 앞뒤를 0 으로 채워, 시점 t 의 창이 [t - history + 1, t + future] 를 덮게 함
        padded = np.pad(values, ((cfg.history - 1, cfg.future - 1), (0, 0)))
        windows = np.lib.stride_tricks.sliding_window_view(padded, cfg.window, axis=0).transpose(0, 2, 1)[rows]
        windows = _average_pool(windows, cfg.input_stride)          # (len(rows), input_width, feature)

        if cfg.time_index and cfg.time_index_onehot:
            # 시간 위치를 overview 와 같은 축의 one-hot 으로: "지금 창이 궤적의 어디쯤인가" 를 1 하나로 표시
            channel = self.features.index('LinearTimeIndex')
            windows[:, :, channel] = 0.0
            windows[np.arange(len(rows)), (rows * (cfg.input_width / n)).astype(int), channel] = 1.0

        if self.overview:
            # overview: 궤적 전체를 input_width 개 구간으로 나눠 구간별 표준편차 → 그 객체의 모든 창에 같은 값
            segment = len(df) // cfg.input_width
            border = (len(df) % cfg.input_width) // 2
            series = df[self.overview].to_numpy()[border:border + cfg.input_width * segment]
            overview = np.std(np.split(series, cfg.input_width), axis=1)        # (input_width, overview 수)
            overview = np.broadcast_to(overview.astype(np.float32), (len(rows),) + overview.shape)
            windows = np.concatenate([windows, overview], axis=-1)
        return windows

    def tf_dataset(self, keys, label_cols, batch_size, shuffle=True, **sampling):
        """학습용 tf.data.Dataset. 라벨은 {출력 층 이름: 값} dict (출력 층 이름 = 라벨 열 이름)."""
        X, y, _ = self.arrays(keys, label_cols, **sampling)
        with tf.device('CPU'):
            ds = tf.data.Dataset.from_tensor_slices(({'local_in': X}, {c: y[:, i] for i, c in enumerate(label_cols)}))
        if shuffle:
            ds = ds.shuffle(ds.cardinality(), seed=self.seed)
        return ds.batch(batch_size)


# ──────────────────────────────────────────────────────────────────────
def _sin_name(col):
    return col[:-5] + '(sin)'           # 'Longitude (deg)' → 'Longitude (sin)'


def _diff(x, col, legacy):
    """차분. 각도 열은 360도 넘김 (예: 359 → 1) 이 큰 값으로 튀지 않게 보정한다."""
    d = np.diff(x, prepend=x[0])
    if legacy:
        return d                        # 대회 제출 모델: 보정이 적용되지 않은 채 학습됨
    if col in ['Longitude (deg)', 'Argument of Periapsis (deg)', 'RAAN (deg)']:
        upper, lower = 180, -180
    elif col == 'True Anomaly (deg)':
        upper, lower = 270, -90         # 보통 증가하지만 작은 감소도 가능
    else:
        return d
    d[d > upper] -= 360
    d[d < lower] += 360
    return d


def _average_pool(windows, stride):
    """시간 축을 stride 개씩 묶어 평균 (원 코드의 convolve_input_stride). 왼쪽 가장자리는 첫 값으로 채운다.

    windows: (n, L, f) → (n, L / stride, f)   (160 → 80, 576 → 144)
    """
    if stride == 1:
        return np.ascontiguousarray(windows)
    n, length, f = windows.shape
    padded = np.concatenate([np.repeat(windows[:, :1], stride // 2, axis=1), windows], axis=1)
    blocks = (length + stride // 2 - stride) // stride + 1
    padded = padded[:, :blocks * stride].reshape(n, blocks, stride, f)
    return padded.mean(axis=2, dtype=np.float64).astype(np.float32)


def _add_localizer_labels(df, shape, weight):
    """localizer 의 회귀 라벨 {dir}_Node_Location_nb 를 만든다 (논문의 gradient encoding).

    node 시점에 shape[0] (=100), 앞뒤로 shape[1:], 그 밖은 0. node 종류별 배율 weight 를 곱한다
    (AD·IK localizer 는 ID 의 배율이 0 → ID 는 학습 대상에서 빠짐).
    """
    profile = np.asarray(shape[::-1] + shape[1:], dtype=np.float32)     # [16, 24, ..., 100, ..., 24, 16]
    half = len(shape) - 1
    for d in DIRECTIONS:
        col = f'{d}_Node_Location_nb'
        is_target = df[f'{d}_Node_Location'] & df[f'{d}_Node'].isin(['ID', 'AD', 'IK'])
        df[col] = is_target.astype(np.float32)
        for t in df.loc[is_target, 'TimeIndex'].to_numpy():
            df.loc[t - half:t + half, col] = profile * weight[df.loc[t, f'{d}_Node']]
