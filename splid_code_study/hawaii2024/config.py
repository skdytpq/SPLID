"""Hawaii2024 (1위) 의 모델 3종 설정.

원 코드에서는 이 값들이 세 군데에 흩어져 있다.
  - submission.py            추론 때의 입력 설정 (feature, 창 크기, 임계값)
  - localizer_sweeper.py     AD·IK localizer 의 학습 설정 (wandb sweep 의 값)
  - classifier_sweeper.py    classifier 의 학습 설정
ID localizer 의 학습 설정은 원 저장소에 없다. 아래 ID 의 값은 다음에서 복원했다.
  - 입력: submission.py
  - 층 구성, dropout, L2, learning rate: 제공된 모델 파일 (hdf5) 에 저장된 model_config · training_config
  - 학습 일정 (표본 간격, epoch 수): 확인할 곳이 없어 AD·IK localizer 와 같게 둠

용어
  - history / future: 창의 과거·미래 step 수 (1 step = 2시간). history 는 현재 시점을 포함한다.
  - input_stride:     창 안에서 몇 step 마다 1개로 줄일지 (구간 평균으로 줄임)
  - overview:         궤적 전체를 창 길이만큼의 구간으로 나눠 구간별 표준편차를 낸 요약 (장기 변화용)
"""
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class ModelConfig:
    name: str                       # 가중치 파일 이름의 앞부분 (<name>_model.hdf5, <name>_scaler.pkl)

    # ── 입력 ─────────────────────────────────────────────────────────────
    raw_features: List[str]         # 변환 없이 (표준화만 해서) 넣는 열
    diff_features: List[str]        # 차분 (x[t] - x[t-1]) 을 넣는 열
    sin_features: List[str]         # sin 값을 넣는 각도 열
    overview_std: List[str]         # overview (구간별 표준편차) 로 넣는 열
    time_index: bool                # 궤적 안에서 현재 창의 위치를 넣을지
    time_index_onehot: bool         # True: overview 와 같은 축의 one-hot, False: -1 ~ 1 선형 값
    history: int
    future: int
    input_stride: int

    # ── 모델 ─────────────────────────────────────────────────────────────
    conv_layers: List[Tuple[int, int, int]] = field(default_factory=list)   # (filters, kernel, stride), 방향별 branch 마다
    lstm_units: int = 0             # classifier 의 LSTM 크기 (방향별 branch 마다)
    lstm_pool: int = 1              # LSTM 출력의 max pooling 크기
    dense_layers: List[int] = field(default_factory=lambda: [64, 32])
    batchnorm: bool = False
    dropout_conv: float = 0.0
    dropout_dense: float = 0.0
    l2: float = 0.0
    learning_rate: Tuple = (0.001,)  # (lr,) 고정 또는 (lr, decay_steps, decay_rate) 계단식 감소
    model_seed: int = 0

    # ── 학습 ─────────────────────────────────────────────────────────────
    train_val_split: float = 0.8
    exclude_objects: List[int] = field(default_factory=list)   # 저자가 번호로 뺀 학습 객체
    batch_size: int = 128
    # localizer 라벨: node 시점 100, 앞뒤로 70, 49, 34, 24, 16 (논문의 gradient encoding)
    label_shape: List[float] = field(default_factory=list)
    label_weight: dict = field(default_factory=dict)            # node 종류별 라벨 배율 (0 이면 그 종류는 학습 대상 아님)
    # localizer 학습 일정: (표본 간격, 시작 offset, epoch 수). node 주변 시점은 간격과 무관하게 항상 포함
    training_sets: List[Tuple[int, int, int]] = field(default_factory=list)
    nodes_to_score: List[str] = field(default_factory=list)     # 검증 채점 대상 node
    # classifier 학습
    max_epochs: int = 500
    early_stopping: int = 69
    sample_stride: int = 1000       # classifier 표본: 모든 node 시점 + 이 간격의 일반 시점
    sample_offset: int = 250

    # ── 추론 후처리 ──────────────────────────────────────────────────────
    thresholds: Tuple[float, float] = (50.0, 50.0)   # (EW, NS) 출력 임계값
    merge_distance: int = 2         # 이 간격 미만으로 붙은 탐지는 하나로 합침

    @property
    def window(self):
        return self.history + self.future

    @property
    def input_width(self):          # 모델에 들어가는 시간 축 길이
        return -(-self.window // self.input_stride)


_LOCALIZER_INPUT = dict(
    raw_features=['Eccentricity', 'Semimajor Axis (m)', 'Inclination (deg)', 'RAAN (deg)', 'Latitude (deg)'],
    diff_features=['Inclination (deg)', 'True Anomaly (deg)', 'Longitude (deg)'],
    sin_features=['Argument of Periapsis (deg)'],
)
_LOCALIZER_MODEL = dict(
    conv_layers=[(64, 7, 1), (64, 7, 1), (48, 7, 2)],
    dense_layers=[64, 32],
    batchnorm=True,
    dropout_dense=0.05,
    model_seed=0,
)
_LOCALIZER_TRAIN = dict(
    train_val_split=0.95,
    exclude_objects=[30, 113, 1012, 1383, 1385, 1386, 1471, 1473, 1474],
    batch_size=2048,
    label_shape=[100.0, 70.0, 49.0, 34.0, 24.0, 16.0],
    training_sets=[(5, 0, 45), (5, 3, 45)],     # 45 epoch 씩 두 번, 두 번째는 표본 위치를 3 step 옮김
)

# AD·IK localizer: 추진 maneuver 가 남기는 국소 흔적을 찾음 → 짧고 촘촘한 창 (과거 10.7일 + 미래 2.7일)
ADIK = ModelConfig(
    name='localizer_adik',
    **_LOCALIZER_INPUT,
    overview_std=[], time_index=False, time_index_onehot=True,
    history=128, future=32, input_stride=2,
    **_LOCALIZER_MODEL, dropout_conv=0.05, l2=0.00025, learning_rate=(0.005,),
    **_LOCALIZER_TRAIN,
    label_weight={'ID': 0.0, 'IK': 1.0, 'AD': 1.0, 'SS': 0.0}, nodes_to_score=['AD', 'IK'],
    thresholds=(45.0, 50.0),
)

# ID localizer: station-keeping 이 끝나는 시점 = 장기 거동 변화 → 긴 창 (과거 26.7일 + 미래 21.3일) + overview
ID = ModelConfig(
    name='localizer_id',
    **_LOCALIZER_INPUT,
    overview_std=['Inclination (deg)'], time_index=True, time_index_onehot=True,
    history=320, future=256, input_stride=4,
    **_LOCALIZER_MODEL, dropout_conv=0.1, l2=0.0001, learning_rate=(0.005, 3000, 0.9),     # 모델 파일에서 읽은 값
    **_LOCALIZER_TRAIN,
    label_weight={'ID': 1.0, 'IK': 0.0, 'AD': 0.0, 'SS': 0.0}, nodes_to_score=['ID'],
    thresholds=(40.0, 50.0),
)

# classifier: node 이후의 mode type (NK/CK/EK/HK) 판정 → 미래 쪽이 긴 창 (과거 2.7일 + 미래 21.3일)
CLASSIFIER = ModelConfig(
    name='classifier',
    raw_features=['Eccentricity', 'Semimajor Axis (m)', 'RAAN (deg)', 'True Anomaly (deg)', 'Latitude (deg)'],
    diff_features=['Eccentricity', 'Semimajor Axis (m)', 'Inclination (deg)', 'True Anomaly (deg)'],
    sin_features=['Argument of Periapsis (deg)', 'Longitude (deg)'],
    overview_std=['Argument of Periapsis (sin)', 'Latitude (deg)', 'Longitude (sin)'],
    time_index=True, time_index_onehot=True,    # 대회 제출 모델은 False (predict 의 legacy 옵션)
    history=32, future=256, input_stride=1,
    lstm_units=64, lstm_pool=6, dense_layers=[64, 32], l2=0.0001,
    learning_rate=(0.0035, 700, 0.9), model_seed=42,    # sweeper 값. 제공된 fixed 모델 파일에는 (0.002, 700, 0.9) 로 기록됨
    train_val_split=0.8,
    exclude_objects=[9, 10, 13, 19, 30, 113, 1012, 1383, 1385, 1386, 1471, 1473, 1474],
    batch_size=128, max_epochs=500, early_stopping=69, sample_stride=1000, sample_offset=250,
)

MODELS = {'adik': ADIK, 'id': ID, 'classifier': CLASSIFIER}

DATA_SEED = 11      # 학습/검증 객체를 나누는 seed (원 sweeper 값)
TRAIN_SEED = 42     # python / numpy / tensorflow 전역 seed
