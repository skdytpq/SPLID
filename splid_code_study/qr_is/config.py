"""QR_Is (3위) 설정. 원 코드에서는 main_train.py · main_submission.py · utils.py 안에 값으로 흩어져 있다."""

# 원시 입력 15개 (csv 의 열)
RAW_COLUMNS = ['Eccentricity', 'Semimajor Axis (m)', 'Inclination (deg)', 'RAAN (deg)', 'Argument of Periapsis (deg)',
               'True Anomaly (deg)', 'Latitude (deg)', 'Longitude (deg)', 'Altitude (m)',
               'X (m)', 'Y (m)', 'Z (m)', 'Vx (m/s)', 'Vy (m/s)', 'Vz (m/s)']

# ── feature (1 step = 2시간) ────────────────────────────────────────────
TIME_CYCLES = [6, 12, 36, 84, 168]          # 주기 index: 12시간, 1일, 3일, 1주, 2주마다 1..n 을 반복
LAG_STEPS = 6                               # lag: 앞뒤 6 step (±12시간) 의 값을 그대로 가져옴
DIFF_STEPS = [-2, -1, 1, 2]                 # 차분: x[t] - x[t-j]
PCT_STEPS = [-1, 1]                         # 변화율: (x[t] - x[t-j]) / x[t-j]
ENVELOPE_WINDOW = 36                        # envelope: 가운데 정렬 3일 창의 (최대 - 최소)
ROLLING_WINDOW = 12                         # 이동 평균·표준편차: 가운데 정렬 1일 창

ENVELOPE_COLUMNS = ['Latitude (deg)', 'Longitude (deg)', 'Altitude (m)']
DIFF_COLUMNS = ['Eccentricity', 'Semimajor Axis (m)', 'Inclination (deg)', 'RAAN (deg)', 'Argument of Periapsis (deg)',
                'True Anomaly (deg)', 'Latitude (deg)', 'Longitude (deg)', 'Altitude (m)',
                'Latitude (deg)_enveloppe', 'Longitude (deg)_enveloppe', 'Altitude (m)_enveloppe']
PCT_COLUMNS = ['Vx (m/s)', 'Vy (m/s)', 'Vz (m/s)']
ROLLING_COLUMNS = ['Eccentricity', 'Semimajor Axis (m)', 'Inclination (deg)', 'RAAN (deg)', 'Argument of Periapsis (deg)',
                   'True Anomaly (deg)', 'Altitude (m)', 'Vx (m/s)', 'Vy (m/s)', 'Vz (m/s)']
# 위성 단위 통계: 궤적 전체의 평균·표준편차 (그 위성의 모든 시점에 같은 값)
OBJECT_MEAN_COLUMNS = ['Eccentricity', 'Semimajor Axis (m)', 'Inclination (deg)', 'RAAN (deg)',
                       'Argument of Periapsis (deg)', 'True Anomaly (deg)', 'Altitude (m)']
OBJECT_STD_COLUMNS = ['X (m)', 'Y (m)', 'Z (m)', 'Vx (m/s)', 'Vy (m/s)', 'Vz (m/s)', 'Eccentricity', 'Altitude (m)',
                      'Inclination (deg)']
# heuristic 예측의 one-hot 열 중, 데이터에 그 class 가 없어도 0 으로 만들어 두는 열 (학습·추론의 열을 맞추기 위함)
HEURISTIC_CLASSES = {'EW': ['AD-NK', 'ID-NK', 'IK-CK', 'IK-EK', 'SS-CK', 'SS-EK', 'SS-NK'],
                     'NS': ['ID-NK', 'IK-CK', 'IK-EK', 'SS-CK', 'SS-EK', 'SS-NK']}

# feature 가 아닌 열
NON_FEATURE_COLUMNS = ['TimeIndex', 'Timestamp', 'ObjectID', 'EW', 'NS',
                       'EW_baseline_heuristic', 'NS_baseline_heuristic',
                       'EW_baseline_heuristic_ffill', 'NS_baseline_heuristic_ffill']

# ── 모델 ────────────────────────────────────────────────────────────────
N_ESTIMATORS = 1000                         # CatBoost tree 수 (모델 3개 각각)
SEED = 42

# ── 예측: 이중 임계값 (precision 용, recall 용) ─────────────────────────
THRESHOLDS = {'EW': (0.1, 0.11), 'NS': (0.1, 0.26)}
