# SPLID

SPLID 챌린지 (2024 MIT ARCLab Prize for AI Innovation in Space) 상위 팀의 공개 코드를 읽고 해석하기 위한 저장소.
GEO 위성의 궤적에서 운용 mode 가 바뀌는 시점 (node) 과 그 뒤의 mode type 을 찾는 문제이며, 1위 (Hawaii2024) 와 3위 (QR_Is) 의 코드를
`config` · `dataloader` · `model` · `train` · `predict` · `eval` · `main` 구조로 다시 짜서 [`splid_code_study/`](splid_code_study/README.md) 에 둠.

- 실행 방법, 폴더 구조, 원 코드와의 대조 결과: [`splid_code_study/README.md`](splid_code_study/README.md)
- 팀별 코드 해설: [`hawaii2024/README.md`](splid_code_study/hawaii2024/README.md), [`qr_is/README.md`](splid_code_study/qr_is/README.md)
- 2위 (Millennial-IUP, XGBoost) 는 공개된 코드를 찾지 못해 포함하지 않음

## 1. 들어 있는 코드

| 폴더 | 내용 | 방식 |
|---|---|---|
| `splid_code_study/hawaii2024/` | 1위 Hawaii2024. TensorFlow 2.15 | node 시점을 CNN 2개로 탐지하고, 탐지된 시점의 mode type 을 LSTM 으로 분류 |
| `splid_code_study/qr_is/` | 3위 QR_Is. CatBoost 1.2 | 시점마다 feature 약 320개를 만들어 "이 시점이 어떤 node 인가" 를 CatBoost 3개로 분류 |
| `splid_code_study/common/` | 주최 측 채점기 | 예측이 정답의 ±6 step 안이고 Node·Type 이 같으면 TP. F2 = 5TP / (5TP + 4FN + FP) |
| `splid_code_study/data_example/` | 예시 입력 1기 (시험 객체 1962) | 궤적 csv 와 정답 node |
| `splid_code_study/output_example/` | 예시 출력 | 위 객체에 대한 두 팀의 예측 csv 와 채점 결과 |

두 팀 폴더의 파일 구성은 같음.

| 파일 | 역할 |
|---|---|
| `config.py` | 입력 feature, 모델 구조, 학습 설정, 임계값 |
| `dataloader.py` | csv 읽기와 입력 만들기 (QR_Is 는 `features.py`, `heuristic.py` 로 나뉨) |
| `model.py` | 모델 구조 |
| `train.py` | 학습 |
| `predict.py` | 추론과 후처리 (Hawaii2024 는 `postprocess.py` 로 나뉨) |
| `eval.py` | 채점 |
| `main.py` | 실행 진입점: `train` / `predict` / `eval` |
| `weights/` | 저자가 공개한 학습된 모델 |

## 2. 데이터

두 팀 모두 SPLID 데이터셋의 `phase_2` 만 사용. 외부 데이터 없음.

| 묶음 | 객체 수 | 구성 | 용도 |
|---|---|---|---|
| `training/` + `train_label.csv` | 1,900기 | 시뮬레이션 1,870 + VCM 기반 30 | 학습 (Hawaii2024 는 일부를 검증용으로 뗌) |
| `test/` + `test_label.csv` | 501기 | 시뮬레이션 130 + VCM 기반 76 + TLE 기반 296 의 502기 중 1기 없음 | 최종 채점 |

- 객체 1기 = csv 1개. 2시간 간격, 약 6개월 (약 2,170행). 행 번호가 `TimeIndex`
- 열: `Timestamp`, Keplerian 6요소 (`Eccentricity`, `Semimajor Axis (m)`, `Inclination (deg)`, `RAAN (deg)`, `Argument of Periapsis (deg)`, `True Anomaly (deg)`), `Latitude (deg)`, `Longitude (deg)`, `Altitude (m)`, J2000 위치·속도 (`X/Y/Z (m)`, `Vx/Vy/Vz (m/s)`)
- 정답: node 목록 `[ObjectID, TimeIndex, Direction, Node, Type]`
  - `Direction`: EW (경도 방향), NS (위도 방향)
  - `Node`: SS (관측 시작), ID (drift 시작), AD (drift 조정), IK (station-keeping 시작)
  - `Type`: node 뒤의 mode. NK (station-keeping 안 함), CK / EK / HK (chemical / electric / hybrid 추진)
- 데이터셋 전체는 이 저장소에 없음. 내려받는 곳은 원 Hawaii2024 저장소의 README 참고. 예시 1기만 `data_example/` 에 포함

## 3. Hawaii2024 (1위) 의 학습

"node 가 언제인가" 와 "그 뒤 mode 가 무엇인가" 를 다른 모델로 나눠, 모델 3개를 따로 학습.

| | AD·IK localizer | ID localizer | classifier |
|---|---|---|---|
| 찾는 것 | AD, IK node 의 시점 | ID node 의 시점 | node 뒤의 mode type (NK / CK / EK / HK) |
| 구조 | Conv1D 3층 branch 2개 → dense 64, 32 → EW·NS 출력 | 왼쪽과 같음 | LSTM 64 branch 2개 → max pooling → dense 64, 32 → EW·NS 출력 |
| 입력 창 | 과거 128 + 미래 32 step (10.7일 + 2.7일) | 과거 320 + 미래 256 step (26.7일 + 21.3일) | 과거 32 + 미래 256 step (2.7일 + 21.3일) |
| 모델 입력 크기 | 80 × 9 (2 step 씩 평균) | 144 × 11 (4 step 씩 평균) | 288 × 15 |
| 라벨 | node 시점 100, 앞뒤 5 step 은 70 · 49 · 34 · 24 · 16, 나머지 0 | 왼쪽과 같음 (ID node 만) | 그 시점의 mode type, 방향별 4 class |
| loss | MSE (회귀) | MSE (회귀) | sparse categorical cross-entropy |
| 학습 / 검증 객체 | 1,796 / 95 (9기 제외) | 1,796 / 95 (9기 제외) | 1,511 / 376 (13기 제외) |
| 학습 표본 | node 주변 시점 전부 + 나머지는 5 step 마다 | 왼쪽과 같음 | node 시점 전부 + 1,000 step 마다 |
| batch / optimizer | 2,048 / Adam, lr 0.005 | 2,048 / Adam, lr 0.005 에서 3,000 step 마다 0.9배 | 128 / Adam, lr 0.0035 에서 700 step 마다 0.9배 |
| epoch | 45 + 45 (두 번째는 표본 위치를 3 step 옮김) | 왼쪽과 같음 | 최대 500, 검증 EW type 정확도가 69 epoch 동안 안 오르면 중단 |
| 정규화 | BatchNorm, dropout 0.05, L2 0.00025 | BatchNorm, dropout 0.1 (conv) · 0.05 (dense), L2 0.0001 | L2 0.0001 |

입력을 만드는 순서 (`dataloader.SplidDataset`)

1. 각도 열에 sin 적용 (`Argument of Periapsis`, classifier 는 `Longitude` 도)
2. 차분 `x[t] − x[t−1]` 추가 (localizer: `Inclination`, `True Anomaly`, `Longitude`)
3. 학습 객체의 값으로 맞춘 StandardScaler 로 표준화
4. 시점마다 [과거, 미래] 창을 잘라 냄. 궤적 양 끝은 0 으로 채움
5. ID localizer 와 classifier 는 궤적 전체의 구간별 표준편차 (overview) 와 "현재 창이 궤적의 어디인가" 를 나타내는 one-hot 을 채널로 추가

추론 (`predict.run`): localizer 출력이 임계값 (AD·IK: EW 45 / NS 50, ID: EW 40 / NS 50) 을 넘는 연속 구간의 가운데를 node 시점으로 정함
→ 그 시점과 시작 시점에서 classifier 로 type 분류 → type 의 앞뒤 순서로 node 종류 결정 (NK→SK 는 IK, SK→NK 는 ID, NK→NK 는 AD) → EW 가 NK 인 동안의 NS node 제거.

```bash
cd splid_code_study
python -m hawaii2024.main train --model adik       --data <phase_2/training> --labels <phase_2/train_label.csv> --out runs/hawaii
python -m hawaii2024.main train --model id         --data <phase_2/training> --labels <phase_2/train_label.csv> --out runs/hawaii
python -m hawaii2024.main train --model classifier --data <phase_2/training> --labels <phase_2/train_label.csv> --out runs/hawaii
python -m hawaii2024.main predict --data <phase_2/test> --weights runs/hawaii --out pred.csv
python -m hawaii2024.main eval    --pred pred.csv --labels <phase_2/test_label.csv>
```

## 4. QR_Is (3위) 의 학습

궤적을 "행 = 한 시점" 인 표 하나로 만들고, 시점마다 class 하나를 고르는 multi-class 분류로 풂. 검증 세트 없이 1,900기 전부 (약 410만 행) 로 학습.

- 라벨: node 가 있는 시점만 `IK-EK` 같은 "Node-Type", 나머지 시점은 `Nothing`. 방향별로 EW 10 class, NS 9 class
- feature 약 320개 (`features.add_features`, 1 step = 2시간)

| 종류 | 내용 |
|---|---|
| 원시 값 | csv 의 15개 열 |
| lag · lead | 앞뒤 6 step 의 값 |
| 차분 · 변화율 | ±1, ±2 step 차분, 속도의 ±1 step 변화율 |
| envelope | 위도 · 경도 · 고도의 36 step (3일) 창 안의 최대 − 최소 |
| 이동 통계 | 12 step (1일) 창의 평균 · 표준편차 |
| 위성 단위 통계 | 궤적 전체의 평균 · 표준편차 (그 위성의 모든 시점에 같은 값) |
| 주기 index | 6, 12, 36, 84, 168 step 주기로 반복하는 값 |
| heuristic 예측 | 주최 측 heuristic baseline 의 예측을 one-hot 으로 |

- 모델: CatBoost 분류기 3개를 차례로 학습 (`model.StackedCatBoost`). 각각 tree 1,000개, seed 42, 나머지는 CatBoost 기본값
  1. NS 1차 모델: feature → NS class
  2. EW 모델: feature + 1번의 NS class 확률 → EW class
  3. NS 최종 모델: feature + 2번의 EW class 확률 → NS class
- 추론 (`predict.py`): 최고 확률이 0.1 미만이면 `Nothing`. 1위가 `Nothing` 이어도 2위 확률이 EW 0.11 / NS 0.26 을 넘으면 2위 class 선택 → class 가 바뀌는 시점만 node 로 기록

```bash
cd splid_code_study
python -m qr_is.main train   --data <phase_2/training> --labels <phase_2/train_label.csv> --out runs/qr_is
python -m qr_is.main predict --data <phase_2/test> --weights runs/qr_is --out pred.csv
python -m qr_is.main eval    --pred pred.csv --labels <phase_2/test_label.csv>
```

전체 학습은 CPU 로 약 5시간, 메모리 수십 GB 필요.

## 5. 성능 (시험 501기, F2)

| | 논문 | 저자 제공 모델을 이 저장소의 코드로 추론 | 원 코드로 1,900기 재학습 |
|---|---|---|---|
| Hawaii2024 | 0.805 | 0.805 (`weights/submitted`), 0.819 (`weights/fixed`) | 0.814 |
| QR_Is | 0.793 | 0.802 | 0.797 |

- 저자 제공 모델로 낸 예측 파일은 원 코드의 예측 파일과 모든 행이 같음 (QR_Is 는 원 코드를 같은 파일 순서로 실행했을 때)
- QR_Is 의 0.802 는 원 코드의 파일 정렬 문제를 고친 값. 원 코드 그대로는 0.798 (내용은 [`qr_is/README.md`](splid_code_study/qr_is/README.md))
- 이 저장소의 코드로 한 1,900기 재학습 결과는 아직 반영하지 않음. 소규모 학습 (Hawaii2024 159기, QR_Is 60기) → 추론 → 채점의 동작은 확인함
- 두 팀 모두 시뮬레이션 궤적 위주로 학습되어, VCM · TLE 기반 시험 궤적 340기에서는 ID · IK · AD node 118개 중 0 ~ 2개만 탐지

## 6. 출처

- Hawaii2024: https://github.com/DavidBaldsiefen/splid-challenge (MIT License)
- QR_Is: https://github.com/FrancoisPorcher/mit-challenge (LICENSE 파일 없음)
- 채점기 · heuristic baseline: 주최 측 splid-devkit (MIT License)
- 자세한 내용: [`splid_code_study/licenses/NOTICE.md`](splid_code_study/licenses/NOTICE.md)
- 논문: Siew et al., "Satellite Pattern-of-Life Identification Challenge: Competition Design and Results", AMOS 2024
