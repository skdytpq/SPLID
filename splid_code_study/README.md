# SPLID 챌린지 상위 팀 코드 읽기

SPLID 챌린지 (2024 MIT ARCLab Prize for AI Innovation in Space, GEO 위성의 운용 모드 전환 시점 탐지) 상위 팀의 공개 코드를
**읽고 해석하기 쉽게** 일반적인 딥러닝 프로젝트 구조 (`config` · `dataloader` · `model` · `train` · `predict` · `eval` · `main`) 로 다시 짠 저장소.

- 계산 내용은 원 코드와 같게 유지했고, 같은 결과가 나오는지 대조했다 (아래 "확인한 것")
- 팀마다 폴더 하나, 파일 이름과 실행 방법은 두 팀이 같다
- 예시 데이터 1기와 그 예측 결과가 들어 있어, 전체 데이터셋 없이 추론·채점을 바로 실행해 볼 수 있다

| 순위 | 팀 | 접근 | 이 저장소 | 논문 F2 |
|---|---|---|---|---|
| 1 | Hawaii2024 | 시점 탐지 (CNN 회귀) + type 분류 (LSTM) | [`hawaii2024/`](hawaii2024/README.md) | 0.805 |
| 2 | Millennial-IUP | XGBoost, 전환 시점 라벨 | 공개된 코드를 찾지 못해 포함하지 않음 | 0.810 |
| 3 | QR_Is | feature 약 320개 + CatBoost 3단 연결 | [`qr_is/`](qr_is/README.md) | 0.793 |

## 폴더 구조

```
splid_code_study/
├── README.md
├── common/
│   └── evaluator.py          공식 채점기 (두 팀 공통): ±6 step 안에서 짝짓고 Node·Type 이 같으면 TP
├── hawaii2024/               1위. TensorFlow 2.15
│   ├── README.md             코드 해설 (흐름, 파일별 역할, 핵심 아이디어의 코드 위치, 원 코드와 달라진 점)
│   ├── config.py             모델 3종 설정
│   ├── dataloader.py         csv → feature 변환 → 창
│   ├── model.py              localizer (CNN), classifier (LSTM)
│   ├── train.py              학습
│   ├── predict.py            추론 전체 흐름
│   ├── postprocess.py        모델 출력 → node 목록
│   ├── eval.py               채점
│   ├── main.py               실행 진입점
│   ├── requirements.txt
│   └── weights/              저자 제공 모델: submitted (대회 제출), fixed (버그 수정 후 재학습)
├── qr_is/                    3위. CatBoost
│   ├── README.md
│   ├── config.py
│   ├── dataloader.py         csv → 표 (행 = 한 시점)
│   ├── features.py           feature engineering
│   ├── heuristic.py          주최 측 heuristic baseline (CatBoost 의 입력으로 쓰임)
│   ├── model.py              CatBoost 3개의 연결
│   ├── train.py · predict.py · eval.py · main.py
│   ├── requirements.txt
│   └── weights/              저자 제공 모델
├── data_example/             예시 입력: 시험 객체 1962
│   ├── test/01962.csv        궤적 (2,173행 = 약 6개월, 2시간 간격)
│   └── test_label.csv        정답 node
├── output_example/           예시 출력: 위 객체에 대한 두 팀의 예측과 채점 결과
└── licenses/                 출처·라이선스
```

## 문제와 데이터 형식

- 입력: GEO 위성 1기의 궤적. 열 = `Timestamp` + Keplerian 6요소 (`Eccentricity`, `Semimajor Axis (m)`, `Inclination (deg)`, `RAAN (deg)`, `Argument of Periapsis (deg)`, `True Anomaly (deg)`) + `Latitude (deg)`, `Longitude (deg)`, `Altitude (m)` + J2000 위치·속도 (`X/Y/Z (m)`, `Vx/Vy/Vz (m/s)`). 행 번호가 `TimeIndex`
- 출력·정답: node 목록 `[ObjectID, TimeIndex, Direction, Node, Type]`
  - `Direction`: EW (경도 방향), NS (위도 방향). 정답 파일의 마지막 행 ES 는 관측 끝 표시
  - `Node`: SS (관측 시작), ID (drift 시작), AD (drift 조정), IK (station-keeping 시작)
  - `Type`: node 뒤의 mode — NK (station-keeping 안 함), CK / EK / HK (chemical / electric / hybrid 추진 station-keeping)
- 지표: F2 = 5·TP / (5·TP + 4·FN + FP). 예측이 정답의 ±6 step (12시간) 안이고 Node·Type 이 모두 같으면 TP

예시 객체 1962 의 정답 (`data_example/test_label.csv`)

| TimeIndex | Direction | Node | Type | 뜻 |
|---|---|---|---|---|
| 0 | EW, NS | SS | NK | 관측 시작, 두 방향 모두 drift 중 |
| 969 | EW | AD | NK | drift 조정 |
| 976 | EW | IK | EK | EW station-keeping 시작 (electric) |
| 1005 | NS | IK | CK | NS station-keeping 시작 (chemical) |
| 1945 | EW, NS | ID | NK | station 을 떠나 drift 시작 |

`output_example/` 의 예측 (`hawaii2024_pred.csv`, `qr_is_pred.csv`) 은 두 팀 모두 이 7개 node 를 모두 맞힌다 (`*_eval.txt`).

## 바로 실행해 보기

Python 3.10 기준. 두 팀의 의존 패키지가 달라 환경을 따로 만드는 것을 권장한다.

```bash
# 1위 Hawaii2024
pip install -r hawaii2024/requirements.txt
python -m hawaii2024.main predict --data data_example/test --weights hawaii2024/weights/fixed --out pred_hawaii.csv
python -m hawaii2024.main eval    --pred pred_hawaii.csv --labels data_example/test_label.csv

# 3위 QR_Is
pip install -r qr_is/requirements.txt
python -m qr_is.main predict --data data_example/test --weights qr_is/weights --out pred_qr_is.csv
python -m qr_is.main eval    --pred pred_qr_is.csv --labels data_example/test_label.csv
```

모든 명령은 이 폴더 (저장소 최상위) 에서 실행한다. 학습 명령은 각 팀의 README 참고.

## 전체 데이터셋

학습과 시험 세트 전체 평가에는 SPLID 데이터셋의 `phase_2` 가 필요하다 (원 Hawaii2024 저장소의 README 에 내려받는 링크가 있음).

```
phase_2/
├── training/          1,900기 (시뮬레이션 1,870 + VCM 30)
├── train_label.csv
├── test/              501기 (시뮬레이션 130 + VCM 76 + TLE 296 중 1기 제외)
└── test_label.csv
```

## 확인한 것

| 항목 | 결과 |
|---|---|
| Hawaii2024 추론, `weights/fixed`, 시험 501기 | F2 0.819 (TP 945 / FP 488 / FN 139). 원 코드의 예측 파일과 1,433행 모두 같음 |
| Hawaii2024 추론, `weights/submitted --legacy`, 시험 501기 | F2 0.805 = 논문 수치 (TP 914 / FP 528 / FN 145). 원 코드의 예측 파일과 1,442행 모두 같음 |
| Hawaii2024 학습용 배열, 모델 구조 | 원 코드의 출력과 완전 일치, 제공 모델 파일과 구조 일치 |
| QR_Is 추론, 저자 제공 모델, 시험 501기 | F2 0.802 (TP 984 / FP 361 / FN 214). 원 코드를 Linux 에서 그대로 실행하면 0.798 — 차이는 아래 버그 수정 때문 |
| QR_Is, 원 코드를 같은 (정렬된) 파일 순서로 실행한 결과와 비교 | 예측 파일 1,345행 모두 같음 |
| 학습 → 추론 → 채점 실행 (소규모) | 동작 확인. Hawaii2024: 학습 159기 → 시험 162기 F2 0.819, QR_Is: 학습 60기 → 시험 162기 F2 0.726 |
| 전체 1,900기 재학습 | 이 저장소의 코드로는 아직 확인하지 않음 (원 코드로는 확인: Hawaii2024 F2 0.814, QR_Is F2 0.797) |

## 원 코드에서 찾은 문제

읽으면서 확인한 것 중 결과에 영향을 주는 것. 자세한 내용은 각 팀 README.

| 팀 | 문제 | 이 저장소의 처리 |
|---|---|---|
| QR_Is | csv 를 정렬하지 않고 읽어, envelope·이동 통계 35열이 다른 위성의 행에 붙을 수 있음 (pandas 2 의 `groupby.apply` 결과 순서와 표의 행 순서가 다를 때) | ObjectID 순으로 정렬해 읽음 |
| QR_Is | 추론 때 feature 열을 순서로 넘겨, 객체 수가 적으면 학습 때와 열 순서가 달라짐 | 열을 이름으로 선택 |
| QR_Is | lag·차분의 빈 값 채우기가 객체 경계를 넘어감, 뒤 단계 모델이 학습 데이터 자체의 예측 확률로 학습 | 원 코드 그대로 둠 (주석으로 표시) |
| Hawaii2024 | 대회 제출 모델은 각도 차분의 360도 넘김이 보정되지 않은 채 학습됨 (저자가 이후 수정) | `--legacy` 로 재현, 기본은 수정된 방식 |
| Hawaii2024 | ID localizer 의 학습 설정이 저장소에 없음, classifier 의 학습 설정이 추론 설정과 다름 | 제공 모델 파일에서 복원, 추론 설정에 맞춤 |
| 공통 | 학습 자료가 시뮬레이션 위주. VCM·TLE 기반 시험 궤적 340기에서는 두 팀 모두 ID·IK·AD node 118개 중 0 ~ 2개만 탐지 | — |

## 출처와 라이선스

[`licenses/NOTICE.md`](licenses/NOTICE.md) 참고. QR_Is 의 원 저장소에는 라이선스가 명시되어 있지 않으므로 공개 저장소로 올릴 때는 확인이 필요하다.

참고 논문
- ① Siew et al., "AI SSA Challenge Problem: Satellite Pattern-of-Life Characterization Dataset and Benchmark Suite", AMOS 2023
- ② Siew et al., "Satellite Pattern-of-Life Identification Challenge: Competition Design and Results", AMOS 2024
