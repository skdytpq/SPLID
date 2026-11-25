# QR_Is (3위) — feature engineering + CatBoost 3단 연결

원 저장소: https://github.com/FrancoisPorcher/mit-challenge (Francois Porcher, Isaac Haik — LICENSE 파일 없음)
방법 설명: 논문 ② 3.3 절, 저자 보고서 "A simple model and advanced strategies are all you need"

## 한 줄 요약

시점마다 약 320개의 feature 를 만들어 "이 시점이 어떤 node 인가 (또는 node 가 아닌가)" 를 CatBoost 로 분류한다.

```
궤적 csv
   │  dataloader.py   객체별 csv → 표 1개 (행 = 한 시점), 주기 index, heuristic baseline 예측
   │  features.py     lag · 차분 · 변화율 · envelope · 이동 통계 · 위성 단위 통계 → 약 320열
   ▼
① NS 1차 모델 ──NS 확률──▶ ② EW 모델 ──EW 확률──▶ ③ NS 최종 모델        model.py
   ▼
시점별 class 확률 (EW 10 class, NS 9 class: Nothing, SS-NK, SS-CK, ..., IK-EK, AD-NK)
   │  predict.dual_threshold   precision 용·recall 용 임계값 2개로 class 결정
   │  predict.to_nodes         'Nothing' 을 직전 예측으로 채우고, 값이 바뀌는 시점만 node 로
   ▼
예측 csv  [ObjectID, TimeIndex, Direction, Node, Type]
```

## 파일

| 파일 | 내용 | 원 코드 |
|---|---|---|
| `config.py` | feature 설정 (창·간격·대상 열), tree 수, 임계값 | 여러 파일에 흩어진 값 |
| `dataloader.py` | csv 읽기, 주기 index, heuristic 호출, 라벨 붙이기 | `utils.py` `tabularize_data()` 앞부분 |
| `features.py` | feature engineering | `utils.py` `tabularize_data()` 뒷부분 |
| `heuristic.py` | 주최 측 heuristic baseline (내용 변경 없음) | `utils.py` 의 heuristic 함수들, `node.py` |
| `model.py` | CatBoost 3개의 학습·추론 순서, 저장·불러오기 | `main_train.py`, `main_submission.py` |
| `train.py` | 학습 | `main_train.py` |
| `predict.py` | 추론, 이중 임계값, 시점별 class → node 목록 | `main_submission.py`, `utils.py` |
| `eval.py` | 최종 채점 | `evaluation.py` |
| `main.py` | 실행 진입점 (`train` / `predict` / `eval`) | — |
| `weights/` | 저자 제공 모델 (CatBoost 3개 + LabelEncoder 2개) | `trained_model_for_submission/` |

## 읽는 순서

1. `dataloader.py` — 표와 라벨의 형태 ('Nothing' 이 대부분인 시점별 multi-class 라벨)
2. `features.py` — 어떤 feature 를 만드는지. 파일 머리의 목록과 `config.py` 를 같이 본다
3. `model.py` — 세 모델이 확률을 어떻게 주고받는지
4. `predict.py` — `dual_threshold` → `to_nodes`
5. `heuristic.py` 는 주최 측 코드라 "객체 1기 → node 예측 4열" 을 내는 상자로 보고 넘어가도 된다

## 핵심 아이디어와 코드 위치

| 아이디어 | 설명 | 위치 |
|---|---|---|
| 전환 시점에만 라벨 | node 시점만 'IK-EK' 등, 나머지는 'Nothing'. devkit baseline 처럼 모든 시점에 mode 라벨을 채우지 않음 | `dataloader.load_table` |
| lag·lead | 앞뒤 6 step (±12시간) 의 원시 값을 그대로 넣어 tree 가 시간 문맥을 보게 함 | `features.add_features`, `config.LAG_STEPS` |
| envelope | 위도·경도·고도의 3일 창 (최대 − 최소). station-keeping 범위가 바뀌는 곳에서 변함 | `features.add_features` |
| 위성 단위 통계 | 궤적 전체의 평균·표준편차를 그 위성의 모든 시점에 넣음. 변동이 큰 위성일수록 node 가 많다는 관찰 | 위와 같음 |
| 주기 index | 12시간 ~ 2주 주기로 1..n 을 반복하는 값 (transformer 의 위치 encoding 과 비슷한 역할) | `dataloader.load_table`, `config.TIME_CYCLES` |
| model stacking | 주최 측 heuristic 의 예측을 one-hot feature 로 추가 | `heuristic.add_baseline_heuristic`, `features.add_features` |
| recombination | NS 1차 → EW → NS 최종. 앞 모델의 class 확률을 다음 모델의 입력에 더함 | `model.StackedCatBoost` |
| 이중 임계값 | 최고 확률이 0.1 미만이면 'Nothing'. 1위가 'Nothing' 이어도 2위 확률이 EW 0.11 / NS 0.26 을 넘으면 2위 선택 | `predict.dual_threshold`, `config.THRESHOLDS` |

## 실행

저장소 최상위 폴더에서 실행 (Python 3.10, `pip install -r qr_is/requirements.txt`).

```bash
python -m qr_is.main predict --data data_example/test --weights qr_is/weights --out pred.csv
python -m qr_is.main eval    --pred pred.csv --labels data_example/test_label.csv

python -m qr_is.main train   --data <phase_2/training> --labels <phase_2/train_label.csv> --out runs/qr_is
python -m qr_is.main predict --data <phase_2/test> --weights runs/qr_is --out pred.csv
```

전체 1,900기 학습은 CPU 로 약 5시간, 메모리는 수십 GB 가 필요하다 (표가 410만 행 × 330열). `--max-objects N` 으로 줄여 동작을 확인할 수 있다.

## 원 코드와 달라진 점

**고친 것 1 — 파일 정렬 (결과가 달라짐)**

원 코드는 csv 를 `Path.glob()` 이 주는 순서대로 읽어 표를 만든다. 그런데 envelope·이동 평균·이동 표준편차는
`groupby('ObjectID').apply(...)` 로 계산하고, pandas 2 에서 이 결과는 ObjectID 오름차순으로 나온다. 원 코드는 그 결과를
`added_data.index = merged_data.index` 로 행 순서대로 표에 붙이므로, 표가 ObjectID 오름차순이 아니면
**한 위성의 envelope·이동 통계 35열 (envelope 3 + envelope 차분 12 + 이동 통계 20) 이 다른 위성의 행에 붙는다.**

- Linux 의 ext4 처럼 glob 순서가 정렬되어 있지 않은 환경에서 발생. 파일 이름이 `1.csv`, `10.csv` 처럼 자릿수가 다르면 이름순으로 읽는 Windows 에서도 발생
- 이 저장소는 `dataloader._list_files` 에서 ObjectID 순으로 정렬해 읽고, `features.add_features` 에서 정렬 여부를 확인한다
- 저자 제공 모델로 시험 501기를 채점하면 정렬 후 F2 0.802 (TP 984 / FP 361 / FN 214), 원 코드를 Linux 서버에서 그대로 실행하면 F2 0.798 (TP 980 / FP 375 / FN 216)
- 저자 제공 모델이 어떤 순서로 학습됐는지는 알 수 없다

**고친 것 2 — 열을 이름으로 선택**

원 코드는 추론 때 일부 단계에서 feature 열을 "만들어진 순서" 로 넘긴다. heuristic one-hot 열은 데이터에 없는 class 가 있으면
맨 뒤에 추가되므로, 객체 수가 적으면 (예: 예시 1기) 학습 때와 열 순서가 달라진다. 여기서는 항상 모델이 기억하는 열 이름
(`model.feature_names_`) 으로 고른다. 시험 501기 전체에서는 결과가 같다.

**결과에 영향이 없는 것**
- 이중 임계값을 배열 연산으로 계산 (원 코드는 110만 행을 Python 반복), 'Nothing' class 번호는 LabelEncoder 에서 찾음 (원 코드는 "가장 많이 예측된 class" 로 가정)
- 객체별 표를 한 번에 이어 붙임 (원 코드는 파일마다 누적 concat)
- 위성 단위 통계를 `groupby.transform` 으로 계산
- 예측 파일 경로를 인자로 받음 (원 코드는 데이터 폴더의 상위 폴더에 저장)

## 확인한 것

| 항목 | 결과 |
|---|---|
| 추론, 저자 제공 모델, 시험 501기 | F2 0.802, TP 984 / FP 361 / FN 214 |
| 원 코드를 같은 (정렬된) 파일 순서로 실행한 결과와 비교, 시험 501기 | 예측 파일 1,345행 모두 같음 |
| 학습 → 추론 → 채점 실행, 학습 60기 (ObjectID 12 간격) | 동작 확인. 시험 162기 (ID 1901 ~ 2063) F2 0.726 |
| 전체 1,900기 재학습 | 이 저장소의 코드로는 아직 확인하지 않음. 학습의 feature 생성은 추론과 같은 코드 (`features.add_features`) |

## 알아둘 점 (원 코드의 동작을 그대로 둔 부분)

- lag·차분·변화율의 빈 값을 표 전체에 대해 `ffill().bfill()` 로 채운다. 각 객체의 처음 몇 행에는 표에서 바로 앞 객체의 마지막 값이 들어간다 (`features.py` 의 주석 참고). 시작 node (SS) 의 type 판정이 앞 객체에 영향을 받을 수 있다
- 2·3번째 모델은 앞 모델이 자기 학습 데이터에 대해 낸 확률로 학습한다 (out-of-fold 가 아님). 학습 loss 가 0 에 가깝게 내려간다
- 검증 세트가 없다. 임계값 (0.1, 0.11, 0.26) 은 저자가 정한 값
- `predict.to_nodes` 의 채우기 (`ffill`) 도 표 전체에 대해 수행된다
- `heuristic.py` 는 pandas 2.2 의 동작 (`df[col].fillna(..., inplace=True)`) 에 기대고 있어 pandas 3 에서는 수정이 필요하다
- 학습 자료가 시뮬레이션 위주라 VCM·TLE 기반 궤적에서는 ID·IK·AD node 를 거의 찾지 못한다 (시험 340기에서 118개 중 0 ~ 1개)
