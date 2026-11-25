# Hawaii2024 (1위) — CNN localizer + LSTM classifier

원 저장소: https://github.com/DavidBaldsiefen/splid-challenge (David Baldsiefen, MIT License)
방법 설명: 논문 ② "Satellite Pattern-of-Life Identification Challenge: Competition Design and Results" 3.1 절

## 한 줄 요약

"node 가 언제인가" (시점 탐지) 와 "그 뒤 mode 가 무엇인가" (type 분류) 를 다른 모델로 나눠 푼다.

```
궤적 csv (2시간 간격, 약 2,170 step)
   │  dataloader.py   feature 변환 → 시점마다 [과거, 미래] 창
   ▼
① AD·IK localizer (CNN)   창 160 step → 80×9      ─┐ 시점마다 0 ~ 100 출력
② ID localizer   (CNN)   창 576 step → 144×11     ─┤ → 임계값 → 연속 구간의 가운데 = node 시점
   ▼                                               │   postprocess.detections_from_outputs
③ classifier (LSTM)      창 288 step → 288×15      ◀┘ node 시점 + 시작 시점 (SS) 에서 type 분류
   ▼
④ 규칙   type 의 앞뒤 순서 → node 종류 (SS / ID / AD / IK)          postprocess.assign_nodes
         EW 가 NK 인 동안의 NS node 제거                             postprocess.drop_ns_outside_ew_sk
   ▼
예측 csv  [ObjectID, TimeIndex, Direction, Node, Type]
```

## 파일

| 파일 | 내용 | 원 코드 |
|---|---|---|
| `config.py` | 모델 3종의 입력·구조·학습·임계값 설정 | `submission.py`, `*_sweeper.py` 의 값 |
| `dataloader.py` | csv 읽기, feature 변환, 표준화, localizer 라벨, 창 만들기 | `base/datahandler.py` |
| `model.py` | localizer (CNN), classifier (LSTM) 구조 | `base/prediction_models.py` |
| `train.py` | 모델 1종 학습 + 검증 세트 점검 | `localizer_sweeper.py`, `classifier_sweeper.py` |
| `predict.py` | 추론 전체 흐름 (①②③④) | `submission.py`, `base/localizer.py`, `base/classifier.py` |
| `postprocess.py` | 임계값·구간 중앙, 탐지 합치기, type → node 규칙, NS 제거 | `base/localizer.py`, `base/classifier.py` |
| `eval.py` | 최종 채점, localizer·classifier 단독 평가 | `submission.py`, `*_sweeper.py` 끝부분 |
| `main.py` | 실행 진입점 (`train` / `predict` / `eval`) | — |
| `weights/submitted/` | 대회 제출 모델 (F2 0.805) | `models/*_model.hdf5` |
| `weights/fixed/` | 저자가 차분 버그를 고쳐 다시 학습한 모델 (F2 0.819) | `models/*_fixedDiffTransform.hdf5` |

## 읽는 순서

1. `config.py` — 세 모델이 무엇이 다른지 (창 길이, 입력 feature, 라벨) 를 먼저 본다
2. `dataloader.py` — `SplidDataset.__init__` (feature 변환) → `_windows` (창) → `_add_localizer_labels` (라벨)
3. `model.py` — `build_localizer`, `build_classifier`
4. `predict.py` 의 `run()` — 전체 흐름. 각 단계에서 `postprocess.py` 의 함수를 따라간다
5. `train.py` — 표본을 어떻게 뽑아 학습하는지

## 핵심 아이디어와 코드 위치

| 아이디어 | 설명 | 위치 |
|---|---|---|
| gradient encoding 라벨 | node 시점 100, 앞뒤 5 step 은 70·49·34·24·16, 나머지 0 을 회귀 (MSE). 0/1 분류보다 불균형이 덜하고 가까운 예측에 벌점이 작다 | `dataloader._add_localizer_labels`, `config.label_shape` |
| node 종류별 모델 | AD·IK 는 maneuver 의 국소 흔적 → 짧은 창 (과거 10.7일 + 미래 2.7일). ID 는 station-keeping 이 끊기는 장기 변화 → 긴 창 (26.7일 + 21.3일). 라벨 배율로 대상을 나눔 | `config.ADIK`, `config.ID` 의 `label_weight` |
| 차분 feature | inclination, true anomaly, 경도의 직전 시점과의 차이. 급격한 국소 변화를 강조 | `dataloader._diff` |
| overview feature | 궤적 전체를 창 길이만큼의 구간으로 나눈 구간별 표준편차. 주기적 station-keeping 이 봉우리로 보임 | `dataloader.SplidDataset._windows` |
| 시간 위치 one-hot | "지금 창이 궤적의 어디인가" 를 overview 와 같은 축에 1 하나로 표시 | 위와 같음 |
| 구간 평균으로 줄이기 | 창을 stride 개씩 평균해 길이를 줄임 (160 → 80, 576 → 144) | `dataloader._average_pool` |
| 두 갈래 branch | 같은 입력을 구조가 같은 두 conv (LSTM) branch 에 넣고 dense 는 공유, 출력은 EW·NS 두 개 | `model.py` |
| 표본 추출 | node 주변 시점은 전부 + 나머지는 5번째마다. 학습 단계마다 시작 위치를 바꿈 | `SplidDataset.arrays`, `config.training_sets` |
| classifier 는 node 에서만 학습 | 추론 때도 탐지된 시점만 분류하므로 학습 표본도 거의 node 시점만 | `train._train_classifier` |
| 임계값 + 구간 중앙 | 출력이 임계값을 넘는 연속 구간의 가운데 한 점을 node 로 | `postprocess.detections_from_outputs` |
| type → node 규칙 | NK→NK 는 AD, SK→NK 는 ID, NK→SK 는 IK | `postprocess.assign_nodes` |

## 실행

저장소 최상위 폴더에서 실행 (Python 3.10, `pip install -r hawaii2024/requirements.txt`).

```bash
# 추론 + 채점 (예시 객체 1기, CPU 로 수 초)
python -m hawaii2024.main predict --data data_example/test --weights hawaii2024/weights/fixed --out pred.csv
python -m hawaii2024.main eval    --pred pred.csv --labels data_example/test_label.csv

# 대회 제출 모델은 --legacy 를 붙인다 (그 모델이 학습된 입력 방식을 재현)
python -m hawaii2024.main predict --data data_example/test --weights hawaii2024/weights/submitted --legacy --out pred.csv

# 학습: 모델 3종을 각각 실행. 같은 --out 폴더를 predict 의 --weights 로 쓴다
python -m hawaii2024.main train --model adik       --data <phase_2/training> --labels <phase_2/train_label.csv> --out runs/hawaii
python -m hawaii2024.main train --model id         --data <phase_2/training> --labels <phase_2/train_label.csv> --out runs/hawaii
python -m hawaii2024.main train --model classifier --data <phase_2/training> --labels <phase_2/train_label.csv> --out runs/hawaii
python -m hawaii2024.main predict --data <phase_2/test> --weights runs/hawaii --out pred.csv
```

`--max-objects N` 을 붙이면 앞의 N 기만 써서 빠르게 동작을 확인할 수 있다.

## 원 코드와 달라진 점

결과에 영향이 없는 것 (아래 "확인한 것" 참고)
- wandb sweep 을 걷어내고 설정을 `config.py` 한 곳에 모음
- 최종 모델이 쓰지 않는 옵션 삭제: lowpass filter, daytime·yeartime feature, oneshot localizer, 2-입력 모델, SGD·RMSprop, binary 라벨, majority 방식 분류
- 구간 평균을 배열 연산으로 계산 (원 코드는 1차원 배열마다 Python 함수 호출)
- classifier 를 탐지된 시점에서만 실행 (원 코드는 모든 시점을 분류한 뒤 골라 씀). 501기 추론이 CPU 에서 약 2분
- 모델 파일을 h5py 로 열어 Keras 에 넘김 (경로에 한글이 있으면 Windows 의 TensorFlow 가 파일을 못 여는 문제 회피)

재학습 결과에 영향이 있는 것
- ID localizer 의 학습 설정은 원 저장소에 없어 복원함. 입력은 `submission.py`, 층·dropout·L2·learning rate 는 제공된 모델 파일에 저장된 값, 학습 일정은 AD·IK 와 같게 둠
- classifier 의 `time_index_onehot` 은 fixed 모델의 추론 설정 (True) 에 맞춤. 원 `classifier_sweeper.py` 는 False 라서 그대로 학습하면 추론 입력과 어긋남
- 초기 가중치의 난수 순서가 원 코드와 다름 (seed 는 같지만 원 코드는 쓰지 않는 층을 만들며 난수를 더 소모). 분포는 같음
- epoch 마다 메모리를 비우는 callback, 학습 후 학습·시험 세트 평가는 뺌

## 확인한 것

| 항목 | 결과 |
|---|---|
| 추론, `weights/fixed`, 시험 501기 | F2 0.819, TP 945 / FP 488 / FN 139. 원 코드의 예측 파일과 1,433행 모두 같음 |
| 추론, `weights/submitted --legacy`, 시험 501기 | F2 0.805 (논문 수치), TP 914 / FP 528 / FN 145. 원 코드의 예측 파일과 1,442행 모두 같음 |
| 학습용 배열 (입력·라벨·시점), 모델 3종, 학습 객체 60기 | 원 `DatasetGenerator` 의 출력과 완전 일치 |
| 모델 구조, 모델 3종 | 제공된 fixed 모델 파일과 층 구성·파라미터 수 일치, 같은 가중치를 넣으면 출력 차이 0 |
| 학습 → 추론 → 채점 실행, 학습 159기 (ObjectID 12 간격) | 동작 확인. 검증 F2: AD·IK 0.958, ID 0.882, type 정확도 0.855. 시험 162기 (ID 1901 ~ 2063) F2 0.819 |
| 전체 1,900기 재학습 | 이 저장소의 코드로는 아직 확인하지 않음 |

## 알아둘 점 (원 코드의 동작을 그대로 둔 부분)

- `exclude_objects`: 저자가 번호로 지정해 뺀 학습 객체. 다른 데이터에는 의미가 없다
- 임계값 (EW 45 / NS 50, ID 는 40 / 50) 은 저자가 손으로 정한 값
- 검증 세트는 학습 세트에서 무작위로 떼어낸 것이라 시뮬레이션 궤적이 대부분이다. 검증 점수가 높아도 VCM·TLE 기반 궤적의 성능은 낮다 (시험 340기에서 ID·IK·AD node 탐지 118개 중 1 ~ 2개)
- `postprocess.detections_from_outputs` 는 여러 객체를 한 표에서 처리해, 앞 객체의 마지막 탐지와 다음 객체의 첫 탐지가 TimeIndex 로 이어지면 한 묶음으로 본다 (실제로는 거의 일어나지 않음)
