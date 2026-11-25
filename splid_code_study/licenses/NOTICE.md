# 출처와 라이선스

이 저장소는 SPLID 챌린지 (2024 MIT ARCLab Prize for AI Innovation in Space) 상위 팀의 공개 코드를 공부하기 위해 구조를 다시 짠 것이다.
알고리즘과 학습된 모델의 저작권은 각 원저자에게 있다.

| 이 저장소의 위치 | 원 출처 | 라이선스 |
|---|---|---|
| `hawaii2024/` (코드, `weights/`) | https://github.com/DavidBaldsiefen/splid-challenge | MIT, (c) 2024 Gerd David Baldsiefen → `LICENSE_splid-challenge_Hawaii2024.txt` |
| `qr_is/` (코드, `weights/`) | https://github.com/FrancoisPorcher/mit-challenge | 원 저장소에 LICENSE 파일 없음 |
| `common/evaluator.py`, `qr_is/heuristic.py` | https://github.com/ARCLab-MIT/splid-devkit (채점기, heuristic baseline) | MIT, (c) 2023 Peng Mun Siew → `LICENSE_splid-devkit.txt` |
| `data_example/` | SPLID 데이터셋 (MIT ARCLab) phase_2 시험 세트의 객체 1962 | 데이터셋 배포 조건을 따름 |

주의: `qr_is/` 의 원 저장소에는 라이선스가 명시되어 있지 않다. 라이선스가 없는 코드는 기본적으로 저작권자의 허락 없이
재배포할 수 없으므로, 이 폴더를 공개 저장소에 올리려면 원저자의 허락을 받거나 비공개 저장소로 두어야 한다.
