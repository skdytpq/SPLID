"""Hawaii2024 (1위) 실행 진입점. 저장소 최상위 폴더에서 실행한다.

  python -m hawaii2024.main predict --data data_example/test --weights hawaii2024/weights/fixed --out pred.csv
  python -m hawaii2024.main eval    --pred pred.csv --labels data_example/test_label.csv
  python -m hawaii2024.main train   --model adik --data <phase_2/training> --labels <phase_2/train_label.csv> --out runs/hawaii

train 은 모델 3종 (adik, id, classifier) 을 각각 실행해야 하고, 같은 --out 폴더를 predict 의 --weights 로 쓰면 된다.
"""
import argparse
import os

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)

    p = sub.add_parser('train', help='모델 1종 학습')
    p.add_argument('--model', required=True, choices=['adik', 'id', 'classifier'])
    p.add_argument('--data', required=True, help='학습 궤적 csv 폴더')
    p.add_argument('--labels', required=True, help='학습 라벨 csv')
    p.add_argument('--out', required=True, help='가중치를 저장할 폴더')
    p.add_argument('--max-objects', type=int, help='앞에서부터 이 수만큼의 객체만 사용 (동작 확인용)')

    p = sub.add_parser('predict', help='궤적 폴더 → 예측 csv')
    p.add_argument('--data', required=True, help='궤적 csv 폴더')
    p.add_argument('--weights', required=True, help='가중치 폴더 (weights/fixed, weights/submitted, 또는 train 의 --out)')
    p.add_argument('--legacy', action='store_true', help='weights/submitted (대회 제출 모델) 을 쓸 때 지정')
    p.add_argument('--out', required=True, help='예측 csv 경로')
    p.add_argument('--max-objects', type=int)

    p = sub.add_parser('eval', help='예측 csv 채점')
    p.add_argument('--pred', required=True)
    p.add_argument('--labels', required=True)

    args = parser.parse_args()
    if args.command == 'train':
        from . import train
        train.run(args.model, args.data, args.labels, args.out, args.max_objects)
    elif args.command == 'predict':
        from . import predict
        result = predict.run(args.data, args.weights, args.legacy, args.max_objects)
        result.to_csv(args.out, index=False)
        print(f'저장: {args.out}')
    else:
        from . import eval as evaluation
        evaluation.evaluate(args.pred, args.labels)


if __name__ == '__main__':
    main()
