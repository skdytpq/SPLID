"""QR_Is (3위) 실행 진입점. 저장소 최상위 폴더에서 실행한다.

  python -m qr_is.main predict --data data_example/test --weights qr_is/weights --out pred.csv
  python -m qr_is.main eval    --pred pred.csv --labels data_example/test_label.csv
  python -m qr_is.main train   --data <phase_2/training> --labels <phase_2/train_label.csv> --out runs/qr_is
"""
import argparse
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)

    p = sub.add_parser('train', help='CatBoost 3개 학습')
    p.add_argument('--data', required=True, help='학습 궤적 csv 폴더')
    p.add_argument('--labels', required=True, help='학습 라벨 csv')
    p.add_argument('--out', required=True, help='모델을 저장할 폴더')
    p.add_argument('--max-objects', type=int, help='ObjectID 가 작은 순으로 이 수만큼만 사용 (동작 확인용)')

    p = sub.add_parser('predict', help='궤적 폴더 → 예측 csv')
    p.add_argument('--data', required=True, help='궤적 csv 폴더')
    p.add_argument('--weights', required=True, help='모델 폴더 (qr_is/weights 또는 train 의 --out)')
    p.add_argument('--out', required=True, help='예측 csv 경로')
    p.add_argument('--max-objects', type=int)

    p = sub.add_parser('eval', help='예측 csv 채점')
    p.add_argument('--pred', required=True)
    p.add_argument('--labels', required=True)

    args = parser.parse_args()
    if args.command == 'train':
        from . import train
        train.run(args.data, args.labels, args.out, args.max_objects)
    elif args.command == 'predict':
        from . import predict
        result = predict.run(args.data, args.weights, args.max_objects)
        result.to_csv(args.out, index=False)
        print(f'저장: {args.out}')
    else:
        from . import eval as evaluation
        evaluation.evaluate(args.pred, args.labels)


if __name__ == '__main__':
    main()
