import argparse
from pathlib import Path
from common.evaluation import evaluate_run


def main():
    parser = argparse.ArgumentParser(description='Evaluate a frozen, validation-selected checkpoint.')
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=Path('data'))
    parser.add_argument('--split', choices=['validation', 'test'], default='test')
    parser.add_argument('--confirm-test', action='store_true')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--force', action='store_true', help='Explicitly replace a previous evaluation.')
    args = parser.parse_args()
    evaluate_run(args.run, args.data_root, args.split, args.confirm_test, args.device, args.num_workers, args.force)


if __name__ == '__main__':
    main()
