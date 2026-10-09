"""Sequential experiments for a single GPU. No trainer.fit(), no hidden test tuning."""
import argparse
from pathlib import Path
from common.config import load_experiments
from common.engine import train
from common.evaluation import evaluate_run, require_test_permission


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', choices=['required', 'improvement', 'all'], default='required')
    parser.add_argument('--exercise', choices=['E1', 'E2', 'E3'])
    parser.add_argument('--seeds', type=int, nargs='+', default=[42])
    parser.add_argument('--phase', choices=['train', 'test'], default='train')
    parser.add_argument('--data-root', type=Path, default=Path('data'))
    parser.add_argument('--output-root', type=Path, default=Path('runs'))
    parser.add_argument('--device', default='auto')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--confirm-test', action='store_true')
    parser.add_argument('--list', action='store_true', help='Print configurations only; no data is accessed.')
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error('Do not repeat a seed.')
    experiments = [c for c in load_experiments()
                   if (args.group == 'all' or c.group == args.group)
                   and (not args.exercise or c.exercise == args.exercise)]
    if args.list:
        for config in experiments:
            print(f'{config.id:30s} | {config.label:34s} | epochs={config.epochs} | {config.group}')
        print(f'{len(experiments)} configurations x {len(args.seeds)} seeds')
        return
    if args.phase == 'test':
        require_test_permission('test', args.confirm_test)
    for config in experiments:
        for seed in args.seeds:
            run = args.output_root / config.id / f'seed_{seed}'
            if args.phase == 'train':
                train(config, seed, args.data_root, args.output_root, args.device, args.num_workers,
                      args.resume and (run / 'last.pt').exists(), args.deterministic)
            else:
                evaluate_run(run, args.data_root, 'test', args.confirm_test, args.device, args.num_workers)


if __name__ == '__main__':
    main()
