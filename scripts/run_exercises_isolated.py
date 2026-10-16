"""Run fixed E1-E3 experiments serially, releasing worker resources between runs."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from common.config import load_experiments


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', choices=['required', 'improvement', 'all'], default='required')
    parser.add_argument('--exercise', choices=['E1', 'E2', 'E3'])
    parser.add_argument('--seeds', type=int, nargs='+', default=[42])
    parser.add_argument('--phase', choices=['train', 'test'], default='train')
    parser.add_argument('--data-root', type=Path, default=Path('data'))
    parser.add_argument('--output-root', type=Path, default=Path('runs/exercises'))
    parser.add_argument('--device', default='auto')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--confirm-test', action='store_true')
    parser.add_argument('--list', action='store_true')
    args = parser.parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds):
        parser.error('Do not repeat a seed.')
    experiments = [config for config in load_experiments()
                   if (args.group == 'all' or config.group == args.group)
                   and (not args.exercise or config.exercise == args.exercise)]
    if args.list:
        for config in experiments:
            print(f'{config.id:30s} | {config.label:34s} | epochs={config.epochs} | {config.group}')
        print(f'{len(experiments)} configurations x {len(args.seeds)} seeds')
        return
    if args.phase == 'test' and not args.confirm_test:
        parser.error('Held-out test requires --confirm-test after freezing the protocol.')
    for config in experiments:
        for seed in args.seeds:
            run = args.output_root / config.id / f'seed_{seed}'
            if args.phase == 'train':
                command = [sys.executable, '-m', f'{config.exercise}.train',
                           '--experiment', config.id, '--seed', str(seed),
                           '--output-root', str(args.output_root)]
                if args.resume and (run / 'last.pt').is_file():
                    command.append('--resume')
                if args.deterministic:
                    command.append('--deterministic')
            else:
                command = [sys.executable, '-m', 'scripts.evaluate', '--run', str(run),
                           '--split', 'test', '--confirm-test']
            command.extend(['--data-root', str(args.data_root), '--device', args.device,
                            '--num-workers', str(args.num_workers)])
            print(f'Isolated {args.phase}: {config.id} seed={seed}', flush=True)
            # Retain the existing entrypoints and their provenance/resume guards.
            # Process exit releases pinned/persistent DataLoader worker handles.
            subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
