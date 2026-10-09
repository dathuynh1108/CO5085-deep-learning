from __future__ import annotations
import argparse
from pathlib import Path
from common.config import get_experiment, load_experiments
from common.engine import train


def training_main(exercise: str | None = None):
    parser = argparse.ArgumentParser(description='Train from scratch; select checkpoints by validation only.')
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data-root', type=Path, default=Path('data'))
    parser.add_argument('--output-root', type=Path, default=Path('runs'))
    parser.add_argument('--device', default='auto')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--no-download', action='store_true')
    args = parser.parse_args()
    config = get_experiment(args.experiment, epochs=args.epochs, batch_size=args.batch_size)
    if exercise and config.exercise != exercise:
        parser.error(f'{args.experiment} is not an {exercise} experiment.')
    train(config, args.seed, args.data_root, args.output_root, args.device,
          args.num_workers, args.resume, args.deterministic, not args.no_download)
