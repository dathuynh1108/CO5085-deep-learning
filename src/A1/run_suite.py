"""Sequential eight-configuration / three-seed suite; test requires consent."""
import argparse
from pathlib import Path
from A1.protocol import CONFIG, ROOT, load_protocol


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--list', action='store_true')
    result.add_argument('--phase', choices=['train', 'test'], default='train')
    result.add_argument('--confirm-test', action='store_true')
    result.add_argument('--protocol', type=Path, default=CONFIG, help='JSON protocol')
    result.add_argument('--experiments', nargs='+', help='Preset IDs; default all from the protocol')
    result.add_argument('--seeds', nargs='+', type=int, help='Default: seeds declared by the protocol')
    result.add_argument('--device', default='cuda')
    result.add_argument('--num-workers', type=int, default=8)
    result.add_argument('--data-root', type=Path, default=ROOT / 'data')
    result.add_argument('--output-root', type=Path)
    result.add_argument('--resume', action='store_true')
    result.add_argument('--deterministic', action='store_true')
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    protocol = load_protocol(args.protocol)
    seeds = args.seeds if args.seeds is not None else protocol['seeds']
    if args.list:
        for experiment in protocol['experiments']:
            print(f"{experiment['id']}: {experiment['regime']}, weights={experiment['weights']}, lr={experiment['lr']:g}")
        print(f"Default: {len(protocol['experiments']) * len(protocol['seeds'])} runs; "
              f"seeds {' '.join(map(str, protocol['seeds']))}; {protocol['epochs']} epochs; "
              f"BF16; image {protocol['image_size']}; train only")
        return
    if args.protocol.resolve() != CONFIG and args.output_root is None:
        parser().error('A separate protocol requires --output-root')
    output_root = args.output_root or ROOT / 'runs/assignment1'
    if args.phase == 'test' and not args.confirm_test:
        parser().error('Held-out evaluation requires --phase test --confirm-test after freezing the protocol')
    if args.num_workers < 0 or len(set(seeds)) != len(seeds) or any(s < 0 for s in seeds):
        parser().error('Workers must be nonnegative; seeds must be unique nonnegative integers')
    ids = {e['id'] for e in protocol['experiments']}
    if args.experiments and (set(args.experiments) - ids or len(set(args.experiments)) != len(args.experiments)):
        parser().error('Unknown or duplicate experiment IDs')
    from common.runtime import choose_device
    from A1.engine import train_run, test_run
    import torch
    device = choose_device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('Protocol requires BF16-capable CUDA; no automatic precision fallback')
    selected = [e for e in protocol['experiments'] if not args.experiments or e['id'] in args.experiments]
    for experiment in selected:
        for seed in seeds:
            run = output_root / experiment['id'] / f'seed_{seed}'
            options = {'run': run, 'protocol': protocol, 'experiment': experiment, 'seed': seed,
                       'data_root': args.data_root.resolve(), 'device': device, 'workers': args.num_workers,
                       'deterministic': args.deterministic, 'config': args.protocol}
            if args.phase == 'train':
                train_run(**options, resume=args.resume)
            else:
                test_run(**options)
            if device.type == 'cuda':
                torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
