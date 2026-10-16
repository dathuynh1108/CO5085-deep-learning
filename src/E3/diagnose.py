"""Validation-only gradient and sequence-order diagnostics after training."""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
from torch.nn import functional as F
from torch.utils.data import Subset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common.data import prepare_data, make_loader
from common.evaluation import load_frozen_run
from common.runtime import atomic_json, atomic_csv, choose_device
from common.tokenization import image_sequence
from E3.models import unroll_recurrent


def diagnose(run: Path, data_root: Path, device_name: str, max_samples: int, batch_size: int):
    if min(max_samples, batch_size) < 1:
        raise ValueError('Sample and batch counts must be positive.')
    device = choose_device(device_name)
    model, config, state, _, model_hash = load_frozen_run(run, device)
    if config.exercise != 'E3':
        raise ValueError('Only E3 checkpoints support recurrent-state diagnostics.')
    _, validation, metadata = prepare_data(data_root, config.split_seed)
    if metadata['fingerprint'] != state['dataset']['fingerprint']:
        raise ValueError('Validation split changed.')
    n = min(max_samples, len(validation))
    subset = Subset(validation, range(n))
    loader = make_loader(subset, batch_size, 0, device)
    steps = model.sequence_length
    totals = torch.zeros(steps, 4, dtype=torch.float64)
    correct = {'original': 0, 'reversed': 0, 'permuted': 0}
    permutation = torch.randperm(steps, generator=torch.Generator().manual_seed(2026)).to(device)
    for images, labels, _ in loader:
        images, labels = images.to(device), labels.to(device)
        sequence = image_sequence(images, model.tokenizer, model.patch_size).detach().requires_grad_(True)
        model.zero_grad(set_to_none=True)
        outputs, hs, cs, gates = unroll_recurrent(model, sequence)
        logits = model.classify(outputs)
        # Sum reduction makes each example's state-gradient independent of the
        # batch size. This is a diagnostic objective, not a training update.
        F.cross_entropy(logits, labels, reduction='sum').backward()
        for t in range(steps):
            totals[t, 0] += hs[t].grad.detach().norm(dim=1).sum().cpu().double()
            if cs:
                totals[t, 1] += cs[t].grad.detach().norm(dim=1).sum().cpu().double()
            totals[t, 2] += sequence.grad[:, t].detach().norm(dim=1).sum().cpu().double()
            totals[t, 3] += gates[t].detach().mean(dim=1).sum().cpu().double()
        correct['original'] += (logits.detach().argmax(1) == labels).sum().item()
        with torch.inference_mode():
            for name, changed in [('reversed', sequence.detach().flip(1)),
                                  ('permuted', sequence.detach()[:, permutation])]:
                output, _ = model.rnn(changed)
                correct[name] += (model.classify(output).argmax(1) == labels).sum().item()
    totals /= n
    rows = [dict(step=t+1, hidden_grad_norm=float(totals[t, 0]),
                 cell_grad_norm=float(totals[t, 1]) if model.kind == 'lstm' else None,
                 input_grad_norm=float(totals[t, 2]), memory_gate_mean=float(totals[t, 3]))
            for t in range(steps)]
    directory = Path(run) / 'diagnostics'; directory.mkdir(exist_ok=True)
    atomic_csv(directory / 'temporal_gradients.csv', rows)
    blocks = model.rnn.weight_hh_l0.detach().chunk(4 if model.kind == 'lstm' else 3)
    singular_values = [torch.linalg.svdvals(w).cpu().tolist() for w in blocks]
    result = dict(split='validation', n_samples=n, sample_selection='first original-index-sorted validation indices',
                  signature=state['signature'], model_sha256=model_hash, sequence_length=steps,
                  accuracy={k: v/n for k,v in correct.items()},
                  permutation=permutation.cpu().tolist(), recurrent_singular_values=singular_values,
                  warning='Order perturbations are distribution shifts, not a standalone proof of long-term memory.')
    atomic_json(directory / 'summary.json', result)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.semilogy(range(1,steps+1), totals[:,0].clamp_min(1e-15), label='hidden state')
    if model.kind == 'lstm':
        ax.semilogy(range(1,steps+1), totals[:,1].clamp_min(1e-15), label='cell state')
    ax.semilogy(range(1,steps+1), totals[:,2].clamp_min(1e-15), label='input token')
    ax.set(xlabel='Sequence step', ylabel='Mean per-example gradient norm')
    ax.legend(); ax.grid(alpha=.25); fig.tight_layout()
    fig.savefig(directory / 'temporal_gradients.png', dpi=150); plt.close(fig)
    print(result['accuracy'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=Path('data'))
    parser.add_argument('--device', default='auto')
    parser.add_argument('--max-samples', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    diagnose(args.run, args.data_root, args.device, args.max_samples, args.batch_size)


if __name__ == '__main__':
    main()
