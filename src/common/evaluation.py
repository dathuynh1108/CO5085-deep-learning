"""Held-out evaluation of a frozen validation-selected checkpoint."""
from __future__ import annotations
import json
import hashlib
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from common.config import Experiment
from common.data import prepare_data, test_data, make_loader
from common.factory import build_model
from common.metrics import classification_metrics
from common.plots import plot_confusion, plot_mistakes
from common.runtime import (atomic_csv, atomic_json, choose_device, synchronize,
                            environment_info, state_fingerprint, seed_everything)


def require_test_permission(split: str, confirmed: bool) -> None:
    if split not in {'test', 'validation'}:
        raise ValueError('Evaluation split must be validation or test.')
    if split == 'test' and not confirmed:
        raise ValueError('Freeze the experiment choices, then pass --confirm-test. Do not tune on test scores.')


def load_frozen_run(directory: Path, device: torch.device):
    directory = Path(directory)
    if (directory / '.train.lock').exists():
        raise RuntimeError('The run is being trained; evaluate only a completed, unlocked run.')
    summary = json.loads((directory / 'metrics.json').read_text(encoding='utf-8'))
    if summary.get('status') != 'trained':
        raise ValueError('Training has not completed.')
    state = torch.load(directory / 'best.pt', map_location='cpu', weights_only=True)
    if state['signature'] != summary['signature']:
        raise ValueError('Checkpoint and metrics refer to different runs.')
    model_hash = state_fingerprint(state['model'])
    if summary.get('best_model_sha256', model_hash) != model_hash:
        raise ValueError('The validation-selected checkpoint was changed after training.')
    run_info = json.loads((directory / 'run.json').read_text(encoding='utf-8'))
    seed_everything(int(state['seed']), bool(run_info['deterministic']))
    config = Experiment(**state['experiment'])
    model = build_model(config).to(device)
    model.load_state_dict(state['model']); model.eval()
    return model, config, state, summary, model_hash


def evaluate_run(directory: Path, data_root: Path, split: str = 'test', confirm_test: bool = False,
                 device_name: str = 'auto', workers: int = 4, force: bool = False,
                 download: bool = True) -> dict:
    require_test_permission(split, confirm_test)
    directory = Path(directory)
    device = choose_device(device_name)
    model, config, state, summary, model_hash = load_frozen_run(directory, device)
    output = directory / f'evaluation_{split}.json'
    if output.exists() and not force:
        prior = json.loads(output.read_text(encoding='utf-8'))
        if prior['signature'] != state['signature'] or prior['model_sha256'] != model_hash:
            raise ValueError('Existing evaluation belongs to a different checkpoint. Investigate before using --force.')
        print(f'{directory}: reuse existing {split} evaluation', flush=True)
        return prior
    _, validation, metadata = prepare_data(Path(data_root), config.split_seed, download)
    if metadata['fingerprint'] != state['dataset']['fingerprint']:
        raise ValueError('Evaluation uses a different training/validation split.')
    dataset = test_data(Path(data_root), metadata, download) if split == 'test' else validation
    loader = make_loader(dataset, config.batch_size, workers, device)
    targets, predictions, confidences, ids, mistakes, timings = [], [], [], [], [], []
    loss_sum = 0.
    with torch.inference_mode():
        for images, labels, indices in loader:
            images = images.to(device, non_blocking=True)
            labels_device = labels.to(device, non_blocking=True)
            synchronize(device); start = time.perf_counter()
            logits = model(images)
            synchronize(device); elapsed = time.perf_counter() - start
            timings.append((len(labels), elapsed))
            probability, predicted = logits.softmax(1).max(1)
            loss_sum += F.cross_entropy(logits, labels_device, reduction='sum').item()
            p, c, y, idx = predicted.cpu().numpy(), probability.cpu().numpy(), labels.numpy(), indices.numpy()
            targets.extend(y.tolist()); predictions.extend(p.tolist())
            confidences.extend(c.tolist()); ids.extend(idx.tolist())
            wrong = np.flatnonzero(p != y)
            for j in wrong[np.argsort(c[wrong])[-8:]]:
                mistakes.append(dict(index=int(idx[j]), target=int(y[j]), prediction=int(p[j]),
                                     confidence=float(c[j]), image=dataset.images[int(idx[j])].numpy()))
            mistakes = sorted(mistakes, key=lambda x: x['confidence'], reverse=True)[:8]
    metrics = classification_metrics(np.asarray(targets), np.asarray(predictions))
    steady = timings[2:] or timings
    image_count, inference_seconds = sum(x[0] for x in steady), sum(x[1] for x in steady)
    fingerprint = hashlib.sha256(dataset.images.numpy().tobytes() + dataset.labels.numpy().tobytes()
                                 + dataset.indices.tobytes()).hexdigest()
    result = dict(metrics, cross_entropy=loss_sum/len(targets), split=split, signature=state['signature'],
                  model_sha256=model_hash, dataset_sha256=fingerprint, seed=state['seed'],
                  best_epoch=state['epoch'], inference_images_per_second=image_count/inference_seconds,
                  benchmark_excluded_initial_batches=2 if len(timings) > 2 else 0,
                  benchmark_includes_transfer=False, batch_size=config.batch_size,
                  environment=environment_info(device))
    atomic_csv(directory / f'predictions_{split}.csv', [
        dict(index=i, target=y, prediction=p, confidence=c)
        for i, y, p, c in zip(ids, targets, predictions, confidences)])
    plots = directory / 'plots'; plots.mkdir(exist_ok=True)
    plot_confusion(metrics['confusion_matrix'], plots / f'confusion_{split}.png')
    plot_mistakes(mistakes, plots / f'mistakes_{split}.png')
    atomic_json(output, result)
    print(f"{directory.name} {split}: accuracy={result['accuracy']:.4%}, macro-F1={result['macro_f1']:.4f}", flush=True)
    return result
