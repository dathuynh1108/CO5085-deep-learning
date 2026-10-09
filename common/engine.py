"""Explicit PyTorch training. Test evaluation is deliberately a separate command."""
from __future__ import annotations
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
import torch
from torch.nn import functional as F
from common.config import Experiment, ROOT
from common.data import prepare_data, make_loader
from common.factory import build_model, parameter_count
from common.plots import plot_history
from common.runtime import (atomic_json, atomic_csv, atomic_torch_save, capture_rng,
                            restore_rng, seed_everything, choose_device, synchronize,
                            environment_info, json_hash, state_fingerprint)
from E3.models import orthogonality_penalty


def source_fingerprint() -> str:
    digest = hashlib.sha256()
    for folder in ['common', 'E1', 'E2', 'E3']:
        for path in sorted((ROOT / folder).glob('*.py')):
            digest.update(str(path.relative_to(ROOT)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


@contextmanager
def run_lock(directory: Path):
    lock = directory / '.train.lock'
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError(f'Run is locked: {lock}. Check for a live process before removing a stale lock.') from error
    with os.fdopen(fd, 'w') as f:
        f.write(str(os.getpid()))
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


def cpu_state_dict(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def better_checkpoint(accuracy: float, loss: float, best: dict) -> bool:
    return accuracy > best['accuracy'] or (accuracy == best['accuracy'] and loss < best['loss'])


def train_epoch(model, loader, optimizer, config: Experiment, device: torch.device):
    model.train()
    n, correct, ce_sum, objective_sum, penalty_sum, norm_sum, batches = 0, 0, 0., 0., 0., 0., 0
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device); started = time.perf_counter()
    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        ce = F.cross_entropy(logits, labels)
        penalty = orthogonality_penalty(model) if config.orth_lambda else ce.new_zeros(())
        objective = ce + config.orth_lambda * penalty
        if not torch.isfinite(objective):
            raise FloatingPointError('Non-finite loss. Run stopped without inventing replacement metrics.')
        objective.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_norm, error_if_nonfinite=True)
        optimizer.step()
        size = labels.numel()
        n += size; batches += 1
        correct += (logits.detach().argmax(1) == labels).sum().item()
        ce_sum += ce.detach().item() * size
        objective_sum += objective.detach().item() * size
        penalty_sum += penalty.detach().item() * size
        norm_sum += float(grad_norm)
    synchronize(device); elapsed = time.perf_counter() - started
    if not n:
        raise RuntimeError('Empty training loader.')
    return dict(train_ce=ce_sum/n, train_objective=objective_sum/n, train_accuracy=correct/n,
                orthogonality_penalty=penalty_sum/n, grad_norm_before_clip=norm_sum/batches,
                train_seconds=elapsed, train_images_per_second=n/elapsed,
                peak_memory_mib=torch.cuda.max_memory_allocated(device)/2**20 if device.type == 'cuda' else None)


@torch.inference_mode()
def validate(model, loader, device: torch.device):
    model.eval()
    n, correct, total = 0, 0, 0.
    synchronize(device); started = time.perf_counter()
    for images, labels, _ in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        logits = model(images)
        total += F.cross_entropy(logits, labels, reduction='sum').item()
        correct += (logits.argmax(1) == labels).sum().item()
        n += labels.numel()
    synchronize(device)
    if not n:
        raise RuntimeError('Empty validation loader.')
    return dict(validation_ce=total/n, validation_accuracy=correct/n,
                validation_seconds=time.perf_counter()-started)


def train(config: Experiment, seed: int, data_root: Path, output_root: Path,
          device_name: str = 'auto', workers: int = 4, resume: bool = False,
          deterministic: bool = False, download: bool = True) -> Path:
    if not 0 <= seed < 2**32:
        raise ValueError('seed must be in [0, 2**32).')
    directory = Path(output_root) / config.id / f'seed_{seed}'
    if directory.exists() and any(directory.iterdir()) and not resume:
        raise FileExistsError(f'{directory} already contains a run. Use --resume or a NEW --output-root.')
    if resume and not (directory / 'last.pt').exists():
        raise FileNotFoundError(f'No resumable checkpoint: {directory / "last.pt"}')
    directory.mkdir(parents=True, exist_ok=True)
    with run_lock(directory):
        seed_everything(seed, deterministic)
        device = choose_device(device_name)
        train_set, val_set, metadata = prepare_data(Path(data_root), config.split_seed, download)
        signature_data = dict(experiment=config.to_dict(), seed=seed, data_fingerprint=metadata['fingerprint'],
                              source_fingerprint=source_fingerprint(), deterministic=deterministic, num_workers=workers)
        signature = json_hash(signature_data)
        model = build_model(config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=config.lr * .01)
        shuffle_generator = torch.Generator().manual_seed(seed)
        train_loader = make_loader(train_set, config.batch_size, workers, device, True, shuffle_generator)
        val_loader = make_loader(val_set, config.batch_size, workers, device)
        history, best_state, best_epoch, start_epoch = [], None, 0, 1
        best = {'accuracy': -1., 'loss': float('inf')}
        info = dict(signature_data, signature=signature, dataset=metadata,
                    n_parameters=parameter_count(model), environment=environment_info(device),
                    dtype='float32', created_unix=time.time())
        if resume:
            # Only resume checkpoints produced locally by this project are trusted.
            state = torch.load(directory / 'last.pt', map_location='cpu', weights_only=False)
            original_info = json.loads((directory / 'run.json').read_text(encoding='utf-8'))
            for key in ['python', 'torch', 'cuda_runtime', 'cudnn', 'device', 'device_name', 'torch_num_threads']:
                if original_info['environment'].get(key) != info['environment'].get(key):
                    raise ValueError(f'Resume environment changed: {key}. Use the original environment.')
            if state['signature'] != signature:
                raise ValueError('Resume signature differs (code/config/seed/data). Start a new output root.')
            model.load_state_dict(state['model'])
            optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
            history, best, best_epoch = state['history'], state['best'], state['best_epoch']
            best_state, start_epoch = state['best_model'], state['epoch'] + 1
            shuffle_generator.set_state(state['shuffle_rng'].cpu())
            restore_rng(state['rng'])
            # Roll the inference checkpoint back to the last committed epoch if a
            # process died between writing best.pt and last.pt.
            atomic_torch_save(directory / 'best.pt', dict(model=best_state, experiment=config.to_dict(),
                dataset=metadata, signature=signature, seed=seed, epoch=best_epoch, validation=best))
        else:
            atomic_json(directory / 'run.json', info)
        print(f'{config.id} | seed={seed} | {device} | parameters={parameter_count(model):,}', flush=True)
        for epoch in range(start_epoch, config.epochs + 1):
            lr = optimizer.param_groups[0]['lr']
            training = train_epoch(model, train_loader, optimizer, config, device)
            validation = validate(model, val_loader, device)
            row = dict(epoch=epoch, learning_rate=lr, **training, **validation)
            history.append(row)
            if better_checkpoint(validation['validation_accuracy'], validation['validation_ce'], best):
                best = {'accuracy': validation['validation_accuracy'], 'loss': validation['validation_ce']}
                best_epoch, best_state = epoch, cpu_state_dict(model)
                atomic_torch_save(directory / 'best.pt', dict(model=best_state, experiment=config.to_dict(),
                    dataset=metadata, signature=signature, seed=seed, epoch=best_epoch, validation=best))
            scheduler.step()
            atomic_torch_save(directory / 'last.pt', dict(epoch=epoch, signature=signature,
                model=cpu_state_dict(model), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                best=best, best_epoch=best_epoch, best_model=best_state, history=history,
                shuffle_rng=shuffle_generator.get_state(), rng=capture_rng()))
            atomic_csv(directory / 'history.csv', history)
            print(f"  {epoch:02d}/{config.epochs}: CE={training['train_ce']:.4f} | "
                  f"val={validation['validation_accuracy']:.4%} | {training['train_seconds']:.1f}s", flush=True)
        if not history:
            raise RuntimeError('No completed epochs.')
        measured = history[1:] or history  # exclude the first epoch's initialization overhead
        summary = dict(status='trained', signature=signature, experiment=config.to_dict(), seed=seed,
                       n_parameters=parameter_count(model), best_epoch=best_epoch,
                       best_validation_accuracy=best['accuracy'], best_validation_ce=best['loss'],
                       completed_epochs=len(history), median_train_seconds=statistics.median(r['train_seconds'] for r in measured),
                       total_train_seconds=sum(r['train_seconds'] for r in history),
                       total_validation_seconds=sum(r['validation_seconds'] for r in history),
                       environment=dict(info['environment'], num_workers=workers), data_fingerprint=metadata['fingerprint'],
                       best_model_sha256=state_fingerprint(best_state))
        atomic_json(directory / 'metrics.json', summary)
        plot_history(history, directory / 'plots')
    return directory
