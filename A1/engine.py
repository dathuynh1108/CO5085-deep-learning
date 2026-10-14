"""Manual epoch loop, immutable run signatures, and epoch-boundary resume."""
from contextlib import contextmanager
from pathlib import Path
import json
import os
import time
import torch
import torchvision
from torch import nn
from common.runtime import (atomic_json, atomic_csv, atomic_torch_save, capture_rng,
                            restore_rng, seed_everything, state_fingerprint, json_hash,
                            file_hash, environment_info, synchronize)
from A1.protocol import ROOT, CONFIG
from A1.models import build_model, training_mode, frozen_state, parameter_counts
from A1.data import development_data, heldout_data, loader
from A1.metrics import classification_metrics

HISTORY_FIELDS = ['epoch', 'lr', 'train_ce', 'train_accuracy', 'train_macro_f1',
                  'val_ce', 'val_accuracy', 'val_macro_f1', 'train_seconds', 'epoch_seconds',
                  'peak_gpu_memory_bytes']


def source_identity():
    files = [ROOT / 'A1' / name for name in
             ['__init__.py', 'protocol.py', 'models.py', 'data.py', 'metrics.py', 'engine.py', 'run_suite.py', 'visuals.py']]
    files += [ROOT / 'common/runtime.py']
    return {str(path.relative_to(ROOT)): file_hash(path) for path in files}


def make_signature(protocol, experiment, seed, data, environment, workers, deterministic):
    # Git metadata is recorded as provenance but unrelated checkout edits do not
    # invalidate A1. Relevant source content is hashed independently.
    stable_environment = {k: v for k, v in environment.items() if k not in {'git_commit', 'git_dirty'}}
    return {'protocol': protocol, 'experiment': experiment, 'seed': seed,
            'config_sha256': file_hash(CONFIG), 'source': source_identity(), 'data': data,
            'environment': stable_environment, 'num_workers': workers, 'deterministic': deterministic}


def require_signature(actual, expected):
    if actual != expected:
        raise RuntimeError('Run signature mismatch: config/source/data/seed/environment/loader changed. Use a new output root.')


@contextmanager
def run_lock(run):
    run.mkdir(parents=True, exist_ok=True)
    path = run / '.a1.lock'
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(f'Run locked: {path}; verify the owning process before removing a stale lock') from exc
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(str(os.getpid()))
        yield
    finally:
        path.unlink()


def save_last(path, signature, model, optimizer, scheduler, generators, history, best, initial, frozen_hash):
    atomic_torch_save(path, {'signature': signature, 'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                            'rng': capture_rng(), 'generators': {k: g.get_state() for k, g in generators.items()},
                            'history': history, 'best': best, 'initial_state_sha256': initial,
                            'frozen_state_sha256': frozen_hash})


def restore_last(path, signature, model, optimizer, scheduler, generators):
    state = torch.load(path, map_location='cpu', weights_only=False)
    require_signature(state['signature'], signature)
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    # AdamW step counters remain CPU for the default non-capturable optimizer;
    # moment tensors must follow parameters on CUDA after CPU checkpoint load.
    for parameter, values in optimizer.state.items():
        for name in ['exp_avg', 'exp_avg_sq', 'max_exp_avg_sq']:
            if name in values:
                values[name] = values[name].to(parameter.device)
    scheduler.load_state_dict(state['scheduler'])
    for key, generator in generators.items():
        generator.set_state(state['generators'][key].cpu())
    restore_rng(state['rng'])
    # Do not retain another model-sized checkpoint object during training.
    return {k: state[k] for k in ['history', 'best', 'initial_state_sha256', 'frozen_state_sha256']}


def reconcile_best(run, signature, model, state):
    """Recover a best write interrupted after committing last.pt."""
    best = state['best']
    if best is None:
        return
    path = run / 'best.pt'
    matches = False
    if path.exists():
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        require_signature(checkpoint['signature'], signature)
        matches = checkpoint['selection'] == best and state_fingerprint(checkpoint['model']) == best['state_sha256']
        del checkpoint
    if matches:
        return
    if best['epoch'] != len(state['history']) or state_fingerprint(model.state_dict()) != best['state_sha256']:
        raise RuntimeError('Best checkpoint is missing/corrupt and cannot be recovered from last.pt')
    atomic_torch_save(path, {'signature': signature, 'model': model.state_dict(), 'selection': best})


def better(accuracy, ce, best):
    return best is None or accuracy > best['accuracy'] or (accuracy == best['accuracy'] and ce < best['ce'])


def epoch_pass(model, batches, device, optimizer=None, clip=1.0, predictions=False):
    if optimizer is None:
        model.eval()
    else:
        training_mode(model)
    total_loss = 0.0
    targets, guesses, confidence = [], [], []
    criterion = nn.CrossEntropyLoss()
    with torch.set_grad_enabled(optimizer is not None):
        for images, labels in batches:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits = model(images)
                loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite CE; refusing to checkpoint this epoch')
            if optimizer is not None:
                loss.backward()
                nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], clip,
                                         error_if_nonfinite=True)
                optimizer.step()
            total_loss += loss.detach().item() * len(labels)
            targets.extend(labels.detach().cpu().tolist())
            guesses.extend(logits.detach().argmax(1).cpu().tolist())
            if predictions:
                confidence.extend(logits.detach().float().softmax(1).max(1).values.cpu().tolist())
    result = classification_metrics(targets, guesses)
    result['ce'] = total_loss / len(targets)
    if predictions:
        result['predictions'] = [{'index': i, 'target': t, 'prediction': p, 'confidence': c}
                                 for i, (t, p, c) in enumerate(zip(targets, guesses, confidence))]
    return result


def write_train_evidence(run, state, signature, counts, environment):
    history, best = state['history'], state['best']
    atomic_csv(run / 'history.csv', history, HISTORY_FIELDS)
    metadata = {'signature': signature, 'signature_sha256': json_hash(signature), 'environment': environment,
                'initial_state_sha256': state['initial_state_sha256'],
                'frozen_state_sha256': state['frozen_state_sha256'], **counts,
                'completed_epochs': len(history), 'expected_epochs': signature['protocol']['epochs'],
                'status': 'trained' if len(history) == signature['protocol']['epochs'] else 'incomplete',
                'best_validation': best, 'train_seconds': sum(row['train_seconds'] for row in history),
                'epoch_seconds': sum(row['epoch_seconds'] for row in history),
                'peak_gpu_memory_bytes': max((row['peak_gpu_memory_bytes'] for row in history), default=0),
                'timing_scope': 'train_seconds: synchronized train pass including data loading; epoch_seconds: train + validation; excludes construction/checkpoint/export',
                'precision': signature['protocol']['precision'], 'test_evaluated': (run / 'evaluation_test.json').exists()}
    atomic_json(run / 'metadata.json', metadata)
    return metadata


def train_run(run, protocol, experiment, seed, data_root, device, workers, resume=False, deterministic=False):
    with run_lock(run):
        if any(run.iterdir()) and not resume:
            if any(p.name != '.a1.lock' for p in run.iterdir()):
                raise RuntimeError(f'Refusing to overwrite existing run {run}; use --resume')
        if resume and not (run / 'last.pt').exists():
            # --resume permits starting a missing run in a partially complete suite.
            if any(p.name != '.a1.lock' for p in run.iterdir()):
                raise RuntimeError('Existing artifacts without last.pt; use a new output root')
        seed_everything(seed, deterministic)
        train, val, split, identity = development_data(data_root, protocol)
        env = environment_info(device)
        env['torchvision'] = torchvision.__version__
        env['visible_cuda_devices'] = os.environ.get('CUDA_VISIBLE_DEVICES')
        env['cuda_device_count'] = torch.cuda.device_count() if device.type == 'cuda' else 0
        signature = make_signature(protocol, experiment, seed, identity, env, workers, deterministic)
        existing = (run / 'last.pt').exists()
        if existing:
            if (run / 'metadata.json').exists():
                require_signature(json.loads((run / 'metadata.json').read_text())['signature'], signature)
        model = build_model(experiment, initialize_pretrained=not existing).to(device)
        counts = parameter_counts(model)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=experiment['lr'],
                                     weight_decay=protocol['weight_decay'], betas=tuple(protocol['betas']), eps=protocol['eps'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=protocol['epochs'])
        generators = {'train': torch.Generator().manual_seed(seed), 'val': torch.Generator().manual_seed(seed + 1)}
        train_batches = loader(train, protocol['batch_size'], workers, generators['train'], True, device)
        val_batches = loader(val, protocol['batch_size'], workers, generators['val'], False, device)
        if existing:
            state = restore_last(run / 'last.pt', signature, model, optimizer, scheduler, generators)
            reconcile_best(run, signature, model, state)
        else:
            state = {'history': [], 'best': None, 'initial_state_sha256': state_fingerprint(model.state_dict()),
                     'frozen_state_sha256': state_fingerprint(frozen_state(model))}
            atomic_json(run / 'split.json', split)
            save_last(run / 'last.pt', signature, model, optimizer, scheduler, generators, **{
                'history': [], 'best': None, 'initial': state['initial_state_sha256'], 'frozen_hash': state['frozen_state_sha256']})
        require_signature(state_fingerprint(frozen_state(model)), state['frozen_state_sha256'])
        for epoch in range(len(state['history']) + 1, protocol['epochs'] + 1):
            train.set_epoch(epoch, seed)
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            synchronize(device)
            start = time.perf_counter()
            lr = optimizer.param_groups[0]['lr']
            training = epoch_pass(model, train_batches, device, optimizer, protocol['gradient_clip'])
            synchronize(device)
            train_seconds = time.perf_counter() - start
            validation = epoch_pass(model, val_batches, device)
            synchronize(device)
            epoch_seconds = time.perf_counter() - start
            require_signature(state_fingerprint(frozen_state(model)), state['frozen_state_sha256'])
            row = {'epoch': epoch, 'lr': lr, **{f'train_{key}': training[key] for key in ['ce', 'accuracy', 'macro_f1']},
                   **{f'val_{key}': validation[key] for key in ['ce', 'accuracy', 'macro_f1']},
                   'train_seconds': train_seconds, 'epoch_seconds': epoch_seconds,
                   'peak_gpu_memory_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0}
            if better(validation['accuracy'], validation['ce'], state['best']):
                state['best'] = {'epoch': epoch, 'accuracy': validation['accuracy'], 'ce': validation['ce'],
                                 'macro_f1': validation['macro_f1'], 'state_sha256': state_fingerprint(model.state_dict())}
            scheduler.step()
            state['history'].append(row)
            # Commit last before writing best. If interrupted between writes,
            # the selected weights can be reconstructed from last on resume.
            save_last(run / 'last.pt', signature, model, optimizer, scheduler, generators,
                      state['history'], state['best'], state['initial_state_sha256'], state['frozen_state_sha256'])
            if state['best']['epoch'] == epoch:
                atomic_torch_save(run / 'best.pt', {'signature': signature, 'model': model.state_dict(), 'selection': state['best']})
            write_train_evidence(run, state, signature, counts, env)
            print(f'{experiment["id"]} seed={seed} epoch={epoch}/{protocol["epochs"]} val_acc={validation["accuracy"]:.4f}', flush=True)
        metadata = write_train_evidence(run, state, signature, counts, env)
        from A1.visuals import curves
        curves(run, state['history'])
        return metadata


def test_run(run, protocol, experiment, seed, data_root, device, workers, deterministic=False):
    with run_lock(run):
        if not (run / 'metadata.json').exists():
            raise RuntimeError('Train this run before evaluating test')
        metadata = json.loads((run / 'metadata.json').read_text())
        if metadata['status'] != 'trained' or metadata['completed_epochs'] != protocol['epochs']:
            raise RuntimeError('All configured epochs must finish before test')
        seed_everything(seed, deterministic)
        # Check the development identity before opening held-out images.
        development = development_data(data_root, protocol)
        identity = development[3]
        del development
        env = environment_info(device)
        env['torchvision'] = torchvision.__version__
        env['visible_cuda_devices'] = os.environ.get('CUDA_VISIBLE_DEVICES')
        env['cuda_device_count'] = torch.cuda.device_count() if device.type == 'cuda' else 0
        signature = make_signature(protocol, experiment, seed, identity, env, workers, deterministic)
        require_signature(metadata['signature'], signature)
        dataset, test_identity = heldout_data(data_root, protocol)
        best_hash = file_hash(run / 'best.pt')
        evaluation_identity = {'signature_sha256': json_hash(signature), 'best_file_sha256': best_hash,
                               'test_data': test_identity, 'batch_size': protocol['batch_size']}
        path = run / 'evaluation_test.json'
        if path.exists():
            evaluation = json.loads(path.read_text())
            require_signature(evaluation['identity'], evaluation_identity)
            return evaluation
        model = build_model(experiment, initialize_pretrained=False).to(device)
        checkpoint = torch.load(run / 'best.pt', map_location='cpu', weights_only=False)
        require_signature(checkpoint['signature'], signature)
        require_signature(checkpoint['selection'], metadata['best_validation'])
        model.load_state_dict(checkpoint['model'])
        require_signature(state_fingerprint(model.state_dict()), metadata['best_validation']['state_sha256'])
        del checkpoint
        batches = loader(dataset, protocol['batch_size'], workers, torch.Generator().manual_seed(seed + 2), False, device)
        synchronize(device)
        start = time.perf_counter()
        evaluation = epoch_pass(model, batches, device, predictions=True)
        synchronize(device)
        evaluation['test_seconds'] = time.perf_counter() - start
        predictions = evaluation.pop('predictions')
        for row in evaluation['per_class']:
            row['class_name'] = dataset.classes[row['class_id']]
        evaluation.update({'identity': evaluation_identity, 'selected_epoch': metadata['best_validation']['epoch'],
                           'best_state_sha256': metadata['best_validation']['state_sha256']})
        atomic_csv(run / 'predictions_test.csv', predictions)
        atomic_csv(run / 'per_class_test.csv', evaluation['per_class'])
        atomic_csv(run / 'confusion_test.csv', [{'true_class': i, **{str(j): n for j, n in enumerate(row)}}
                                               for i, row in enumerate(evaluation['confusion'])])
        from A1.visuals import evaluation_figures
        evaluation_figures(run, dataset, predictions, evaluation['confusion'])
        evaluation['artifacts'] = {name: file_hash(run / name) for name in
                                  ['predictions_test.csv', 'per_class_test.csv', 'confusion_test.csv']}
        atomic_json(path, evaluation)
        metadata['test_evaluated'] = True
        atomic_json(run / 'metadata.json', metadata)
        return evaluation
