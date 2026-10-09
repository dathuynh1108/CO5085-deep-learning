"""Reproducibility and atomic artifact writes."""
from __future__ import annotations
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import tempfile
import numpy as np
import torch


def json_hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write('\n')
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_torch_save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
    os.close(fd)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        if not rows:
            raise ValueError('Empty CSV needs explicit fieldnames.')
        fieldnames = list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader(); writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def seed_everything(seed: int, deterministic: bool = False) -> None:
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(deterministic)


def capture_rng() -> dict:
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state: dict) -> None:
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] and torch.cuda.is_available():
        if len(state['cuda']) != torch.cuda.device_count():
            raise RuntimeError('Exact resume requires the original number of visible CUDA devices.')
        torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])


def choose_device(name: str) -> torch.device:
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(name)
    if device.type not in {'cuda', 'cpu'}:
        raise ValueError('This experiment supports CPU or CUDA.')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available. Install the matching PyTorch wheel or select --device cpu.')
    return device


def synchronize(device: torch.device) -> None:
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def environment_info(device: torch.device) -> dict:
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {'python': platform.python_version(), 'platform': platform.platform(),
            'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
            'cudnn': torch.backends.cudnn.version(), 'device': str(device),
            'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else platform.processor(),
            'git_commit': commit, 'git_dirty': dirty,
            'torch_num_threads': torch.get_num_threads(),
            'tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
            'tf32_cudnn': torch.backends.cudnn.allow_tf32}


def state_fingerprint(state: dict) -> str:
    """Stable tensor-content hash, independent of torch.save ZIP metadata."""
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()
