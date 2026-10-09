"""Explicit, validated experiment definitions."""
from __future__ import annotations
from dataclasses import dataclass, asdict, replace
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Experiment:
    id: str
    exercise: str
    model: str
    label: str
    group: str = 'required'
    epochs: int = 30
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    clip_norm: float = 1.0
    split_seed: int = 2026
    tokenizer: str = 'rows'
    backend: str = 'manual'
    hidden_size: int = 128
    dim: int = 64
    heads: int = 4
    depth: int = 2
    patch_size: int = 4
    pooling: str = 'last'
    orthogonal_init: bool = False
    chrono_init: bool = False
    orth_lambda: float = 0.0

    def __post_init__(self):
        choices = {'E1': {'softmax', 'mlp', 'cnn'}, 'E2': {'transformer'}, 'E3': {'lstm', 'gru'}}
        if self.exercise not in choices or self.model not in choices[self.exercise]:
            raise ValueError('Model does not belong to the specified exercise.')
        if not self.id or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_' for c in self.id):
            raise ValueError('Experiment id must be a lowercase path-safe identifier.')
        if min(self.epochs, self.batch_size, self.hidden_size, self.depth, self.heads, self.dim) < 1:
            raise ValueError('Epochs, batch size and model dimensions must be positive.')
        if not all(math.isfinite(v) for v in [self.lr, self.weight_decay, self.clip_norm, self.orth_lambda]):
            raise ValueError('Hyperparameters must be finite.')
        if self.lr <= 0 or self.clip_norm <= 0 or self.weight_decay < 0 or self.orth_lambda < 0:
            raise ValueError('Invalid learning rate, regularization, or clipping norm.')
        if self.dim % self.heads or self.patch_size <= 0 or 28 % self.patch_size:
            raise ValueError('Invalid attention head dimension or image patch size.')
        if self.backend not in {'manual', 'torch'} or self.pooling not in {'last', 'mean'}:
            raise ValueError('Unknown attention backend or pooling.')
        if self.tokenizer not in {'rows', 'patches', 'cnn'}:
            raise ValueError('Unknown image tokenizer.')
        if self.exercise == 'E3' and self.tokenizer == 'cnn':
            raise ValueError('E3 uses raw rows/patches, not CNN features.')
        if (self.orth_lambda or self.orthogonal_init or self.chrono_init) and self.exercise != 'E3':
            raise ValueError('Recurrent improvements are restricted to E3.')
        if self.chrono_init and self.model != 'lstm':
            raise ValueError('The chrono experiment is LSTM-only.')

    def to_dict(self) -> dict:
        return asdict(self)


def load_experiments(path: Path | None = None) -> list[Experiment]:
    path = path or ROOT / 'configs' / 'experiments.json'
    rows = [Experiment(**row) for row in json.loads(Path(path).read_text(encoding='utf-8'))]
    if len({x.id for x in rows}) != len(rows):
        raise ValueError('Duplicate experiment id.')
    return rows


def get_experiment(identifier: str, **overrides) -> Experiment:
    for row in load_experiments():
        if row.id == identifier:
            return replace(row, **{k: v for k, v in overrides.items() if v is not None})
    raise ValueError(f'Unknown experiment {identifier!r}. Run python -m scripts.run_suite --list.')
