"""One stratified training/validation split shared by all three exercises."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
from common.runtime import atomic_json, json_hash

CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def stratified_split(labels: np.ndarray, validation_fraction: float, seed: int):
    y = np.asarray(labels)
    if y.ndim != 1 or not 0 < validation_fraction < 1 or not len(y):
        raise ValueError('Invalid label vector or validation fraction.')
    rng = np.random.default_rng(seed)
    train, validation = [], []
    for cls in np.unique(y):
        ids = np.flatnonzero(y == cls)
        n_val = int(round(len(ids) * validation_fraction))
        if n_val < 1 or n_val >= len(ids):
            raise ValueError('Each class needs at least one train and one validation sample.')
        rng.shuffle(ids)
        validation.extend(ids[:n_val]); train.extend(ids[n_val:])
    return np.sort(np.asarray(train, dtype=np.int64)), np.sort(np.asarray(validation, dtype=np.int64))


def channel_statistics(images: torch.Tensor, indices: np.ndarray, chunk_size: int = 2048):
    if images.ndim != 3 or images.dtype != torch.uint8 or not len(indices):
        raise ValueError('Expected nonempty training indices and uint8 grayscale images [N,H,W].')
    total, squared, count = 0.0, 0.0, 0
    for start in range(0, len(indices), chunk_size):
        x = images[torch.as_tensor(indices[start:start + chunk_size])].double() / 255
        total += x.sum().item(); squared += x.square().sum().item(); count += x.numel()
    mean = total / count
    std = max(squared / count - mean * mean, 0.0) ** 0.5
    if std <= 1e-12:
        raise ValueError('Training images have zero variance.')
    return mean, std


class NormalizedImages(Dataset):
    def __init__(self, images: torch.Tensor, labels: torch.Tensor, indices: np.ndarray,
                 mean: float, std: float):
        self.images, self.labels = images, labels
        self.indices, self.mean, self.std = indices, mean, std

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        original = int(self.indices[index])
        image = (self.images[original].unsqueeze(0).float() / 255 - self.mean) / self.std
        return image, int(self.labels[original]), original


def fashion_mnist_type():
    """Keep torchvision's loader/checksums, with the dataset owner's HTTPS mirror."""
    from torchvision.datasets import FashionMNIST as TorchFashionMNIST

    class FashionMNIST(TorchFashionMNIST):
        mirrors = ['https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/',
                   *TorchFashionMNIST.mirrors]
    return FashionMNIST


def validate_metadata(metadata: dict, definition: dict) -> None:
    if any(metadata.get(k) != v for k, v in definition.items()):
        raise ValueError('Dataset/split definition changed. Use a new data root.')
    values = {**definition, 'mean': metadata['mean'], 'std': metadata['std']}
    if (not all(math.isfinite(metadata[k]) for k in ['mean', 'std'])
            or metadata['std'] <= 1e-12 or json_hash(values) != metadata['fingerprint']):
        raise ValueError('Cached normalization metadata changed or is invalid.')


def prepare_data(root: Path, split_seed: int = 2026, download: bool = True):
    FashionMNIST = fashion_mnist_type()
    root = Path(root)
    base = FashionMNIST(str(root / 'raw'), train=True, download=download)
    labels = base.targets.numpy()
    train, validation = stratified_split(labels, 0.1, split_seed)
    # This digest records exact pixel/label content, not only the dataset name.
    digest = hashlib.sha256(base.data.numpy().tobytes() + labels.tobytes()).hexdigest()
    definition = {'dataset': 'FashionMNIST', 'version': 2, 'split_seed': split_seed,
                  'validation_fraction': 0.1, 'source_sha256': digest,
                  'train_indices_sha256': hashlib.sha256(train.tobytes()).hexdigest(),
                  'validation_indices_sha256': hashlib.sha256(validation.tobytes()).hexdigest()}
    cached = root / 'processed' / f'fashion-mnist-split-{split_seed}.json'
    if cached.exists():
        metadata = json.loads(cached.read_text(encoding='utf-8'))
        validate_metadata(metadata, definition)
    else:
        mean, std = channel_statistics(base.data, train)
        fingerprint = json_hash(dict(definition, mean=mean, std=std))
        metadata = dict(definition, fingerprint=fingerprint, mean=mean, std=std,
                        n_train=len(train), n_validation=len(validation), class_names=CLASS_NAMES)
        atomic_json(cached, metadata)
        # Explicit indices are stored for auditing; they are not committed to Git.
        np.savez_compressed(cached.with_suffix('.npz'), train=train, validation=validation)
    args = (metadata['mean'], metadata['std'])
    return (NormalizedImages(base.data, base.targets, train, *args),
            NormalizedImages(base.data, base.targets, validation, *args), metadata)


def test_data(root: Path, metadata: dict, download: bool = True):
    FashionMNIST = fashion_mnist_type()
    base = FashionMNIST(str(Path(root) / 'raw'), train=False, download=download)
    return NormalizedImages(base.data, base.targets, np.arange(len(base)), metadata['mean'], metadata['std'])


def make_loader(dataset: Dataset, batch_size: int, workers: int, device: torch.device,
                shuffle: bool = False, generator: torch.Generator | None = None):
    if batch_size < 1 or workers < 0:
        raise ValueError('batch_size must be positive and num_workers nonnegative.')
    sampler = RandomSampler(dataset, generator=generator) if shuffle else None
    worker_generator = torch.Generator().manual_seed(9017)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=False,
                      num_workers=workers, pin_memory=device.type == 'cuda',
                      persistent_workers=workers > 0, generator=worker_generator, drop_last=False)
