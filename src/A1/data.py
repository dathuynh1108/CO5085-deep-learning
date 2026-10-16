"""All official CIFAR-100 images, with a fixed stratified development split."""
import hashlib
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR100
from torchvision import transforms as T
from common.runtime import json_hash, file_hash


def stratified_split(labels, seed=2026, val_per_class=50):
    labels = np.asarray(labels)
    if labels.shape != (50000,) or not np.array_equal(np.unique(labels), np.arange(100)):
        raise ValueError('Expected all 50000 official training labels and all 100 classes')
    rng = np.random.default_rng(seed)
    train, val = [], []
    for cls in range(100):
        indices = np.flatnonzero(labels == cls)
        if len(indices) != 500:
            raise ValueError('Expected exactly 500 train images per class')
        rng.shuffle(indices)
        val.extend(indices[:val_per_class].tolist())
        train.extend(indices[val_per_class:].tolist())
    return sorted(train), sorted(val)


class EpochSubset(Subset):
    """Augmentation is a pure function of train seed, epoch and official index.

    Workers are recreated for each iterator (persistent_workers=False), so they
    receive the current epoch. This also works with workers=0 without consuming
    the model's RNG stream. Worker scheduling cannot alter augmentation.
    """
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self.epoch, self.seed = 0, 0

    def set_epoch(self, epoch, seed):
        self.epoch, self.seed = epoch, seed

    def __getitem__(self, index):
        official_index = self.indices[index]
        digest = hashlib.sha256(f'A1:{self.seed}:{self.epoch}:{official_index}'.encode()).digest()
        seed = int.from_bytes(digest[:8], 'little') % (2**63 - 1)
        python_state, numpy_state = random.getstate(), np.random.get_state()
        try:
            with torch.random.fork_rng(devices=[]):
                # CPU generator only: never initialize CUDA inside a worker.
                torch.random.default_generator.manual_seed(seed)
                random.seed(seed)
                np.random.seed(seed % 2**32)
                return self.dataset[official_index]
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)

    def __getitems__(self, indices):
        # Override Subset's batched fast path; it must not bypass seeded access.
        return [self[index] for index in indices]


def transform(protocol, train=False):
    steps = [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()] if train else []
    return T.Compose(steps + [T.Resize((protocol['image_size'], protocol['image_size']),
                                      interpolation=T.InterpolationMode.BILINEAR, antialias=True),
                              T.ToTensor(), T.Normalize(protocol['mean'], protocol['std'])])


def seed_worker(_):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def array_hash(array):
    array = np.ascontiguousarray(array)
    return hashlib.sha256(array.tobytes()).hexdigest()


def development_data(root, protocol):
    # No implicit download and no construction/evaluation of the test dataset.
    train_base = CIFAR100(str(root), train=True, download=False, transform=transform(protocol, True))
    val_base = CIFAR100(str(root), train=True, download=False, transform=transform(protocol))
    train, val = stratified_split(train_base.targets, protocol['split_seed'])
    split = {'seed': protocol['split_seed'], 'train': train, 'val': val,
             'train_count': len(train), 'val_count': len(val), 'classes': train_base.classes}
    identity = {'images_sha256': array_hash(train_base.data),
                'labels_sha256': array_hash(np.asarray(train_base.targets, dtype=np.int64)),
                'split_sha256': json_hash(split), 'meta_sha256': file_hash(root / 'cifar-100-python/meta')}
    return EpochSubset(train_base, train), Subset(val_base, val), split, identity


def heldout_data(root, protocol):
    dataset = CIFAR100(str(root), train=False, download=False, transform=transform(protocol))
    if len(dataset) != 10000 or not np.array_equal(np.bincount(dataset.targets, minlength=100), np.full(100, 100)):
        raise ValueError('Expected all official test images and 100 images per class')
    return dataset, {'images_sha256': array_hash(dataset.data),
                     'labels_sha256': array_hash(np.asarray(dataset.targets, dtype=np.int64)),
                     'classes': dataset.classes, 'count': len(dataset)}


def loader(dataset, batch_size, workers, generator, shuffle=False, device=None):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      generator=generator, worker_init_fn=seed_worker, persistent_workers=False,
                      pin_memory=device is not None and device.type == 'cuda', drop_last=False)
