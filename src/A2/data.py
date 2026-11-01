"""Prepare the Oxford trimaps with a fixed breed-stratified validation split."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.datasets import OxfordIIITPet

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/a2_experiments.json"
IGNORE = 255


def read_config():
    return json.loads(CONFIG.read_text())


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_partition(path):
    pairs = [(line.split()[0], int(line.split()[1])) for line in path.read_text().splitlines()
             if line.strip() and not line.startswith("#")]
    if len({name for name, _ in pairs}) != len(pairs):
        raise ValueError(f"Repeated sample in {path}")
    return dict(pairs)


def split_names(trainval, test, seed, validation_per_breed):
    if trainval.keys() & test.keys():
        raise ValueError("Official split identifiers overlap")
    breeds = defaultdict(list)
    for name, breed in sorted(trainval.items()):
        breeds[breed].append(name)
    rng = np.random.default_rng(seed)
    valid = set()
    for members in breeds.values():
        if len(members) <= validation_per_breed:
            raise ValueError("Insufficient training samples for a breed")
        valid.update(rng.choice(members, validation_per_breed, replace=False).tolist())
    return {
        "train": sorted(trainval.keys() - valid),
        "validation": sorted(valid),
        "test": sorted(test),
    }


def read_sample(base, name):
    with Image.open(base / "images" / f"{name}.jpg") as source:
        image = source.convert("RGB")
    with Image.open(base / "annotations/trimaps" / f"{name}.png") as source:
        mask = source.copy()
    if image.size != mask.size:
        raise ValueError(f"Image and trimap sizes differ for {name}: {image.size}, {mask.size}")
    return image, mask


def letterbox(image, mask, size):
    width, height = image.size
    scale = size / max(width, height)
    w, h = max(1, round(width * scale)), max(1, round(height * scale))
    rgb = np.empty((size, size, 3), dtype=np.uint8)
    rgb[:] = (124, 116, 104)
    rgb[:h, :w] = np.asarray(image.resize((w, h), Image.Resampling.BILINEAR))
    raw = np.asarray(mask.resize((w, h), Image.Resampling.NEAREST))
    return rgb, encode_mask(raw, (size, size))


def encode_mask(raw, shape):
    mask = np.full(shape, IGNORE, dtype=np.uint8)
    labels = mask[:raw.shape[0], :raw.shape[1]]
    for original in (1, 2, 3):
        labels[raw == original] = original - 1
    return mask


def prepare(data_root, cache, config):
    for partition in ("trainval", "test"):
        OxfordIIITPet(data_root, split=partition, target_types="segmentation", download=True)
    base = data_root / "oxford-iiit-pet"
    trainval = read_partition(base / "annotations/trainval.txt")
    test = read_partition(base / "annotations/test.txt")
    splits = split_names(trainval, test, config["split_seed"], config["validation_per_breed"])
    cache.mkdir(parents=True, exist_ok=False)
    manifest = {
        "config_sha256": sha256(CONFIG),
        "files": {name: sha256(base / name) for name in ("images.tar.gz", "annotations.tar.gz")},
        "split_seed": config["split_seed"],
        "size": config["size"],
        "splits": {},
        "excluded_cross_split_duplicates": [],
    }
    pixels_seen = {}
    # Process evaluation splits last. Any exact RGB match to an earlier split
    # stays exclusively in that earlier split.
    for partition, names in splits.items():
        size = config["size"]
        names_kept, images, masks, widths, heights, boundary_shares = [], [], [], [], [], []
        labels = Counter()
        fingerprint = hashlib.sha256()
        for name in names:
            image, mask = read_sample(base, name)
            pixel_hash = hashlib.sha256(f"{image.size}:".encode() + image.tobytes()).digest()
            earlier = pixels_seen.setdefault(pixel_hash, (partition, name))
            if earlier[0] != partition:
                manifest["excluded_cross_split_duplicates"].append(
                    {"excluded": name, "split": partition, "kept": earlier[1], "kept_split": earlier[0]}
                )
                continue
            rgb, target = letterbox(image, mask, size)
            names_kept.append(name)
            images.append(rgb)
            masks.append(target)
            fingerprint.update(name.encode() + b"\0" + rgb.tobytes() + target.tobytes())
            if partition != "test":
                widths.append(image.width)
                heights.append(image.height)
                counts = np.bincount(target.ravel(), minlength=256)
                labels.update({i: int(counts[i]) for i in (0, 1, 2, IGNORE)})
                boundary_shares.append(float(counts[2] / counts[:3].sum()))
        if not names_kept:
            raise ValueError(f"No samples in {partition}")
        np.save(cache / f"{partition}_images.npy", np.stack(images))
        np.save(cache / f"{partition}_masks.npy", np.stack(masks))
        manifest["splits"][partition] = {
            "names": names_kept, "size": len(names_kept), "tensor_sha256": fingerprint.hexdigest(),
            "breed_counts": dict(Counter((test if partition == "test" else trainval)[n] for n in names_kept)),
        }
        if partition != "test":
            manifest["splits"][partition]["eda"] = {
                "valid_pixel_counts": [labels[i] for i in range(3)], "ignored_pixels": labels[IGNORE],
                "width_quartiles": np.percentile(widths, [0, 25, 50, 75, 100]).tolist(),
                "height_quartiles": np.percentile(heights, [0, 25, 50, 75, 100]).tolist(),
                "boundary_share_quartiles": np.percentile(boundary_shares, [0, 25, 50, 75, 100]).tolist(),
            }
    (cache / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


class PetTrimaps(Dataset):
    def __init__(self, cache, partition):
        self.images = np.load(cache / f"{partition}_images.npy", mmap_mode="r")
        self.masks = np.load(cache / f"{partition}_masks.npy", mmap_mode="r")
        self.augment = partition == "train"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image, mask = self.images[index], self.masks[index]
        if self.augment and torch.rand(()).item() < 0.5:
            image, mask = image[:, ::-1], mask[:, ::-1]
        return torch.from_numpy(image.copy()).permute(2, 0, 1), torch.from_numpy(mask.copy())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--cache", type=Path, default=ROOT / "data/a2_prepared")
    args = parser.parse_args()
    manifest = prepare(args.data_root, args.cache, read_config())
    print({key: split["size"] for key, split in manifest["splits"].items()}, flush=True)


if __name__ == "__main__":
    main()
