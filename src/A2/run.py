"""Train both fusion variants, then explicitly freeze and evaluate checkpoints."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision

from A2.data import CONFIG, ROOT, PetTrimaps, read_config, sha256
from A2.models import PetSegmenter, segmentation_loss
from common.runtime import (
    atomic_csv, atomic_json, atomic_torch_save, choose_device, environment_info,
    seed_everything, state_fingerprint, synchronize,
)


def signature(cache):
    return {
        "config_sha256": sha256(CONFIG),
        "data_manifest_sha256": sha256(cache / "manifest.json"),
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in sorted((ROOT / "src/A2").glob("*.py"))
            if path.name in ("__init__.py", "data.py", "models.py", "run.py")
        } | {"src/common/runtime.py": sha256(ROOT / "src/common/runtime.py")},
    }


def run_paths(root, config):
    for seed in config["seeds"]:
        for variant in config["variants"]:
            yield variant, seed, root / variant / f"seed_{seed}"


def load_dataset(cache, partition, manifest):
    dataset = PetTrimaps(cache, partition)
    fingerprint = hashlib.sha256()
    split = manifest["splits"][partition]
    if len(dataset) != len(dataset.masks) or len(dataset) != len(split["names"]):
        raise ValueError(f"Invalid cache size for {partition}")
    for name, rgb, mask in zip(split["names"], dataset.images, dataset.masks, strict=True):
        fingerprint.update(name.encode() + b"\0" + rgb.tobytes() + mask.tobytes())
    if fingerprint.hexdigest() != split["tensor_sha256"]:
        raise ValueError(f"Cache content changed for {partition}")
    return dataset


def loader(dataset, config, workers, device, seed, shuffle=False):
    return DataLoader(
        dataset, batch_size=config["batch_size"], shuffle=shuffle, num_workers=workers,
        persistent_workers=workers > 0, pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(seed),
    )


def normalize(image, device):
    image = image.to(device, non_blocking=True, memory_format=torch.channels_last).float().div_(255)
    mean = image.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = image.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (image - mean) / std


def confusion_by_image(prediction, target):
    # One bincount for all images. Class code 9 represents an ignored target.
    ids = np.where(target < 3, target.astype(np.int64) * 3 + prediction, 9)
    ids += np.arange(len(ids), dtype=np.int64)[:, None, None] * 10
    return np.bincount(ids.ravel(), minlength=len(ids) * 10).reshape(-1, 10)[:, :9].reshape(-1, 3, 3)


def scores(matrix):
    hit = np.diag(matrix)
    mass = matrix.sum(0) + matrix.sum(1)
    valid = mass > 0
    iou = hit[valid] / (mass[valid] - hit[valid])
    dice = 2 * hit[valid] / mass[valid]
    return {
        "miou": float(iou.mean()), "macro_dice": float(dice.mean()),
        "iou": [float(hit[i] / (mass[i] - hit[i])) if valid[i] else None for i in range(3)],
        "dice": [float(2 * hit[i] / mass[i]) if valid[i] else None for i in range(3)],
    }


@torch.inference_mode()
def evaluate(model, batches, device, *, collect_images=False):
    model.eval()
    total = np.zeros((3, 3), dtype=np.int64)
    gate_sum = np.zeros(3)
    per_image = []
    for image, target in batches:
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits, edge = model(normalize(image, device))
        target = target.numpy()
        prediction = logits.argmax(1).to(torch.uint8).cpu().numpy()
        confusion = confusion_by_image(prediction, target)
        total += confusion.sum(0)
        if collect_images:
            gate = edge.float().sigmoid().cpu().numpy()
            for label in range(3):
                gate_sum[label] += gate[target == label].sum(dtype=np.float64)
            per_image.extend(confusion.tolist())
    result = {"confusion": total.tolist(), **scores(total)}
    if collect_images:
        result["edge_score_by_target"] = (gate_sum / total.sum(1)).tolist()
    return result, per_image


def train_run(variant, seed, path, config, identity, train_set, val_set, device, workers):
    path.mkdir(parents=True, exist_ok=False)
    seed_everything(seed)
    model = PetSegmenter(variant, pretrained=True)
    initial_state = state_fingerprint(model.state_dict())
    model.to(device=device, memory_format=torch.channels_last)
    heads = [param for name, param in model.named_parameters() if not name.startswith("encoder.")]
    optimizer = torch.optim.AdamW(
        [{"params": model.encoder.parameters(), "lr": config["encoder_lr"]},
         {"params": heads, "lr": config["head_lr"]}],
        weight_decay=config["weight_decay"],
    )
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    train_loader = loader(train_set, config, workers, device, seed, shuffle=True)
    val_loader = loader(val_set, config, workers, device, seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    metadata = {
        **identity, "config": config, "variant": variant, "seed": seed, "workers": workers,
        "initial_state_sha256": initial_state, "params": sum(p.numel() for p in model.parameters()),
        "runtime": environment_info(device) | {
            "torchvision": torchvision.__version__, "numpy": np.__version__,
            "precision": "bfloat16 autocast" if device.type == "cuda" else "float32",
        },
    }
    atomic_json(path / "protocol.json", metadata)
    rows = []
    best_score = -1
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        synchronize(device)
        start = time.perf_counter()
        loss_sum = torch.zeros((), device=device)
        for image, target in train_loader:
            target = target.to(device, non_blocking=True, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits, edge = model(normalize(image, device))
            loss = segmentation_loss(logits, edge, target, config["boundary_weight"])
            loss.backward()
            optimizer.step()
            loss_sum += loss.detach() * len(image)
        synchronize(device)
        train_seconds = time.perf_counter() - start
        loss_value = float(loss_sum / len(train_set))
        if not np.isfinite(loss_value):
            raise RuntimeError(f"Non-finite training loss in {path} epoch {epoch}")
        val_metrics, _ = evaluate(model, val_loader, device)
        if val_metrics["miou"] > best_score:
            best_epoch, best_score = epoch, val_metrics["miou"]
            atomic_torch_save(path / "best.pt", model.state_dict())
        rows.append({
            "epoch": epoch, "train_loss": loss_value, "train_seconds": train_seconds,
            "val_miou": val_metrics["miou"], "val_macro_dice": val_metrics["macro_dice"],
            "val_transition_iou": val_metrics["iou"][2],
            "encoder_lr": optimizer.param_groups[0]["lr"], "head_lr": optimizer.param_groups[1]["lr"],
        })
        atomic_csv(path / "history.csv", rows)
        print(f"{variant} seed={seed} epoch={epoch:02d} loss={loss_value:.4f} "
              f"val_mIoU={val_metrics['miou']:.4f} train={train_seconds:.1f}s", flush=True)
        schedule.step()
    atomic_json(path / "run.json", {
        **metadata, "epochs": len(rows), "best_epoch": best_epoch, "best_val_miou": best_score,
        "total_train_seconds": sum(row["train_seconds"] for row in rows),
        "peak_allocated_mib": (
            torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
        ),
        "checkpoint_sha256": sha256(path / "best.pt"),
        "history_sha256": sha256(path / "history.csv"),
    })


def freeze(root, config, identity):
    frozen = {"identity": identity, "runs": {}}
    initial_states = {}
    for variant, seed, path in run_paths(root, config):
        record = json.loads((path / "run.json").read_text())
        if (
            any(record[k] != value for k, value in identity.items())
            or (record["variant"], record["seed"]) != (variant, seed)
            or record["epochs"] != config["epochs"]
            or sha256(path / "best.pt") != record["checkpoint_sha256"]
            or sha256(path / "history.csv") != record["history_sha256"]
        ):
            raise ValueError(f"Experiment identity or evidence changed: {path}")
        if initial_states.setdefault(seed, record["initial_state_sha256"]) != record["initial_state_sha256"]:
            raise ValueError(f"Variants did not share initialization for seed {seed}")
        frozen["runs"][str(path.relative_to(root))] = sha256(path / "run.json")
    return frozen


def test_run(variant, path, dataset, names, config, workers, device):
    model = PetSegmenter(variant).to(device=device, memory_format=torch.channels_last)
    model.load_state_dict(torch.load(path / "best.pt", map_location=device, weights_only=True))
    result, matrices = evaluate(model, loader(dataset, config, workers, device, 0), device,
                                collect_images=True)
    rows = []
    for name, matrix in zip(names, matrices, strict=True):
        row = {"name": name}
        row.update({f"c{actual}{pred}": count for actual, counts in enumerate(matrix)
                    for pred, count in enumerate(counts)})
        rows.append(row)
    atomic_csv(path / "test_per_image.csv", rows)
    atomic_json(path / "test.json", {
        **result, "samples": len(dataset), "checkpoint_sha256": sha256(path / "best.pt"),
        "per_image_sha256": sha256(path / "test_per_image.csv"),
    })
    print(f"test {path.relative_to(path.parents[1])}: mIoU={result['miou']:.4f} "
          f"Dice={result['macro_dice']:.4f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["train", "freeze", "test"])
    parser.add_argument("--cache", type=Path, default=ROOT / "data/a2_prepared")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/assignment2")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--confirm-test", action="store_true")
    args = parser.parse_args()
    if args.phase == "test" and not args.confirm_test:
        parser.error("Freeze the protocol and pass --confirm-test to evaluate held-out data.")
    if args.workers < 0:
        parser.error("--workers must be nonnegative.")
    config = read_config()
    identity = signature(args.cache)
    manifest = json.loads((args.cache / "manifest.json").read_text())
    if manifest["config_sha256"] != identity["config_sha256"]:
        raise ValueError("The data cache was prepared with a different protocol.")
    locked = args.output / "frozen.json"
    if args.phase == "freeze":
        if locked.exists():
            raise FileExistsError(locked)
        atomic_json(locked, freeze(args.output, config, identity))
        return
    device = choose_device(args.device)
    if device.type == "cuda" and not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError("This protocol requires CUDA hardware with bfloat16 support.")
    if args.phase == "test":
        if json.loads(locked.read_text()) != freeze(args.output, config, identity):
            raise ValueError("Frozen protocol changed")
        dataset = load_dataset(args.cache, "test", manifest)
        for variant, _, path in run_paths(args.output, config):
            result_path = path / "test.json"
            if result_path.exists():
                result = json.loads(result_path.read_text())
                if (result["checkpoint_sha256"] != sha256(path / "best.pt")
                        or result["per_image_sha256"] != sha256(path / "test_per_image.csv")):
                    raise ValueError(f"Saved test evidence changed: {path}")
                continue
            test_run(variant, path, dataset, manifest["splits"]["test"]["names"],
                     config, args.workers, device)
        return
    if locked.exists():
        raise ValueError("The experiment has been frozen.")
    train_set = load_dataset(args.cache, "train", manifest)
    val_set = load_dataset(args.cache, "validation", manifest)
    for variant, seed, path in run_paths(args.output, config):
        if (path / "run.json").exists():
            record = json.loads((path / "run.json").read_text())
            if any(record[k] != value for k, value in identity.items()):
                raise ValueError(f"Completed run uses a different protocol: {path}")
            continue
        train_run(variant, seed, path, config, identity, train_set, val_set, device, args.workers)


if __name__ == "__main__":
    main()
