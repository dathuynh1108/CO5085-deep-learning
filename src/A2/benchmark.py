"""Measure CUDA forward latency with resident inputs and fixed batch size one."""

import argparse
import json
from pathlib import Path
import statistics

import numpy as np
import torch

from A2.data import ROOT, read_config, sha256
from A2.models import PetSegmenter
from A2.run import freeze, load_dataset, normalize, run_paths, signature
from common.runtime import atomic_json


@torch.inference_mode()
def measure(model, image):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(50):
            model(image)
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(200)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(200)]
        for start, end in zip(starts, ends, strict=True):
            start.record()
            model(image)
            end.record()
        torch.cuda.synchronize()
    elapsed = [start.elapsed_time(end) for start, end in zip(starts, ends, strict=True)]
    return {
        "median_ms": statistics.median(elapsed), "p95_ms": float(np.percentile(elapsed, 95)),
        "samples_ms": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=ROOT / "runs/assignment2")
    parser.add_argument("--cache", type=Path, default=ROOT / "data/a2_prepared")
    args = parser.parse_args()
    config = read_config()
    frozen = json.loads((args.runs / "frozen.json").read_text())
    if freeze(args.runs, config, signature(args.cache)) != frozen:
        raise ValueError("Frozen protocol changed")
    manifest = json.loads((args.cache / "manifest.json").read_text())
    dataset = load_dataset(args.cache, "test", manifest)
    device = torch.device("cuda")
    image = normalize(dataset[0][0][None], device)
    records = []
    for variant, seed, path in run_paths(args.runs, config):
        model = PetSegmenter(variant).to(device=device, memory_format=torch.channels_last).eval()
        model.load_state_dict(torch.load(path / "best.pt", weights_only=True, map_location=device))
        records.append({
            "variant": variant, "seed": seed, "checkpoint_sha256": sha256(path / "best.pt"),
            **measure(model, image),
        })
    atomic_json(args.runs / "latency.json", {
        "device": torch.cuda.get_device_name(), "torch": torch.__version__, "precision": "bfloat16 autocast",
        "input": list(image.shape), "warmup": 50, "repetitions": 200,
        "sample": manifest["splits"]["test"]["names"][0],
        "scope": "CUDA events around eager forward, with resident input; includes auxiliary boundary output",
        "runs": records,
    })


if __name__ == "__main__":
    main()
