"""Verify frozen A2 evidence and export the report tables and figures."""

import argparse
import csv
import json
from pathlib import Path
import shutil
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import torch

from A2.data import ROOT, read_config, sha256
from A2.models import PetSegmenter
from A2.run import (
    confusion_by_image, freeze, load_dataset, normalize, run_paths, scores, signature,
)
from common.runtime import atomic_json, choose_device

LABELS = ("Vật thể", "Nền", "Dải chuyển tiếp")
NAMES = {"fixed": "Trộn cố định", "guided": "Trộn theo biên"}
COLORS = {"fixed": "#406680", "guided": "#187456"}
MASK_COLORS = ListedColormap(["#2e8a65", "#233c52", "#f2be51"])
MASK_COLORS.set_bad("#ecebe6")
METRICS = ("miou", "macro_dice", "foreground_iou", "background_iou", "transition_iou")


def stat(values):
    values = [float(value) for value in values]
    return {"mean": statistics.mean(values), "sd": statistics.stdev(values), "values": values}


def percentage(value, digits=2):
    return f"{100 * value:.{digits}f}".replace(".", "{,}")


def tex_stat(result, scale=100, digits=2):
    mean = f"{scale * result['mean']:.{digits}f}".replace(".", "{,}")
    sd = f"{scale * result['sd']:.{digits}f}".replace(".", "{,}")
    return f"${mean} \\pm {sd}$"


def read_results(root, destination, config, names):
    rows = {}
    for variant, seed, path in run_paths(root, config):
        run = json.loads((path / "run.json").read_text())
        test = json.loads((path / "test.json").read_text())
        if (test["checkpoint_sha256"] != run["checkpoint_sha256"]
                or test["per_image_sha256"] != sha256(path / "test_per_image.csv")
                or test["samples"] != len(names)):
            raise ValueError(f"Test evidence changed for {path}")
        with (path / "test_per_image.csv").open() as stream:
            pixels = list(csv.DictReader(stream))
        if [record["name"] for record in pixels] != names:
            raise ValueError(f"Evaluation order differs for {path}")
        matrices = np.array([[int(record[f"c{i}{j}"]) for i in range(3) for j in range(3)]
                             for record in pixels]).reshape(-1, 3, 3)
        aggregate = matrices.sum(0)
        if aggregate.tolist() != test["confusion"] or any(
            test[key] != value for key, value in scores(aggregate).items()
        ):
            raise ValueError(f"Metrics disagree with pixel counts for {path}")
        if rows and not np.array_equal(matrices.sum(2), next(iter(rows.values()))["matrices"].sum(2)):
            raise ValueError("Ground truth varies across test evaluations")
        evidence = destination / "evidence" / path.relative_to(root)
        evidence.mkdir(parents=True, exist_ok=True)
        for name in ("run.json", "history.csv", "test.json", "test_per_image.csv"):
            shutil.copy2(path / name, evidence / name)
        with (path / "history.csv").open() as stream:
            history = list(csv.DictReader(stream))
        rows[variant, seed] = {
            "run": run, "test": test, "matrices": matrices, "history": history,
            "measures": {
                **{key: test[key] for key in ("miou", "macro_dice")},
                **dict(zip(METRICS[2:], test["iou"], strict=True)),
                "validation_miou": run["best_val_miou"],
                "train_seconds": run["total_train_seconds"],
                "peak_mib": run["peak_allocated_mib"],
            },
        }
    return rows


def summarize(rows, config, manifest):
    seeds = config["seeds"]
    variants = config["variants"]
    groups = {}
    for variant in variants:
        selected = [rows[variant, seed] for seed in seeds]
        metrics = selected[0]["measures"]
        groups[variant] = {
            key: stat([row["measures"][key] for row in selected])
            for key in metrics if metrics[key] is not None
        } | {
            "params": selected[0]["run"]["params"],
            "selected_epochs": [row["run"]["best_epoch"] for row in selected],
            "edge_score_by_target": [
                stat([row["test"]["edge_score_by_target"][label] for row in selected])
                for label in range(3)
            ],
        }
    paired = {
        key: stat([rows["guided", seed]["measures"][key] - rows["fixed", seed]["measures"][key]
                   for seed in seeds]) for key in METRICS
    }
    support = next(iter(rows.values()))["matrices"].sum(2)
    shares = support[:, 0] / support.sum(1)
    order = np.argsort(shares, kind="stable")
    strata = []
    for members in np.array_split(order, 3):
        strata.append({
            "count": len(members),
            "foreground_share_range": [float(shares[members].min()), float(shares[members].max())],
            "transition_iou": {
                variant: stat([scores(rows[variant, seed]["matrices"][members].sum(0))["iou"][2]
                               for seed in seeds]) for variant in variants
            },
        })
    return {
        "train_runs": len(rows), "test_runs": len(rows), "seeds": seeds,
        "splits": {name: data["size"] for name, data in manifest["splits"].items()},
        "groups": groups, "guided_minus_fixed": paired, "foreground_strata": strata,
        "runtime": next(iter(rows.values()))["run"]["runtime"],
    }


def plot_data(cache, manifest, output):
    stats = manifest["splits"]["train"]["eda"]
    counts = stats["valid_pixel_counts"]
    target = np.load(cache / "train_masks.npy", mmap_mode="r")
    shares = (target == 2).sum((1, 2)) / (target < 3).sum((1, 2))
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 2.6), layout="constrained")
    axes[0].bar(LABELS, [100 * count / sum(counts) for count in counts], color=MASK_COLORS.colors)
    axes[0].set_ylabel("Tỷ lệ điểm ảnh (%)")
    axes[1].hist(100 * shares, bins=25, color=COLORS["guided"])
    axes[1].set(xlabel="Tỷ lệ dải chuyển tiếp trên từng ảnh (%)", ylabel="Số ảnh")
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_curves(rows, config, output):
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3), layout="constrained")
    for ax, key, title in zip(
        axes, ("val_miou", "val_transition_iou"), ("mIoU xác thực (%)", "IoU dải chuyển tiếp (%)"),
        strict=True,
    ):
        for variant in config["variants"]:
            values = 100 * np.array([[float(h[key]) for h in rows[variant, seed]["history"]]
                                     for seed in config["seeds"]])
            epoch = np.arange(1, config["epochs"] + 1)
            mean, sd = values.mean(0), values.std(0, ddof=1)
            ax.plot(epoch, mean, color=COLORS[variant], label=NAMES[variant])
            ax.fill_between(epoch, mean - sd, mean + sd, color=COLORS[variant], alpha=0.14)
        ax.set(xlabel="Epoch", ylabel=title, xticks=[1, 3, 6, 9, 12])
        ax.grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False, loc="lower right")
    fig.savefig(output, dpi=160)
    plt.close(fig)


def select_examples(rows, seed):
    fixed, guided = (rows[variant, seed]["matrices"] for variant in ("fixed", "guided"))
    support = fixed.sum(2)
    candidates = np.flatnonzero(support[:, 2] >= 0.02 * support.sum(1))
    difference = np.array([scores(guided[i])["iou"][2] - scores(fixed[i])["iou"][2]
                           for i in candidates])
    ordered = candidates[np.argsort(difference, kind="stable")]
    return [int(ordered[-1]), int(ordered[len(ordered) // 2]), int(ordered[0])]


@torch.inference_mode()
def plot_examples(root, cache, manifest, rows, config, device, output):
    seed = config["seeds"][0]
    indices = select_examples(rows, seed)
    test = load_dataset(cache, "test", manifest)
    names = manifest["splits"]["test"]["names"]
    example_records, predictions = [], {}
    fig, axes = plt.subplots(3, 5, figsize=(10.4, 6.0), layout="constrained")
    for variant in config["variants"]:
        path = root / variant / f"seed_{seed}" / "best.pt"
        model = PetSegmenter(variant).to(device=device, memory_format=torch.channels_last).eval()
        model.load_state_dict(torch.load(path, weights_only=True, map_location=device))
        for index in indices:
            start = index // config["batch_size"] * config["batch_size"]
            stop = min(start + config["batch_size"], len(test))
            batch = torch.stack([test[i][0] for i in range(start, stop)])
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits, edge = model(normalize(batch, device))
            offset = index - start
            prediction = logits[offset].argmax(0).cpu().numpy()
            matrix = confusion_by_image(prediction[None], test.masks[index][None])[0]
            if not np.array_equal(matrix, rows[variant, seed]["matrices"][index]):
                raise ValueError(f"Example {names[index]} differs from saved evidence. Use the evaluation device.")
            predictions[variant, index] = prediction, edge[offset].float().sigmoid().cpu().numpy()
    for row, index in enumerate(indices):
        target = test.masks[index]
        fixed, _ = predictions["fixed", index]
        guided, gate = predictions["guided", index]
        axes[row, 0].imshow(test.images[index])
        for col, mask in enumerate((target, fixed, guided), 1):
            axes[row, col].imshow(np.ma.masked_where(target == 255, mask), cmap=MASK_COLORS,
                                  vmin=0, vmax=2, interpolation="nearest")
        axes[row, 4].imshow(np.ma.masked_where(target == 255, gate), cmap="magma", vmin=0, vmax=1)
        iou = {variant: scores(rows[variant, seed]["matrices"][index])["iou"][2]
               for variant in config["variants"]}
        axes[row, 0].set_xlabel(names[index], fontsize=8)
        for col, variant in enumerate(config["variants"], 2):
            axes[row, col].set_xlabel(f"IoU chuyển tiếp {iou[variant]:.1%}", fontsize=8)
        for ax in axes[row]:
            ax.set(xticks=[], yticks=[])
            for spine in ax.spines.values():
                spine.set_visible(False)
        example_records.append({"name": names[index], "seed": seed, "transition_iou": iou})
    for ax, label in zip(axes[0], ("Ảnh", "Trimap", "Trộn cố định", "Trộn theo biên", "Hệ số trộn"), strict=True):
        ax.set_title(label, fontsize=10)
    fig.savefig(output, dpi=165)
    plt.close(fig)
    return example_records


def write_tables(destination, summary, manifest):
    groups = summary["groups"]
    table = [
        r"\begin{table}[htbp]\centering\small",
        r"\begin{tabular}{@{}lrrrrr@{}}\toprule",
        r"Biến thể & mIoU & Dice & IoU vật thể & IoU nền & IoU chuyển tiếp\\\midrule",
    ]
    for variant, result in groups.items():
        table.append(" & ".join([NAMES[variant]] + [tex_stat(result[k]) for k in METRICS]) + r"\\")
    table.extend([
        r"\bottomrule\end{tabular}",
        r"\caption{Kết quả kiểm tra theo phần trăm, trình bày trung bình và độ lệch chuẩn mẫu qua ba hạt giống.}",
        r"\label{tab:results}\end{table}", "",
        r"\begin{table}[htbp]\centering\small",
        r"\begin{tabular}{@{}lrrrr@{}}\toprule",
        r"Biến thể & mIoU xác thực (\%) & Epoch được chọn & Thời gian (s) & Đỉnh (MiB)\\\midrule",
    ])
    for variant, result in groups.items():
        table.append(" & ".join([
            NAMES[variant], tex_stat(result["validation_miou"]), ", ".join(map(str, result["selected_epochs"])),
            tex_stat(result["train_seconds"], scale=1, digits=1),
            tex_stat(result["peak_mib"], scale=1, digits=1) if "peak_mib" in result else "--",
        ]) + r"\\")
    table.extend([
        r"\bottomrule\end{tabular}",
        r"\caption{Chi phí của 12 epoch và kết quả xác thực. Epoch theo thứ tự hạt giống 42, 43, 44. "
        r"Thời gian gồm tải batch và cập nhật trọng số. Mức cấp phát GPU cực đại bao gồm lượt xác thực.}",
        r"\label{tab:cost}\end{table}", "",
    ])
    (destination / "results.tex").write_text("\n".join(table))
    eda = manifest["splits"]["train"]["eda"]
    counts = eda["valid_pixel_counts"]
    total = sum(counts)
    (destination / "data_analysis.tex").write_text(
        f"Trung vị chiều rộng và chiều cao ảnh gốc lần lượt là {eda['width_quartiles'][2]:.0f} "
        f"và {eda['height_quartiles'][2]:.0f} pixel. Trên lưới dùng để huấn luyện, vật thể chiếm "
        f"{percentage(counts[0] / total)}\\% điểm ảnh có nhãn, nền chiếm "
        f"{percentage(counts[1] / total)}\\% và dải chuyển tiếp chiếm "
        f"{percentage(counts[2] / total)}\\%. Tỷ lệ dải chuyển tiếp theo từng ảnh có trung vị "
        f"{percentage(eda['boundary_share_quartiles'][2])}\\% và khoảng tứ phân vị "
        f"{percentage(eda['boundary_share_quartiles'][1])}--{percentage(eda['boundary_share_quartiles'][3])}\\%. "
        "Sự mất cân bằng này là lý do dùng BCE cân bằng cho đầu dự đoán biên.\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=ROOT / "runs/assignment2")
    parser.add_argument("--cache", type=Path, default=ROOT / "data/a2_prepared")
    parser.add_argument("--report-root", type=Path, default=ROOT / "reports/assignment2")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    config = read_config()
    frozen = json.loads((args.runs / "frozen.json").read_text())
    if freeze(args.runs, config, signature(args.cache)) != frozen:
        raise ValueError("Frozen protocol changed")
    manifest = json.loads((args.cache / "manifest.json").read_text())
    destination = args.report_root / "generated"
    figures = destination / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    names = manifest["splits"]["test"]["names"]
    rows = read_results(args.runs, destination, config, names)
    latency_path = args.runs / "latency.json"
    if latency_path.is_file():
        latency = json.loads(latency_path.read_text())
        measured = {(run["variant"], run["seed"]): run["checkpoint_sha256"] for run in latency["runs"]}
        expected = {key: row["run"]["checkpoint_sha256"] for key, row in rows.items()}
        if measured != expected:
            raise ValueError("Latency benchmark uses different checkpoints")
        shutil.copy2(latency_path, destination / "latency.json")
    summary = summarize(rows, config, manifest)
    plot_data(args.cache, manifest, figures / "dataset.png")
    plot_curves(rows, config, figures / "curves.png")
    summary["examples"] = plot_examples(args.runs, args.cache, manifest, rows, config,
                                        choose_device(args.device), figures / "examples.png")
    write_tables(destination, summary, manifest)
    atomic_json(destination / "summary.json", summary)
    shutil.copy2(args.cache / "manifest.json", destination / "data_manifest.json")
    shutil.copy2(args.runs / "frozen.json", destination / "frozen.json")
    atomic_json(destination / "manifest.json", {
        "trained_runs": summary["train_runs"], "tested_runs": summary["test_runs"],
        "summary_sha256": sha256(destination / "summary.json"),
        "frozen_sha256": sha256(destination / "frozen.json"),
    })


if __name__ == "__main__":
    main()
