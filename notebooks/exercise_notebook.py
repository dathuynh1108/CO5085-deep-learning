"""Shared artifact validation and plots for the E1–E3 notebooks."""

from __future__ import annotations

import csv
import io
import json
import os
import statistics

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Image, Markdown, display

from common.data import CLASS_NAMES
from common.engine import source_fingerprint
from common.runtime import json_hash


def setting(name, default=None):
    return os.environ.get(f'CO5085_EXERCISES_{name}', default)


def flag(name, default=False):
    value = setting(name)
    if value is None:
        return default
    if value not in {'0', '1'}:
        raise ValueError(f'CO5085_EXERCISES_{name} phải là 0 hoặc 1.')
    return value == '1'


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def verified_training_status(directory, config, seed, metadata, workers, deterministic, environment):
    """Determine reuse eligibility with the same signature as common.engine.train."""
    info_file = directory / 'run.json'
    metrics_file = directory / 'metrics.json'
    if not directory.exists() or not any(directory.iterdir()):
        return 'new'
    if (directory / '.train.lock').exists():
        raise RuntimeError(f'Lượt chạy đang khóa: {directory}')
    if not info_file.is_file():
        raise FileNotFoundError(f'Thiếu run.json: {directory}. Chọn OUTPUT_ROOT mới.')
    signature = json_hash(dict(
        experiment=config.to_dict(), seed=seed, data_fingerprint=metadata['fingerprint'],
        source_fingerprint=source_fingerprint(), deterministic=deterministic, num_workers=workers,
    ))
    info = read_json(info_file)
    if info['signature'] != signature:
        raise ValueError(f'Cấu hình, seed, dữ liệu hoặc mã nguồn không khớp: {directory}. Chọn OUTPUT_ROOT mới.')
    if not metrics_file.is_file():
        return 'incomplete'
    metrics = read_json(metrics_file)
    if (metrics.get('status') != 'trained' or metrics.get('completed_epochs') != config.epochs
            or metrics['signature'] != signature):
        raise ValueError(f'Không thể xác minh huấn luyện hoàn tất: {directory}.')
    for key in ('python', 'torch', 'cuda_runtime', 'cudnn', 'device', 'device_name', 'torch_num_threads'):
        if info['environment'].get(key) != environment.get(key):
            raise ValueError(f'Môi trường huấn luyện khác tại {key}: {directory}. Chọn OUTPUT_ROOT mới.')
    return 'trained'


def show_figure(figure):
    # Explicit PNG output works in headless kernels as well as interactive ones.
    with io.BytesIO() as buffer:
        figure.savefig(buffer, format='png', dpi=110, bbox_inches='tight')
        display(Image(data=buffer.getvalue()))
    plt.close(figure)


def show_results(results):
    lines = [
        '| Cấu hình | Seed | Tham số | Accuracy TB ± SD (%) | Macro-F1 TB ± SD | Giây train/epoch (trung vị) |',
        '| --- | --- | ---: | ---: | ---: | ---: |',
    ]
    for model in dict.fromkeys(row['experiment'] for row in results):
        rows = [row for row in results if row['experiment'] == model]

        def mean_sd(key, scale=1):
            values = [row[key] * scale for row in rows]
            mean = statistics.mean(values)
            return f'{mean:.3f} ± {statistics.stdev(values):.3f}' if len(values) > 1 else f'{mean:.3f}'

        seconds = statistics.median(row['median_train_seconds'] for row in rows)
        seeds = ', '.join(str(row['seed']) for row in rows)
        lines.append(f"| {model} | {seeds} | {rows[0]['n_parameters']:,} | "
                     f"{mean_sd('accuracy', 100)} | {mean_sd('macro_f1')} | {seconds:.2f} |")
    display(Markdown('\n'.join(lines)))


def show_comparison(configs, directories, evaluations, representative_seed, report_dir):
    """Plot actual validation curves and row-normalized confusion for one seed."""
    report_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for config in configs:
        directory = directories[(config.id, representative_seed)]
        with (directory / 'history.csv').open(encoding='utf-8', newline='') as handle:
            history = list(csv.DictReader(handle))
        epochs = [int(row['epoch']) for row in history]
        for ax, metric, ylabel in zip(axes, ('ce', 'accuracy'), ('Cross-entropy', 'Accuracy')):
            ax.plot(epochs, [float(row['validation_' + metric]) for row in history],
                    label=config.label, marker='.' if len(epochs) == 1 else None)
            ax.set(xlabel='Epoch', ylabel=ylabel)
            ax.grid(alpha=.25)
    fig.suptitle(f'Đường cong trên tập validation — seed {representative_seed}')
    axes[1].legend(fontsize=7, loc='best')
    fig.tight_layout()
    fig.savefig(report_dir / 'validation_curves.png', dpi=150)
    show_figure(fig)

    columns = min(3, len(configs))
    rows = (len(configs) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(4.1 * columns + 1, 3.7 * rows),
                             squeeze=False, layout='constrained')
    for ax in axes.flat:
        ax.set_visible(False)
    active_axes = []
    for ax, config in zip(axes.flat, configs):
        ax.set_visible(True)
        active_axes.append(ax)
        result = evaluations[(config.id, representative_seed)]
        counts = np.asarray(result['confusion_matrix'])
        total = counts.sum(axis=1, keepdims=True)
        rates = np.divide(counts, total, out=np.zeros(counts.shape, float), where=total != 0)
        image = ax.imshow(rates, vmin=0, vmax=1, cmap='Blues')
        ax.set(title=config.label, xlabel='Lớp dự đoán', ylabel='Lớp thực')
        ax.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=60, ha='right', fontsize=7)
        ax.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES, fontsize=7)
    fig.suptitle(f"Ma trận nhầm lẫn ({result['split']}) — seed {representative_seed}")
    fig.colorbar(image, ax=active_axes, shrink=.8, label='Tỷ lệ theo lớp thực')
    fig.savefig(report_dir / 'confusion.png', dpi=150)
    show_figure(fig)
