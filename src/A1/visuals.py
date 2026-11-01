"""Figures derived only from measured history and predictions."""
import os
import tempfile
from pathlib import Path
import numpy as np


def pyplot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def save_figure(figure, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(suffix='.png', dir=path.parent)
    os.close(fd)
    try:
        figure.savefig(temporary, dpi=150, bbox_inches='tight')
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    pyplot().close(figure)


def curves(run, history):
    if not history:
        return
    plt = pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, metric in zip(axes, ['ce', 'accuracy']):
        for split in ['train', 'val']:
            axis.plot([row['epoch'] for row in history], [row[f'{split}_{metric}'] for row in history], label=split)
        axis.set(xlabel='Epoch', ylabel=metric)
        axis.legend()
    save_figure(fig, run / 'curves.png')


def evaluation_figures(run, dataset, predictions, confusion):
    plt = pyplot()
    fig, axis = plt.subplots(figsize=(10, 9))
    image = axis.imshow(confusion, cmap='Blues')
    axis.set(xlabel='Predicted fine class (0–99)', ylabel='True fine class (0–99)')
    fig.colorbar(image, ax=axis)
    save_figure(fig, run / 'confusion_test.png')
    errors = sorted((r for r in predictions if r['target'] != r['prediction']),
                    key=lambda r: (-r['confidence'], r['index']))[:8]
    if not errors:
        return
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for axis in axes.flat:
        axis.axis('off')
    for axis, row in zip(axes.flat, errors):
        axis.imshow(dataset.data[row['index']])
        axis.set_title(f"#{row['index']} true: {dataset.classes[row['target']]}\npred: {dataset.classes[row['prediction']]} ({row['confidence']:.2f})", fontsize=8)
    save_figure(fig, run / 'errors_test.png')
