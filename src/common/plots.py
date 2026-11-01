"""Figures are generated only from actual logs/predictions."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from common.data import CLASS_NAMES


def plot_history(history: list[dict], directory: Path) -> None:
    if not history:
        return
    directory.mkdir(parents=True, exist_ok=True)
    epochs = [r['epoch'] for r in history]
    for metric, ylabel in [('ce', 'Cross-entropy'), ('accuracy', 'Accuracy')]:
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        ax.plot(epochs, [r['train_' + metric] for r in history], label='Train')
        ax.plot(epochs, [r['validation_' + metric] for r in history], label='Validation')
        ax.set(xlabel='Epoch', ylabel=ylabel)
        ax.grid(alpha=0.25); ax.legend(); fig.tight_layout()
        fig.savefig(directory / f'{metric}.png', dpi=150); plt.close(fig)


def plot_confusion(matrix: list[list[int]], path: Path) -> None:
    cm = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    image = ax.imshow(cm)
    ax.set_xticks(range(10), CLASS_NAMES, rotation=50, ha='right', fontsize=8)
    ax.set_yticks(range(10), CLASS_NAMES, fontsize=8)
    ax.set(xlabel='Predicted class', ylabel='True class')
    fig.colorbar(image, ax=ax, label='Image count')
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_mistakes(examples: list[dict], path: Path) -> None:
    if not examples:
        # Never show arbitrary images as mistakes when there are no errors.
        return
    examples = sorted(examples, key=lambda e: e['confidence'], reverse=True)[:8]
    width, height, columns = 210, 200, 4
    rows = (len(examples) + columns - 1) // columns
    canvas = Image.new('RGB', (width * columns, height * rows), 'white')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=13)
    for i, ex in enumerate(examples):
        x, y = (i % columns) * width, (i // columns) * height
        image = Image.fromarray(ex['image'].astype(np.uint8)).convert('RGB').resize((128, 128), Image.Resampling.NEAREST)
        canvas.paste(image, (x + 40, y + 4))
        lines = [f"True: {CLASS_NAMES[ex['target']]}", f"Pred: {CLASS_NAMES[ex['prediction']]}",
                 f"p={ex['confidence']:.3f}, id={ex['index']}"]
        draw.multiline_text((x + 5, y + 138), '\n'.join(lines), fill='black', font=font, spacing=2)
    canvas.save(path)
