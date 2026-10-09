"""Small, dependency-light classification metrics."""
from __future__ import annotations
import numpy as np


def classification_metrics(targets: np.ndarray, predictions: np.ndarray,
                           num_classes: int = 10) -> dict:
    y, p = np.asarray(targets), np.asarray(predictions)
    if y.ndim != 1 or p.shape != y.shape or y.size == 0:
        raise ValueError('Expected equal, nonempty one-dimensional label arrays.')
    if not (np.issubdtype(y.dtype, np.integer) and np.issubdtype(p.dtype, np.integer)):
        raise ValueError('Class labels must be integers.')
    if num_classes < 1 or min(y.min(), p.min()) < 0 or max(y.max(), p.max()) >= num_classes:
        raise ValueError('Class label outside the configured range.')
    cm = np.bincount(num_classes * y + p, minlength=num_classes ** 2).reshape(num_classes, num_classes)
    tp, support, predicted = np.diag(cm), cm.sum(1), cm.sum(0)
    precision = np.divide(tp, predicted, out=np.zeros(num_classes), where=predicted != 0)
    recall = np.divide(tp, support, out=np.zeros(num_classes), where=support != 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(num_classes), where=(precision + recall) != 0)
    return {
        'accuracy': float(tp.sum() / y.size), 'macro_f1': float(f1.mean()),
        'n_samples': int(y.size), 'confusion_matrix': cm.tolist(),
        'per_class': [dict(class_id=i, precision=float(precision[i]), recall=float(recall[i]),
                           f1=float(f1[i]), support=int(support[i])) for i in range(num_classes)],
    }
