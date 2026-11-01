"""Metrics include all 100 classes, with zero F1 for undefined classes."""
import numpy as np


def classification_metrics(targets, predictions, classes=100):
    targets, predictions = np.asarray(targets, dtype=np.int64), np.asarray(predictions, dtype=np.int64)
    if targets.shape != predictions.shape or targets.ndim != 1 or not len(targets):
        raise ValueError('Expected matching nonempty label vectors')
    if np.any((targets < 0) | (targets >= classes) | (predictions < 0) | (predictions >= classes)):
        raise ValueError('Class index out of range')
    confusion = np.bincount(targets * classes + predictions, minlength=classes**2).reshape(classes, classes)
    tp = np.diag(confusion)
    support, predicted = confusion.sum(1), confusion.sum(0)
    precision = np.divide(tp, predicted, out=np.zeros(classes), where=predicted != 0)
    recall = np.divide(tp, support, out=np.zeros(classes), where=support != 0)
    f1 = np.divide(2 * tp, support + predicted, out=np.zeros(classes), where=(support + predicted) != 0)
    return {'accuracy': float(tp.sum() / len(targets)), 'macro_f1': float(f1.mean()),
            'confusion': confusion.tolist(),
            'per_class': [{'class_id': i, 'support': int(support[i]), 'precision': float(precision[i]),
                           'recall': float(recall[i]), 'f1': float(f1[i])} for i in range(classes)]}
