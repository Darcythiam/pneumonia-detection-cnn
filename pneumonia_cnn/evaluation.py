"""Threshold selection on validation data and evaluation at a frozen threshold."""

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             precision_score, recall_score, roc_auc_score, roc_curve,
                             brier_score_loss)


def _validate_inputs(labels, probabilities):
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if y.size == 0 or y.size != p.size or set(y.tolist()) != {0, 1}:
        raise ValueError("Expected equal-length labels and probabilities with both classes")
    if not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("Predicted probabilities must be finite values in [0, 1]")
    return y, p


def choose_threshold(validation_labels, validation_probabilities) -> float:
    """Maximize Youden's J using validation data only."""
    y, p = _validate_inputs(validation_labels, validation_probabilities)
    fpr, tpr, thresholds = roc_curve(y, p)
    valid = np.isfinite(thresholds)
    best = np.argmax(np.where(valid, tpr - fpr, -np.inf))
    return float(thresholds[best])


def evaluate(labels, probabilities, threshold: float) -> dict:
    y, p = _validate_inputs(labels, probabilities)
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    prediction = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    bins = np.minimum((p * 10).astype(int), 9)
    reliability = []
    for index in range(10):
        mask = bins == index
        if np.any(mask):
            reliability.append({"bin": index, "count": int(mask.sum()),
                                "mean_probability": float(p[mask].mean()),
                                "observed_frequency": float(y[mask].mean())})
    ece = sum(row["count"] / len(y) * abs(row["mean_probability"] -
              row["observed_frequency"]) for row in reliability)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, prediction)),
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "brier_score": float(brier_score_loss(y, p)),
        "ece_10_bins": float(ece),
        "reliability_bins": reliability,
        "pneumonia_precision": float(precision_score(y, prediction, zero_division=0)),
        "pneumonia_recall": float(recall_score(y, prediction, zero_division=0)),
        "normal_specificity": float(tn / (tn + fp)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "support": {"normal": int(tn + fp), "pneumonia": int(fn + tp)},
    }
