"""Evaluation metrics for binary matching models."""

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_metrics(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    y_probability: list[float] | np.ndarray | None = None,
) -> dict[str, Any]:
    """Return JSON-serializable binary classification metrics."""
    truth = np.asarray(y_true, dtype=int)
    prediction = np.asarray(y_pred, dtype=int)
    if truth.size == 0 or truth.size != prediction.size:
        raise ValueError("y_true and y_pred must have the same non-zero length.")
    matrix = confusion_matrix(truth, prediction, labels=[0, 1])
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(truth, prediction)),
        "precision": float(precision_score(truth, prediction, zero_division=0)),
        "recall": float(recall_score(truth, prediction, zero_division=0)),
        "f1": float(f1_score(truth, prediction, zero_division=0)),
        "confusion_matrix": matrix.tolist(),
        "evaluation_rows": int(truth.size),
    }
    if y_probability is not None and np.unique(truth).size == 2:
        metrics["roc_auc"] = float(roc_auc_score(truth, np.asarray(y_probability)))
    else:
        metrics["roc_auc"] = None
    return metrics
