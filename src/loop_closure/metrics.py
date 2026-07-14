"""Metrics for imbalanced loop-closure detection experiments.

The original notebooks computed ROC-AUC from binary predictions. This module
requires continuous scores and keeps threshold selection separate from final
held-out test evaluation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


@dataclass(frozen=True)
class EvaluationResult:
    threshold: float
    average_precision: float
    roc_auc: float
    max_f1: float
    max_f1_threshold: float
    precision: float
    recall: float
    f1: float
    accuracy: float
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int
    prevalence: float
    sample_count: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _validate_arrays(y_true: Iterable[int], scores: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(list(y_true), dtype=int)
    s = np.asarray(list(scores), dtype=float)

    if y.ndim != 1 or s.ndim != 1:
        raise ValueError("y_true and scores must be one-dimensional")
    if y.size == 0:
        raise ValueError("At least one sample is required")
    if y.size != s.size:
        raise ValueError("y_true and scores must have equal length")
    if not np.isfinite(s).all():
        raise ValueError("scores contain NaN or infinite values")
    labels = set(np.unique(y).tolist())
    if not labels.issubset({0, 1}) or labels != {0, 1}:
        raise ValueError("y_true must contain both binary classes 0 and 1")
    return y, s


def select_threshold_by_f1(y_true: Iterable[int], scores: Iterable[float]) -> tuple[float, float]:
    """Select the score threshold that maximizes F1 on calibration data.

    Larger scores must indicate a stronger loop-closure match. For Euclidean
    distances, use score = -distance before calling this function.
    """
    y, s = _validate_arrays(y_true, scores)
    precision, recall, thresholds = precision_recall_curve(y, s)
    if thresholds.size == 0:
        raise ValueError("Unable to determine a threshold from the supplied scores")

    f1_values = 2.0 * precision[:-1] * recall[:-1] / np.maximum(
        precision[:-1] + recall[:-1], np.finfo(float).eps
    )
    best_index = int(np.nanargmax(f1_values))
    return float(thresholds[best_index]), float(f1_values[best_index])


def evaluate_binary_scores(
    y_true: Iterable[int],
    scores: Iterable[float],
    *,
    threshold: float,
) -> EvaluationResult:
    """Evaluate continuous matching scores at a frozen operational threshold."""
    y, s = _validate_arrays(y_true, scores)
    predictions = (s >= threshold).astype(int)

    best_threshold, best_f1 = select_threshold_by_f1(y, s)
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()

    # Calling roc_curve here validates that continuous scores form a usable curve.
    roc_curve(y, s)

    return EvaluationResult(
        threshold=float(threshold),
        average_precision=float(average_precision_score(y, s)),
        roc_auc=float(roc_auc_score(y, s)),
        max_f1=float(best_f1),
        max_f1_threshold=float(best_threshold),
        precision=float(precision_score(y, predictions, zero_division=0)),
        recall=float(recall_score(y, predictions, zero_division=0)),
        f1=float(f1_score(y, predictions, zero_division=0)),
        accuracy=float(accuracy_score(y, predictions)),
        true_positive=int(tp),
        false_positive=int(fp),
        true_negative=int(tn),
        false_negative=int(fn),
        prevalence=float(np.mean(y)),
        sample_count=int(y.size),
    )
