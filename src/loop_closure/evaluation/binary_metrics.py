"""Métricas binárias para avaliação de fechamento de ciclo.

O protocolo assume scores orientados de modo que valores maiores indiquem
maior evidência de fechamento de ciclo.

O limiar operacional é calibrado exclusivamente no conjunto de calibração
pela maximização do F1. Teste e stress devem reutilizar o mesmo limiar.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _validate_binary_inputs(
    labels: np.ndarray,
    scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(labels)
    y_score = np.asarray(scores, dtype=np.float64)

    if y_true.ndim != 1:
        raise ValueError(
            "labels deve ser um vetor unidimensional."
        )

    if y_score.ndim != 1:
        raise ValueError(
            "scores deve ser um vetor unidimensional."
        )

    if len(y_true) == 0:
        raise ValueError(
            "labels e scores não podem estar vazios."
        )

    if len(y_true) != len(y_score):
        raise ValueError(
            "labels e scores devem possuir o mesmo tamanho."
        )

    if not np.isfinite(y_score).all():
        raise ValueError(
            "scores contém valor não finito."
        )

    try:
        y_true = y_true.astype(
            np.int64,
            copy=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "labels deve conter apenas 0 e 1."
        ) from error

    unique_labels = set(
        int(value)
        for value in np.unique(y_true)
    )

    if not unique_labels.issubset({0, 1}):
        raise ValueError(
            "labels deve conter apenas 0 e 1."
        )

    if unique_labels != {0, 1}:
        raise ValueError(
            "A avaliação exige exemplos positivos e negativos."
        )

    return y_true, y_score


def _binary_clf_curve(
    labels: np.ndarray,
    scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true, y_score = _validate_binary_inputs(
        labels,
        scores,
    )

    descending_order = np.argsort(
        y_score,
        kind="mergesort",
    )[::-1]

    sorted_scores = y_score[descending_order]
    sorted_labels = y_true[descending_order]

    distinct_indices = np.where(
        np.diff(sorted_scores)
    )[0]

    threshold_indices = np.concatenate(
        (
            distinct_indices,
            np.asarray(
                [len(sorted_labels) - 1],
                dtype=np.int64,
            ),
        )
    )

    true_positives = np.cumsum(
        sorted_labels,
        dtype=np.int64,
    )[threshold_indices]

    false_positives = (
        1
        + threshold_indices
        - true_positives
    )

    thresholds = sorted_scores[
        threshold_indices
    ]

    return (
        false_posititives_as_float(
            false_positives
        ),
        true_positives.astype(
            np.float64,
            copy=False,
        ),
        thresholds,
    )


def false_posititives_as_float(
    values: np.ndarray,
) -> np.ndarray:
    """Normaliza a contagem de falsos positivos para float64."""
    return np.asarray(
        values,
        dtype=np.float64,
    )


def roc_curve_binary(
    labels: np.ndarray,
    scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    false_positives, true_positives, thresholds = (
        _binary_clf_curve(
            labels,
            scores,
        )
    )

    negative_count = false_positives[-1]
    positive_count = true_positives[-1]

    fpr = np.concatenate(
        (
            np.asarray([0.0]),
            false_positives / negative_count,
        )
    )

    tpr = np.concatenate(
        (
            np.asarray([0.0]),
            true_positives / positive_count,
        )
    )

    curve_thresholds = np.concatenate(
        (
            np.asarray([np.inf]),
            thresholds,
        )
    )

    return fpr, tpr, curve_thresholds


def precision_recall_curve_binary(
    labels: np.ndarray,
    scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    false_positives, true_positives, thresholds = (
        _binary_clf_curve(
            labels,
            scores,
        )
    )

    predicted_positives = (
        true_positives
        + false_positives
    )

    precision = np.divide(
        true_positives,
        predicted_positives,
        out=np.zeros_like(
            true_positives,
            dtype=np.float64,
        ),
        where=predicted_positives != 0,
    )

    recall = (
        true_positives
        / true_positives[-1]
    )

    # Formato compatível com a convenção usual:
    # recall decrescente e ponto final (precision=1, recall=0).
    return (
        np.concatenate(
            (
                precision[::-1],
                np.asarray([1.0]),
            )
        ),
        np.concatenate(
            (
                recall[::-1],
                np.asarray([0.0]),
            )
        ),
        thresholds[::-1],
    )


def _trapezoid(
    y: np.ndarray,
    x: np.ndarray,
) -> float:
    """Integra numericamente usando a API disponível do NumPy."""
    trapezoid_function = getattr(
        np,
        "trapezoid",
        None,
    )

    if trapezoid_function is not None:
        return float(
            trapezoid_function(y, x)
        )

    # Compatibilidade com versões antigas do NumPy.
    trapz_function = getattr(
        np,
        "trapz",
        None,
    )

    if trapz_function is None:
        raise RuntimeError(
            "O NumPy não fornece trapezoid nem trapz."
        )

    return float(
        trapz_function(y, x)
    )


def compute_ranking_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
) -> dict[str, float | int]:
    y_true, y_score = _validate_binary_inputs(
        labels,
        scores,
    )

    fpr, tpr, _ = roc_curve_binary(
        y_true,
        y_score,
    )

    precision, recall, _ = (
        precision_recall_curve_binary(
            y_true,
            y_score,
        )
    )

    roc_auc = _trapezoid(
        tpr,
        fpr,
    )

    pr_auc = _trapezoid(
        precision[::-1],
        recall[::-1],
    )

    average_precision = float(
        -np.sum(
            np.diff(recall)
            * precision[:-1]
        )
    )

    positive_count = int(
        np.sum(y_true == 1)
    )
    negative_count = int(
        np.sum(y_true == 0)
    )

    return {
        "sample_count": int(len(y_true)),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "positive_prevalence": (
            positive_count / len(y_true)
        ),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "average_precision": average_precision,
    }


def calibrate_threshold_max_f1(
    labels: np.ndarray,
    scores: np.ndarray,
) -> dict[str, float | str]:
    y_true, y_score = _validate_binary_inputs(
        labels,
        scores,
    )

    precision, recall, thresholds = (
        precision_recall_curve_binary(
            y_true,
            y_score,
        )
    )

    candidate_precision = precision[:-1]
    candidate_recall = recall[:-1]

    denominator = (
        candidate_precision
        + candidate_recall
    )

    f1_values = np.divide(
        2.0
        * candidate_precision
        * candidate_recall,
        denominator,
        out=np.zeros_like(
            denominator,
            dtype=np.float64,
        ),
        where=denominator != 0,
    )

    maximum_f1 = float(
        np.max(f1_values)
    )

    candidate_indices = np.flatnonzero(
        np.isclose(
            f1_values,
            maximum_f1,
            rtol=1e-12,
            atol=1e-12,
        )
    )

    # Desempate conservador:
    # 1. maior precisão;
    # 2. maior limiar.
    maximum_precision = float(
        np.max(
            candidate_precision[
                candidate_indices
            ]
        )
    )

    precision_tied_indices = (
        candidate_indices[
            np.isclose(
                candidate_precision[
                    candidate_indices
                ],
                maximum_precision,
                rtol=1e-12,
                atol=1e-12,
            )
        ]
    )

    best_index = int(
        precision_tied_indices[
            np.argmax(
                thresholds[
                    precision_tied_indices
                ]
            )
        ]
    )

    return {
        "criterion": "maximum_f1",
        "tie_breaker": (
            "higher_precision_then_higher_threshold"
        ),
        "threshold": float(
            thresholds[best_index]
        ),
        "precision": float(
            candidate_precision[best_index]
        ),
        "recall": float(
            candidate_recall[best_index]
        ),
        "f1": float(
            f1_values[best_index]
        ),
    }


def evaluate_at_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    y_true, y_score = _validate_binary_inputs(
        labels,
        scores,
    )

    normalized_threshold = float(
        threshold
    )

    if not math.isfinite(
        normalized_threshold
    ):
        raise ValueError(
            "threshold deve ser finito."
        )

    predictions = (
        y_score >= normalized_threshold
    ).astype(np.int64)

    true_positive = int(
        np.sum(
            (y_true == 1)
            & (predictions == 1)
        )
    )
    false_positive = int(
        np.sum(
            (y_true == 0)
            & (predictions == 1)
        )
    )
    true_negative = int(
        np.sum(
            (y_true == 0)
            & (predictions == 0)
        )
    )
    false_negative = int(
        np.sum(
            (y_true == 1)
            & (predictions == 0)
        )
    )

    def safe_divide(
        numerator: float,
        denominator: float,
    ) -> float:
        if denominator == 0:
            return 0.0
        return numerator / denominator

    precision = safe_divide(
        true_positive,
        true_positive + false_positive,
    )
    recall = safe_divide(
        true_positive,
        true_positive + false_negative,
    )
    specificity = safe_divide(
        true_negative,
        true_negative + false_positive,
    )
    accuracy = safe_divide(
        true_positive + true_negative,
        len(y_true),
    )

    f1 = safe_divide(
        2.0 * precision * recall,
        precision + recall,
    )

    balanced_accuracy = (
        recall + specificity
    ) / 2.0

    mcc_numerator = (
        true_positive * true_negative
        - false_positive * false_negative
    )

    mcc_denominator = math.sqrt(
        (true_positive + false_positive)
        * (true_positive + false_negative)
        * (true_negative + false_positive)
        * (true_negative + false_negative)
    )

    matthews_correlation = safe_divide(
        mcc_numerator,
        mcc_denominator,
    )

    return {
        "threshold": normalized_threshold,
        "prediction_rule": (
            "positive_if_score_greater_than_or_equal_to_threshold"
        ),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "matthews_correlation_coefficient": (
            matthews_correlation
        ),
    }


def evaluate_binary_scores(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    threshold: float | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "ranking": compute_ranking_metrics(
            labels,
            scores,
        )
    }

    if threshold is not None:
        result["operating_point"] = (
            evaluate_at_threshold(
                labels,
                scores,
                threshold,
            )
        )

    return result
