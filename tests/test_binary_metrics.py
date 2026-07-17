from __future__ import annotations

import numpy as np
import pytest

from loop_closure.evaluation.binary_metrics import (
    calibrate_threshold_max_f1,
    compute_ranking_metrics,
    evaluate_at_threshold,
    evaluate_binary_scores,
    precision_recall_curve_binary,
    roc_curve_binary,
)


LABELS = np.asarray(
    [0, 0, 1, 1],
    dtype=np.int64,
)

SCORES = np.asarray(
    [0.1, 0.4, 0.35, 0.8],
    dtype=np.float64,
)


def test_known_ranking_metrics() -> None:
    metrics = compute_ranking_metrics(
        LABELS,
        SCORES,
    )

    assert metrics["sample_count"] == 4
    assert metrics["positive_count"] == 2
    assert metrics["negative_count"] == 2
    assert metrics[
        "positive_prevalence"
    ] == pytest.approx(0.5)

    assert metrics["roc_auc"] == pytest.approx(
        0.75
    )
    assert metrics[
        "average_precision"
    ] == pytest.approx(
        5.0 / 6.0
    )
    assert metrics["pr_auc"] == pytest.approx(
        19.0 / 24.0
    )


def test_roc_curve_has_expected_boundaries() -> None:
    fpr, tpr, thresholds = roc_curve_binary(
        LABELS,
        SCORES,
    )

    assert fpr[0] == pytest.approx(0.0)
    assert tpr[0] == pytest.approx(0.0)
    assert np.isinf(thresholds[0])

    assert fpr[-1] == pytest.approx(1.0)
    assert tpr[-1] == pytest.approx(1.0)


def test_precision_recall_curve_has_expected_boundaries() -> None:
    precision, recall, thresholds = (
        precision_recall_curve_binary(
            LABELS,
            SCORES,
        )
    )

    assert precision[-1] == pytest.approx(
        1.0
    )
    assert recall[-1] == pytest.approx(
        0.0
    )
    assert len(precision) == (
        len(thresholds) + 1
    )
    assert len(recall) == (
        len(thresholds) + 1
    )


def test_threshold_is_calibrated_by_maximum_f1() -> None:
    result = calibrate_threshold_max_f1(
        LABELS,
        SCORES,
    )

    assert result["criterion"] == (
        "maximum_f1"
    )
    assert result["threshold"] == pytest.approx(
        0.35
    )
    assert result["precision"] == pytest.approx(
        2.0 / 3.0
    )
    assert result["recall"] == pytest.approx(
        1.0
    )
    assert result["f1"] == pytest.approx(
        0.8
    )


def test_confusion_matrix_and_metrics() -> None:
    result = evaluate_at_threshold(
        LABELS,
        SCORES,
        threshold=0.35,
    )

    assert result["true_positive"] == 2
    assert result["false_positive"] == 1
    assert result["true_negative"] == 1
    assert result["false_negative"] == 0

    assert result["precision"] == pytest.approx(
        2.0 / 3.0
    )
    assert result["recall"] == pytest.approx(
        1.0
    )
    assert result["specificity"] == pytest.approx(
        0.5
    )
    assert result["f1"] == pytest.approx(
        0.8
    )
    assert result["accuracy"] == pytest.approx(
        0.75
    )
    assert result[
        "balanced_accuracy"
    ] == pytest.approx(
        0.75
    )


@pytest.mark.parametrize(
    "labels",
    [
        np.asarray([0, 2]),
        np.asarray([-1, 1]),
        np.asarray([0, 1, 3]),
    ],
)
def test_rejects_non_binary_labels(
    labels: np.ndarray,
) -> None:
    with pytest.raises(
        ValueError,
        match="0 e 1",
    ):
        compute_ranking_metrics(
            labels,
            np.arange(
                len(labels),
                dtype=np.float64,
            ),
        )


@pytest.mark.parametrize(
    "labels",
    [
        np.asarray([0, 0, 0]),
        np.asarray([1, 1, 1]),
    ],
)
def test_rejects_single_class(
    labels: np.ndarray,
) -> None:
    with pytest.raises(
        ValueError,
        match="positivos e negativos",
    ):
        compute_ranking_metrics(
            labels,
            np.asarray(
                [0.1, 0.2, 0.3],
            ),
        )


def test_rejects_non_finite_scores() -> None:
    with pytest.raises(
        ValueError,
        match="não finito",
    ):
        compute_ranking_metrics(
            LABELS,
            np.asarray(
                [0.1, np.nan, 0.3, 0.4],
            ),
        )


def test_rejects_length_mismatch() -> None:
    with pytest.raises(
        ValueError,
        match="mesmo tamanho",
    ):
        compute_ranking_metrics(
            LABELS,
            np.asarray([0.1, 0.2]),
        )


def test_higher_similarity_scores_improve_ranking() -> None:
    labels = np.asarray(
        [1, 0, 1, 0],
    )
    well_oriented_scores = np.asarray(
        [4.0, 3.0, 2.0, 1.0],
    )
    inverted_scores = (
        -well_oriented_scores
    )

    good = compute_ranking_metrics(
        labels,
        well_oriented_scores,
    )
    bad = compute_ranking_metrics(
        labels,
        inverted_scores,
    )

    assert (
        good["average_precision"]
        > bad["average_precision"]
    )
    assert (
        good["roc_auc"]
        > bad["roc_auc"]
    )


def test_threshold_comparison_is_inclusive() -> None:
    labels = np.asarray(
        [1, 0],
    )
    scores = np.asarray(
        [0.5, 0.4],
    )

    result = evaluate_at_threshold(
        labels,
        scores,
        threshold=0.5,
    )

    assert result["true_positive"] == 1
    assert result["false_negative"] == 0


def test_combined_evaluation_contains_ranking_and_operating_point() -> None:
    result = evaluate_binary_scores(
        LABELS,
        SCORES,
        threshold=0.35,
    )

    assert "ranking" in result
    assert "operating_point" in result
    assert result["ranking"][
        "average_precision"
    ] == pytest.approx(
        5.0 / 6.0
    )
    assert result["operating_point"][
        "f1"
    ] == pytest.approx(
        0.8
    )
