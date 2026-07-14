import numpy as np

from loop_closure.metrics import evaluate_binary_scores, select_threshold_by_f1


def test_continuous_scores_produce_valid_metrics():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.05, 0.10, 0.40, 0.55, 0.80, 0.95])
    threshold, max_f1 = select_threshold_by_f1(y_true, scores)
    result = evaluate_binary_scores(y_true, scores, threshold=threshold)

    assert 0.0 <= result.average_precision <= 1.0
    assert 0.0 <= result.roc_auc <= 1.0
    assert 0.0 <= max_f1 <= 1.0
    assert result.sample_count == 6
    assert result.true_positive + result.false_positive + result.true_negative + result.false_negative == 6


def test_distance_must_be_converted_to_similarity_score():
    y_true = np.array([1, 1, 0, 0])
    distances = np.array([0.1, 0.2, 0.8, 0.9])
    scores = -distances
    threshold, _ = select_threshold_by_f1(y_true, scores)
    result = evaluate_binary_scores(y_true, scores, threshold=threshold)
    assert result.roc_auc == 1.0
