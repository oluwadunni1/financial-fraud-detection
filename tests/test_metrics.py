"""Metric tests, against hand-computed values.

Expected numbers here are worked out by hand rather than captured from a run --
a metric that records whatever the code currently returns cannot catch the code
being wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from fraud.models.metrics import (
    auc_pr,
    confusion_at_threshold,
    evaluate_scores,
    precision_at_k,
    recall_at_k,
    threshold_at_fpr,
)

# Ranked by score: 0.9(+) 0.8(-) 0.7(+) 0.6(-) 0.5(-) ... 2 positives of 6.
Y_TRUE = np.array([1, 0, 1, 0, 0, 0])
Y_SCORE = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])


def test_precision_at_k_hand_computed():
    assert precision_at_k(Y_TRUE, Y_SCORE, 1) == 1.0        # top-1 is fraud
    assert precision_at_k(Y_TRUE, Y_SCORE, 2) == 0.5        # 1 of 2
    assert precision_at_k(Y_TRUE, Y_SCORE, 3) == pytest.approx(2 / 3)
    assert precision_at_k(Y_TRUE, Y_SCORE, 6) == pytest.approx(2 / 6)


def test_recall_at_k_hand_computed():
    assert recall_at_k(Y_TRUE, Y_SCORE, 1) == 0.5   # 1 of 2 positives
    assert recall_at_k(Y_TRUE, Y_SCORE, 3) == 1.0   # both captured
    assert recall_at_k(Y_TRUE, Y_SCORE, 6) == 1.0


def test_k_larger_than_dataset_is_clamped():
    assert precision_at_k(Y_TRUE, Y_SCORE, 1000) == pytest.approx(2 / 6)


def test_auc_pr_perfect_ranking_is_one():
    perfect = np.array([0.9, 0.8, 0.1, 0.05, 0.01, 0.0])
    y = np.array([1, 1, 0, 0, 0, 0])
    assert auc_pr(y, perfect) == pytest.approx(1.0)


def test_auc_pr_of_inverted_ranking_is_poor():
    """Worst ranking must score below the base rate of 1/3."""
    y = np.array([1, 1, 0, 0, 0, 0])
    inverted = np.array([0.0, 0.01, 0.05, 0.1, 0.8, 0.9])
    assert auc_pr(y, inverted) < 1 / 3


def test_auc_pr_floor_is_the_base_rate():
    """Random scores on a 10% positive set land near 0.10, not 0.5.

    This is why AUC-PR is the headline: its chance level tracks the imbalance,
    so 'better than chance' is unambiguous.
    """
    rng = np.random.default_rng(0)
    y = np.zeros(20_000, dtype=int)
    y[: 2_000] = 1
    rng.shuffle(y)
    assert auc_pr(y, rng.random(20_000)) == pytest.approx(0.10, abs=0.02)


def test_metrics_on_a_split_without_positives_are_nan():
    """2020 has 336,500 rows and zero fraud -- it must not crash or fake a score."""
    y = np.zeros(10, dtype=int)
    scores = np.linspace(0, 1, 10)
    assert np.isnan(auc_pr(y, scores))
    assert np.isnan(recall_at_k(y, scores, 3))
    assert np.isnan(threshold_at_fpr(y, scores, 0.01))


def test_threshold_at_fpr_stays_within_budget():
    rng = np.random.default_rng(1)
    y = np.zeros(5_000, dtype=int)
    y[:100] = 1
    rng.shuffle(y)
    scores = rng.random(5_000) + y * 0.4  # positives score higher

    threshold = threshold_at_fpr(y, scores, 0.01)
    predicted = scores >= threshold
    negatives = (y == 0).sum()
    observed_fpr = (predicted & (y == 0)).sum() / negatives
    assert observed_fpr <= 0.01 + 1e-9


def test_confusion_counts_sum_to_the_dataset():
    c = confusion_at_threshold(Y_TRUE, Y_SCORE, 0.75)
    assert c["tp"] + c["fp"] + c["tn"] + c["fn"] == len(Y_TRUE)
    # At 0.75: 0.9(+) and 0.8(-) predicted positive.
    assert c == {"tp": 1, "fp": 1, "tn": 3, "fn": 1}


def test_evaluate_scores_uses_a_supplied_threshold():
    """Test must be scored at the threshold chosen on val, not its own."""
    loose = evaluate_scores(Y_TRUE, Y_SCORE, ks=[2], target_fpr=0.5, threshold=0.75)
    assert loose["threshold"] == 0.75
    assert loose["confusion"]["tp"] == 1


def test_evaluate_scores_reports_base_rate_and_shape():
    result = evaluate_scores(Y_TRUE, Y_SCORE, ks=[1, 3], target_fpr=0.25)
    assert result["rows"] == 6
    assert result["positives"] == 2
    assert result["base_rate"] == pytest.approx(2 / 6)
    assert "precision_at_1" in result and "recall_at_3" in result
    assert "auc_pr" in result and "confusion" in result


def test_evaluate_scores_is_json_serialisable():
    """The evaluate stage writes this straight to a DVC-tracked report."""
    import json

    result = evaluate_scores(Y_TRUE, Y_SCORE, ks=[2], target_fpr=0.25)
    json.loads(json.dumps(result))  # raises if a numpy type leaked through
