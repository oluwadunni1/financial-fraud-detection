"""Evaluation metrics for a ~0.12% positive rate.

Accuracy is useless here: predicting "never fraud" scores 99.88%. What matters
is how the model ranks, and what it costs to act on that ranking:

- **AUC-PR** (average precision) -- the headline. Its floor is the base rate
  itself, so a model at chance scores ~0.0012 and any real signal is visible.
  ROC-AUC looks flattering under this much imbalance and is not used as the
  headline for that reason.
- **precision@k** -- if an analyst can review k alerts a day, what fraction are
  real? This is the number an operations team actually feels.
- **recall at a fixed FPR** -- how much fraud is caught when the false-positive
  budget is pinned. Pinning the budget is what makes two models comparable.

The decision threshold is chosen on validation and then applied unchanged to
test. Choosing it on test would be fitting to the thing being measured.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def auc_pr(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area under the precision-recall curve (average precision)."""
    if len(np.unique(y_true)) < 2:
        # A split with no positives (2020 alone) has no PR curve to speak of.
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def auc_roc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Reported for comparability with published numbers, never as headline."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """Fraction of the top-k highest-scoring transactions that are fraud."""
    k = min(k, len(y_score))
    if k == 0:
        return float("nan")
    top = np.argpartition(-y_score, k - 1)[:k]
    return float(y_true[top].mean())


def recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """Fraction of all fraud captured within the top-k alerts."""
    positives = y_true.sum()
    if positives == 0:
        return float("nan")
    k = min(k, len(y_score))
    top = np.argpartition(-y_score, k - 1)[:k]
    return float(y_true[top].sum() / positives)


def threshold_at_fpr(
    y_true: np.ndarray, y_score: np.ndarray, target_fpr: float
) -> float:
    """Lowest threshold whose false-positive rate stays within budget.

    Chosen on validation; the caller then applies it to test unchanged.
    """
    if len(np.unique(y_true)) < 2:
        return float("nan")
    fpr, _, thresholds = roc_curve(y_true, y_score)
    within = np.where(fpr <= target_fpr)[0]
    # roc_curve's first threshold is +inf (predict nothing); index 0 is always
    # within budget, so `within` is never empty.
    return float(thresholds[within[-1]])


def confusion_at_threshold(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float
) -> dict[str, int]:
    predicted = y_score >= threshold
    actual = y_true.astype(bool)
    return {
        "tp": int((predicted & actual).sum()),
        "fp": int((predicted & ~actual).sum()),
        "tn": int((~predicted & ~actual).sum()),
        "fn": int((~predicted & actual).sum()),
    }


def evaluate_scores(
    y_true: np.ndarray,
    y_score: np.ndarray,
    ks: Sequence[int],
    target_fpr: float,
    threshold: float | None = None,
) -> dict:
    """Full metric set for one split.

    Args:
        threshold: pass the value chosen on validation to score test at the same
            operating point. When None, one is chosen from this split.
    """
    y_true = np.asarray(y_true).astype(np.int8)
    y_score = np.asarray(y_score, dtype=np.float64)

    if threshold is None:
        threshold = threshold_at_fpr(y_true, y_score, target_fpr)

    positives = int(y_true.sum())
    result: dict = {
        "rows": int(len(y_true)),
        "positives": positives,
        "base_rate": float(positives / len(y_true)) if len(y_true) else float("nan"),
        "auc_pr": auc_pr(y_true, y_score),
        "auc_roc": auc_roc(y_true, y_score),
        "threshold": float(threshold),
    }

    for k in ks:
        result[f"precision_at_{k}"] = precision_at_k(y_true, y_score, k)
        result[f"recall_at_{k}"] = recall_at_k(y_true, y_score, k)

    confusion = confusion_at_threshold(y_true, y_score, threshold)
    result["confusion"] = confusion
    denominator = confusion["tp"] + confusion["fp"]
    result["precision_at_threshold"] = (
        confusion["tp"] / denominator if denominator else float("nan")
    )
    result[f"recall_at_fpr_{target_fpr}"] = (
        confusion["tp"] / positives if positives else float("nan")
    )
    negatives = confusion["fp"] + confusion["tn"]
    result["actual_fpr"] = confusion["fp"] / negatives if negatives else float("nan")
    return result
