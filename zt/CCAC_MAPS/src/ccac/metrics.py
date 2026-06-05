"""
Evaluation metrics for CCAC MAPS 5-class anxiety level classification.

Replaces the problematic global Macro-F1 with a three-dimensional framework:

  1. macro_auc          — threshold-independent discrimination (PRIMARY)
  2. quadratic_weighted_kappa — ordinal correctness (severity order matters)
  3. min_class_recall   — collapse detection gate (any class recall==0 → FAIL)

Composite: robust_score = macro_auc × coverage_penalty

Rationale: Macro-F1 conflates discrimination ability with calibration quality.
When train/test distributions are inverted (normal 76.5% → moderate 76.4%),
a model that predicts everything as a single class can score highest Macro-F1
while learning nothing useful.  AUC is threshold- and distribution-independent,
so it isolates what the model actually learned from how we threshold it.

See docs/metric_analysis.md for the full diagnosis.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)

# Competition label IDs are not ordinal severity values:
#   0=中度, 1=正常, 2=轻度, 3=重度, 4=非常严重
# True severity order is:
#   正常 < 轻度 < 中度 < 重度 < 非常严重
SEVERITY_RANK_BY_INDEX = np.asarray([2, 0, 1, 3, 4], dtype=np.int64)


def severity_ranks(indices: np.ndarray, num_classes: int | None = None) -> np.ndarray:
    """Map competition class indices to ordinal severity ranks.

    The public submission label IDs are categorical IDs, not an ordinal scale.
    This helper prevents ordinal metrics/losses from accidentally treating
    class id 0 as less severe than class id 1.
    """
    indices = indices.astype(np.int64)
    if num_classes is None:
        num_classes = int(indices.max()) + 1 if len(indices) else len(SEVERITY_RANK_BY_INDEX)
    ranks = SEVERITY_RANK_BY_INDEX[:num_classes]
    return ranks[indices]

# ---------------------------------------------------------------------------
# Primary: threshold-independent discrimination
# ---------------------------------------------------------------------------


def macro_auc(
    probabilities: np.ndarray,
    labels: np.ndarray,
    num_classes: int | None = None,
) -> float:
    """One-vs-Rest ROC-AUC averaged across classes.

    A model that predicts everything as one class → AUC ≈ 0.5 (random).
    Perfect separation → AUC = 1.0.

    Parameters
    ----------
    probabilities : (N, C) float array of predicted probabilities.
    labels : (N,) int array of ground-truth class indices.
    num_classes : int or None. Inferred from probabilities if None.

    Returns
    -------
    float in [0.0, 1.0].  0.5 = random, 1.0 = perfect.
    Returns 0.5 when no class can be evaluated (degenerate labels).
    """
    if num_classes is None:
        num_classes = probabilities.shape[1]

    if len(probabilities) == 0 or num_classes < 2:
        return 0.5

    scores: list[float] = []
    for c in range(num_classes):
        y_true = (labels == c).astype(int)
        unique_vals = np.unique(y_true)
        if len(unique_vals) < 2:
            # Class never appears or is every sample — skip
            continue
        try:
            scores.append(float(roc_auc_score(y_true, probabilities[:, c])))
        except ValueError:
            # Model predicts constant probability for this class
            scores.append(0.5)

    return float(np.mean(scores)) if scores else 0.5


# ---------------------------------------------------------------------------
# Ordinal: Quadratic Weighted Kappa
# ---------------------------------------------------------------------------


def quadratic_weighted_kappa(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int | None = None,
) -> float:
    """Cohen's kappa with quadratic (squared-distance) weighting.

    The CCAC MAPS submission label IDs are not ordinal. This function maps
    class IDs to severity ranks before computing distance, using:
    1=正常, 2=轻度, 0=中度, 3=重度, 4=非常严重.

    QWK = 1 - (observed weighted disagreement / expected weighted disagreement)

    Parameters
    ----------
    y_true : (N,) int array of ground-truth class indices.
    y_pred : (N,) int array of predicted class indices.
    num_classes : int or None. Inferred from data if None.

    Returns
    -------
    float in [-1.0, 1.0].  1.0 = perfect agreement, 0.0 = chance-level,
    negative = worse than chance.
    """
    if num_classes is None:
        num_classes = max(int(y_true.max()), int(y_pred.max())) + 1

    N = num_classes
    y_true_rank = severity_ranks(y_true, N)
    y_pred_rank = severity_ranks(y_pred, N)

    # Quadratic weight matrix: w_ij = (i - j)^2 / (N-1)^2  (normalised)
    w = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            w[i, j] = (float(i) - float(j)) ** 2 / ((N - 1) ** 2)

    hist_true = np.bincount(y_true_rank.astype(np.int64), minlength=N).astype(np.float64)
    hist_pred = np.bincount(y_pred_rank.astype(np.int64), minlength=N).astype(np.float64)
    n = len(y_true)

    # Observed confusion matrix
    O = np.zeros((N, N), dtype=np.float64)
    for t, p in zip(y_true_rank.astype(np.int64), y_pred_rank.astype(np.int64)):
        O[t, p] += 1.0

    # Expected confusion matrix under independence
    E = np.outer(hist_true, hist_pred) / n if n > 0 else np.zeros((N, N))

    num = np.sum(w * O)
    den = np.sum(w * E)
    if den == 0.0:
        return 1.0 if num == 0.0 else 0.0
    return float(1.0 - num / den)


# ---------------------------------------------------------------------------
# Gate: minimum per-class recall
# ---------------------------------------------------------------------------


def per_class_recalls(
    labels: np.ndarray,
    predictions: np.ndarray,
    num_classes: int | None = None,
) -> list[float]:
    """Recall for each class.  Returns list of length num_classes."""
    if num_classes is None:
        num_classes = max(labels.max(), predictions.max()) + 1
    return [
        float(recall_score(labels == c, predictions == c, zero_division=0.0))
        for c in range(num_classes)
    ]


def min_class_recall(
    labels: np.ndarray,
    predictions: np.ndarray,
    num_classes: int | None = None,
) -> float:
    """Minimum recall across all classes.

    0.0 means at least one class is completely ignored — model is COLLAPSED.
    """
    recalls = per_class_recalls(labels, predictions, num_classes)
    return float(min(recalls)) if recalls else 0.0


def num_zero_recall_classes(
    labels: np.ndarray,
    predictions: np.ndarray,
    num_classes: int | None = None,
) -> int:
    """Count how many classes have zero recall."""
    recalls = per_class_recalls(labels, predictions, num_classes)
    return sum(1 for r in recalls if r == 0.0)


# ---------------------------------------------------------------------------
# Composite: robust_score
# ---------------------------------------------------------------------------


def robust_score(
    probabilities: np.ndarray,
    labels: np.ndarray,
    num_classes: int | None = None,
    coverage_penalty: float = 0.5,
) -> dict[str, float]:
    """Three-dimensional evaluation combining discrimination, ordinal, and coverage.

    Parameters
    ----------
    probabilities : (N, C) float array.
    labels : (N,) int array.
    num_classes : int or None.
    coverage_penalty : float
        Factor applied to AUC when any class has zero recall.
        0.5 = halve the score; 1.0 = no penalty.

    Returns
    -------
    dict with keys:
        macro_auc, qwk, min_class_recall, num_zero_recall_classes,
        per_class_recall, per_class_f1, accuracy, macro_f1 (legacy),
        robust_score
    """
    if num_classes is None:
        num_classes = probabilities.shape[1]

    predictions = probabilities.argmax(axis=1).astype(np.int64)
    labels = labels.astype(np.int64)

    auc = macro_auc(probabilities, labels, num_classes)
    qwk = quadratic_weighted_kappa(labels, predictions, num_classes)
    recalls = per_class_recalls(labels, predictions, num_classes)
    min_rec = min(recalls) if recalls else 0.0
    n_zero = sum(1 for r in recalls if r == 0.0)
    per_class_f1 = [
        float(f1_score(labels == c, predictions == c, zero_division=0.0))
        for c in range(num_classes)
    ]

    # Legacy metrics for comparison
    macro_f1_val = float(f1_score(labels, predictions, average="macro", zero_division=0.0))
    accuracy_val = float(accuracy_score(labels, predictions))

    # Coverage penalty
    penalty = coverage_penalty if n_zero > 0 else 1.0
    rs = auc * penalty

    return {
        "macro_auc": auc,
        "qwk": qwk,
        "min_class_recall": min_rec,
        "num_zero_recall_classes": n_zero,
        "per_class_recall": recalls,
        "per_class_f1": per_class_f1,
        "accuracy": accuracy_val,
        "macro_f1": macro_f1_val,
        "robust_score": rs,
    }


# ---------------------------------------------------------------------------
# Extended classification report (replaces _classification_metrics)
# ---------------------------------------------------------------------------

_LABEL_NAMES = {
    0: "中度",
    1: "正常",
    2: "轻度",
    3: "重度",
    4: "非常严重",
}


def extended_classification_report(
    probabilities: np.ndarray,
    labels: np.ndarray,
    num_classes: int = 5,
    label_names: dict[int, str] | None = None,
) -> dict:
    """Full evaluation report combining legacy metrics + new framework.

    Call this from _evaluate() instead of the old _classification_metrics().
    """
    if label_names is None:
        label_names = _LABEL_NAMES

    predictions = probabilities.argmax(axis=1).astype(np.int64)
    labels = labels.astype(np.int64)

    robust = robust_score(probabilities, labels, num_classes)

    # Per-class breakdown indexed by class-name for readability
    per_class = {}
    for c in range(num_classes):
        name = label_names.get(c, f"class_{c}")
        per_class[name] = {
            "recall": robust["per_class_recall"][c],
            "f1": robust["per_class_f1"][c],
            "support": int((labels == c).sum()),
        }

    return {
        # Legacy
        "accuracy": robust["accuracy"],
        "macro_f1": robust["macro_f1"],
        # New primary
        "macro_auc": robust["macro_auc"],
        "qwk": robust["qwk"],
        "robust_score": robust["robust_score"],
        # Gate
        "min_class_recall": robust["min_class_recall"],
        "num_zero_recall_classes": robust["num_zero_recall_classes"],
        "collapsed": robust["num_zero_recall_classes"] > 0,
        # Detail
        "per_class": per_class,
        "per_class_recall": robust["per_class_recall"],
        "per_class_f1": robust["per_class_f1"],
    }


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def format_metrics_table(metrics: dict, label_names: dict[int, str] | None = None) -> str:
    """Pretty-print the extended metrics as a multi-line string."""
    if label_names is None:
        label_names = _LABEL_NAMES

    lines: list[str] = []
    lines.append("=" * 62)
    lines.append(f"{'Metric':<30} {'Value':>12}")
    lines.append("=" * 62)
    lines.append(f"{'accuracy':<30} {metrics['accuracy']:>12.4f}")
    lines.append(f"{'macro_f1 (legacy)':<30} {metrics['macro_f1']:>12.4f}")
    lines.append("-" * 62)
    lines.append(f"{'macro_auc (NEW primary)':<30} {metrics['macro_auc']:>12.4f}")
    lines.append(f"{'qwk (ordinal)':<30} {metrics['qwk']:>12.4f}")
    lines.append(f"{'min_class_recall (gate)':<30} {metrics['min_class_recall']:>12.4f}")
    lines.append(f"{'num_zero_recall_classes':<30} {metrics['num_zero_recall_classes']:>12}")
    lines.append(f"{'collapsed':<30} {str(metrics['collapsed']):>12}")
    lines.append("-" * 62)
    lines.append(f"{'robust_score':<30} {metrics['robust_score']:>12.4f}")
    lines.append("=" * 62)

    if "per_class" in metrics:
        lines.append("")
        lines.append("Per-class breakdown:")
        lines.append(f"{'Class':<12} {'Recall':>8} {'F1':>8} {'Support':>8}")
        lines.append("-" * 40)
        for c_name, c_data in metrics["per_class"].items():
            lines.append(
                f"{c_name:<12} {c_data['recall']:>8.4f} "
                f"{c_data['f1']:>8.4f} {c_data['support']:>8}"
            )

    return "\n".join(lines)


def compare_metrics(
    old_metrics: dict,
    new_metrics: dict,
    label_names: dict[int, str] | None = None,
) -> str:
    """Side-by-side comparison of old vs new metrics for a single experiment."""
    if label_names is None:
        label_names = _LABEL_NAMES

    lines = [
        "Old (Macro-F1) vs New (Robust) Comparison",
        "=" * 56,
        f"{'Metric':<28} {'Old':>12} {'New':>12}",
        "=" * 56,
    ]

    # Map old keys to new keys for comparison
    comparisons = [
        ("accuracy", "accuracy"),
        ("macro_f1", "macro_f1"),
    ]
    for old_key, new_key in comparisons:
        old_val = old_metrics.get(old_key, float("nan"))
        new_val = new_metrics.get(new_key, float("nan"))
        lines.append(f"{old_key:<28} {old_val:>12.4f} {new_val:>12.4f}")

    lines.append("-" * 56)
    lines.append(f"{'macro_auc (NEW)':<28} {'—':>12} {new_metrics['macro_auc']:>12.4f}")
    lines.append(f"{'qwk (NEW)':<28} {'—':>12} {new_metrics['qwk']:>12.4f}")
    lines.append(f"{'robust_score (NEW)':<28} {'—':>12} {new_metrics['robust_score']:>12.4f}")
    lines.append(f"{'collapsed (NEW)':<28} {'—':>12} {str(new_metrics['collapsed']):>12}")

    return "\n".join(lines)
