from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, f1_score


def classification_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, predictions, average="weighted", zero_division=0)),
        "min_class_f1": min_class_f1(labels, predictions),
    }


def min_class_f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    scores = f1_score(labels, predictions, average=None, zero_division=0)
    if len(scores) == 0:
        return 0.0
    return float(np.min(scores))


def format_classification_report(labels: np.ndarray, predictions: np.ndarray, class_names: list[str]) -> str:
    return classification_report(labels, predictions, target_names=class_names, zero_division=0)
