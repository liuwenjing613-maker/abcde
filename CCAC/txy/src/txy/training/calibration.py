from __future__ import annotations

import itertools

import numpy as np

from txy.training.metrics import classification_metrics


def apply_class_bias(logits: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return logits + bias.reshape(1, -1)


def search_class_bias(
    logits: np.ndarray,
    labels: np.ndarray,
    grid: tuple[float, ...] = (-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0),
    max_classes: int | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    num_classes = logits.shape[1]
    if max_classes is not None:
        num_classes = min(num_classes, logits.shape[1])

    best_bias = np.zeros(logits.shape[1], dtype=np.float32)
    best_metrics = classification_metrics(labels, logits.argmax(axis=1))
    best_metrics["macro_f1"] = -1.0

    # Coarse grid search on each class bias independently, then refine jointly on top classes.
    for class_idx in range(num_classes):
        for value in grid:
            bias = best_bias.copy()
            bias[class_idx] = value
            adjusted = apply_class_bias(logits, bias)
            metrics = classification_metrics(labels, adjusted.argmax(axis=1))
            if metrics["macro_f1"] > best_metrics["macro_f1"]:
                best_bias = bias
                best_metrics = metrics

    # Small joint search on pairs of classes with largest |bias|.
    candidate_indices = list(range(num_classes))
    pair_grid = tuple(v for v in grid if abs(v) <= 1.0)
    for i, j in itertools.combinations(candidate_indices, 2):
        for bi, bj in itertools.product(pair_grid, pair_grid):
            bias = best_bias.copy()
            bias[i] = bi
            bias[j] = bj
            adjusted = apply_class_bias(logits, bias)
            metrics = classification_metrics(labels, adjusted.argmax(axis=1))
            if metrics["macro_f1"] > best_metrics["macro_f1"]:
                best_bias = bias
                best_metrics = metrics

    return best_bias.astype(np.float32), best_metrics
