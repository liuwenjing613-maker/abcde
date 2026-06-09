from __future__ import annotations

import numpy as np
import pandas as pd


def compute_train_prior_bias(
    labels: pd.Series,
    label_mapping: dict[str, int],
    num_classes: int | None = None,
) -> np.ndarray:
    """
    Fixed log-prior shift from train label frequencies.
    Does not use validation grid search — safer for test submission.
    """
    if num_classes is None:
        num_classes = len(label_mapping)
    counts = np.zeros(num_classes, dtype=np.float64)
    for label, index in label_mapping.items():
        if 0 <= int(index) < num_classes:
            counts[int(index)] = float((labels.astype(str).str.strip() == label).sum())
    counts = np.where(counts == 0, 1.0, counts)
    prior = counts / counts.sum()
    log_prior = np.log(prior)
    log_uniform = np.log(1.0 / num_classes)
    return (log_prior - log_uniform).astype(np.float32)


def apply_prior_bias(logits: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return logits + bias.reshape(1, -1)
