from __future__ import annotations

import numpy as np


def blend_logits(
    logits_list: list[np.ndarray],
    weights: list[float] | None = None,
) -> np.ndarray:
    if not logits_list:
        raise ValueError("logits_list is empty")
    stacked = np.stack(logits_list, axis=0).astype(np.float32)
    if weights is None:
        weights_arr = np.ones(stacked.shape[0], dtype=np.float32) / stacked.shape[0]
    else:
        weights_arr = np.asarray(weights, dtype=np.float32)
        weights_arr = weights_arr / weights_arr.sum()
    return (stacked * weights_arr.reshape(-1, 1, 1)).sum(axis=0)


def class_wise_blend(
    logits_list: list[np.ndarray],
    class_weights: np.ndarray,
) -> np.ndarray:
    """
    logits_list: [M, N, C]
    class_weights: [M, C] per-model per-class weights.
    """
    stacked = np.stack(logits_list, axis=0).astype(np.float32)
    weights = class_weights.astype(np.float32)
    weights = weights / weights.sum(axis=0, keepdims=True).clip(min=1e-6)
    return (stacked * weights[:, np.newaxis, :]).sum(axis=0)
