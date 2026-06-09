from __future__ import annotations

import numpy as np
import pandas as pd

from txy.constants import INDEX_TO_LEVEL, LEVEL_TO_INDEX, SUBMISSION_LEVEL_TO_INDEX

NUM_CLASSES = 5


def encode_ordinal_labels(series: pd.Series) -> tuple[np.ndarray, dict[str, int]]:
    """Unified DASS ordinal encoding: 0=正常 ... 4=非常严重."""
    normalized = series.astype(str).str.strip()
    unknown = set(normalized.unique()) - set(SUBMISSION_LEVEL_TO_INDEX)
    if unknown:
        raise ValueError(f"unknown labels: {sorted(unknown)}")
    encoded = normalized.map(SUBMISSION_LEVEL_TO_INDEX).to_numpy(dtype=np.int64)
    return encoded, dict(SUBMISSION_LEVEL_TO_INDEX)


def decode_ordinal_labels(indices: np.ndarray | list[int]) -> list[str]:
    return [INDEX_TO_LEVEL[int(i)] for i in indices]


def ordinal_mapping_json() -> dict[str, int]:
    return dict(SUBMISSION_LEVEL_TO_INDEX)
