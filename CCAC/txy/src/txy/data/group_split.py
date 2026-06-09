from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold


def make_subject_id(frame: pd.DataFrame) -> pd.Series:
    return frame[["anon_school", "anon_class", "anon_person"]].agg("/".join, axis=1)


def make_group_id(frame: pd.DataFrame, group_by: str = "school_class") -> np.ndarray:
    if group_by == "school":
        return frame["anon_school"].astype(str).to_numpy()
    if group_by == "school_class":
        return frame[["anon_school", "anon_class"]].agg("/".join, axis=1).to_numpy()
    raise ValueError(f"unsupported group_by: {group_by}")


def build_group_folds(
    labels: np.ndarray,
    groups: np.ndarray,
    num_folds: int,
    seed: int,
    stratified: bool = True,
) -> list[tuple[np.ndarray, np.ndarray]]:
    unique_groups = np.unique(groups)
    fold_count = max(2, min(num_folds, len(unique_groups)))
    if fold_count < 2:
        raise ValueError("need at least 2 groups for group split")

    if stratified:
        class_counts = pd.Series(labels).value_counts()
        if len(class_counts) > 1 and class_counts.min() >= fold_count:
            splitter = StratifiedGroupKFold(n_splits=fold_count, shuffle=True, random_state=seed)
            return list(splitter.split(np.zeros(len(labels)), labels, groups))

    splitter = GroupKFold(n_splits=fold_count)
    return list(splitter.split(np.zeros(len(labels)), labels, groups))
