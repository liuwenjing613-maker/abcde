from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from txy.constants import LEVEL_TO_INDEX, SUBMISSION_LEVEL_TO_INDEX
from txy.data.feature_io import make_subject_id


def probs_to_logits(probs: np.ndarray) -> np.ndarray:
    return np.log(probs.clip(1e-6, 1.0)).astype(np.float32)


def remap_probs_to_ordinal(probs: np.ndarray, source_mapping: dict[str, int]) -> np.ndarray:
    """Remap probability columns from arbitrary train index order to DASS ordinal 0..4."""
    ordinal = np.zeros_like(probs, dtype=np.float32)
    for label_name, ordinal_idx in SUBMISSION_LEVEL_TO_INDEX.items():
        source_idx = source_mapping.get(label_name)
        if source_idx is None:
            raise KeyError(f"label {label_name} missing in source_mapping")
        ordinal[:, int(ordinal_idx)] = probs[:, int(source_idx)]
    return ordinal


def remap_logits_to_ordinal(logits: np.ndarray, source_mapping: dict[str, int]) -> np.ndarray:
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = probs / probs.sum(axis=1, keepdims=True)
    ordinal_probs = remap_probs_to_ordinal(probs.astype(np.float32), source_mapping)
    return probs_to_logits(ordinal_probs)


def load_label_mapping(path: Path | None) -> dict[str, int] | None:
    if path is None or not path.exists():
        return None
    return {k: int(v) for k, v in json.loads(path.read_text(encoding="utf-8")).items()}


def is_ordinal_mapping(mapping: dict[str, int] | None) -> bool:
    return mapping == SUBMISSION_LEVEL_TO_INDEX or mapping == LEVEL_TO_INDEX


def load_oof_aligned(
    path: Path,
    subject_ids: list[str],
    label_mapping_path: Path | None = None,
    num_classes: int = 5,
) -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(path)
    if "subject_id" not in df.columns:
        raise ValueError(f"{path} missing subject_id column")
    df = df.set_index("subject_id").loc[subject_ids].reset_index()

    mapping = load_label_mapping(label_mapping_path or path.parent / "label_mapping.json")
    logit_cols = [f"logit_class_{i}" for i in range(num_classes)]
    prob_cols = [f"prob_class_{i}" for i in range(num_classes)]

    if all(c in df.columns for c in logit_cols):
        logits = df[logit_cols].to_numpy(dtype=np.float32)
        if mapping and not is_ordinal_mapping(mapping):
            logits = remap_logits_to_ordinal(logits, mapping)
    elif all(c in df.columns for c in prob_cols):
        probs = df[prob_cols].to_numpy(dtype=np.float32)
        if mapping and not is_ordinal_mapping(mapping):
            probs = remap_probs_to_ordinal(probs, mapping)
        logits = probs_to_logits(probs)
    else:
        raise ValueError(f"{path} needs logit_class_* or prob_class_* columns")

    return df, logits


def load_test_logits_aligned(
    path: Path,
    test_frame: pd.DataFrame,
    label_mapping_path: Path | None = None,
    num_classes: int = 5,
) -> np.ndarray:
    df = pd.read_csv(path)
    keys = ["anon_school", "anon_class", "anon_person"]
    if not all(k in df.columns for k in keys):
        raise ValueError(f"{path} missing subject key columns")
    merged = test_frame[keys].merge(df, on=keys, how="left")
    if merged.isna().any().any():
        missing = int(merged.isna().any(axis=1).sum())
        raise ValueError(f"{path} missing predictions for {missing} test subjects")

    mapping = load_label_mapping(label_mapping_path or path.parent / "label_mapping.json")
    prob_cols = [f"prob_class_{i}" for i in range(num_classes)]
    logit_cols = [f"logit_class_{i}" for i in range(num_classes)]

    if all(c in merged.columns for c in logit_cols):
        logits = merged[logit_cols].to_numpy(dtype=np.float32)
        if mapping and not is_ordinal_mapping(mapping):
            logits = remap_logits_to_ordinal(logits, mapping)
    elif all(c in merged.columns for c in prob_cols):
        probs = merged[prob_cols].to_numpy(dtype=np.float32)
        if mapping and not is_ordinal_mapping(mapping):
            probs = remap_probs_to_ordinal(probs, mapping)
        logits = probs_to_logits(probs)
    elif "label" in merged.columns:
        # Text labels only — cannot blend; raise
        raise ValueError(f"{path} has labels only; need prob/logit columns for fusion")
    else:
        raise ValueError(f"{path} needs prob_class_* or logit_class_*")
    return logits


def build_train_subject_order(dataset_path: Path) -> tuple[pd.DataFrame, list[str]]:
    frame = pd.read_csv(dataset_path / "train_val" / "labels.csv")
    frame = frame.dropna(subset=["t4_anxiety_level"]).reset_index(drop=True)
    frame["subject_id"] = make_subject_id(frame)
    return frame, frame["subject_id"].astype(str).tolist()
