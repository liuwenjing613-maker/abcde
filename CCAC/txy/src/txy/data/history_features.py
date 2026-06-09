from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from txy.constants import (
    HISTORY_LEVEL_COLS,
    HISTORY_SCORE_COLS,
    LEVEL_TO_INDEX,
    STAGES,
)


@dataclass
class HistoryFeatureBuilder:
    """Build tabular history features from T1/T2/T3 DASS scores and levels."""

    score_columns: list[str]
    level_columns: list[str]
    feature_names: list[str]
    level_vocab_size: int = 5

    @classmethod
    def from_labels_frame(cls, frame: pd.DataFrame) -> "HistoryFeatureBuilder":
        score_columns = [c for c in HISTORY_SCORE_COLS if c in frame.columns]
        level_columns = [c for c in HISTORY_LEVEL_COLS if c in frame.columns]
        if not score_columns:
            raise ValueError("labels frame has no history score columns")
        builder = cls(
            score_columns=score_columns,
            level_columns=level_columns,
            feature_names=[],
        )
        builder.feature_names = builder._derive_names()
        return builder

    def _derive_names(self) -> list[str]:
        names: list[str] = []
        names.extend(self.score_columns)
        names.extend([f"{col}_idx" for col in self.level_columns])
        for dim in ("depression", "anxiety", "stress"):
            stage_cols = [f"{stage.lower()}_{dim}_score" for stage in STAGES]
            stage_cols = [c for c in stage_cols if c in self.score_columns]
            if len(stage_cols) == 3:
                names.extend(
                    [
                        f"{dim}_mean_t1t3",
                        f"{dim}_max_t1t3",
                        f"{dim}_slope_t1t3",
                        f"{dim}_delta_t3_t1",
                        f"{dim}_delta_t3_t2",
                        f"{dim}_rising_t1t3",
                        f"{dim}_normal_to_abnormal",
                    ]
                )
        return names

    def transform(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        n = len(frame)
        blocks: list[np.ndarray] = []

        score_block = np.zeros((n, len(self.score_columns)), dtype=np.float32)
        for col_index, col in enumerate(self.score_columns):
            if col in frame.columns:
                score_block[:, col_index] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0).to_numpy(
                    dtype=np.float32
                )
        blocks.append(score_block)

        if self.level_columns:
            level_idx = np.zeros((n, len(self.level_columns)), dtype=np.float32)
            for col_index, col in enumerate(self.level_columns):
                if col in frame.columns:
                    mapped = frame[col].astype(str).str.strip().map(LEVEL_TO_INDEX)
                    level_idx[:, col_index] = mapped.fillna(0).to_numpy(dtype=np.float32)
            blocks.append(level_idx)

        for dim in ("depression", "anxiety", "stress"):
            stage_cols = [f"{stage.lower()}_{dim}_score" for stage in STAGES]
            stage_cols = [c for c in stage_cols if c in self.score_columns]
            if len(stage_cols) != 3:
                continue
            values = np.zeros((n, 3), dtype=np.float32)
            for j, col in enumerate(stage_cols):
                if col in frame.columns:
                    values[:, j] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
            t1, t2, t3 = values[:, 0], values[:, 1], values[:, 2]
            mean = values.mean(axis=1, keepdims=True)
            vmax = values.max(axis=1, keepdims=True)
            slope = ((t3 - t1) / 2.0).reshape(-1, 1)
            d31 = (t3 - t1).reshape(-1, 1)
            d32 = (t3 - t2).reshape(-1, 1)
            rising = ((t1 < t2) & (t2 < t3)).astype(np.float32).reshape(-1, 1)
            normal_to_abnormal = ((t1 <= 4.0) & (t3 > 4.0)).astype(np.float32).reshape(-1, 1)
            blocks.extend([mean, vmax, slope, d31, d32, rising, normal_to_abnormal])

        features = np.concatenate(blocks, axis=1).astype(np.float32)
        level_indices = np.zeros((n, len(self.level_columns)), dtype=np.int64)
        if self.level_columns:
            for col_index, col in enumerate(self.level_columns):
                if col in frame.columns:
                    mapped = frame[col].astype(str).str.strip().map(LEVEL_TO_INDEX)
                    level_indices[:, col_index] = mapped.fillna(0).to_numpy(dtype=np.int64)
        return features, level_indices
