#!/usr/bin/env python3
"""Sample validation splits that mimic the public evaluation class prior.

The normal 5-fold validation split follows the train distribution, while the
public leaderboard support is nearly inverted. This utility writes repeated
validation samples that can be used for calibration and stress tests.

Modes:
  exact_public_with_replacement:
    382 validation rows per fold with exact public support counts. This requires
    replacement because train_val has only 186 "中度" subjects but public support
    has 292.

  max_public_no_replacement:
    Largest per-fold no-replacement sample that preserves the public support
    ratio as closely as possible under train_val class counts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


LABEL_COLUMN = "t4_anxiety_level"
ID_COLUMNS = ["anon_school", "anon_class", "anon_person"]

# CodaBench public support, mapped to label names.
PUBLIC_SUPPORT = {
    "中度": 292,
    "正常": 15,
    "轻度": 46,
    "重度": 14,
    "非常严重": 15,
}

DISPLAY_ORDER = ["正常", "中度", "轻度", "重度", "非常严重"]


def _target_counts(mode: str, labels: pd.Series) -> dict[str, int]:
    if mode == "exact_public_with_replacement":
        return dict(PUBLIC_SUPPORT)

    if mode != "max_public_no_replacement":
        raise ValueError(f"unsupported mode: {mode}")

    available = labels.value_counts().to_dict()
    scale = min(available[label] / count for label, count in PUBLIC_SUPPORT.items())
    counts = {label: int(np.floor(count * scale)) for label, count in PUBLIC_SUPPORT.items()}
    # Use all limiting-class samples after flooring so the largest class is not
    # underused due to floating-point noise.
    limiting = min(PUBLIC_SUPPORT, key=lambda label: available[label] / PUBLIC_SUPPORT[label])
    counts[limiting] = available[limiting]
    return counts


def _sample_val_frame(
    frame: pd.DataFrame,
    target_counts: dict[str, int],
    rng: np.random.Generator,
    replace: bool,
) -> pd.DataFrame:
    parts = []
    for label, count in target_counts.items():
        class_frame = frame[frame[LABEL_COLUMN].astype(str).str.strip() == label]
        if count > len(class_frame) and not replace:
            raise ValueError(f"cannot sample {count} rows of {label} without replacement")
        sampled_pos = rng.choice(len(class_frame), size=count, replace=replace)
        sampled = class_frame.iloc[sampled_pos].copy()
        sampled["sampled_label"] = label
        sampled["sample_copy"] = np.arange(count, dtype=np.int64)
        parts.append(sampled)
    val = pd.concat(parts, ignore_index=True)
    val = val.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1))).reset_index(drop=True)
    val["subject_id"] = val[ID_COLUMNS].astype(str).agg("/".join, axis=1)
    val["repeat_count"] = val.groupby("source_row")["source_row"].transform("size").astype(np.int64)
    val["repeat_index"] = val.groupby("source_row").cumcount().astype(np.int64)
    return val


def _write_summary(output_dir: Path, split_summaries: list[dict], train_counts: dict[str, int]) -> None:
    summary = {
        "label_column": LABEL_COLUMN,
        "public_support": PUBLIC_SUPPORT,
        "train_counts": train_counts,
        "folds": split_summaries,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-csv", default="datasets/train_val/labels.csv")
    parser.add_argument("--output-dir", default="artifacts/exp/eval_prior_val_splits")
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mode",
        choices=["exact_public_with_replacement", "max_public_no_replacement"],
        default="exact_public_with_replacement",
    )
    args = parser.parse_args()

    labels_csv = Path(args.labels_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(labels_csv).reset_index(names="source_row")
    frame[LABEL_COLUMN] = frame[LABEL_COLUMN].astype(str).str.strip()

    replace = args.mode == "exact_public_with_replacement"
    target_counts = _target_counts(args.mode, frame[LABEL_COLUMN])
    train_counts = {label: int((frame[LABEL_COLUMN] == label).sum()) for label in DISPLAY_ORDER}

    split_summaries = []
    for fold in range(1, args.num_folds + 1):
        rng = np.random.default_rng(args.seed + fold - 1)
        val = _sample_val_frame(frame, target_counts, rng, replace=replace)
        unique_val_rows = set(val["source_row"].astype(int).tolist())
        train = frame[~frame["source_row"].isin(unique_val_rows)].copy().reset_index(drop=True)

        val_path = output_dir / f"fold_{fold}_val.csv"
        train_path = output_dir / f"fold_{fold}_train.csv"
        val.to_csv(val_path, index=False, encoding="utf-8")
        train.to_csv(train_path, index=False, encoding="utf-8")

        counts = val[LABEL_COLUMN].value_counts().to_dict()
        duplicate_rows = int(len(val) - val["source_row"].nunique())
        fold_summary = {
            "fold": fold,
            "mode": args.mode,
            "val_rows": int(len(val)),
            "val_unique_subjects": int(val["source_row"].nunique()),
            "val_duplicate_rows": duplicate_rows,
            "max_repeat_count": int(val["repeat_count"].max()),
            "val_counts": {label: int(counts.get(label, 0)) for label in DISPLAY_ORDER},
            "train_rows_after_unique_val_removal": int(len(train)),
            "val_csv": str(val_path),
            "train_csv": str(train_path),
        }
        split_summaries.append(fold_summary)

    _write_summary(output_dir, split_summaries, train_counts)

    print(f"Mode: {args.mode}")
    print(f"Output: {output_dir}")
    print(f"Target counts: {target_counts}")
    for item in split_summaries:
        print(
            f"Fold {item['fold']}: val_rows={item['val_rows']} "
            f"unique={item['val_unique_subjects']} dup_rows={item['val_duplicate_rows']} "
            f"counts={item['val_counts']}"
        )


if __name__ == "__main__":
    main()
