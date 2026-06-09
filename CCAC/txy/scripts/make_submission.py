#!/usr/bin/env python3
"""
Build CodaBench submission: one CSV (submission.csv) inside a zip.

Platform requirements:
  - Columns: anon_school, anon_class, anon_person, label
  - label: integer 0..4 using DASS ordinal order:
      0=正常, 1=轻度, 2=中度, 3=重度, 4=非常严重
  - Exactly one row per official test subject (default 382 rows)
"""
from __future__ import annotations

import argparse
import json
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from txy.constants import (
    LEVEL_TO_INDEX,
    SUBMISSION_LEVEL_TO_INDEX,
    train_class_index_to_submission_label,
)
from txy.training.prior_calibration import apply_prior_bias, compute_train_prior_bias


def encode_submission_label(value) -> int:
    if pd.isna(value):
        raise ValueError("missing label value")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        code = int(value)
        if code in SUBMISSION_LEVEL_TO_INDEX.values():
            return code
        # Possibly a train-time class index; remap to submission ordinal.
        return train_class_index_to_submission_label(code)
    text = str(value).strip()
    if text.isdigit():
        return encode_submission_label(int(text))
    if text not in SUBMISSION_LEVEL_TO_INDEX:
        raise ValueError(f"unknown label text: {text}")
    return int(SUBMISSION_LEVEL_TO_INDEX[text])


def load_official_test_subjects(path: Path, expected_rows: int | None) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = ["anon_school", "anon_class", "anon_person"]
    missing = set(cols).difference(df.columns)
    if missing:
        raise ValueError(f"test subjects missing columns: {sorted(missing)}")
    df = df[cols].drop_duplicates().reset_index(drop=True)
    if expected_rows is not None:
        if len(df) < expected_rows:
            raise ValueError(f"test subjects has {len(df)} rows, expected {expected_rows}")
        df = df.iloc[:expected_rows].copy()
    return df


def predictions_from_source(
    source_csv: Path,
    dataset_path: Path,
    calibration: str,
) -> pd.DataFrame:
    df = pd.read_csv(source_csv)
    keys = ["anon_school", "anon_class", "anon_person"]
    for col in keys:
        if col not in df.columns:
            raise ValueError(f"predictions source missing column: {col}")

    if "label" in df.columns:
        labels = [encode_submission_label(v) for v in df["label"]]
        out = df[keys].copy()
        out["label"] = labels
        return out

    if "pred_label" in df.columns:
        labels = [encode_submission_label(v) for v in df["pred_label"]]
        out = df[keys].copy()
        out["label"] = labels
        return out

    prob_cols = sorted(
        [c for c in df.columns if c.startswith("prob_class_")],
        key=lambda x: int(x.split("_")[-1]),
    )
    if not prob_cols:
        raise ValueError("predictions source needs label, pred_label, or prob_class_* columns")

    probs = df[prob_cols].to_numpy(dtype=np.float32)
    logits = np.log(probs.clip(1e-6, 1.0))
    if calibration == "train_prior":
        train_labels = pd.read_csv(dataset_path / "train_val" / "labels.csv")["t4_anxiety_level"]
        bias = compute_train_prior_bias(train_labels, LEVEL_TO_INDEX, len(prob_cols))
        logits = apply_prior_bias(logits, bias)

    # With unified ordinal encoding, class index equals submission label (0=正常..4=非常严重).
    train_argmax = logits.argmax(axis=1)
    submission_labels = [train_class_index_to_submission_label(int(i)) for i in train_argmax]
    out = df[keys].copy()
    out["label"] = submission_labels
    return out


def merge_to_official_test(official: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    merged = official.merge(preds, on=["anon_school", "anon_class", "anon_person"], how="left")
    missing = int(merged["label"].isna().sum())
    if missing:
        raise ValueError(f"missing predictions for {missing} official test subjects")
    merged["label"] = merged["label"].astype(int)
    return merged[["anon_school", "anon_class", "anon_person", "label"]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CodaBench submission zip (CSV only)")
    parser.add_argument("--dataset-path", type=str, default="/home/adodas/dataset_ccac")
    parser.add_argument("--artifact-dir", type=str, default="artifacts/residual")
    parser.add_argument("--predictions-source", type=str, default=None)
    parser.add_argument("--test-subjects", type=str, default=None)
    parser.add_argument("--expected-rows", type=int, default=382)
    parser.add_argument("--calibration", type=str, default="raw", choices=["raw", "train_prior"])
    parser.add_argument("--csv-name", type=str, default="submission.csv")
    parser.add_argument("--output-zip", type=str, default=None)
    parser.add_argument("--skip-infer", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.25)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    dataset_path = Path(args.dataset_path).resolve()
    artifact_dir = Path(args.artifact_dir).resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_path = Path(args.output_zip or root / "submissions" / f"CCAC_submission_{stamp}.zip")

    preds_full = Path(
        args.predictions_source
        or artifact_dir / "test_predictions_submission.csv"
    )
    if not args.skip_infer and not preds_full.exists():
        import os
        import subprocess

        train_cfg = json.loads((artifact_dir / "train_config.json").read_text(encoding="utf-8")) if (
            artifact_dir / "train_config.json"
        ).exists() else {}
        if train_cfg.get("kd_weight") is not None or train_cfg.get("model_kind") == "stagewise_v4":
            infer_script = "infer_stagewise_v4.py"
        elif train_cfg.get("anchor_model_type") or train_cfg.get("history_dropout_prob") is not None:
            infer_script = "infer_residual_v3.py"
        else:
            infer_script = "infer_residual.py"
        infer_cmd = [
            "python",
            str(root / "scripts" / infer_script),
            "--dataset-path",
            str(dataset_path),
            "--artifact-dir",
            str(artifact_dir),
            "--calibration",
            args.calibration,
            "--output",
            str(preds_full),
        ]
        if infer_script == "infer_residual.py":
            infer_cmd.extend(["--alpha", str(args.alpha)])
        subprocess.run(
            infer_cmd,
            cwd=str(root),
            env={**os.environ, "PYTHONPATH": str(root / "src")},
            check=True,
        )

    test_subjects_path = Path(args.test_subjects or dataset_path / "test" / "subjects.csv")
    official = load_official_test_subjects(test_subjects_path, args.expected_rows)
    preds = predictions_from_source(preds_full, dataset_path, args.calibration)
    submission = merge_to_official_test(official, preds)

    staging = root / "submissions" / f"_staging_{stamp}"
    staging.mkdir(parents=True, exist_ok=True)
    csv_path = staging / args.csv_name
    submission.to_csv(csv_path, index=False, encoding="utf-8")

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path, arcname=args.csv_name)

    int_to_text = {int(v): k for k, v in SUBMISSION_LEVEL_TO_INDEX.items()}
    label_counts = submission["label"].value_counts().sort_index().to_dict()
    print(
        json.dumps(
            {
                "zip_path": str(zip_path),
                "csv_name": args.csv_name,
                "num_rows": int(len(submission)),
                "encoding": "submission ordinal: 0=正常,1=轻度,2=中度,3=重度,4=非常严重",
                "label_counts_int": {int(k): int(v) for k, v in label_counts.items()},
                "label_counts_named": {int_to_text[int(k)]: int(v) for k, v in label_counts.items()},
                "first_rows": submission.head(5).to_dict(orient="records"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
