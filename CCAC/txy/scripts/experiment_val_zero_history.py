#!/usr/bin/env python3
"""Experiment A: compare val with real tabular vs zero-history (test-like) evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Report val_normal vs val_zero_history from artifact fold metrics")
    parser.add_argument("--artifact-dir", type=str, default="artifacts/residual_v3")
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir).resolve()
    metrics_path = artifact_dir / "fold_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"missing {metrics_path}; train residual v3 first")

    df = pd.read_csv(metrics_path)
    report = {
        "artifact_dir": str(artifact_dir),
        "val_normal_macro_f1_mean": float(df["macro_f1"].mean()) if "macro_f1" in df.columns else None,
        "val_zero_history_macro_f1_mean": float(df["val_zero_history_macro_f1"].mean()),
        "tabular_anchor_macro_f1_mean": float(df["tabular_anchor_macro_f1"].mean()) if "tabular_anchor_macro_f1" in df.columns else None,
        "per_fold": df.to_dict(orient="records"),
    }
    if "calibrated_macro_f1" in df.columns:
        report["val_normal_calibrated_macro_f1_mean"] = float(df["calibrated_macro_f1"].mean())

    summary_path = artifact_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        report["test_has_history"] = summary.get("test_has_history")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
