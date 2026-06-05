#!/usr/bin/env python3
"""
Re-evaluate all historical experiments with the new metric framework.

Reads OOF prediction CSVs from artifacts/exp/*/oof_predictions.csv,
computes macro_auc, QWK, min_class_recall, and robust_score alongside
the legacy macro_f1 for direct comparison.

Usage:
    python scripts/reevaluate_experiments.py              # all experiments
    python scripts/reevaluate_experiments.py --exp nd3_focal_g2  # single
    python scripts/reevaluate_experiments.py --top 10      # top-N by robust_score
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "src"))

from ccac.metrics import (
    extended_classification_report,
    format_metrics_table,
    macro_auc,
    min_class_recall,
    quadratic_weighted_kappa,
    robust_score,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ARTIFACTS_DIR = _project_root / "artifacts" / "exp"
LABEL_MAPPING = {"中度": 0, "正常": 1, "轻度": 2, "重度": 3, "非常严重": 4}
LABEL_NAMES = {v: k for k, v in LABEL_MAPPING.items()}
NUM_CLASSES = 5

# Known OOF Macro-F1 values from docs/experiments/no_dass_experiments.md / summary.json
# (used when the CSV is missing or for cross-reference)
KNOWN_OOF_MF1: dict[str, float] = {
    "deep_residual_gpu": 0.363,   # has DASS
    "nd1_baseline": 0.232,        # BiGRU no-DASS (no CSV — computed from folds)
    "nd3_focal_g2": 0.264,
    "nd9_transformer": 0.274,
    "nd11_distillation": 0.329,
    "nd12_ensemble": 0.287,
    "nd15_distill_transformer": 0.310,
    "nd16_multiteacher": 0.308,
    "deep_residual": 0.244,       # γ=1.0 no-DASS
    "no_dass": 0.232,
    "nd5_basic": 0.253,
    "nd8_ldam": 0.257,
    "nd13_tcn": 0.170,
}

# Submission public-leaderboard MF1 (from docs/experiments/submission_log.md)
SUBMISSION_MF1: dict[str, float] = {
    "deep_residual_gpu": 0.0908,  # sub1 — DASS collapse
    "nd3_focal_g2": 0.1805,       # sub4 — DeepRes γ=2.0 + prior calib
    "nd11_distillation": 0.1769,  # sub3 — KD + prior calib
    "no_dass": 0.0486,            # sub2 — BiGRU baseline
    "nd11_distillation_raw": 0.0328,  # sub5 — KD no calib
}


# ---------------------------------------------------------------------------
# Core: load an experiment and compute metrics
# ---------------------------------------------------------------------------


def load_oof_csv(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Load OOF predictions CSV.

    Returns (labels, predictions, probabilities) or None on failure.
    labels: (N,) int64
    predictions: (N,) int64
    probabilities: (N, 5) float32
    """
    if not csv_path.exists():
        return None

    try:
        df = pd.read_csv(str(csv_path))
    except Exception:
        return None

    # Map string labels to integers
    label_map = LABEL_MAPPING
    labels = df["true_label"].map(label_map).to_numpy(dtype=np.int64)

    # Predictions (use pred_label column if present, else argmax of probs)
    if "pred_label" in df.columns:
        predictions = df["pred_label"].map(label_map).to_numpy(dtype=np.int64)
    else:
        predictions = np.full(len(df), -1, dtype=np.int64)

    # Probabilities
    prob_cols = [f"prob_class_{c}" for c in range(NUM_CLASSES)]
    if all(c in df.columns for c in prob_cols):
        probabilities = df[prob_cols].to_numpy(dtype=np.float32)
    else:
        probabilities = np.zeros((len(df), NUM_CLASSES), dtype=np.float32)

    # If predictions are missing, derive from probabilities
    if (predictions < 0).any():
        predictions = probabilities.argmax(axis=1).astype(np.int64)

    # Drop NaN labels (shouldn't happen, but be safe)
    valid = ~np.isnan(probabilities).any(axis=1) & (labels >= 0)
    return labels[valid], predictions[valid], probabilities[valid]


def evaluate_experiment(exp_name: str, exp_dir: Path) -> dict | None:
    """Compute full extended metrics for one experiment."""
    csv_path = exp_dir / "oof_predictions.csv"
    loaded = load_oof_csv(csv_path)
    if loaded is None:
        return None

    labels, predictions, probabilities = loaded
    if len(labels) == 0:
        return None

    report = extended_classification_report(
        probabilities, labels, NUM_CLASSES, LABEL_NAMES
    )

    # Attach metadata
    report["experiment"] = exp_name
    report["n_samples"] = len(labels)

    # Cross-reference known values
    report["known_oof_mf1"] = KNOWN_OOF_MF1.get(exp_name)
    report["submission_mf1"] = SUBMISSION_MF1.get(exp_name)

    return report


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def print_comparison_table(results: list[dict]) -> None:
    """Print a ranked comparison table of all experiments."""
    # Sort by robust_score descending
    sorted_results = sorted(results, key=lambda r: r["robust_score"], reverse=True)

    header = (
        f"{'Rank':<5} {'Experiment':<28} {'FeatSet':<10} {'Macro-F1':>10} {'Macro-AUC':>10} "
        f"{'QWK':>8} {'Min-Rec':>8} {'Zero':>5} {'Robust':>10} {'Status':>8}"
    )
    sep = "=" * len(header)

    print("\nRe-evaluation of Historical Experiments with New Metric Framework")
    print(sep)
    print(header)
    print(sep)

    for rank, r in enumerate(sorted_results, 1):
        status = "❌ COLLAPSED" if r["collapsed"] else "✓ OK"
        featset = r.get("feature_set", "—")
        print(
            f"{rank:<5} {r['experiment']:<28} {featset:<10} "
            f"{r['macro_f1']:>10.4f} {r['macro_auc']:>10.4f} "
            f"{r['qwk']:>8.4f} {r['min_class_recall']:>8.4f} "
            f"{r['num_zero_recall_classes']:>5} {r['robust_score']:>10.4f} "
            f"{status:>8}"
        )

    print(sep)

    # Summary
    collapsed = [r for r in sorted_results if r["collapsed"]]
    ok = [r for r in sorted_results if not r["collapsed"]]
    print(f"\n{len(ok)}/{len(sorted_results)} experiments pass coverage gate")
    print(f"{len(collapsed)}/{len(sorted_results)} experiments have at least one zero-recall class")
    print()

    # Rank-shift analysis
    print("Rank shift vs legacy Macro-F1:")
    print(f"{'Experiment':<28} {'MF1→Rank':>10} {'Robust→Rank':>12} {'Shift':>8}")
    print("-" * 60)
    by_mf1 = sorted(results, key=lambda r: r["macro_f1"], reverse=True)
    mf1_ranks = {r["experiment"]: i + 1 for i, r in enumerate(by_mf1)}
    robust_ranks = {r["experiment"]: i + 1 for i, r in enumerate(sorted_results)}
    for r in sorted_results:
        name = r["experiment"]
        shift = mf1_ranks[name] - robust_ranks[name]
        direction = "↑" if shift > 0 else ("↓" if shift < 0 else "—")
        print(
            f"{name:<28} {mf1_ranks[name]:>10} {robust_ranks[name]:>12} "
            f"{shift:+d} {direction:>5}"
        )


def print_single_experiment(report: dict) -> None:
    """Print detailed metrics for a single experiment."""
    print(format_metrics_table(report, LABEL_NAMES))

    if report.get("known_oof_mf1") is not None:
        print(f"\nKnown OOF MF1:  {report['known_oof_mf1']:.4f}")
        print(f"Computed MF1:   {report['macro_f1']:.4f}")
        if abs(report["known_oof_mf1"] - report["macro_f1"]) > 0.01:
            print("  ⚠  Discrepancy — CSV may differ from training run")

    if report.get("submission_mf1") is not None:
        print(f"Submission MF1:  {report['submission_mf1']:.4f}")
        delta = report["macro_f1"] - report["submission_mf1"]
        print(f"OOF→Test gap:    {delta:+.4f}")

    # Per-class F1 comparison
    print("\nPer-class F1:")
    for c in range(NUM_CLASSES):
        name = LABEL_NAMES.get(c, f"class_{c}")
        f1 = report["per_class_f1"][c]
        rec = report["per_class_recall"][c]
        bar = "█" * int(f1 * 20) if f1 > 0 else "▏"
        print(f"  {name:<8} F1={f1:.4f}  Rec={rec:.4f}  {bar}")


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def export_results(results: list[dict], output_path: Path) -> None:
    """Export results as JSON and CSV."""
    # JSON
    json_path = output_path.with_suffix(".json")
    # Convert numpy values for JSON serialization
    serializable = []
    for r in results:
        d = dict(r)
        d["per_class_recall"] = [float(x) for x in d["per_class_recall"]]
        d["per_class_f1"] = [float(x) for x in d["per_class_f1"]]
        for k, v in d.get("per_class", {}).items():
            d["per_class"][k] = {kk: float(vv) if isinstance(vv, (np.floating, np.integer)) else vv for kk, vv in v.items()}
        serializable.append(d)
    json_path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Exported JSON to {json_path}")

    # CSV summary
    csv_path = output_path.with_suffix(".csv")
    rows = []
    for r in results:
        rows.append({
            "experiment": r["experiment"],
            "n_samples": r["n_samples"],
            "macro_f1": round(r["macro_f1"], 4),
            "macro_auc": round(r["macro_auc"], 4),
            "qwk": round(r["qwk"], 4),
            "min_class_recall": round(r["min_class_recall"], 4),
            "num_zero_recall_classes": r["num_zero_recall_classes"],
            "robust_score": round(r["robust_score"], 4),
            "collapsed": r["collapsed"],
            "known_oof_mf1": r.get("known_oof_mf1"),
            "submission_mf1": r.get("submission_mf1"),
        })
    pd.DataFrame(rows).to_csv(str(csv_path), index=False)
    print(f"Exported CSV to {csv_path}")


# ---------------------------------------------------------------------------
# Feature-set classification
# ---------------------------------------------------------------------------

# Experiments that DO use DASS features (invalid for test-time comparison
# because DASS is absent from the test set).
DASS_EXPERIMENTS: set[str] = {
    "basic_features",
    "contrastive",
    "deep_residual",
    "deep_residual_gpu",
    "final",
    "final_wider",
    "ldam",
    "mixup",
    "swa",
    "transformer",
    "xgboost",
    "ensemble_2model",
    "ensemble_equal",
    "ensemble_final",
    "nodeep_no_dass",
}

# ND-* experiments that do NOT use DASS (valid for test-time comparison).
NO_DASS_EXPERIMENTS: set[str] = {
    "no_dass",                  # ND-1: BiGRU baseline
    "nd3_focal_g2",             # ND-3: DeepResidual γ=2.0
    "nd4_cw2",                  # ND-4: class_weight=2.0
    "nd5_basic",                # ND-5: +basic features
    "nd6_dass_dropout",         # ND-6: DASS dropout (evaluated without DASS)
    "nd8_ldam",                 # ND-8: LDAM loss
    "nd9_transformer",          # ND-9: Transformer
    "nd10_prior_calibration",   # ND-10: prior calibration
    "nd11_distillation",        # ND-11: knowledge distillation
    "nd12_ensemble",            # ND-12: multi-architecture ensemble
    "nd13_tcn",                 # ND-13: TCN
    "nd14_ensemble_distilled",  # ND-14: distilled ensemble
    "nd15_distill_transformer", # ND-15: Transformer student
    "nd16_multiteacher",        # ND-16: multi-teacher distillation
    "nd_resample",              # oversampling
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-evaluate historical experiments with new metric framework"
    )
    parser.add_argument(
        "--exp", type=str, default=None,
        help="Single experiment to evaluate (directory name under artifacts/exp/)",
    )
    parser.add_argument(
        "--top", type=int, default=0,
        help="Show top-N experiments by robust_score",
    )
    parser.add_argument(
        "--export", type=str, default=None,
        help="Export results to this path (without extension; writes .json + .csv)",
    )
    parser.add_argument(
        "--no-dass", action="store_true",
        help="Only include experiments that do NOT use DASS features "
             "(valid for test-time comparison). Excludes DASS-dependent "
             "experiments whose OOF metrics are inflated by unavailable features.",
    )
    parser.add_argument(
        "--dass-only", action="store_true",
        help="Only include experiments that use DASS features.",
    )
    args = parser.parse_args()

    artifacts_dir = ARTIFACTS_DIR
    if not artifacts_dir.exists():
        print(f"Artifacts directory not found: {artifacts_dir}")
        sys.exit(1)

    # Single experiment mode
    if args.exp:
        exp_dir = artifacts_dir / args.exp
        if not exp_dir.exists():
            print(f"Experiment directory not found: {exp_dir}")
            sys.exit(1)
        report = evaluate_experiment(args.exp, exp_dir)
        if report is None:
            print(f"No OOF predictions found for {args.exp}")
            sys.exit(1)
        print_single_experiment(report)
        return

    # Batch mode: scan all experiments
    results: list[dict] = []
    skipped: list[str] = []

    for exp_dir in sorted(artifacts_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        name = exp_dir.name
        report = evaluate_experiment(name, exp_dir)
        if report is None:
            skipped.append(name)
            continue
        results.append(report)

    if not results:
        print("No experiments with OOF predictions found.")
        if skipped:
            print(f"Skipped {len(skipped)} dirs with no oof_predictions.csv: {skipped}")
        sys.exit(1)

    # Tag experiments by feature set (always, before any filtering)
    for r in results:
        if r["experiment"] in NO_DASS_EXPERIMENTS:
            r["feature_set"] = "AV-only ✓"
        elif r["experiment"] in DASS_EXPERIMENTS:
            r["feature_set"] = "DASS ⚠"
        else:
            r["feature_set"] = "unknown"

    # Feature-set filtering
    if args.no_dass:
        results = [r for r in results if r["feature_set"] == "AV-only ✓"]
        print(f"\nFiltered to {len(results)} AV-only (no-DASS) experiments.\n")
    elif args.dass_only:
        results = [r for r in results if r["feature_set"] == "DASS ⚠"]
        print(f"\nFiltered to {len(results)} DASS experiments.\n")

    if not results:
        print("No experiments match the filter criteria.")
        sys.exit(1)

    # Print comparison
    if args.top > 0:
        results = sorted(results, key=lambda r: r["robust_score"], reverse=True)[:args.top]

    print_comparison_table(results)

    if skipped:
        print(f"\nSkipped {len(skipped)} directories (no oof_predictions.csv):")
        for s in skipped:
            print(f"  - {s}")

    # Export if requested
    if args.export:
        export_path = Path(args.export)
        export_results(results, export_path)


if __name__ == "__main__":
    main()
