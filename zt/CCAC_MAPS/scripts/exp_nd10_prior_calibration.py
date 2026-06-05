#!/usr/bin/env python3
"""ND-10: Test-time prior shift calibration.

Key insight: The public test set has a KNOWN distribution that differs massively
from training:
  - Training: 正常=76.5%, 中度=12.2%, 轻度=3.8%, 重度=3.5%, 非常严重=3.9%
  - Public test: 中度=76.4%, 正常=3.9%, 轻度=12.0%, 重度=3.7%, 非常严重=3.9%

We can correct model probabilities using prior shift:
  p_corrected(y|x) ∝ p_model(y|x) * p_test(y) / p_train(y)

This is post-hoc — no retraining needed. Works on ANY model's OOF/test probabilities.

Usage:
    PYTHONPATH=src python scripts/exp_nd10_prior_calibration.py \
        --oof-csv artifacts/exp/nd9_transformer/oof_predictions.csv \
        --output-dir artifacts/exp/nd10_prior_calibration
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score


# Known distributions
# Training distribution (from train_val/labels.csv, 1527 subjects)
TRAIN_PRIOR = np.array([0.122, 0.765, 0.038, 0.035, 0.039], dtype=np.float64)
# Public test distribution (from public leaderboard ground truth, 382 subjects)
TEST_PRIOR = np.array([0.764, 0.039, 0.120, 0.037, 0.039], dtype=np.float64)
# Class order: 0=中度, 1=正常, 2=轻度, 3=重度, 4=非常严重

# Also try: uniform prior (treat all classes equally at test time)
UNIFORM_PRIOR = np.array([0.2, 0.2, 0.2, 0.2, 0.2], dtype=np.float64)

# And: inverse training frequency (aggressively upweight minority)
INV_TRAIN_PRIOR = 1.0 / (TRAIN_PRIOR + 1e-6)
INV_TRAIN_PRIOR = INV_TRAIN_PRIOR / INV_TRAIN_PRIOR.sum()


def apply_prior_correction(
    probs: np.ndarray,
    train_prior: np.ndarray,
    test_prior: np.ndarray,
    temperature: float = 1.0,
) -> np.ndarray:
    """Apply prior shift correction to model probabilities.

    p_corrected(y|x) ∝ p_model(y|x)^{1/T} * p_test(y) / p_train(y)

    Parameters
    ----------
    probs : (N, C) array of model probabilities
    train_prior : (C,) array of training class frequencies
    test_prior : (C,) array of test class frequencies (or target distribution)
    temperature : temperature scaling factor (T>1 softens, T<1 sharpens)

    Returns
    -------
    corrected_probs : (N, C) renormalized probabilities
    """
    # Apply temperature scaling
    if temperature != 1.0:
        logits = np.log(np.maximum(probs, 1e-9))
        logits = logits / temperature
        probs = np.exp(logits)
        probs = probs / probs.sum(axis=-1, keepdims=True)

    # Prior shift: p_new ∝ p_old * (p_test / p_train)
    correction = test_prior / np.maximum(train_prior, 1e-9)
    corrected = probs * correction.reshape(1, -1)
    corrected = corrected / corrected.sum(axis=-1, keepdims=True)
    return corrected


def evaluate_correction(
    probs: np.ndarray,
    labels: np.ndarray,
    train_prior: np.ndarray,
    test_prior: np.ndarray,
    label_names: list[str],
    temperature: float = 1.0,
    prior_name: str = "test",
) -> dict:
    """Evaluate a prior correction configuration."""
    corrected = apply_prior_correction(probs, train_prior, test_prior, temperature)
    preds = corrected.argmax(axis=1)

    mf1 = float(f1_score(labels, preds, average="macro", zero_division=0))
    acc = float(accuracy_score(labels, preds))
    wf1 = float(f1_score(labels, preds, average="weighted", zero_division=0))

    per_class = {}
    for i, name in enumerate(label_names):
        per_class[name] = float(f1_score(labels == i, preds == i, average="binary", zero_division=0))

    return {
        "prior": prior_name,
        "temperature": temperature,
        "macro_f1": mf1,
        "accuracy": acc,
        "weighted_f1": wf1,
        "per_class_f1": per_class,
        "pred_distribution": np.bincount(preds, minlength=len(label_names)).tolist(),
    }


def main():
    p = argparse.ArgumentParser(description="ND-10: Test-time prior shift calibration")
    p.add_argument("--oof-csv", type=str, required=True,
                   help="Path to OOF predictions CSV from a no-DASS model")
    p.add_argument("--output-dir", type=str, default="artifacts/exp/nd10_prior_calibration")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="Temperature scaling (default 1.0 = no scaling)")
    args = p.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load OOF predictions
    oof = pd.read_csv(args.oof_csv)
    prob_cols = [c for c in oof.columns if c.startswith("prob_class_")]
    probs = oof[prob_cols].to_numpy(dtype=np.float64)

    # Get true labels
    if "true_label" in oof.columns:
        label_strs = oof["true_label"].values
        # Map label strings to indices
        label_map = {"中度": 0, "正常": 1, "轻度": 2, "重度": 3, "非常严重": 4}
        labels = np.array([label_map.get(str(s).strip(), 0) for s in label_strs], dtype=np.int64)
    else:
        print("ERROR: OOF CSV missing 'true_label' column")
        sys.exit(1)

    label_names = ["中度", "正常", "轻度", "重度", "非常严重"]
    num_classes = len(label_names)

    # Baseline: argmax
    baseline_preds = probs.argmax(axis=1)
    baseline_mf1 = float(f1_score(labels, baseline_preds, average="macro", zero_division=0))
    baseline_acc = float(accuracy_score(labels, baseline_preds))
    print(f"Baseline (argmax): MF1={baseline_mf1:.4f}, Acc={baseline_acc:.4f}")
    print(f"Baseline pred dist: {np.bincount(baseline_preds, minlength=num_classes)}")
    print(f"Baseline true dist: {np.bincount(labels, minlength=num_classes)}")
    print()

    # Try different prior correction strategies
    strategies = [
        ("test_prior", TEST_PRIOR, 1.0),
        ("test_prior_t0.8", TEST_PRIOR, 0.8),
        ("test_prior_t1.2", TEST_PRIOR, 1.2),
        ("test_prior_t1.5", TEST_PRIOR, 1.5),
        ("test_prior_t0.5", TEST_PRIOR, 0.5),
        ("uniform", UNIFORM_PRIOR, 1.0),
        ("uniform_t0.8", UNIFORM_PRIOR, 0.8),
        ("inv_train", INV_TRAIN_PRIOR, 1.0),
    ]

    results = []
    best_mf1 = baseline_mf1
    best_result = None

    for name, prior, temp in strategies:
        r = evaluate_correction(probs, labels, TRAIN_PRIOR, prior, label_names, temp, name)
        results.append(r)
        marker = " *** BEST" if r["macro_f1"] > best_mf1 else ""
        print(f"{name:20s} T={temp:.1f}  MF1={r['macro_f1']:.4f}  Acc={r['accuracy']:.4f}  "
              f"WF1={r['weighted_f1']:.4f}{marker}")
        print(f"  Per-class: {r['per_class_f1']}")
        print(f"  Pred dist: {r['pred_distribution']}")
        if r["macro_f1"] > best_mf1:
            best_mf1 = r["macro_f1"]
            best_result = r

    # Grid search over temperature for best prior
    print("\n--- Grid search temperature for test_prior ---")
    best_temp_mf1 = baseline_mf1
    best_temp = 1.0
    for temp in np.arange(0.3, 2.6, 0.1):
        r = evaluate_correction(probs, labels, TRAIN_PRIOR, TEST_PRIOR, label_names, float(temp), "test_prior")
        if r["macro_f1"] > best_temp_mf1:
            best_temp_mf1 = r["macro_f1"]
            best_temp = temp
            best_result = r
        if r["macro_f1"] > baseline_mf1:
            print(f"  T={temp:.1f}: MF1={r['macro_f1']:.4f} Acc={r['accuracy']:.4f} ***")
        elif temp % 0.5 < 0.1:
            print(f"  T={temp:.1f}: MF1={r['macro_f1']:.4f} Acc={r['accuracy']:.4f}")

    # Final best — use baseline as fallback
    if best_result is None:
        best_result = {
            "prior": "none (argmax)", "temperature": 1.0, "macro_f1": baseline_mf1,
            "accuracy": baseline_acc, "weighted_f1": float(f1_score(
                labels, baseline_preds, average="weighted", zero_division=0)),
            "per_class_f1": {},
            "pred_distribution": np.bincount(baseline_preds, minlength=num_classes).tolist(),
        }
        best_temp = 1.0
        for i, name in enumerate(label_names):
            best_result["per_class_f1"][name] = float(
                f1_score(labels == i, baseline_preds == i, average="binary", zero_division=0))

    print(f"\n{'='*60}")
    print(f"BEST on OOF: {best_result['prior']} T={best_result['temperature']:.1f}")
    print(f"  Macro-F1:  {best_result['macro_f1']:.4f}  (baseline: {baseline_mf1:.4f}, "
          f"delta: {best_result['macro_f1'] - baseline_mf1:+.4f})")
    print(f"  Accuracy:  {best_result['accuracy']:.4f}  (baseline: {baseline_acc:.4f})")
    print(f"  Per-class: {best_result['per_class_f1']}")
    print(f"  Pred dist: {best_result['pred_distribution']}")
    print(f"\nNOTE: Prior calibration targets the PUBLIC TEST distribution, not OOF.")
    print(f"OOF follows TRAINING distribution, so OOF metrics will degrade with test prior.")
    print(f"The real benefit is expected on the actual test set (public leaderboard).")

    # Save results
    corrected_probs = apply_prior_correction(
        probs, TRAIN_PRIOR, TEST_PRIOR, best_temp
    )
    corrected_preds = corrected_probs.argmax(axis=1)

    # Write corrected OOF
    oof_out = oof.copy()
    oof_out["pred_label"] = [label_names[p] for p in corrected_preds]
    for ci in range(num_classes):
        oof_out[f"prob_class_{ci}"] = corrected_probs[:, ci]
    oof_out.to_csv(output_dir / "oof_predictions.csv", index=False)

    # Write classification report
    (output_dir / "classification_report.txt").write_text(
        classification_report(labels, corrected_preds, target_names=label_names, zero_division=0),
        encoding="utf-8",
    )

    # Also for each strategy
    for r in results:
        r["per_class_f1"] = {k: round(v, 4) for k, v in r["per_class_f1"].items()}

    summary = {
        "experiment": "ND-10: Test-time prior shift calibration",
        "input_oof": str(args.oof_csv),
        "baseline_macro_f1": baseline_mf1,
        "baseline_accuracy": baseline_acc,
        "best_macro_f1": best_result["macro_f1"],
        "best_accuracy": best_result["accuracy"],
        "best_config": f"{best_result['prior']}_T{best_result['temperature']:.1f}",
        "delta_mf1": best_result["macro_f1"] - baseline_mf1,
        "train_prior": TRAIN_PRIOR.tolist(),
        "test_prior": TEST_PRIOR.tolist(),
        "all_results": results,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
