#!/usr/bin/env python3
"""ND-12: Multi-architecture ensemble of no-DASS models.

Combines OOF probabilities from diverse architectures:
  - DeepResidual (ND-3: γ=2.0) — cross-stage attention + diff features
  - Transformer (ND-9: γ=2.0) — self-attention temporal encoder
  - BiGRU baseline (ND-1) — simple recurrent encoder

Different architectures make different errors — ensemble averages them out.

Usage:
    PYTHONPATH=src python scripts/exp_nd12_ensemble.py \
        --output-dir artifacts/exp/nd12_ensemble
"""

import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
from itertools import combinations


# Known OOF paths for no-DASS models
DEFAULT_MODELS = {
    "ND3_DeepResidual": "artifacts/exp/nd3_focal_g2/oof_predictions.csv",
    "ND9_Transformer": "artifacts/exp/nd9_transformer/oof_predictions.csv",
    "ND1_BiGRU": "artifacts/baselines/anxiety_wavlm_dinov2_small/oof_predictions.csv",
    "ND2_DeepRes_g1": "artifacts/exp/no_dass/oof_predictions.csv",
    "ND5_Basic": "artifacts/exp/nd5_basic/oof_predictions.csv",
}

LABEL_MAP = {"中度": 0, "正常": 1, "轻度": 2, "重度": 3, "非常严重": 4}
LABEL_NAMES = ["中度", "正常", "轻度", "重度", "非常严重"]


def load_oof(path, label_map):
    """Load OOF probabilities and labels from a model's OOF CSV."""
    oof = pd.read_csv(path)
    prob_cols = [c for c in oof.columns if c.startswith("prob_class_")]
    probs = oof[prob_cols].to_numpy(dtype=np.float64)
    labels = np.array([label_map.get(str(s).strip(), 0) for s in oof["true_label"].values])
    return probs, labels


def grid_search_weights(probs_list, labels, step=0.05):
    """Grid search optimal ensemble weights to maximize macro-F1."""
    n = len(probs_list)
    best_mf1 = -1
    best_weights = None

    if n == 2:
        for w in np.arange(0, 1.001, step):
            weights = [w, 1 - w]
            ensemble = sum(w * p for w, p in zip(weights, probs_list))
            preds = ensemble.argmax(axis=1)
            mf1 = f1_score(labels, preds, average="macro", zero_division=0)
            if mf1 > best_mf1:
                best_mf1 = mf1
                best_weights = weights
    elif n == 3:
        for w1 in np.arange(0, 1.001, step):
            for w2 in np.arange(0, 1.001 - w1, step):
                w3 = 1 - w1 - w2
                weights = [w1, w2, w3]
                ensemble = sum(w * p for w, p in zip(weights, probs_list))
                preds = ensemble.argmax(axis=1)
                mf1 = f1_score(labels, preds, average="macro", zero_division=0)
                if mf1 > best_mf1:
                    best_mf1 = mf1
                    best_weights = weights
    else:
        # Uniform as fallback
        weights = [1.0 / n] * n
        ensemble = sum(w * p for w, p in zip(weights, probs_list))
        preds = ensemble.argmax(axis=1)
        best_mf1 = f1_score(labels, preds, average="macro", zero_division=0)
        best_weights = weights

    return best_weights, best_mf1


def evaluate_ensemble(probs_list, weights, labels):
    """Evaluate ensemble with given weights."""
    ensemble = sum(w * p for w, p in zip(weights, probs_list))
    preds = ensemble.argmax(axis=1)
    mf1 = float(f1_score(labels, preds, average="macro", zero_division=0))
    acc = float(accuracy_score(labels, preds))
    wf1 = float(f1_score(labels, preds, average="weighted", zero_division=0))
    per_class = {}
    for i, name in enumerate(LABEL_NAMES):
        per_class[name] = float(f1_score(labels == i, preds == i, average="binary", zero_division=0))
    return {"macro_f1": mf1, "accuracy": acc, "weighted_f1": wf1,
            "per_class_f1": per_class, "pred_distribution": np.bincount(preds, minlength=5).tolist()}


def main():
    p = argparse.ArgumentParser(description="ND-12: Multi-architecture ensemble")
    p.add_argument("--output-dir", default="artifacts/exp/nd12_ensemble")
    p.add_argument("--model-oofs", nargs="*",
                   help="ModelName=path pairs, e.g. ND3=artifacts/exp/nd3_focal_g2/oof_predictions.csv")
    p.add_argument("--grid-step", type=float, default=0.05)
    args = p.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse model OOFs
    if args.model_oofs:
        model_paths = {}
        for item in args.model_oofs:
            name, path = item.split("=", 1)
            model_paths[name] = path
    else:
        # Check which defaults exist
        model_paths = {}
        for name, path in DEFAULT_MODELS.items():
            if Path(path).exists():
                model_paths[name] = path

    if len(model_paths) < 2:
        print(f"ERROR: Need at least 2 models, found {len(model_paths)}")
        print("Available: ", list(model_paths.keys()))
        sys.exit(1)

    print(f"Loading {len(model_paths)} models:")
    probs_list = []
    labels = None
    model_names = []
    individual_metrics = []

    for name, path in model_paths.items():
        probs, lbls = load_oof(Path(path), LABEL_MAP)
        if labels is None:
            labels = lbls
        probs_list.append(probs)
        model_names.append(name)

        # Individual metrics
        preds = probs.argmax(axis=1)
        mf1 = float(f1_score(labels, preds, average="macro", zero_division=0))
        acc = float(accuracy_score(labels, preds))
        wf1 = float(f1_score(labels, preds, average="weighted", zero_division=0))
        individual_metrics.append({"name": name, "macro_f1": mf1, "accuracy": acc, "weighted_f1": wf1})
        print(f"  {name:30s} MF1={mf1:.4f}  Acc={acc:.4f}  WF1={wf1:.4f}  "
              f"pred_dist={np.bincount(preds, minlength=5)}")

    print()

    results = []

    # Try all combinations of 2+ models
    all_indices = list(range(len(model_names)))
    for r in range(2, len(all_indices) + 1):
        for combo in combinations(all_indices, r):
            combo_names = [model_names[i] for i in combo]
            combo_probs = [probs_list[i] for i in combo]
            combo_label = " + ".join(combo_names)

            # Uniform
            uniform_w = [1.0 / len(combo)] * len(combo)
            uniform_m = evaluate_ensemble(combo_probs, uniform_w, labels)
            results.append({"ensemble": combo_label, "method": "uniform",
                           "weights": uniform_w, **uniform_m})

            # Grid search
            best_w, best_mf1 = grid_search_weights(combo_probs, labels, args.grid_step)
            best_m = evaluate_ensemble(combo_probs, best_w, labels)
            results.append({"ensemble": combo_label, "method": "grid_search",
                           "weights": [round(w, 4) for w in best_w], **best_m})

            print(f"{combo_label}")
            print(f"  Uniform:     MF1={uniform_m['macro_f1']:.4f} Acc={uniform_m['accuracy']:.4f}")
            print(f"  Grid search: MF1={best_m['macro_f1']:.4f} Acc={best_m['accuracy']:.4f} "
                  f"weights={[round(w, 3) for w in best_w]}")

    # Find best
    best = max(results, key=lambda r: r["macro_f1"])
    print(f"\n{'='*60}")
    print(f"BEST: {best['ensemble']} ({best['method']})")
    print(f"  Macro-F1: {best['macro_f1']:.4f}")
    print(f"  Accuracy: {best['accuracy']:.4f}")
    print(f"  Weights:  {best['weights']}")
    print(f"  Per-class: {best['per_class_f1']}")
    print(f"  Pred dist: {best['pred_distribution']}")

    # Save best ensemble OOF
    best_combo_names = best["ensemble"].split(" + ")
    best_indices = [model_names.index(n) for n in best_combo_names]
    best_probs = [probs_list[i] for i in best_indices]
    best_ensemble_probs = sum(w * p for w, p in zip(best["weights"], best_probs))
    best_preds = best_ensemble_probs.argmax(axis=1)

    # Use the first model's OOF CSV as template for subject IDs
    first_path = list(model_paths.values())[0]
    oof_template = pd.read_csv(first_path)
    oof_out = oof_template[["subject_id", "true_label"]].copy()
    oof_out["pred_label"] = [LABEL_NAMES[p] for p in best_preds]
    for ci in range(5):
        oof_out[f"prob_class_{ci}"] = best_ensemble_probs[:, ci]
    oof_out.to_csv(output_dir / "oof_predictions.csv", index=False)

    (output_dir / "classification_report.txt").write_text(
        classification_report(labels, best_preds, target_names=LABEL_NAMES, zero_division=0),
        encoding="utf-8")

    # Compare with individual models
    best_individual = max(individual_metrics, key=lambda m: m["macro_f1"])

    summary = {
        "experiment": "ND-12: Multi-architecture ensemble",
        "models_used": model_names,
        "individual_metrics": individual_metrics,
        "best_individual_mf1": best_individual["macro_f1"],
        "best_ensemble": best,
        "delta_vs_best_individual": best["macro_f1"] - best_individual["macro_f1"],
        "all_results": results,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2))

    print(f"\nDelta vs best individual ({best_individual['name']}): "
          f"{best['macro_f1'] - best_individual['macro_f1']:+.4f}")
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
