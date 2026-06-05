#!/usr/bin/env python3
"""Train multiple DASS+Focal models with different audio/video feature pairs
and ensemble their predictions.

Usage
-----
PYTHONPATH=src python scripts/train_ensemble.py \
    --dataset-path datasets \
    --output-dir artifacts/ensemble \
    --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ccac.baselines.anxiety_baseline import (
    BaselineConfig,
    _is_release_dataset,
    _load_release_train_val,
    _encode_labels,
    _build_folds,
)
from ccac.baselines.dass_baseline import (
    DASSConfig,
    train_dass_baseline,
    _extract_dass_features,
)

# Diverse feature pairs — different audio models × different video models
DEFAULT_PAIRS: list[tuple[str, str]] = [
    ("audio_wavlm_base", "video_dinov2_small"),         # default baseline
    ("audio_wavlm_base", "video_dinov2_base"),           # bigger video
    ("audio_wav2vec2_xlsr_chinese", "video_dinov2_small"),  # bigger audio
    ("audio_chinese_hubert_base", "video_clip_large"),    # different both
]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train ensemble of feature-pair models")
    p.add_argument("--dataset-path", type=str, default="datasets")
    p.add_argument("--output-dir", type=str, default="artifacts/ensemble")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dass-scheme", type=str, default="none")
    p.add_argument("--focal-gamma", type=float, default=1.0)
    p.add_argument("--pairs", type=str, nargs="*",
                   help="Space-separated audio:video pairs, e.g. audio_wavlm_base:video_dinov2_small")
    return p


def parse_pairs(args) -> list[tuple[str, str]]:
    if args.pairs:
        return [tuple(p.split(":")) for p in args.pairs]
    return DEFAULT_PAIRS


def main() -> None:
    args = build_parser().parse_args()
    pairs = parse_pairs(args)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Ensemble: {len(pairs)} feature pairs")
    for audio, video in pairs:
        print(f"  {audio} + {video}")

    # --- Train each model ---
    pair_results: list[dict] = []
    for i, (audio, video) in enumerate(pairs):
        pair_dir = output_dir / f"pair_{i}_{audio}__{video}"
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(pairs)}] {audio} + {video}")
        print(f"{'='*60}")

        config = BaselineConfig(
            dataset_path=str(Path(args.dataset_path).resolve()),
            output_dir=str(pair_dir),
            audio_feature_name=audio,
            video_feature_name=video,
            device=args.device,
        )
        dass_config = DASSConfig(
            dass_scheme=args.dass_scheme,
            focal_gamma=args.focal_gamma,
        )

        result = train_dass_baseline(config, dass_config)
        result["audio"] = audio
        result["video"] = video
        pair_results.append(result)
        print(f"  -> Macro-F1: {result['overall_oof_metrics']['macro_f1']:.4f}")

    # --- Ensemble OOF ---
    dataset_path = Path(args.dataset_path).resolve()
    if _is_release_dataset(dataset_path):
        temp_config = BaselineConfig(
            dataset_path=str(dataset_path),
            output_dir=str(output_dir),
        )
        frame, _, _, label_mapping, _ = _load_release_train_val(temp_config, dataset_path)
        labels = frame["_label_index"].to_numpy(dtype=np.int64)
    else:
        print("Skipping OOF ensemble: not a release dataset")
        return

    # Load OOF probabilities from each pair
    all_oof_probs = []
    for i, (audio, video) in enumerate(pairs):
        pair_dir = output_dir / f"pair_{i}_{audio}__{video}"
        oof_csv = pair_dir / "oof_predictions.csv"
        if not oof_csv.exists():
            print(f"  WARNING: {oof_csv} not found, skipping")
            continue
        oof = pd.read_csv(oof_csv)
        prob_cols = [c for c in oof.columns if c.startswith("prob_class_")]
        probs = oof[prob_cols].to_numpy(dtype=np.float32)
        all_oof_probs.append(probs)
        print(f"  {audio}+{video}: MF1={pair_results[i]['overall_oof_metrics']['macro_f1']:.4f}")

    if not all_oof_probs:
        print("No OOF predictions found")
        return

    ensemble_probs = np.mean(np.stack(all_oof_probs, axis=0), axis=0)
    ensemble_preds = ensemble_probs.argmax(axis=1)

    from sklearn.metrics import accuracy_score, f1_score, classification_report
    label_by_index = {i: label for label, i in label_mapping.items()}
    pred_labels = [label_by_index[int(p)] for p in ensemble_preds]
    true_labels = [label_by_index[int(l)] for l in labels]

    print(f"\n{'='*60}")
    print("ENSEMBLE OOF RESULTS")
    print(f"{'='*60}")
    print(f"Pairs: {len(all_oof_probs)}")
    print(f"Macro-F1:  {f1_score(labels, ensemble_preds, average='macro', zero_division=0):.4f}")
    print(f"Accuracy:  {accuracy_score(labels, ensemble_preds):.4f}")
    print(f"Weighted:  {f1_score(labels, ensemble_preds, average='weighted', zero_division=0):.4f}")
    print(f"\n{classification_report(labels, ensemble_preds, target_names=list(label_mapping.keys()), zero_division=0)}")

    # Save ensemble OOF
    oof_df = frame[[
        c for c in frame.columns if c in ["anon_school", "anon_class", "anon_person", "subject_id"]
    ]].copy()
    if "subject_id" not in oof_df.columns:
        oof_df["subject_id"] = frame[["anon_school", "anon_class", "anon_person"]].agg("/".join, axis=1)
    oof_df["true_label"] = true_labels
    oof_df["pred_label"] = pred_labels
    for ci in range(len(label_mapping)):
        oof_df[f"prob_class_{ci}"] = ensemble_probs[:, ci]
    oof_df.to_csv(output_dir / "ensemble_oof_predictions.csv", index=False)

    # --- Ensemble test ---
    all_test_probs = []
    for i, (audio, video) in enumerate(pairs):
        pair_dir = output_dir / f"pair_{i}_{audio}__{video}"
        test_csv = pair_dir / "test_predictions.csv"
        if not test_csv.exists():
            continue
        test = pd.read_csv(test_csv)
        prob_cols = [c for c in test.columns if c.startswith("prob_class_")]
        all_test_probs.append(test[prob_cols].to_numpy(dtype=np.float32))

    if all_test_probs:
        test_ensemble = np.mean(np.stack(all_test_probs, axis=0), axis=0)
        test_preds = test_ensemble.argmax(axis=1)
        test_df = test[["anon_school", "anon_class", "anon_person"]].copy()
        if "subject_id" in test.columns:
            test_df["subject_id"] = test["subject_id"]
        test_df["pred_label"] = [label_by_index[int(p)] for p in test_preds]
        for ci in range(len(label_mapping)):
            test_df[f"prob_class_{ci}"] = test_ensemble[:, ci]
        test_df.to_csv(output_dir / "ensemble_test_predictions.csv", index=False)
        print(f"\nEnsemble test predictions saved: {output_dir / 'ensemble_test_predictions.csv'}")

    # --- Summary ---
    summary = {
        "pairs": [f"{a}+{v}" for a, v in pairs],
        "individual_macro_f1": [r["overall_oof_metrics"]["macro_f1"] for r in pair_results],
        "ensemble_macro_f1": float(f1_score(labels, ensemble_preds, average="macro", zero_division=0)),
        "ensemble_accuracy": float(accuracy_score(labels, ensemble_preds)),
        "ensemble_weighted_f1": float(f1_score(labels, ensemble_preds, average="weighted", zero_division=0)),
    }
    (output_dir / "ensemble_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nSummary: {output_dir / 'ensemble_summary.json'}")


if __name__ == "__main__":
    main()
