#!/usr/bin/env python3
"""Trial 1: Transformer-based temporal encoder for CCAC MAPS.

Usage:
    PYTHONPATH=src python scripts/exp_transformer.py \
        --dataset-path datasets \
        --output-dir artifacts/exp/transformer \
        --device cuda
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ccac.baselines.anxiety_baseline import BaselineConfig
from ccac.experiments.transformer_temporal import TransformerConfig, train_transformer_baseline


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Trial 1: Transformer temporal encoder")
    p.add_argument("--dataset-path", type=str, default="datasets")
    p.add_argument("--output-dir", type=str, default="artifacts/exp/transformer")
    p.add_argument("--audio-feature-name", type=str, default="audio_wavlm_base")
    p.add_argument("--video-feature-name", type=str, default="video_clip_base")
    p.add_argument("--target-label-column", type=str, default="t4_anxiety_level")
    p.add_argument("--dass-scheme", type=str, default="none",
                   choices=["none", "scores_a", "scores_das", "encoder"])
    p.add_argument("--focal-gamma", type=float, default=1.0)
    p.add_argument("--transformer-dim", type=int, default=256)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--no-calibrate", action="store_true")
    # Standard hyperparams
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--class-weight-power", type=float, default=1.0)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--num-folds", type=int, default=5)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--no-feature-cache", action="store_true")
    return p


def main():
    args = build_parser().parse_args()

    baseline_config = BaselineConfig(
        dataset_path=str(Path(args.dataset_path).resolve()),
        output_dir=str(Path(args.output_dir).resolve()),
        audio_feature_name=args.audio_feature_name,
        video_feature_name=args.video_feature_name,
        target_label_column=args.target_label_column,
        hidden_dim=args.hidden_dim,
        temporal_hidden_dim=args.transformer_dim,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        class_weight_power=args.class_weight_power,
        label_smoothing=args.label_smoothing,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        num_folds=args.num_folds,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
        feature_cache=not args.no_feature_cache,
    )

    transformer_config = TransformerConfig(
        dass_scheme=args.dass_scheme,
        focal_gamma=args.focal_gamma,
        transformer_dim=args.transformer_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        calibrate_thresholds=not args.no_calibrate,
    )

    result = train_transformer_baseline(baseline_config, transformer_config)
    print(json.dumps(result["overall_oof_metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
