#!/usr/bin/env python3
"""Trial 21: Stochastic Weight Averaging for anxiety classification.

Usage:
    PYTHONPATH=src python scripts/exp_swa.py \
        --dataset-path datasets \
        --output-dir artifacts/exp/swa \
        --device cuda
"""

from __future__ import annotations

import argparse, json
from pathlib import Path
from ccac.baselines.anxiety_baseline import BaselineConfig
from ccac.experiments.swa_training import SWAConfig, train_swa


def main():
    p = argparse.ArgumentParser(description="Trial 21: SWA Training")
    p.add_argument("--dataset-path", type=str, default="datasets")
    p.add_argument("--output-dir", type=str, default="artifacts/exp/swa")
    p.add_argument("--audio-feature-name", type=str, default="audio_wavlm_base")
    p.add_argument("--video-feature-name", type=str, default="video_clip_base")
    p.add_argument("--dass-scheme", type=str, default="none")
    p.add_argument("--focal-gamma", type=float, default=1.0)
    p.add_argument("--swa-start-epoch", type=int, default=40)
    p.add_argument("--swa-frequency", type=int, default=1)
    p.add_argument("--swa-lr", type=float, default=1e-4)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--num-residual-blocks", type=int, default=3)
    p.add_argument("--no-calibrate", action="store_true")
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
    args = p.parse_args()

    bc = BaselineConfig(
        dataset_path=str(Path(args.dataset_path).resolve()),
        output_dir=str(Path(args.output_dir).resolve()),
        audio_feature_name=args.audio_feature_name,
        video_feature_name=args.video_feature_name,
        hidden_dim=args.hidden_dim, dropout=args.dropout,
        learning_rate=args.learning_rate, weight_decay=args.weight_decay,
        class_weight_power=args.class_weight_power,
        label_smoothing=args.label_smoothing,
        batch_size=args.batch_size, epochs=args.epochs,
        patience=args.patience, num_folds=args.num_folds,
        num_workers=args.num_workers, seed=args.seed,
        device=args.device, feature_cache=not args.no_feature_cache,
    )
    sc = SWAConfig(
        dass_scheme=args.dass_scheme,
        focal_gamma=args.focal_gamma,
        num_heads=args.num_heads,
        num_residual_blocks=args.num_residual_blocks,
        calibrate_thresholds=not args.no_calibrate,
        swa_start_epoch=args.swa_start_epoch,
        swa_frequency=args.swa_frequency,
        swa_lr=args.swa_lr,
    )

    result = train_swa(bc, sc)
    print(json.dumps(result["overall_oof_metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
