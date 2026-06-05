#!/usr/bin/env python3
"""Trial 3: Add audio_basic + video_basic as additional per-clip features.

Usage:
    PYTHONPATH=src python scripts/exp_basic_features.py \
        --dataset-path datasets \
        --output-dir artifacts/exp/basic_features \
        --device cuda
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
from ccac.baselines.anxiety_baseline import BaselineConfig
from ccac.experiments.basic_features import BasicFeaturesConfig, train_basic_features


def main():
    p = argparse.ArgumentParser(description="Trial 3: +audio_basic +video_basic")
    p.add_argument("--dataset-path", type=str, default="datasets")
    p.add_argument("--output-dir", type=str, default="artifacts/exp/basic_features")
    p.add_argument("--audio-feature-name", type=str, default="audio_wavlm_base")
    p.add_argument("--video-feature-name", type=str, default="video_clip_base")
    p.add_argument("--dass-scheme", type=str, default="none")
    p.add_argument("--focal-gamma", type=float, default=1.0)
    p.add_argument("--no-audio-basic", action="store_true")
    p.add_argument("--no-video-basic", action="store_true")
    p.add_argument("--no-calibrate", action="store_true")
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--num-folds", type=int, default=5)
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
        batch_size=args.batch_size, epochs=args.epochs,
        patience=args.patience, num_folds=args.num_folds,
        seed=args.seed, device=args.device,
        feature_cache=not args.no_feature_cache,
    )
    bfc = BasicFeaturesConfig(
        dass_scheme=args.dass_scheme, focal_gamma=args.focal_gamma,
        use_audio_basic=not args.no_audio_basic,
        use_video_basic=not args.no_video_basic,
        calibrate_thresholds=not args.no_calibrate,
    )

    result = train_basic_features(bc, bfc)
    print(json.dumps(result["overall_oof_metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
