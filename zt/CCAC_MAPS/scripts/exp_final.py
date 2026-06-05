#!/usr/bin/env python3
"""Final optimized model: wider DeeperResidual + basic features + 8 heads + 4 blocks."""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from ccac.baselines.anxiety_baseline import BaselineConfig
from ccac.experiments.basic_features import BasicFeaturesConfig, train_basic_features

def main():
    p = argparse.ArgumentParser(description="Final optimized model")
    p.add_argument("--dataset-path", type=str, default="datasets")
    p.add_argument("--output-dir", type=str, default="artifacts/exp/final")
    p.add_argument("--audio-feature-name", type=str, default="audio_wavlm_base")
    p.add_argument("--video-feature-name", type=str, default="video_clip_base")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--hidden-dim", type=int, default=320)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--num-residual-blocks", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--num-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-audio-basic", action="store_true")
    p.add_argument("--no-video-basic", action="store_true")
    p.add_argument("--no-calibrate", action="store_true")
    args = p.parse_args()

    print(f"CUDA available: {__import__('torch').cuda.is_available()}")
    print(f"Hidden dim: {args.hidden_dim}, Heads: {args.num_heads}, Blocks: {args.num_residual_blocks}")

    bc = BaselineConfig(
        dataset_path=str(Path(args.dataset_path).resolve()),
        output_dir=str(Path(args.output_dir).resolve()),
        audio_feature_name=args.audio_feature_name,
        video_feature_name=args.video_feature_name,
        hidden_dim=args.hidden_dim, dropout=args.dropout,
        learning_rate=args.learning_rate, batch_size=args.batch_size,
        epochs=args.epochs, patience=args.patience,
        num_folds=args.num_folds, seed=args.seed, device=args.device,
        feature_cache=True,
    )
    bfc = BasicFeaturesConfig(
        dass_scheme="none", focal_gamma=1.0,
        num_heads=args.num_heads, num_residual_blocks=args.num_residual_blocks,
        use_audio_basic=not args.no_audio_basic,
        use_video_basic=not args.no_video_basic,
        calibrate_thresholds=not args.no_calibrate,
    )

    result = train_basic_features(bc, bfc)
    m = result["overall_oof_metrics"]
    print(f"\nFINAL RESULT: MF1={m['macro_f1']:.4f} Acc={m['accuracy']:.4f} WF1={m['weighted_f1']:.4f}")
    return result

if __name__ == "__main__":
    main()
