#!/usr/bin/env python3
"""ND-13: Temporal Convolutional Network (TCN) for stage modeling.

Usage:
    PYTHONPATH=src python scripts/exp_nd13_tcn.py \
        --dataset-path datasets \
        --output-dir artifacts/exp/nd13_tcn \
        --device cuda
"""

import argparse, json
from pathlib import Path
from ccac.baselines.anxiety_baseline import BaselineConfig
from ccac.experiments.tcn_temporal import TCNConfig, train_tcn


def main():
    p = argparse.ArgumentParser(description="ND-13: TCN Temporal Encoder")
    p.add_argument("--dataset-path", default="datasets")
    p.add_argument("--output-dir", default="artifacts/exp/nd13_tcn")
    p.add_argument("--audio-feature-name", default="audio_wavlm_base")
    p.add_argument("--video-feature-name", default="video_clip_base")
    p.add_argument("--device", default="cuda")
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--tcn-layers", type=int, default=3)
    p.add_argument("--kernel-size", type=int, default=3)
    p.add_argument("--num-residual-blocks", type=int, default=2)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--num-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-calibrate", action="store_true")
    args = p.parse_args()

    bc = BaselineConfig(
        dataset_path=str(Path(args.dataset_path).resolve()),
        output_dir=str(Path(args.output_dir).resolve()),
        audio_feature_name=args.audio_feature_name,
        video_feature_name=args.video_feature_name,
        hidden_dim=args.hidden_dim, dropout=args.dropout,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size, epochs=args.epochs,
        patience=args.patience, num_folds=args.num_folds,
        seed=args.seed, device=args.device,
    )
    tc = TCNConfig(
        focal_gamma=args.focal_gamma,
        tcn_layers=args.tcn_layers,
        kernel_size=args.kernel_size,
        num_residual_blocks=args.num_residual_blocks,
        calibrate_thresholds=not args.no_calibrate,
    )

    result = train_tcn(bc, tc)
    print(json.dumps(result["overall_oof_metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
