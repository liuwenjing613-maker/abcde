#!/usr/bin/env python3
"""ND-11: Knowledge Distillation — DASS teacher → No-DASS student.

Usage:
    PYTHONPATH=src python scripts/exp_nd11_distillation.py \
        --dataset-path datasets \
        --output-dir artifacts/exp/nd11_distillation \
        --teacher-oof artifacts/dass/focal_g1/oof_predictions.csv \
        --device cuda
"""

import argparse, json, sys
from pathlib import Path

from ccac.baselines.anxiety_baseline import BaselineConfig
from ccac.experiments.knowledge_distillation import DistillationConfig, train_distillation


def main():
    p = argparse.ArgumentParser(description="ND-11: Knowledge Distillation")
    p.add_argument("--dataset-path", default="datasets")
    p.add_argument("--output-dir", default="artifacts/exp/nd11_distillation")
    p.add_argument("--teacher-oof", required=True,
                   help="Path to teacher OOF predictions CSV (e.g. artifacts/dass/focal_g1/oof_predictions.csv)")
    p.add_argument("--audio-feature-name", default="audio_wavlm_base")
    p.add_argument("--video-feature-name", default="video_clip_base")
    p.add_argument("--device", default="cuda")
    p.add_argument("--alpha", type=float, default=0.9)
    p.add_argument("--temperature", type=float, default=3.0)
    p.add_argument("--student-focal-gamma", type=float, default=2.0)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--num-residual-blocks", type=int, default=3)
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
    dc = DistillationConfig(
        alpha=args.alpha, temperature=args.temperature,
        student_focal_gamma=args.student_focal_gamma,
        num_heads=args.num_heads,
        num_residual_blocks=args.num_residual_blocks,
        calibrate_thresholds=not args.no_calibrate,
    )

    result = train_distillation(bc, dc, teacher_oof_path=args.teacher_oof)
    print(json.dumps(result["overall_oof_metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
