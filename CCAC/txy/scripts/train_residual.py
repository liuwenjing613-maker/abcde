#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from txy.models.residual import ResidualFusionModel
from txy.models.stagewise import StageWiseLongitudinalModel
from txy.training.trainer import TrainConfig, train_longitudinal


def build_residual_factory(alpha: float):
    def factory(audio_dim, video_dim, history_score_dim, history_level_slots, num_classes):
        stagewise = StageWiseLongitudinalModel(
            audio_dim=audio_dim,
            video_dim=video_dim,
            history_score_dim=history_score_dim,
            history_level_slots=history_level_slots,
            num_classes=num_classes,
            use_history=True,
        )
        return ResidualFusionModel(
            tabular_dim=history_score_dim,
            stagewise_model=stagewise,
            num_classes=num_classes,
            alpha=alpha,
        )

    return factory


def main() -> None:
    parser = argparse.ArgumentParser(description="Train residual tabular+multimodal fusion (Experiment 4)")
    parser.add_argument("--dataset-path", type=str, default="/home/adodas/dataset_ccac")
    parser.add_argument("--output-dir", type=str, default="artifacts/residual_v2")
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--class-weight-power", type=float, default=1.5)
    parser.add_argument("--calibrate-bias", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = TrainConfig(
        dataset_path=str(Path(args.dataset_path).resolve()),
        output_dir=str(Path(args.output_dir).resolve()),
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_folds=args.num_folds,
        seed=args.seed,
        use_history=True,
        class_weight_power=args.class_weight_power,
        calibrate_bias=args.calibrate_bias,
    )
    result = train_longitudinal(
        config,
        build_residual_factory(args.alpha),
        model_kind="residual",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
