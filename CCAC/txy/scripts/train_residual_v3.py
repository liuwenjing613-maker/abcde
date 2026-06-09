#!/usr/bin/env python3
"""Train Residual v3: external tabular anchor + multimodal with history dropout."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from txy.training.trainer_v3 import TrainV3Config, train_residual_v3


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Residual v3 (external tabular anchor)")
    parser.add_argument("--dataset-path", type=str, default="/home/adodas/dataset_ccac")
    parser.add_argument("--output-dir", type=str, default="artifacts/residual_v3")
    parser.add_argument("--anchor-model-type", type=str, default="lightgbm", choices=["lightgbm", "mlp"])
    parser.add_argument("--alpha-with-history", type=float, default=0.25)
    parser.add_argument("--alpha-missing-history", type=float, default=1.0)
    parser.add_argument("--history-dropout-prob", type=float, default=0.5)
    parser.add_argument("--class-weight-power", type=float, default=1.0)
    parser.add_argument("--calibrate-bias", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = TrainV3Config(
        dataset_path=str(Path(args.dataset_path).resolve()),
        output_dir=str(Path(args.output_dir).resolve()),
        anchor_model_type=args.anchor_model_type,
        alpha_with_history=args.alpha_with_history,
        alpha_missing_history=args.alpha_missing_history,
        history_dropout_prob=args.history_dropout_prob,
        class_weight_power=args.class_weight_power,
        calibrate_bias=args.calibrate_bias,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_folds=args.num_folds,
        seed=args.seed,
    )
    result = train_residual_v3(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
