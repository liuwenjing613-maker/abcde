#!/usr/bin/env python3
"""Train StageWise v4: mm-only student + tabular teacher distillation + ordinal aux."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from txy.training.trainer_v4 import TrainV4Config, train_stagewise_v4


def main() -> None:
    parser = argparse.ArgumentParser(description="Train StageWise v4 (tabular teacher distillation)")
    parser.add_argument("--dataset-path", type=str, default="/home/adodas/dataset_ccac")
    parser.add_argument("--output-dir", type=str, default="artifacts/stagewise_v4")
    parser.add_argument("--anchor-model-type", type=str, default="lightgbm", choices=["lightgbm", "mlp"])
    parser.add_argument("--kd-temperature", type=float, default=2.0)
    parser.add_argument("--kd-weight", type=float, default=0.4)
    parser.add_argument("--ordinal-weight", type=float, default=0.2)
    parser.add_argument("--class-weight-power", type=float, default=1.0)
    parser.add_argument("--calibrate-bias", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = TrainV4Config(
        dataset_path=str(Path(args.dataset_path).resolve()),
        output_dir=str(Path(args.output_dir).resolve()),
        anchor_model_type=args.anchor_model_type,
        kd_temperature=args.kd_temperature,
        kd_weight=args.kd_weight,
        ordinal_weight=args.ordinal_weight,
        class_weight_power=args.class_weight_power,
        calibrate_bias=args.calibrate_bias,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_folds=args.num_folds,
        seed=args.seed,
    )
    result = train_stagewise_v4(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
