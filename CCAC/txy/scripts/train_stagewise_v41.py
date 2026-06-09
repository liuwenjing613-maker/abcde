#!/usr/bin/env python3
"""Train StageWise v4.1: ensemble teacher + balanced sampler + tuned optimization."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from txy.training.trainer_v41 import TrainV41Config, train_stagewise_v41


def main() -> None:
    parser = argparse.ArgumentParser(description="Train StageWise v4.1")
    parser.add_argument("--dataset-path", type=str, default="/home/adodas/dataset_ccac")
    parser.add_argument("--output-dir", type=str, default="artifacts/stagewise_v41")
    parser.add_argument("--loss-mode", type=str, default="ce_kd_ordinal",
                        choices=["ce_only", "ce_kd", "ce_ordinal", "ce_kd_ordinal"])
    parser.add_argument("--kd-weight", type=float, default=0.5)
    parser.add_argument("--ordinal-weight", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--no-balanced-sampler", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = TrainV41Config(
        dataset_path=str(Path(args.dataset_path).resolve()),
        output_dir=str(Path(args.output_dir).resolve()),
        loss_mode=args.loss_mode,
        kd_weight=args.kd_weight,
        ordinal_weight=args.ordinal_weight,
        learning_rate=args.learning_rate,
        dropout=args.dropout,
        epochs=args.epochs,
        patience=args.patience,
        use_balanced_sampler=not args.no_balanced_sampler,
        device=args.device,
        seed=args.seed,
    )
    result = train_stagewise_v41(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
