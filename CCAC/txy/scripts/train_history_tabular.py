#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from txy.training.tabular_trainer import TabularConfig, train_history_tabular


def main() -> None:
    parser = argparse.ArgumentParser(description="Train history-only tabular baseline (Experiment 1)")
    parser.add_argument("--dataset-path", type=str, default="/home/adodas/dataset_ccac")
    parser.add_argument("--output-dir", type=str, default="artifacts/history_tabular")
    parser.add_argument("--model-type", type=str, default="lightgbm", choices=["lightgbm", "mlp"])
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--group-by", type=str, default="school_class", choices=["school", "school_class"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = TabularConfig(
        dataset_path=str(Path(args.dataset_path).resolve()),
        output_dir=str(Path(args.output_dir).resolve()),
        model_type=args.model_type,
        num_folds=args.num_folds,
        group_by=args.group_by,
        seed=args.seed,
    )
    result = train_history_tabular(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
