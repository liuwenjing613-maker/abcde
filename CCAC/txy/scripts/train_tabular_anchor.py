#!/usr/bin/env python3
"""Pre-train external tabular anchor and cache OOF logits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from txy.training.tabular_anchor import TabularAnchorConfig, train_tabular_anchor


def main() -> None:
    parser = argparse.ArgumentParser(description="Train external tabular anchor (CatBoost/LightGBM)")
    parser.add_argument("--dataset-path", type=str, default="/home/adodas/dataset_ccac")
    parser.add_argument("--output-dir", type=str, default="artifacts/tabular_anchor")
    parser.add_argument("--model-type", type=str, default="lightgbm", choices=["lightgbm", "mlp"])
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = TabularAnchorConfig(
        dataset_path=str(Path(args.dataset_path).resolve()),
        output_dir=str(Path(args.output_dir).resolve()),
        model_type=args.model_type,
        num_folds=args.num_folds,
        seed=args.seed,
    )
    result = train_tabular_anchor(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
