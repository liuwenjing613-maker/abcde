#!/usr/bin/env python3
"""Train shallow multimodal LightGBM on summary + PCA features."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from txy.training.shallow_mm_trainer import ShallowMMTrainConfig, train_shallow_multimodal


def main() -> None:
    parser = argparse.ArgumentParser(description="Train shallow multimodal LightGBM")
    parser.add_argument("--dataset-path", type=str, default="/home/adodas/dataset_ccac")
    parser.add_argument("--output-dir", type=str, default="artifacts/shallow_mm")
    parser.add_argument("--audio-pca-dim", type=int, default=128)
    parser.add_argument("--video-pca-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = ShallowMMTrainConfig(
        dataset_path=str(Path(args.dataset_path).resolve()),
        output_dir=str(Path(args.output_dir).resolve()),
        audio_pca_dim=args.audio_pca_dim,
        video_pca_dim=args.video_pca_dim,
        seed=args.seed,
    )
    result = train_shallow_multimodal(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
