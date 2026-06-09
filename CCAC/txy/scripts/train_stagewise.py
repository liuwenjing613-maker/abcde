#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from txy.models.stagewise import StageWiseLongitudinalModel
from txy.training.trainer import TrainConfig, train_longitudinal


def build_model_factory(multimodal_only: bool, use_history: bool):
    def factory(audio_dim, video_dim, history_score_dim, history_level_slots, num_classes):
        return StageWiseLongitudinalModel(
            audio_dim=audio_dim,
            video_dim=video_dim,
            history_score_dim=history_score_dim,
            history_level_slots=history_level_slots,
            num_classes=num_classes,
            use_history=use_history and not multimodal_only,
        )

    return factory


def main() -> None:
    parser = argparse.ArgumentParser(description="Train StageWiseLongitudinal model (Experiment 2/3)")
    parser.add_argument("--dataset-path", type=str, default="/home/adodas/dataset_ccac")
    parser.add_argument("--output-dir", type=str, default="artifacts/stagewise")
    parser.add_argument("--multimodal-only", action="store_true", help="Experiment 2: no history features")
    parser.add_argument("--no-history", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--group-by", type=str, default="school_class")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    use_history = not args.no_history
    config = TrainConfig(
        dataset_path=str(Path(args.dataset_path).resolve()),
        output_dir=str(Path(args.output_dir).resolve()),
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_folds=args.num_folds,
        group_by=args.group_by,
        seed=args.seed,
        use_history=use_history,
        multimodal_only=args.multimodal_only,
    )
    model_kind = "multimodal_only" if args.multimodal_only else "stagewise"
    result = train_longitudinal(config, build_model_factory(args.multimodal_only, use_history), model_kind=model_kind)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
