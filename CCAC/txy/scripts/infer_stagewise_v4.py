#!/usr/bin/env python3
"""Infer StageWise v4 on test (mm-only + optional bias calibration)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from txy.constants import INDEX_TO_LEVEL, TARGET_LABEL_COLUMN
from txy.data.feature_io import load_or_build_multimodal, make_subject_id
from txy.data.history_features import HistoryFeatureBuilder
from txy.data.labels import NUM_CLASSES
from txy.data.longitudinal_dataset import LongitudinalPersonDataset
from txy.models.stagewise_v4 import StageWiseV4Model
from txy.training.calibration import apply_class_bias
from txy.training.prior_calibration import apply_prior_bias, compute_train_prior_bias
from txy.training.trainer import TrainConfig, _apply_multimodal_scalers, _resolve_device
from txy.training.trainer_v3 import _IndexedV3Dataset, _collate_v3
from txy.training.trainer_v4 import TrainV4Config, _predict_v4


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer StageWise v4 on test subjects")
    parser.add_argument("--dataset-path", type=str, default="/home/adodas/dataset_ccac")
    parser.add_argument("--artifact-dir", type=str, default="artifacts/stagewise_v4")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--bias-mode",
        type=str,
        default="oof",
        choices=["none", "oof", "shrink", "prior"],
        help="none: raw logits; oof: fold class bias; shrink: 0.5*bias; prior: train prior shift",
    )
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir).resolve()
    dataset_root = Path(args.dataset_path).resolve()
    device = _resolve_device(args.device)
    train_config = json.loads((artifact_dir / "train_config.json").read_text(encoding="utf-8"))
    v4_config = TrainV4Config(**{k: v for k, v in train_config.items() if k in TrainV4Config.__dataclass_fields__})

    train_frame = pd.read_csv(dataset_root / "train_val" / "labels.csv")
    history_builder = HistoryFeatureBuilder.from_labels_frame(train_frame)

    test_frame = pd.read_csv(dataset_root / "test" / "subjects.csv")
    test_frame["subject_id"] = make_subject_id(test_frame)
    history_scores, history_levels = history_builder.transform(test_frame)

    mm_config = TrainConfig(dataset_path=str(dataset_root), output_dir=str(artifact_dir))
    audio, video, clip_mask, fused, audio_dim, video_dim = load_or_build_multimodal(
        dataset_root,
        "test",
        test_frame,
        mm_config.audio_feature_name,
        mm_config.video_feature_name,
        use_cache=mm_config.feature_cache,
    )

    fold_dirs = sorted(
        [p for p in artifact_dir.iterdir() if p.is_dir() and p.name.startswith("fold_")],
        key=lambda p: int(p.name.split("_")[1]),
    )
    if not fold_dirs:
        raise FileNotFoundError(f"no fold checkpoints under {artifact_dir}")

    dummy_tab = np.zeros((len(test_frame), NUM_CLASSES), dtype=np.float32)
    all_logits = []
    for fold_dir in fold_dirs:
        ckpt_path = fold_dir / f"best_{mm_config.checkpoint_metric}.pt"
        if not ckpt_path.exists():
            ckpt_path = next(fold_dir.glob("best_*.pt"), None)
        if ckpt_path is None:
            raise FileNotFoundError(f"missing checkpoint in {fold_dir}")
        state = torch.load(ckpt_path, map_location=device, weights_only=False)

        mm_scaler = (
            np.asarray(state["mm_scaler_mean"], dtype=np.float32),
            np.asarray(state["mm_scaler_std"], dtype=np.float32),
        )
        scaled_audio, scaled_video, scaled_fused = _apply_multimodal_scalers(
            audio, video, fused, mm_scaler, audio_dim, video_dim
        )

        dataset = LongitudinalPersonDataset(
            scaled_audio,
            scaled_video,
            scaled_fused,
            clip_mask,
            history_scores,
            history_levels,
            np.zeros(len(test_frame), dtype=np.int64),
            test_frame["subject_id"].astype(str).tolist(),
            train=False,
        )
        indexed = _IndexedV3Dataset(dataset, np.arange(len(test_frame)), dummy_tab)
        loader = DataLoader(indexed, batch_size=args.batch_size, shuffle=False, collate_fn=_collate_v3)

        model = StageWiseV4Model(
            audio_dim=audio_dim,
            video_dim=video_dim,
            num_classes=NUM_CLASSES,
            hidden_dim=v4_config.hidden_dim,
            temporal_hidden_dim=v4_config.temporal_hidden_dim,
            dropout=v4_config.dropout,
        ).to(device)
        model.load_state_dict(state["model"])
        _, logits = _predict_v4(model, loader, device)

        if args.bias_mode == "oof":
            bias = np.asarray(state.get("class_bias", np.zeros(NUM_CLASSES)), dtype=np.float32)
        elif args.bias_mode == "shrink":
            bias = np.asarray(state.get("class_bias_shrink", state.get("class_bias", np.zeros(NUM_CLASSES))), dtype=np.float32)
        else:
            bias = np.zeros(NUM_CLASSES, dtype=np.float32)
        logits = apply_class_bias(logits, bias)
        all_logits.append(logits)

    ensemble_logits = np.mean(np.stack(all_logits, axis=0), axis=0)
    if args.bias_mode == "prior":
        from txy.constants import LEVEL_TO_INDEX

        prior_bias = compute_train_prior_bias(train_frame[TARGET_LABEL_COLUMN], LEVEL_TO_INDEX, NUM_CLASSES)
        ensemble_logits = apply_prior_bias(ensemble_logits, prior_bias)

    probs = torch.softmax(torch.from_numpy(ensemble_logits), dim=-1).numpy()
    pred_index = ensemble_logits.argmax(axis=1)

    output = test_frame[["anon_school", "anon_class", "anon_person"]].copy()
    output["label"] = [INDEX_TO_LEVEL[int(i)] for i in pred_index]
    for class_index in range(NUM_CLASSES):
        output[f"prob_class_{class_index}"] = probs[:, class_index]

    out_path = Path(args.output or artifact_dir / "test_predictions_submission.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(out_path, index=False, encoding="utf-8")

    int_to_text = {int(k): v for k, v in INDEX_TO_LEVEL.items()}
    label_counts = {int(k): int(v) for k, v in pd.Series(pred_index).value_counts().sort_index().to_dict().items()}
    meta = {
        "bias_mode": args.bias_mode,
        "num_folds": len(fold_dirs),
        "output": str(out_path),
        "label_counts_int": label_counts,
        "label_counts_named": {int_to_text[k]: v for k, v in label_counts.items()},
    }
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
