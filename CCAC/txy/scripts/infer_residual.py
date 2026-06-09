#!/usr/bin/env python3
"""Reproduce residual test predictions from fold checkpoints."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from txy.data.feature_io import load_or_build_multimodal, make_subject_id
from txy.data.history_features import HistoryFeatureBuilder
from txy.data.longitudinal_dataset import LongitudinalPersonDataset, collate_person_batch
from txy.models.residual import ResidualFusionModel
from txy.models.stagewise import StageWiseLongitudinalModel
from txy.training.prior_calibration import apply_prior_bias, compute_train_prior_bias
from txy.training.trainer import (
    TrainConfig,
    _apply_multimodal_scalers,
    _apply_scaler,
    _collate_with_tabular,
    _fit_scaler,
    _predict_logits,
    _resolve_device,
)


def build_model(audio_dim, video_dim, history_dim, level_slots, num_classes, alpha: float, device):
    stagewise = StageWiseLongitudinalModel(
        audio_dim=audio_dim,
        video_dim=video_dim,
        history_score_dim=history_dim,
        history_level_slots=level_slots,
        num_classes=num_classes,
        use_history=True,
    )
    return ResidualFusionModel(
        tabular_dim=history_dim,
        stagewise_model=stagewise,
        num_classes=num_classes,
        alpha=alpha,
    ).to(device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer residual model on test subjects")
    parser.add_argument("--dataset-path", type=str, default="/home/adodas/dataset_ccac")
    parser.add_argument("--artifact-dir", type=str, default="artifacts/residual")
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--calibration",
        type=str,
        default="raw",
        choices=["raw", "train_prior"],
        help="raw: no post-hoc bias; train_prior: fixed train label prior shift only",
    )
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir).resolve()
    dataset_root = Path(args.dataset_path).resolve()
    device = _resolve_device(args.device)

    label_mapping = json.loads((artifact_dir / "label_mapping.json").read_text(encoding="utf-8"))
    num_classes = len(label_mapping)
    label_by_index = {int(index): label for label, index in label_mapping.items()}

    train_frame = pd.read_csv(dataset_root / "train_val" / "labels.csv")
    history_builder = HistoryFeatureBuilder.from_labels_frame(train_frame)
    tabular_features, _ = history_builder.transform(train_frame)

    test_frame = pd.read_csv(dataset_root / "test" / "subjects.csv")
    test_frame["subject_id"] = make_subject_id(test_frame)
    history_scores, history_levels = history_builder.transform(test_frame)
    tabular_test = history_scores

    config = TrainConfig(
        dataset_path=str(dataset_root),
        output_dir=str(artifact_dir),
    )
    audio, video, clip_mask, fused, audio_dim, video_dim = load_or_build_multimodal(
        dataset_root,
        "test",
        test_frame,
        config.audio_feature_name,
        config.video_feature_name,
        use_cache=config.feature_cache,
    )

    fold_dirs = sorted(
        [p for p in artifact_dir.iterdir() if p.is_dir() and p.name.startswith("fold_")],
        key=lambda p: int(p.name.split("_")[1]),
    )
    if not fold_dirs:
        raise FileNotFoundError(f"no fold checkpoints under {artifact_dir}")

    all_logits = []
    for fold_dir in fold_dirs:
        ckpt_path = fold_dir / f"best_{config.checkpoint_metric}.pt"
        if not ckpt_path.exists():
            ckpt_path = next(fold_dir.glob("best_*.pt"), None)
        if ckpt_path is None:
            raise FileNotFoundError(f"missing checkpoint in {fold_dir}")
        state = torch.load(ckpt_path, map_location=device, weights_only=False)

        mm_scaler = (
            np.asarray(state["mm_scaler_mean"], dtype=np.float32),
            np.asarray(state["mm_scaler_std"], dtype=np.float32),
        )
        tab_scaler = (
            np.asarray(state["tab_scaler_mean"], dtype=np.float32),
            np.asarray(state["tab_scaler_std"], dtype=np.float32),
        )
        scaled_audio, scaled_video, scaled_fused = _apply_multimodal_scalers(
            audio, video, fused, mm_scaler, audio_dim, video_dim
        )
        scaled_tab = _apply_scaler(tabular_test, tab_scaler)

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

        from txy.training.trainer import _IndexedDataset

        indexed = _IndexedDataset(dataset, np.arange(len(test_frame)), scaled_tab)

        def collate_fn(items):
            return _collate_with_tabular(items, scaled_tab, np.arange(len(items)))

        loader = DataLoader(indexed, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        model = build_model(
            audio_dim, video_dim, history_scores.shape[1], history_levels.shape[1], num_classes, args.alpha, device
        )
        model.load_state_dict(state["model"])
        _, logits = _predict_logits(model, loader, device)
        all_logits.append(logits)

    ensemble_logits = np.mean(np.stack(all_logits, axis=0), axis=0)

    if args.calibration == "train_prior":
        train_labels = train_frame[config.target_label_column]
        bias = compute_train_prior_bias(train_labels, label_mapping, num_classes)
        ensemble_logits = apply_prior_bias(ensemble_logits, bias)
    else:
        bias = np.zeros(num_classes, dtype=np.float32)

    probs = torch.softmax(torch.from_numpy(ensemble_logits), dim=-1).numpy()
    pred_index = ensemble_logits.argmax(axis=1)

    output = test_frame[["anon_school", "anon_class", "anon_person"]].copy()
    output["label"] = [label_by_index[int(i)] for i in pred_index]
    for class_index in range(num_classes):
        output[f"prob_class_{class_index}"] = probs[:, class_index]

    out_path = Path(args.output or artifact_dir / "test_predictions_submission.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(out_path, index=False, encoding="utf-8")

    meta = {
        "calibration": args.calibration,
        "class_bias_applied": bias.tolist(),
        "num_folds": len(fold_dirs),
        "alpha": args.alpha,
        "output": str(out_path),
    }
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
