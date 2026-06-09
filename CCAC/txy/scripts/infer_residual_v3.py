#!/usr/bin/env python3
"""Infer Residual v3 on test subjects (history_available=False when test lacks tabular)."""
from __future__ import annotations

import argparse
import json
import pickle
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
from txy.models.residual_v3 import ResidualV3Model
from txy.training.calibration import apply_class_bias
from txy.training.prior_calibration import apply_prior_bias, compute_train_prior_bias
from txy.training.tabular_anchor import _predict_logits
from txy.training.trainer import TrainConfig, _apply_multimodal_scalers, _resolve_device
from txy.training.trainer_v3 import (
    TrainV3Config,
    _IndexedV3Dataset,
    _collate_v3,
    _predict_v3,
    _test_has_history,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer Residual v3 on test subjects")
    parser.add_argument("--dataset-path", type=str, default="/home/adodas/dataset_ccac")
    parser.add_argument("--artifact-dir", type=str, default="artifacts/residual_v3")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--calibration",
        type=str,
        default="raw",
        choices=["raw", "train_prior", "oof_bias"],
        help="raw: fold bias only; train_prior: add train prior; oof_bias: same as raw",
    )
    parser.add_argument("--force-no-history", action="store_true", help="Treat test as missing tabular history")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir).resolve()
    dataset_root = Path(args.dataset_path).resolve()
    device = _resolve_device(args.device)
    train_config = json.loads((artifact_dir / "train_config.json").read_text(encoding="utf-8"))
    v3_config = TrainV3Config(**{k: v for k, v in train_config.items() if k in TrainV3Config.__dataclass_fields__})

    train_frame = pd.read_csv(dataset_root / "train_val" / "labels.csv")
    history_builder = HistoryFeatureBuilder.from_labels_frame(train_frame)
    train_tabular, _ = history_builder.transform(train_frame)

    test_frame = pd.read_csv(dataset_root / "test" / "subjects.csv")
    test_frame["subject_id"] = make_subject_id(test_frame)
    history_scores, history_levels = history_builder.transform(test_frame)
    test_has_history = _test_has_history(dataset_root, history_builder) and not args.force_no_history
    if test_has_history:
        test_tabular, _ = history_builder.transform(test_frame)
    else:
        test_tabular = np.zeros((len(test_frame), train_tabular.shape[1]), dtype=np.float32)

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

    all_logits = []
    for fold_dir in fold_dirs:
        ckpt_path = fold_dir / f"best_{mm_config.checkpoint_metric}.pt"
        if not ckpt_path.exists():
            ckpt_path = next(fold_dir.glob("best_*.pt"), None)
        if ckpt_path is None:
            raise FileNotFoundError(f"missing checkpoint in {fold_dir}")
        state = torch.load(ckpt_path, map_location=device, weights_only=False)

        with open(fold_dir / "tabular_anchor.pkl", "rb") as f:
            anchor_model = pickle.load(f)
        tab_logits = _predict_logits(anchor_model, test_tabular)

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
        indexed = _IndexedV3Dataset(dataset, np.arange(len(test_frame)), tab_logits)
        loader = DataLoader(indexed, batch_size=args.batch_size, shuffle=False, collate_fn=_collate_v3)

        model = ResidualV3Model(
            audio_dim=audio_dim,
            video_dim=video_dim,
            num_classes=NUM_CLASSES,
            hidden_dim=v3_config.hidden_dim,
            temporal_hidden_dim=v3_config.temporal_hidden_dim,
            dropout=v3_config.dropout,
            alpha_with_history=v3_config.alpha_with_history,
            alpha_missing_history=v3_config.alpha_missing_history,
        ).to(device)
        model.load_state_dict(state["model"])
        _, logits = _predict_v3(model, loader, device, history_available=test_has_history)
        bias = np.asarray(state.get("class_bias", np.zeros(NUM_CLASSES)), dtype=np.float32)
        logits = apply_class_bias(logits, bias)
        all_logits.append(logits)

    ensemble_logits = np.mean(np.stack(all_logits, axis=0), axis=0)

    if args.calibration == "train_prior":
        train_labels = train_frame[TARGET_LABEL_COLUMN]
        from txy.constants import LEVEL_TO_INDEX

        prior_bias = compute_train_prior_bias(train_labels, LEVEL_TO_INDEX, NUM_CLASSES)
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
        "test_has_history": test_has_history,
        "calibration": args.calibration,
        "num_folds": len(fold_dirs),
        "output": str(out_path),
        "label_counts_int": label_counts,
        "label_counts_named": {int_to_text[k]: v for k, v in label_counts.items()},
    }
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
