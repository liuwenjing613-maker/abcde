#!/usr/bin/env python3
"""Generate a CCAC submission from robust checkpoint fold models.

The robust checkpoint experiments save five DeepResidual no-DASS fold
checkpoints. This script ensembles those checkpoints on the release test set,
writes a submission CSV, validates the required format, and optionally creates
a zip containing `submission.csv`.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ccac.baselines.anxiety_baseline import (
    BaselineConfig,
    _apply_scaler,
    _build_release_features,
    _release_cache_path,
    _resolve_device,
)
from ccac.baselines.dass_baseline import DASSConfig, DASSDataset
from ccac.experiments.deep_residual import DeepResidualModel


LABEL_TO_INT = {"中度": 0, "正常": 1, "轻度": 2, "重度": 3, "非常严重": 4}
REQUIRED_COLUMNS = ["anon_school", "anon_class", "anon_person", "label"]


def _load_test_features(
    dataset_path: Path,
    config: BaselineConfig,
    test_frame: pd.DataFrame,
    audio_feature_name: str,
    video_feature_name: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    cache_path = _release_cache_path(dataset_path, config, split="test")
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=True, mmap_mode="r")
        features = cached["features"].astype(np.float32)
        clip_mask = cached["clip_mask"].astype(bool)
        if features.shape[0] == len(test_frame):
            input_dim = int(cached["input_dim"]) if "input_dim" in cached.files else features.shape[-1]
            return features, clip_mask, input_dim

    features, clip_mask, input_dim = _build_release_features(
        dataset_path,
        "test",
        test_frame,
        audio_feature_name,
        video_feature_name,
    )
    if config.feature_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            features=features,
            clip_mask=clip_mask,
            input_dim=np.asarray(input_dim, dtype=np.int64),
        )
    return features, clip_mask, int(input_dim)


def _validate_submission(output: pd.DataFrame, test_frame: pd.DataFrame) -> None:
    if list(output.columns) != REQUIRED_COLUMNS:
        raise ValueError(f"Bad columns: {list(output.columns)}")
    if len(output) != len(test_frame):
        raise ValueError(f"Bad row count: {len(output)} != {len(test_frame)}")
    for col in REQUIRED_COLUMNS[:3]:
        if not output[col].astype(str).equals(test_frame[col].astype(str)):
            raise ValueError(f"Identifier column is not aligned with test subjects: {col}")
    if output["label"].isna().any():
        raise ValueError("Submission contains missing labels")
    labels = set(output["label"].astype(int).unique().tolist())
    if not labels.issubset({0, 1, 2, 3, 4}):
        raise ValueError(f"Labels outside 0..4: {sorted(labels)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", default="datasets")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-zip")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--audio-feature-name", default="audio_wavlm_base")
    parser.add_argument("--video-feature-name", default="video_clip_base")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--num-subjects",
        type=int,
        default=None,
        help="Limit submission to the first N test subjects; current public submissions use 382.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    model_dir = Path(args.model_dir)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device)

    label_mapping = json.loads((model_dir / "label_mapping.json").read_text(encoding="utf-8"))
    label_by_idx = {int(idx): label for label, idx in label_mapping.items()}
    num_classes = len(label_mapping)

    config = BaselineConfig(
        dataset_path=str(dataset_path),
        output_dir=str(model_dir),
        audio_feature_name=args.audio_feature_name,
        video_feature_name=args.video_feature_name,
    )

    full_test_frame = pd.read_csv(dataset_path / "test" / "subjects.csv")
    full_test_av, full_test_mask, input_dim = _load_test_features(
        dataset_path,
        config,
        full_test_frame,
        args.audio_feature_name,
        args.video_feature_name,
    )
    if args.num_subjects is not None:
        if args.num_subjects <= 0 or args.num_subjects > len(full_test_frame):
            raise ValueError(f"Invalid --num-subjects: {args.num_subjects}")
        test_frame = full_test_frame.iloc[: args.num_subjects].reset_index(drop=True)
        test_av = full_test_av[: args.num_subjects]
        test_mask = full_test_mask[: args.num_subjects]
    else:
        test_frame = full_test_frame
        test_av = full_test_av
        test_mask = full_test_mask
    test_dass = np.zeros((len(test_frame), 0), dtype=np.float32)
    dummy_labels = np.zeros(len(test_frame), dtype=np.int64)

    all_probs: list[np.ndarray] = []
    for fold_id in range(1, 6):
        ckpt_path = model_dir / f"fold_{fold_id}" / "best_model.pt"
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        scaler = (
            np.asarray(state["scaler_mean"], dtype=np.float32),
            np.asarray(state["scaler_std"], dtype=np.float32),
        )
        scaled_av = _apply_scaler(test_av, scaler)
        dataset = DASSDataset(scaled_av, test_mask, test_dass, dummy_labels)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

        model = DeepResidualModel(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=256,
            num_heads=4,
            num_residual_blocks=3,
            dropout=0.2,
            dass_config=DASSConfig(dass_scheme="none"),
        ).to(device)
        model.load_state_dict(state["model"])
        model.eval()

        fold_probs = []
        with torch.no_grad():
            for av, mask, dass, _ in loader:
                logits = model(av.to(device), mask.to(device), dass.to(device))
                fold_probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
        all_probs.append(np.concatenate(fold_probs, axis=0))
        print(f"Fold {fold_id} done: {ckpt_path}")

    probs = np.mean(np.stack(all_probs, axis=0), axis=0)
    pred_idx = probs.argmax(axis=1).astype(int)
    pred_labels = [label_by_idx[int(idx)] for idx in pred_idx]

    output = test_frame[["anon_school", "anon_class", "anon_person"]].copy()
    output["label"] = [LABEL_TO_INT[label] for label in pred_labels]
    _validate_submission(output, test_frame)
    output.to_csv(output_csv, index=False, encoding="utf-8")

    counts = output["label"].value_counts().sort_index().to_dict()
    print(f"Saved CSV: {output_csv}")
    print(f"Rows: {len(output)}, label distribution: {counts}")

    if args.output_zip:
        output_zip = Path(args.output_zip)
        output_zip.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(output_csv, arcname="submission.csv")
        print(f"Saved ZIP: {output_zip}")


if __name__ == "__main__":
    main()
