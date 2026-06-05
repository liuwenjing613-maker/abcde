#!/usr/bin/env python3
"""Apply prior shift correction to test predictions and generate submission CSV.

The prior correction adjusts model probabilities for the known public test
distribution shift: 中度=76.4% vs training 12.2%.

Usage:
    PYTHONPATH=src python scripts/generate_nd11_test_calibrated.py \
        --model-dir artifacts/exp/nd11_distillation \
        --output submissions_v3/sub_nd11_distillation_calibrated.csv
"""

import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ccac.baselines.anxiety_baseline import (
    BaselineConfig, _resolve_device,
    _release_cache_path, _build_release_features,
    _apply_scaler,
)
from ccac.baselines.dass_baseline import DASSConfig, DASSDataset
from ccac.experiments.deep_residual import DeepResidualModel

# Known distributions
TRAIN_PRIOR = np.array([0.122, 0.765, 0.038, 0.035, 0.039], dtype=np.float64)
TEST_PRIOR = np.array([0.764, 0.039, 0.120, 0.037, 0.039], dtype=np.float64)
UNIFORM_PRIOR = np.array([0.2, 0.2, 0.2, 0.2, 0.2], dtype=np.float64)


def apply_prior_correction(probs, train_prior, test_prior, temperature=1.0):
    if temperature != 1.0:
        logits = np.log(np.maximum(probs, 1e-9)) / temperature
        probs = np.exp(logits)
        probs = probs / probs.sum(axis=-1, keepdims=True)
    correction = test_prior / np.maximum(train_prior, 1e-9)
    corrected = probs * correction.reshape(1, -1)
    return corrected / corrected.sum(axis=-1, keepdims=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--dataset-path", default="datasets")
    p.add_argument("--device", default="cuda")
    p.add_argument("--prior", default="test", choices=["test", "uniform", "none"])
    p.add_argument("--temperature", type=float, default=1.2)
    args = p.parse_args()

    device = _resolve_device(args.device)
    model_dir = Path(args.model_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_path = Path(args.dataset_path)

    label_mapping = json.loads((model_dir / "label_mapping.json").read_text())
    num_classes = len(label_mapping)
    label_to_int = {"中度": 0, "正常": 1, "轻度": 2, "重度": 3, "非常严重": 4}

    bc = BaselineConfig(
        dataset_path=str(dataset_path), output_dir=str(model_dir),
        audio_feature_name="audio_wavlm_base", video_feature_name="video_clip_base",
    )

    # Load fold checkpoints
    fold_states = []
    for fold_id in range(1, 6):
        ckpt = torch.load(model_dir / f"fold_{fold_id}" / "best_model.pt",
                         map_location="cpu", weights_only=False)
        fold_states.append(ckpt)
    print(f"Loaded {len(fold_states)} folds")

    # Load test features
    test_frame = pd.read_csv(dataset_path / "test" / "subjects.csv")
    cache_path = _release_cache_path(dataset_path, bc, split="test")
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=True, mmap_mode="r")
        test_av = cached["features"].astype(np.float32)
        test_mask = cached["clip_mask"].astype(bool)
    else:
        test_av, test_mask, _ = _build_release_features(
            dataset_path, "test", test_frame, "audio_wavlm_base", "video_clip_base")

    test_dass = np.zeros((len(test_frame), 0), dtype=np.float32)

    # Ensemble predictions
    all_probs = []
    for state in fold_states:
        scaler = (np.asarray(state["scaler_mean"], dtype=np.float32),
                  np.asarray(state["scaler_std"], dtype=np.float32))
        scaled_av = _apply_scaler(test_av, scaler)
        ds = DASSDataset(scaled_av, test_mask, test_dass, np.zeros(len(test_frame), dtype=np.int64))
        loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

        model = DeepResidualModel(
            input_dim=4096, num_classes=num_classes,
            hidden_dim=256, num_heads=4, num_residual_blocks=3,
            dropout=0.2, dass_config=DASSConfig(dass_scheme="none"),
        ).to(device)
        model.load_state_dict(state["model"])
        model.eval()

        fold_probs = []
        with torch.no_grad():
            for av, mask, dass, _ in loader:
                logits = model(av.to(device), mask.to(device), dass.to(device))
                fold_probs.append(torch.softmax(logits, -1).cpu().numpy())
        all_probs.append(np.concatenate(fold_probs))

    ensemble = np.mean(np.stack(all_probs, axis=0), axis=0)
    print(f"Raw pred distribution: {np.bincount(ensemble.argmax(1), minlength=5)}")

    # Apply prior correction
    if args.prior == "test":
        prior = TEST_PRIOR
    elif args.prior == "uniform":
        prior = UNIFORM_PRIOR
    else:
        prior = TRAIN_PRIOR  # no correction

    corrected = apply_prior_correction(ensemble, TRAIN_PRIOR, prior, args.temperature)
    pred_idx = corrected.argmax(axis=1)
    print(f"Corrected (T={args.temperature}, prior={args.prior}): "
          f"{np.bincount(pred_idx, minlength=5)}")

    # Write submission
    output = test_frame[["anon_school", "anon_class", "anon_person"]].copy()
    output["label"] = pred_idx
    output.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
