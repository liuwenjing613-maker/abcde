#!/usr/bin/env python3
"""Generate test predictions for ND-11 (Knowledge Distillation model).

Usage:
    PYTHONPATH=src python scripts/generate_nd11_test.py \
        --dataset-path datasets \
        --model-dir artifacts/exp/nd11_distillation \
        --output artifacts/exp/nd11_distillation/test_predictions.csv \
        --device cuda
"""

import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ccac.baselines.anxiety_baseline import (
    BaselineConfig, _resolve_device,
    _is_release_dataset, _load_release_train_val,
    _release_cache_path, _build_release_features,
    _apply_scaler,
)
from ccac.baselines.dass_baseline import DASSConfig, DASSDataset
from ccac.experiments.deep_residual import DeepResidualModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-path", default="datasets")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--audio-feature-name", default="audio_wavlm_base")
    p.add_argument("--video-feature-name", default="video_clip_base")
    args = p.parse_args()

    device = _resolve_device(args.device)
    model_dir = Path(args.model_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_path = Path(args.dataset_path)

    # Load label mapping
    label_mapping = json.loads((model_dir / "label_mapping.json").read_text())
    num_classes = len(label_mapping)
    label_by_idx = {i: l for l, i in label_mapping.items()}
    print(f"Label mapping: {label_mapping}")

    # Load baseline config
    bc_data = json.loads((model_dir / "baseline_config.json").read_text())
    input_dim = 4096  # default for wavlm_base + clip_base
    bc = BaselineConfig(
        dataset_path=str(dataset_path),
        output_dir=str(model_dir),
        audio_feature_name=args.audio_feature_name,
        video_feature_name=args.video_feature_name,
    )

    # Load fold models
    fold_states = []
    for fold_id in range(1, 6):
        fold_dir = model_dir / f"fold_{fold_id}"
        ckpt = torch.load(fold_dir / "best_model.pt", map_location="cpu", weights_only=False)
        fold_states.append(ckpt)
    print(f"Loaded {len(fold_states)} fold checkpoints")

    # Load test features
    test_path = dataset_path / "test" / "subjects.csv"
    test_frame = pd.read_csv(test_path)
    test_frame["subject_id"] = test_frame[["anon_school", "anon_class", "anon_person"]].agg("/".join, axis=1)
    print(f"Test subjects: {len(test_frame)}")

    # Try cache first
    cache_path = _release_cache_path(dataset_path, bc, split="test")
    test_av, test_mask = None, None
    if cache_path.exists():
        try:
            cached = np.load(cache_path, allow_pickle=True, mmap_mode="r")
            test_av = cached["features"].astype(np.float32)
            test_mask = cached["clip_mask"].astype(bool)
            if test_av.shape[0] != len(test_frame):
                test_av = test_mask = None
        except Exception:
            pass

    if test_av is None:
        print("Building test features...")
        test_av, test_mask, _ = _build_release_features(
            dataset_path, "test", test_frame,
            args.audio_feature_name, args.video_feature_name,
        )
        if bc.feature_cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_path, features=test_av, clip_mask=test_mask,
                               input_dim=np.asarray(input_dim, dtype=np.int64))

    print(f"Test features: {test_av.shape}")

    # No DASS for test
    test_dass = np.zeros((len(test_frame), 0), dtype=np.float32)

    # Ensemble predictions across folds
    probabilities = []
    for state in fold_states:
        scaler = (np.asarray(state["scaler_mean"], dtype=np.float32),
                  np.asarray(state["scaler_std"], dtype=np.float32))
        scaled_av = _apply_scaler(test_av, scaler)
        dataset = DASSDataset(scaled_av, test_mask, test_dass, np.zeros(len(test_frame), dtype=np.int64))
        loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

        model = DeepResidualModel(
            input_dim=input_dim, num_classes=num_classes,
            hidden_dim=256, num_heads=4, num_residual_blocks=3,
            dropout=0.2, dass_config=DASSConfig(dass_scheme="none"),
        ).to(device)
        model.load_state_dict(state["model"])
        model.eval()

        fold_probs = []
        with torch.no_grad():
            for av, mask, dass, _ in loader:
                av = av.to(device)
                mask = mask.to(device)
                dass = dass.to(device)
                logits = model(av, mask, dass)
                fold_probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
        probabilities.append(np.concatenate(fold_probs, axis=0))
        print(f"  Fold done")

    ensemble = np.mean(np.stack(probabilities, axis=0), axis=0)
    pred_idx = ensemble.argmax(axis=1)
    pred_label = [label_by_idx[int(i)] for i in pred_idx]

    # Count predictions
    unique, counts = np.unique(pred_label, return_counts=True)
    print(f"Prediction distribution: {dict(zip(unique, counts))}")

    # Write output
    output = test_frame[["anon_school", "anon_class", "anon_person"]].copy()
    # Map labels to integers: 中度=0, 正常=1, 轻度=2, 重度=3, 非常严重=4
    label_to_int = {"中度": 0, "正常": 1, "轻度": 2, "重度": 3, "非常严重": 4}
    output["label"] = [label_to_int.get(l, 0) for l in pred_label]
    output.to_csv(output_path, index=False)
    print(f"\nTest predictions saved to {output_path}")
    print(f"Distribution: {dict(zip(*np.unique(output['label'].values, return_counts=True)))}")


if __name__ == "__main__":
    main()
