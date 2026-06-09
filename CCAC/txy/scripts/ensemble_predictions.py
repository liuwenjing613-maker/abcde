#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from txy.ensemble.class_wise import blend_logits, class_wise_blend
from txy.training.metrics import classification_metrics


def _softmax(logits: np.ndarray) -> np.ndarray:
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return (exp / exp.sum(axis=1, keepdims=True)).astype(np.float32)


def load_oof_logits(path: Path, num_classes: int | None = None) -> tuple[pd.DataFrame, np.ndarray, np.ndarray | None]:
    df = pd.read_csv(path)
    if num_classes is None:
        logit_cols = [c for c in df.columns if c.startswith("logit_class_")]
        prob_cols = [c for c in df.columns if c.startswith("prob_class_")]
        num_classes = max(len(logit_cols), len(prob_cols), 5)
    logit_cols = [f"logit_class_{i}" for i in range(num_classes)]
    if all(col in df.columns for col in logit_cols):
        logits = df[logit_cols].to_numpy(dtype=np.float32)
    else:
        prob_cols = [f"prob_class_{i}" for i in range(num_classes)]
        probs = df[prob_cols].to_numpy(dtype=np.float32)
        logits = np.log(probs.clip(1e-6, 1.0))
    labels = None
    if "true_label" in df.columns:
        mapping = {label: index for index, label in enumerate(sorted(df["true_label"].astype(str).unique()))}
        labels = df["true_label"].astype(str).map(mapping).to_numpy(dtype=np.int64)
    return df, logits, labels


def load_test_probs(path: Path, num_classes: int) -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(path)
    prob_cols = [f"prob_class_{i}" for i in range(num_classes)]
    if all(col in df.columns for col in prob_cols):
        probs = df[prob_cols].to_numpy(dtype=np.float32)
        logits = np.log(probs.clip(1e-6, 1.0))
        return df, logits
    logit_cols = [f"logit_class_{i}" for i in range(num_classes)]
    logits = df[logit_cols].to_numpy(dtype=np.float32)
    return df, logits


def main() -> None:
    parser = argparse.ArgumentParser(description="Class-wise logit ensemble (Experiment 6)")
    parser.add_argument("--inputs", nargs="+", required=True, help="oof_predictions.csv paths")
    parser.add_argument("--weights", nargs="+", type=float, default=None, help="model-level weights")
    parser.add_argument("--output", type=str, default="artifacts/ensemble/oof_predictions.csv")
    parser.add_argument(
        "--class-weights",
        type=str,
        default=None,
        help='JSON file with shape [num_models, num_classes], e.g. [[0.6,0.2,...],[...]]',
    )
    parser.add_argument("--test-inputs", nargs="+", default=None, help="test_predictions.csv paths aligned with --inputs")
    parser.add_argument("--test-output", type=str, default=None)
    args = parser.parse_args()

    paths = [Path(p) for p in args.inputs]
    frames: list[pd.DataFrame] = []
    logits_list: list[np.ndarray] = []
    labels: np.ndarray | None = None
    num_classes: int | None = None

    for path in paths:
        df, logits, lbls = load_oof_logits(path, num_classes)
        if num_classes is None:
            num_classes = logits.shape[1]
            labels = lbls
        frames.append(df)
        logits_list.append(logits)

    assert num_classes is not None
    base_df = frames[0]
    if "subject_id" in base_df.columns:
        for index in range(1, len(frames)):
            order = base_df["subject_id"].astype(str).tolist()
            frames[index] = frames[index].set_index("subject_id").loc[order].reset_index()
            logits_list[index] = frames[index][[f"logit_class_{i}" for i in range(num_classes)]].to_numpy(dtype=np.float32) \
                if all(f"logit_class_{i}" in frames[index].columns for i in range(num_classes)) \
                else np.log(frames[index][[f"prob_class_{i}" for i in range(num_classes)]].to_numpy(dtype=np.float32).clip(1e-6, 1.0))

    if args.class_weights:
        class_weights = np.asarray(json.loads(Path(args.class_weights).read_text(encoding="utf-8")), dtype=np.float32)
        blended = class_wise_blend(logits_list, class_weights)
    else:
        blended = blend_logits(logits_list, args.weights)

    probs = _softmax(blended)
    output = base_df[["subject_id", "true_label"]].copy() if "true_label" in base_df.columns else base_df[["subject_id"]].copy()
    for i in range(num_classes):
        output[f"logit_class_{i}"] = blended[:, i]
        output[f"prob_class_{i}"] = probs[:, i]

    if "true_label" in output.columns:
        mapping = {label: index for index, label in enumerate(sorted(output["true_label"].astype(str).unique()))}
        label_by_index = {index: label for label, index in mapping.items()}
        preds = blended.argmax(axis=1)
        output["pred_label"] = [label_by_index[int(i)] for i in preds]
        if labels is not None:
            metrics = classification_metrics(labels, preds)
            print(json.dumps({"oof_metrics": metrics}, ensure_ascii=False, indent=2))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(out_path, index=False, encoding="utf-8")
    print(f"saved ensemble OOF to {out_path}")

    if args.test_inputs:
        test_paths = [Path(p) for p in args.test_inputs]
        test_frames: list[pd.DataFrame] = []
        test_logits_list: list[np.ndarray] = []
        for path in test_paths:
            df, logits = load_test_probs(path, num_classes)
            test_frames.append(df)
            test_logits_list.append(logits)
        test_base = test_frames[0]
        if "subject_id" in test_base.columns:
            order = test_base["subject_id"].astype(str).tolist()
            for index in range(1, len(test_frames)):
                test_frames[index] = test_frames[index].set_index("subject_id").loc[order].reset_index()
                _, logits = load_test_probs(test_paths[index], num_classes)
                test_logits_list[index] = logits
        if args.class_weights:
            test_blended = class_wise_blend(test_logits_list, class_weights)
        else:
            test_blended = blend_logits(test_logits_list, args.weights)
        test_probs = _softmax(test_blended)
        label_map_path = paths[0].parent / "label_mapping.json"
        if label_map_path.exists():
            label_mapping = json.loads(label_map_path.read_text(encoding="utf-8"))
            label_by_index = {index: label for label, index in label_mapping.items()}
        else:
            label_by_index = {i: str(i) for i in range(num_classes)}
        test_out = test_base[["anon_school", "anon_class", "anon_person"]].copy()
        if "subject_id" in test_base.columns:
            test_out["subject_id"] = test_base["subject_id"]
        pred_idx = test_blended.argmax(axis=1)
        test_out["label"] = [label_by_index[int(i)] for i in pred_idx]
        for i in range(num_classes):
            test_out[f"prob_class_{i}"] = test_probs[:, i]
        test_path = Path(args.test_output or out_path.parent / "test_predictions.csv")
        test_path.parent.mkdir(parents=True, exist_ok=True)
        test_out.to_csv(test_path, index=False, encoding="utf-8")
        print(f"saved ensemble test to {test_path}")


if __name__ == "__main__":
    main()
