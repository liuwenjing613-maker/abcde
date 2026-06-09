from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from txy.constants import INDEX_TO_LEVEL, TARGET_LABEL_COLUMN
from txy.data.feature_io import load_or_build_multimodal, make_subject_id
from txy.data.group_split import build_group_folds, make_group_id
from txy.data.labels import NUM_CLASSES, encode_ordinal_labels
from txy.data.shallow_mm_features import ShallowMMConfig, ShallowMultimodalFeatureBuilder
from txy.training.calibration import apply_class_bias, search_class_bias
from txy.training.metrics import classification_metrics, format_classification_report
from txy.training.tabular_anchor import _fit_tabular_model, _predict_logits


@dataclass
class ShallowMMTrainConfig:
    dataset_path: str
    output_dir: str
    model_type: str = "lightgbm"
    audio_pca_dim: int = 128
    video_pca_dim: int = 128
    num_folds: int = 5
    seed: int = 42
    group_by: str = "school_class"
    calibrate_bias: bool = True


def train_shallow_multimodal(config: ShallowMMTrainConfig) -> dict[str, Any]:
    output_dir = Path(config.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = Path(config.dataset_path)

    frame = pd.read_csv(dataset_root / "train_val" / "labels.csv")
    frame = frame.dropna(subset=[TARGET_LABEL_COLUMN]).reset_index(drop=True)
    frame["subject_id"] = make_subject_id(frame)
    labels, label_mapping = encode_ordinal_labels(frame[TARGET_LABEL_COLUMN])
    groups = make_group_id(frame, config.group_by)

    audio, video, clip_mask, _, audio_dim, video_dim = load_or_build_multimodal(
        dataset_root,
        "train_val",
        frame,
        "audio_wavlm_base",
        "video_dinov2_small",
        use_cache=True,
    )

    fold_indices = build_group_folds(labels, groups, config.num_folds, config.seed)
    oof_logits = np.zeros((len(frame), NUM_CLASSES), dtype=np.float32)
    oof_predictions = np.full(len(frame), -1, dtype=np.int64)
    fold_metrics: list[dict[str, Any]] = []

    for fold_id, (train_idx, val_idx) in enumerate(fold_indices, start=1):
        fold_dir = output_dir / f"fold_{fold_id}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        builder = ShallowMultimodalFeatureBuilder()
        mm_cfg = ShallowMMConfig(
            audio_pca_dim=config.audio_pca_dim,
            video_pca_dim=config.video_pca_dim,
            seed=config.seed,
        )
        train_features = builder.transform(
            audio[train_idx], video[train_idx], clip_mask[train_idx], fit_pca=True, config=mm_cfg
        )
        val_features = builder.transform(audio[val_idx], video[val_idx], clip_mask[val_idx], fit_pca=False)

        model = _fit_tabular_model(config.model_type, train_features, labels[train_idx], config.seed)
        logits = _predict_logits(model, val_features)
        bias = np.zeros(NUM_CLASSES, dtype=np.float32)
        metrics = classification_metrics(labels[val_idx], logits.argmax(axis=1))
        if config.calibrate_bias:
            bias, metrics = search_class_bias(logits, labels[val_idx])
            logits = apply_class_bias(logits, bias)

        oof_logits[val_idx] = logits
        oof_predictions[val_idx] = logits.argmax(axis=1)

        with open(fold_dir / "model.pkl", "wb") as f:
            pickle.dump({"model": model, "builder": builder, "class_bias": bias.tolist()}, f)
        (fold_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        fold_metrics.append({"fold": fold_id, **metrics})

    # test predictions
    test_path = dataset_root / "test" / "subjects.csv"
    if test_path.exists():
        test_frame = pd.read_csv(test_path)
        test_frame["subject_id"] = make_subject_id(test_frame)
        t_audio, t_video, t_mask, _, _, _ = load_or_build_multimodal(
            dataset_root, "test", test_frame, "audio_wavlm_base", "video_dinov2_small", use_cache=True
        )
        all_probs = []
        fold_dirs = sorted(output_dir.glob("fold_*"), key=lambda p: int(p.name.split("_")[1]))
        for fold_dir in fold_dirs:
            with open(fold_dir / "model.pkl", "rb") as f:
                payload = pickle.load(f)
            feats = payload["builder"].transform(t_audio, t_video, t_mask, fit_pca=False)
            logits = _predict_logits(payload["model"], feats)
            logits = apply_class_bias(logits, np.asarray(payload["class_bias"], dtype=np.float32))
            probs = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs = probs / probs.sum(axis=1, keepdims=True)
            all_probs.append(probs)
        ensemble = np.mean(np.stack(all_probs, axis=0), axis=0)
        pred_idx = ensemble.argmax(axis=1)
        test_out = test_frame[["anon_school", "anon_class", "anon_person"]].copy()
        test_out["label"] = [INDEX_TO_LEVEL[int(i)] for i in pred_idx]
        for i in range(NUM_CLASSES):
            test_out[f"prob_class_{i}"] = ensemble[:, i]
        test_out.to_csv(output_dir / "test_predictions.csv", index=False, encoding="utf-8")

    class_names = [INDEX_TO_LEVEL[i] for i in range(NUM_CLASSES)]
    overall = classification_metrics(labels, oof_predictions)
    oof_df = pd.DataFrame(
        {
            "subject_id": frame["subject_id"].astype(str),
            "true_label": frame[TARGET_LABEL_COLUMN].astype(str),
            "pred_label": [INDEX_TO_LEVEL[int(i)] for i in oof_predictions],
        }
    )
    for i in range(NUM_CLASSES):
        oof_df[f"logit_class_{i}"] = oof_logits[:, i]

    pd.DataFrame(fold_metrics).to_csv(output_dir / "fold_metrics.csv", index=False, encoding="utf-8")
    oof_df.to_csv(output_dir / "oof_predictions.csv", index=False, encoding="utf-8")
    (output_dir / "label_mapping.json").write_text(json.dumps(label_mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "classification_report.txt").write_text(
        format_classification_report(labels, oof_predictions, class_names), encoding="utf-8"
    )
    summary = {
        "model_kind": "shallow_mm_lgbm",
        "config": asdict(config),
        "feature_dim": len(builder.feature_names),
        "overall_oof_metrics": overall,
        "fold_metrics_mean": pd.DataFrame(fold_metrics).mean(numeric_only=True).to_dict(),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
