from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from txy.constants import TARGET_LABEL_COLUMN
from txy.data.feature_io import make_subject_id
from txy.data.group_split import build_group_folds, make_group_id
from txy.data.history_features import HistoryFeatureBuilder
from txy.data.labels import NUM_CLASSES, encode_ordinal_labels
from txy.training.calibration import apply_class_bias, search_class_bias
from txy.training.metrics import classification_metrics, format_classification_report


@dataclass
class TabularConfig:
    dataset_path: str
    output_dir: str
    target_label_column: str = TARGET_LABEL_COLUMN
    model_type: str = "lightgbm"  # lightgbm | mlp
    num_folds: int = 5
    seed: int = 42
    group_by: str = "school_class"
    calibrate_bias: bool = True


def _fit_model(model_type: str, x_train: np.ndarray, y_train: np.ndarray, num_classes: int, seed: int):
    if model_type == "lightgbm":
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise ImportError("lightgbm is required for model_type=lightgbm; pip install lightgbm or use --model-type mlp") from exc
        model = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=num_classes,
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            class_weight="balanced",
            random_state=seed,
            verbose=-1,
        )
        model.fit(x_train, y_train)
        return model

    if model_type == "mlp":
        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "mlp",
                    MLPClassifier(
                        hidden_layer_sizes=(128, 64),
                        activation="relu",
                        alpha=1e-4,
                        batch_size=64,
                        learning_rate_init=1e-3,
                        max_iter=300,
                        early_stopping=True,
                        validation_fraction=0.1,
                        random_state=seed,
                    ),
                ),
            ]
        )
        model.fit(x_train, y_train)
        return model

    raise ValueError(f"unsupported model_type: {model_type}")


def _predict_proba(model, x: np.ndarray, num_classes: int) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(x)
        if probs.shape[1] == num_classes:
            return probs.astype(np.float32)
    logits = model.decision_function(x)
    if logits.ndim == 1:
        logits = np.stack([-logits, logits], axis=1)
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return (exp / exp.sum(axis=1, keepdims=True)).astype(np.float32)


def train_history_tabular(config: TabularConfig) -> dict[str, Any]:
    output_dir = Path(config.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = Path(config.dataset_path)

    frame = pd.read_csv(dataset_root / "train_val" / "labels.csv")
    frame = frame.dropna(subset=[config.target_label_column]).reset_index(drop=True)
    frame["subject_id"] = make_subject_id(frame)
    labels, label_mapping = encode_ordinal_labels(frame[config.target_label_column])
    groups = make_group_id(frame, config.group_by)

    builder = HistoryFeatureBuilder.from_labels_frame(frame)
    features, _ = builder.transform(frame)
    num_classes = NUM_CLASSES

    fold_indices = build_group_folds(labels, groups, config.num_folds, config.seed)
    oof_logits = np.zeros((len(frame), num_classes), dtype=np.float32)
    oof_predictions = np.full(len(frame), -1, dtype=np.int64)
    fold_metrics: list[dict[str, Any]] = []
    fold_models: list[Any] = []
    fold_biases: list[np.ndarray] = []

    for fold_id, (train_idx, val_idx) in enumerate(fold_indices, start=1):
        fold_dir = output_dir / f"fold_{fold_id}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        model = _fit_model(config.model_type, features[train_idx], labels[train_idx], num_classes, config.seed)
        probs = _predict_proba(model, features[val_idx], num_classes)
        logits = np.log(probs.clip(1e-6, 1.0))

        bias = np.zeros(num_classes, dtype=np.float32)
        metrics = classification_metrics(labels[val_idx], logits.argmax(axis=1))
        if config.calibrate_bias:
            bias, metrics = search_class_bias(logits, labels[val_idx])

        calibrated_logits = apply_class_bias(logits, bias)
        predictions = calibrated_logits.argmax(axis=1)
        oof_logits[val_idx] = calibrated_logits
        oof_predictions[val_idx] = predictions
        fold_models.append(model)
        fold_biases.append(bias)

        fold_record = {"fold": fold_id, "model_type": config.model_type, **metrics}
        fold_metrics.append(fold_record)
        with open(fold_dir / "model.pkl", "wb") as f:
            pickle.dump({"model": model, "class_bias": bias.tolist()}, f)
        (fold_dir / "metrics.json").write_text(json.dumps(fold_record, ensure_ascii=False, indent=2), encoding="utf-8")

    label_by_index = {index: label for label, index in label_mapping.items()}
    class_names = [label for label, _ in sorted(label_mapping.items(), key=lambda item: item[1])]
    overall = classification_metrics(labels, oof_predictions)
    metrics_df = pd.DataFrame(fold_metrics)

    oof_df = pd.DataFrame(
        {
            "subject_id": frame["subject_id"].astype(str),
            "true_label": frame[config.target_label_column].astype(str),
            "pred_label": [label_by_index[int(i)] for i in oof_predictions],
        }
    )
    for class_index in range(num_classes):
        oof_df[f"logit_class_{class_index}"] = oof_logits[:, class_index]

    metrics_df.to_csv(output_dir / "fold_metrics.csv", index=False, encoding="utf-8")
    oof_df.to_csv(output_dir / "oof_predictions.csv", index=False, encoding="utf-8")
    (output_dir / "label_mapping.json").write_text(json.dumps(label_mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "tabular_config.json").write_text(json.dumps(asdict(config), ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "classification_report.txt").write_text(
        format_classification_report(labels, oof_predictions, class_names),
        encoding="utf-8",
    )

    test_path = dataset_root / "test" / "subjects.csv"
    has_test_history = any(col in pd.read_csv(test_path, nrows=1).columns for col in builder.score_columns)
    if test_path.exists() and has_test_history:
        test_frame = pd.read_csv(test_path)
        test_features, _ = builder.transform(test_frame)
        all_probs = []
        for model, bias in zip(fold_models, fold_biases):
            probs = _predict_proba(model, test_features, num_classes)
            logits = np.log(probs.clip(1e-6, 1.0))
            logits = apply_class_bias(logits, bias)
            exp = np.exp(logits - logits.max(axis=1, keepdims=True))
            all_probs.append(exp / exp.sum(axis=1, keepdims=True))
        ensemble = np.mean(np.stack(all_probs, axis=0), axis=0)
        pred_index = ensemble.argmax(axis=1)
        output = test_frame[["anon_school", "anon_class", "anon_person"]].copy()
        output["label"] = [label_by_index[int(i)] for i in pred_index]
        for class_index in range(num_classes):
            output[f"prob_class_{class_index}"] = ensemble[:, class_index]
        output.to_csv(output_dir / "test_predictions.csv", index=False, encoding="utf-8")

    summary = {
        "model_type": config.model_type,
        "config": asdict(config),
        "feature_names": builder.feature_names,
        "fold_metrics_mean": metrics_df.mean(numeric_only=True).to_dict(),
        "overall_oof_metrics": overall,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
