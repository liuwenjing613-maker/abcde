from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from txy.constants import TARGET_LABEL_COLUMN
from txy.data.feature_io import make_subject_id
from txy.data.group_split import build_group_folds, make_group_id
from txy.data.history_features import HistoryFeatureBuilder
from txy.data.labels import encode_ordinal_labels
from txy.training.metrics import classification_metrics


@dataclass
class TabularAnchorConfig:
    dataset_path: str
    output_dir: str
    model_type: str = "mlp"  # mlp | lightgbm
    num_folds: int = 5
    seed: int = 42
    group_by: str = "school_class"


def _fit_tabular_model(model_type: str, x_train: np.ndarray, y_train: np.ndarray, seed: int):
    if model_type == "lightgbm":
        try:
            import lightgbm as lgb
        except ImportError:
            model_type = "mlp"

    if model_type == "lightgbm":
        import lightgbm as lgb

        model = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=5,
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            class_weight="balanced",
            random_state=seed,
            verbose=-1,
        )
        model.fit(x_train, y_train)
        return model

    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(128, 64),
                    max_iter=300,
                    early_stopping=True,
                    validation_fraction=0.1,
                    random_state=seed,
                ),
            ),
        ]
    ).fit(x_train, y_train)


def _predict_logits(model, features: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(features)
        return np.log(probs.clip(1e-6, 1.0)).astype(np.float32)
    raise TypeError("model must support predict_proba")


def train_tabular_anchor(config: TabularAnchorConfig) -> dict[str, Any]:
    output_dir = Path(config.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = Path(config.dataset_path)

    frame = pd.read_csv(dataset_root / "train_val" / "labels.csv")
    frame = frame.dropna(subset=[TARGET_LABEL_COLUMN]).reset_index(drop=True)
    frame["subject_id"] = make_subject_id(frame)
    labels, label_mapping = encode_ordinal_labels(frame[TARGET_LABEL_COLUMN])
    groups = make_group_id(frame, config.group_by)

    builder = HistoryFeatureBuilder.from_labels_frame(frame)
    features, _ = builder.transform(frame)

    fold_indices = build_group_folds(labels, groups, config.num_folds, config.seed)
    oof_logits = np.zeros((len(frame), 5), dtype=np.float32)
    fold_models: list[Any] = []

    for fold_id, (train_idx, val_idx) in enumerate(fold_indices, start=1):
        model = _fit_tabular_model(config.model_type, features[train_idx], labels[train_idx], config.seed)
        oof_logits[val_idx] = _predict_logits(model, features[val_idx])
        fold_dir = output_dir / f"fold_{fold_id}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        with open(fold_dir / "model.pkl", "wb") as f:
            pickle.dump(model, f)
        fold_models.append(model)
        metrics = classification_metrics(labels[val_idx], oof_logits[val_idx].argmax(axis=1))
        (fold_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    overall = classification_metrics(labels, oof_logits.argmax(axis=1))
    np.savez_compressed(
        output_dir / "oof_tabular_logits.npz",
        subject_ids=frame["subject_id"].astype(str).to_numpy(dtype=object),
        logits=oof_logits,
        labels=labels,
    )
    (output_dir / "label_mapping.json").write_text(
        json.dumps(label_mapping, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "anchor_config.json").write_text(
        json.dumps(asdict(config), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = {"overall_oof_metrics": overall, "label_mapping": label_mapping}
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def load_oof_tabular_logits(anchor_dir: Path, subject_ids: list[str]) -> np.ndarray:
    cached = np.load(anchor_dir / "oof_tabular_logits.npz", allow_pickle=True)
    cached_ids = cached["subject_ids"].astype(str).tolist()
    logits = cached["logits"].astype(np.float32)
    id_to_logits = {sid: logits[i] for i, sid in enumerate(cached_ids)}
    return np.stack([id_to_logits[sid] for sid in subject_ids], axis=0)


def predict_tabular_logits(anchor_dir: Path, features: np.ndarray, fold_id: int | None = None) -> np.ndarray:
    fold_dirs = sorted(anchor_dir.glob("fold_*"), key=lambda p: int(p.name.split("_")[1]))
    if fold_id is not None:
        fold_dirs = [anchor_dir / f"fold_{fold_id}"]
    logits_list = []
    for fold_dir in fold_dirs:
        with open(fold_dir / "model.pkl", "rb") as f:
            model = pickle.load(f)
        logits_list.append(_predict_logits(model, features))
    return np.mean(np.stack(logits_list, axis=0), axis=0)
