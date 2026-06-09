from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from txy.constants import INDEX_TO_LEVEL, TARGET_LABEL_COLUMN
from txy.data.feature_io import load_or_build_multimodal, make_subject_id
from txy.data.group_split import build_group_folds, make_group_id
from txy.data.history_features import HistoryFeatureBuilder
from txy.data.labels import NUM_CLASSES, encode_ordinal_labels
from txy.data.longitudinal_dataset import LongitudinalPersonDataset, collate_person_batch
from txy.models.residual_v3 import ResidualV3Model
from txy.training.calibration import apply_class_bias, search_class_bias
from txy.training.losses import build_criterion
from txy.training.metrics import classification_metrics, format_classification_report
from txy.training.tabular_anchor import _fit_tabular_model, _predict_logits
from txy.training.trainer import (
    _apply_multimodal_scalers,
    _fit_scaler,
    _index_tensor_as_numpy,
    _metric_value,
    _move_batch,
    _resolve_device,
    _set_seed,
)


@dataclass
class TrainV3Config:
    dataset_path: str
    output_dir: str
    audio_feature_name: str = "audio_wavlm_base"
    video_feature_name: str = "video_dinov2_small"
    target_label_column: str = TARGET_LABEL_COLUMN
    hidden_dim: int = 256
    temporal_hidden_dim: int = 192
    dropout: float = 0.2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    class_weight_power: float = 1.0
    label_smoothing: float = 0.0
    batch_size: int = 32
    epochs: int = 80
    patience: int = 12
    num_folds: int = 5
    num_workers: int = 0
    seed: int = 42
    device: str = "cuda"
    feature_cache: bool = True
    group_by: str = "school_class"
    stage_drop_prob: float = 0.1
    clip_drop_prob: float = 0.05
    feature_noise_std: float = 0.01
    checkpoint_metric: str = "macro_f1"
    calibrate_bias: bool = True
    anchor_model_type: str = "lightgbm"
    alpha_with_history: float = 0.25
    alpha_missing_history: float = 1.0
    history_dropout_prob: float = 0.5
    loss_weight_final: float = 1.0
    loss_weight_mm: float = 0.5
    loss_weight_tab: float = 0.3


class _IndexedV3Dataset(LongitudinalPersonDataset):
    def __init__(
        self,
        base: LongitudinalPersonDataset,
        indices: np.ndarray,
        tab_logits: np.ndarray,
    ):
        super().__init__(
            audio=_index_tensor_as_numpy(base.audio, indices),
            video=_index_tensor_as_numpy(base.video, indices),
            fused=_index_tensor_as_numpy(base.fused, indices),
            clip_mask=_index_tensor_as_numpy(base.clip_mask, indices),
            history_scores=_index_tensor_as_numpy(base.history_scores, indices),
            history_levels=_index_tensor_as_numpy(base.history_levels, indices),
            labels=_index_tensor_as_numpy(base.labels, indices),
            subject_ids=[base.subject_ids[i] for i in indices],
            stage_drop_prob=base.stage_drop_prob,
            clip_drop_prob=base.clip_drop_prob,
            feature_noise_std=base.feature_noise_std,
            train=base.train,
        )
        self._tab_logits = tab_logits.astype(np.float32)

    def __getitem__(self, index: int) -> dict:
        item = super().__getitem__(index)
        item["logits_tab"] = torch.from_numpy(self._tab_logits[index]).float()
        return item


def _collate_v3(items: list[dict]) -> dict[str, Any]:
    batch = collate_person_batch(items)
    logits_tab = torch.stack([item["logits_tab"] for item in items], dim=0)
    return {
        "audio": batch.audio,
        "video": batch.video,
        "clip_mask": batch.clip_mask,
        "labels": batch.labels,
        "subject_ids": batch.subject_ids,
        "logits_tab": logits_tab,
    }


def _v3_loss(
    outputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    criterion: nn.Module,
    config: TrainV3Config,
) -> torch.Tensor:
    loss_final = criterion(outputs["logits_final"], labels)
    loss_mm = criterion(outputs["logits_mm"], labels)
    loss_tab = criterion(outputs["logits_tab"].detach(), labels)
    return (
        config.loss_weight_final * loss_final
        + config.loss_weight_mm * loss_mm
        + config.loss_weight_tab * loss_tab
    )


@torch.no_grad()
def _predict_v3(
    model: ResidualV3Model,
    loader: DataLoader,
    device: torch.device,
    history_available: bool,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_labels: list[np.ndarray] = []
    all_logits: list[np.ndarray] = []
    flag = torch.tensor(history_available, device=device)
    for batch in loader:
        batch = _move_batch(batch, device)
        outputs = model(
            batch["audio"],
            batch["video"],
            batch["clip_mask"],
            batch["logits_tab"],
            flag.expand(batch["labels"].shape[0]),
        )
        all_labels.append(batch["labels"].cpu().numpy())
        all_logits.append(outputs["logits_final"].cpu().numpy())
    labels = np.concatenate(all_labels, axis=0) if all_labels else np.zeros(0, dtype=np.int64)
    logits = np.concatenate(all_logits, axis=0) if all_logits else np.zeros((0, NUM_CLASSES), dtype=np.float32)
    return labels, logits


def train_residual_v3(config: TrainV3Config) -> dict[str, Any]:
    _set_seed(config.seed)
    device = _resolve_device(config.device)
    output_dir = Path(config.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_root = Path(config.dataset_path)
    frame = pd.read_csv(dataset_root / "train_val" / "labels.csv")
    frame = frame.dropna(subset=[config.target_label_column]).reset_index(drop=True)
    frame["subject_id"] = make_subject_id(frame)
    labels, label_mapping = encode_ordinal_labels(frame[config.target_label_column])
    groups = make_group_id(frame, config.group_by)

    history_builder = HistoryFeatureBuilder.from_labels_frame(frame)
    tabular_features, _ = history_builder.transform(frame)
    history_scores, history_levels = history_builder.transform(frame)

    audio, video, clip_mask, fused, audio_dim, video_dim = load_or_build_multimodal(
        dataset_root,
        "train_val",
        frame,
        config.audio_feature_name,
        config.video_feature_name,
        use_cache=config.feature_cache,
    )

    fold_indices = build_group_folds(labels, groups, config.num_folds, config.seed)
    oof_logits = np.zeros((len(frame), NUM_CLASSES), dtype=np.float32)
    oof_predictions = np.full(len(frame), -1, dtype=np.int64)
    fold_metrics: list[dict[str, Any]] = []
    fold_states: list[dict[str, Any]] = []

    for fold_id, (train_idx, val_idx) in enumerate(fold_indices, start=1):
        fold_dir = output_dir / f"fold_{fold_id}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        anchor_model = _fit_tabular_model(
            config.anchor_model_type,
            tabular_features[train_idx],
            labels[train_idx],
            config.seed,
        )
        train_tab_logits = _predict_logits(anchor_model, tabular_features[train_idx])
        val_tab_logits = _predict_logits(anchor_model, tabular_features[val_idx])
        anchor_metrics = classification_metrics(labels[val_idx], val_tab_logits.argmax(axis=1))
        with open(fold_dir / "tabular_anchor.pkl", "wb") as f:
            import pickle

            pickle.dump(anchor_model, f)
        (fold_dir / "tabular_anchor_metrics.json").write_text(
            json.dumps(anchor_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        mm_scaler = _fit_scaler(fused[train_idx], clip_mask[train_idx])
        train_audio, train_video, train_fused = _apply_multimodal_scalers(
            audio[train_idx], video[train_idx], fused[train_idx], mm_scaler, audio_dim, video_dim
        )
        val_audio, val_video, val_fused = _apply_multimodal_scalers(
            audio[val_idx], video[val_idx], fused[val_idx], mm_scaler, audio_dim, video_dim
        )

        train_base = LongitudinalPersonDataset(
            train_audio,
            train_video,
            train_fused,
            clip_mask[train_idx],
            history_scores[train_idx],
            history_levels[train_idx],
            labels[train_idx],
            frame["subject_id"].astype(str).iloc[train_idx].tolist(),
            stage_drop_prob=config.stage_drop_prob,
            clip_drop_prob=config.clip_drop_prob,
            feature_noise_std=config.feature_noise_std,
            train=True,
        )
        val_base = LongitudinalPersonDataset(
            val_audio,
            val_video,
            val_fused,
            clip_mask[val_idx],
            history_scores[val_idx],
            history_levels[val_idx],
            labels[val_idx],
            frame["subject_id"].astype(str).iloc[val_idx].tolist(),
            train=False,
        )

        train_dataset = _IndexedV3Dataset(train_base, np.arange(len(train_idx)), train_tab_logits)
        val_dataset = _IndexedV3Dataset(val_base, np.arange(len(val_idx)), val_tab_logits)

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            collate_fn=_collate_v3,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=_collate_v3,
        )

        model = ResidualV3Model(
            audio_dim=audio_dim,
            video_dim=video_dim,
            num_classes=NUM_CLASSES,
            hidden_dim=config.hidden_dim,
            temporal_hidden_dim=config.temporal_hidden_dim,
            dropout=config.dropout,
            alpha_with_history=config.alpha_with_history,
            alpha_missing_history=config.alpha_missing_history,
        ).to(device)

        criterion = build_criterion(
            NUM_CLASSES,
            torch.from_numpy(labels[train_idx]),
            config.class_weight_power,
            config.label_smoothing,
            device,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

        best_state = None
        best_metric = -math.inf
        best_epoch = 0
        epochs_without_improvement = 0

        for epoch in range(1, config.epochs + 1):
            model.train()
            for batch in train_loader:
                batch = _move_batch(batch, device)
                history_available = torch.rand(batch["labels"].shape[0], device=device) >= config.history_dropout_prob
                optimizer.zero_grad(set_to_none=True)
                outputs = model(
                    batch["audio"],
                    batch["video"],
                    batch["clip_mask"],
                    batch["logits_tab"],
                    history_available,
                )
                loss = _v3_loss(outputs, batch["labels"], criterion, config)
                loss.backward()
                optimizer.step()

            _, val_logits = _predict_v3(model, val_loader, device, history_available=True)
            val_metrics = classification_metrics(labels[val_idx], val_logits.argmax(axis=1))
            score = _metric_value(val_metrics, config.checkpoint_metric)

            if score > best_metric:
                best_metric = score
                best_epoch = epoch
                epochs_without_improvement = 0
                best_state = {
                    "model": model.state_dict(),
                    "mm_scaler_mean": mm_scaler[0].tolist(),
                    "mm_scaler_std": mm_scaler[1].tolist(),
                    "epoch": epoch,
                    "metrics": val_metrics,
                }
                torch.save(best_state, fold_dir / f"best_{config.checkpoint_metric}.pt")
            else:
                epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break

        if best_state is None:
            raise RuntimeError(f"fold {fold_id} produced no checkpoint")

        model.load_state_dict(best_state["model"])
        _, val_logits_normal = _predict_v3(model, val_loader, device, history_available=True)
        _, val_logits_zero = _predict_v3(model, val_loader, device, history_available=False)
        metrics_normal = classification_metrics(labels[val_idx], val_logits_normal.argmax(axis=1))
        metrics_zero = classification_metrics(labels[val_idx], val_logits_zero.argmax(axis=1))

        bias = np.zeros(NUM_CLASSES, dtype=np.float32)
        calibrated_metrics = metrics_normal
        val_logits_for_oof = val_logits_normal
        if config.calibrate_bias:
            bias, calibrated_metrics = search_class_bias(val_logits_normal, labels[val_idx])
            best_state["class_bias"] = bias.tolist()
            best_state["metrics_calibrated"] = calibrated_metrics
            torch.save(best_state, fold_dir / f"best_{config.checkpoint_metric}.pt")
            val_logits_for_oof = apply_class_bias(val_logits_normal, bias)

        predictions = val_logits_for_oof.argmax(axis=1)
        oof_logits[val_idx] = val_logits_for_oof
        oof_predictions[val_idx] = predictions

        fold_record = {
            "fold": fold_id,
            "best_epoch": best_epoch,
            "model_kind": "residual_v3",
            "tabular_anchor_macro_f1": anchor_metrics["macro_f1"],
            **best_state["metrics"],
            "val_zero_history_macro_f1": metrics_zero["macro_f1"],
            "val_zero_history_accuracy": metrics_zero["accuracy"],
            **{f"calibrated_{k}": v for k, v in calibrated_metrics.items()},
        }
        fold_metrics.append(fold_record)
        fold_states.append(best_state)
        (fold_dir / "metrics.json").write_text(json.dumps(fold_record, ensure_ascii=False, indent=2), encoding="utf-8")

    class_names = [INDEX_TO_LEVEL[i] for i in range(NUM_CLASSES)]
    overall = classification_metrics(labels, oof_predictions)
    metrics_df = pd.DataFrame(fold_metrics)

    oof_df = pd.DataFrame(
        {
            "subject_id": frame["subject_id"].astype(str),
            "true_label": frame[config.target_label_column].astype(str),
            "pred_label": [INDEX_TO_LEVEL[int(i)] for i in oof_predictions],
        }
    )
    for class_index in range(NUM_CLASSES):
        oof_df[f"logit_class_{class_index}"] = oof_logits[:, class_index]
        oof_df[f"prob_class_{class_index}"] = torch.softmax(torch.from_numpy(oof_logits), dim=-1)[:, class_index].numpy()

    metrics_df.to_csv(output_dir / "fold_metrics.csv", index=False, encoding="utf-8")
    oof_df.to_csv(output_dir / "oof_predictions.csv", index=False, encoding="utf-8")
    (output_dir / "label_mapping.json").write_text(json.dumps(label_mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "train_config.json").write_text(json.dumps(asdict(config), ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "classification_report.txt").write_text(
        format_classification_report(labels, oof_predictions, class_names),
        encoding="utf-8",
    )

    test_has_history = _test_has_history(dataset_root, history_builder)
    _write_test_predictions_v3(
        config,
        dataset_root,
        output_dir,
        frame,
        fold_states,
        history_builder,
        tabular_features,
        device,
        test_has_history,
        audio_dim,
        video_dim,
    )

    summary = {
        "model_kind": "residual_v3",
        "config": asdict(config),
        "label_mapping": label_mapping,
        "encoding": "ordinal 0=正常..4=非常严重",
        "test_has_history": test_has_history,
        "feature_dims": {
            "audio_dim": audio_dim,
            "video_dim": video_dim,
            "tabular_dim": int(tabular_features.shape[1]),
        },
        "fold_metrics_mean": metrics_df.mean(numeric_only=True).to_dict(),
        "overall_oof_metrics": overall,
        "val_zero_history_macro_f1_mean": float(metrics_df["val_zero_history_macro_f1"].mean()),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _test_has_history(dataset_root: Path, history_builder: HistoryFeatureBuilder) -> bool:
    test_path = dataset_root / "test" / "subjects.csv"
    if not test_path.exists():
        return False
    cols = pd.read_csv(test_path, nrows=1).columns
    return any(col in cols for col in history_builder.score_columns)


def _write_test_predictions_v3(
    config: TrainV3Config,
    dataset_root: Path,
    output_dir: Path,
    train_frame: pd.DataFrame,
    fold_states: list[dict[str, Any]],
    history_builder: HistoryFeatureBuilder,
    train_tabular_features: np.ndarray,
    device: torch.device,
    test_has_history: bool,
    audio_dim: int,
    video_dim: int,
) -> None:
    test_path = dataset_root / "test" / "subjects.csv"
    if not test_path.exists() or not fold_states:
        return

    test_frame = pd.read_csv(test_path)
    if test_frame.empty:
        return
    test_frame["subject_id"] = make_subject_id(test_frame)
    history_scores, history_levels = history_builder.transform(test_frame)
    if test_has_history:
        test_tabular, _ = history_builder.transform(test_frame)
    else:
        test_tabular = np.zeros_like(train_tabular_features[: len(test_frame)])

    audio, video, clip_mask, fused, _, _ = load_or_build_multimodal(
        dataset_root,
        "test",
        test_frame,
        config.audio_feature_name,
        config.video_feature_name,
        use_cache=config.feature_cache,
    )

    fold_dirs = sorted(
        [p for p in output_dir.iterdir() if p.is_dir() and p.name.startswith("fold_")],
        key=lambda p: int(p.name.split("_")[1]),
    )
    all_probs = []
    history_available = test_has_history

    for fold_dir, state in zip(fold_dirs, fold_states):
        import pickle

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
        loader = DataLoader(indexed, batch_size=config.batch_size, shuffle=False, collate_fn=_collate_v3)

        model = ResidualV3Model(
            audio_dim=audio_dim,
            video_dim=video_dim,
            num_classes=NUM_CLASSES,
            hidden_dim=config.hidden_dim,
            temporal_hidden_dim=config.temporal_hidden_dim,
            dropout=config.dropout,
            alpha_with_history=config.alpha_with_history,
            alpha_missing_history=config.alpha_missing_history,
        ).to(device)
        model.load_state_dict(state["model"])
        _, logits = _predict_v3(model, loader, device, history_available=history_available)
        bias = np.asarray(state.get("class_bias", np.zeros(NUM_CLASSES)), dtype=np.float32)
        logits = apply_class_bias(logits, bias)
        probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
        all_probs.append(probs)

    ensemble = np.mean(np.stack(all_probs, axis=0), axis=0)
    pred_index = ensemble.argmax(axis=1)
    output = test_frame[["anon_school", "anon_class", "anon_person", "subject_id"]].copy()
    output["label"] = [INDEX_TO_LEVEL[int(i)] for i in pred_index]
    for class_index in range(NUM_CLASSES):
        output[f"prob_class_{class_index}"] = ensemble[:, class_index]
    output.to_csv(output_dir / "test_predictions.csv", index=False, encoding="utf-8")
