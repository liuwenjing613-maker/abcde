from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from txy.constants import TARGET_LABEL_COLUMN
from txy.data.feature_io import load_or_build_multimodal, make_subject_id
from txy.data.group_split import build_group_folds, make_group_id
from txy.data.history_features import HistoryFeatureBuilder
from txy.data.labels import encode_ordinal_labels
from txy.data.longitudinal_dataset import LongitudinalPersonDataset, collate_person_batch
from txy.training.calibration import apply_class_bias, search_class_bias
from txy.training.losses import build_criterion
from txy.training.metrics import classification_metrics, format_classification_report


@dataclass
class TrainConfig:
    dataset_path: str
    output_dir: str
    audio_feature_name: str = "audio_wavlm_base"
    video_feature_name: str = "video_dinov2_small"
    target_label_column: str = TARGET_LABEL_COLUMN
    hidden_dim: int = 256
    temporal_hidden_dim: int = 192
    history_hidden_dim: int = 128
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
    use_history: bool = True
    multimodal_only: bool = False
    stage_drop_prob: float = 0.1
    clip_drop_prob: float = 0.05
    feature_noise_std: float = 0.01
    checkpoint_metric: str = "macro_f1"
    calibrate_bias: bool = True


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(name: str) -> torch.device:
    if name.startswith("cuda") and torch.cuda.is_available():
        return torch.device(name)
    return torch.device("cpu")


def _fit_scaler(features: np.ndarray, mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    if features.ndim == 4:
        flat = features.reshape(-1, features.shape[-1])
        if mask is not None:
            flat_mask = mask.reshape(-1)
            valid = flat[flat_mask] if flat_mask.any() else flat
        else:
            valid = flat
    else:
        valid = features
    mean = valid.mean(axis=0)
    std = valid.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def _apply_scaler(features: np.ndarray, scaler: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    mean, std = scaler
    if features.ndim == 4:
        return ((features - mean.reshape(1, 1, 1, -1)) / std.reshape(1, 1, 1, -1)).astype(np.float32)
    return ((features - mean.reshape(1, -1)) / std.reshape(1, -1)).astype(np.float32)


def _apply_multimodal_scalers(
    audio: np.ndarray,
    video: np.ndarray,
    fused: np.ndarray,
    scaler: tuple[np.ndarray, np.ndarray],
    audio_dim: int,
    video_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean, std = scaler
    audio_scaled = _apply_scaler(audio, (mean[:audio_dim], std[:audio_dim]))
    video_scaled = _apply_scaler(video, (mean[audio_dim : audio_dim + video_dim], std[audio_dim : audio_dim + video_dim]))
    fused_scaled = _apply_scaler(fused, scaler)
    return audio_scaled, video_scaled, fused_scaled


def _metric_value(metrics: dict[str, float], name: str) -> float:
    return float(metrics.get(name, -math.inf))


@torch.no_grad()
def _predict_logits(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_labels: list[np.ndarray] = []
    all_logits: list[np.ndarray] = []
    for batch in loader:
        batch = _move_batch(batch, device)
        logits = _forward_model(model, batch)
        all_labels.append(batch["labels"].cpu().numpy())
        all_logits.append(logits.cpu().numpy())
    labels = np.concatenate(all_labels, axis=0) if all_labels else np.zeros(0, dtype=np.int64)
    logits = np.concatenate(all_logits, axis=0) if all_logits else np.zeros((0, 0), dtype=np.float32)
    return labels, logits


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def _forward_model(model: nn.Module, batch: dict[str, Any]) -> torch.Tensor:
    if hasattr(model, "stagewise"):
        return model(
            batch["tabular_features"],
            batch["audio"],
            batch["video"],
            batch["clip_mask"],
            batch["history_scores"],
            batch["history_levels"],
        )
    if batch.get("tabular_features") is not None and hasattr(model, "tabular_head"):
        return model(batch["tabular_features"])
    return model(
        batch["audio"],
        batch["video"],
        batch["clip_mask"],
        batch.get("history_scores"),
        batch.get("history_levels"),
    )


def _collate_with_tabular(items: list[dict], tabular: np.ndarray, indices: np.ndarray) -> dict[str, Any]:
    batch = collate_person_batch(items)
    tabular_tensor = torch.from_numpy(tabular[indices]).float()
    return {
        "audio": batch.audio,
        "video": batch.video,
        "fused": batch.fused,
        "clip_mask": batch.clip_mask,
        "history_scores": batch.history_scores,
        "history_levels": batch.history_levels,
        "labels": batch.labels,
        "subject_ids": batch.subject_ids,
        "tabular_features": tabular_tensor,
    }


def _index_tensor_as_numpy(tensor: torch.Tensor, indices: np.ndarray) -> np.ndarray:
    return tensor[indices].cpu().numpy()


class _IndexedDataset(LongitudinalPersonDataset):
    def __init__(self, base: LongitudinalPersonDataset, indices: np.ndarray, tabular: np.ndarray | None = None):
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
        self._indices = indices
        self._tabular = tabular

    def __getitem__(self, index: int) -> dict:
        item = super().__getitem__(index)
        if self._tabular is not None:
            item["tabular_features"] = torch.from_numpy(self._tabular[self._indices[index]]).float()
        return item


def train_longitudinal(
    config: TrainConfig,
    model_factory: Callable[[int, int, int, int, int], nn.Module],
    model_kind: str = "stagewise",
    tabular_features: np.ndarray | None = None,
) -> dict[str, Any]:
    _set_seed(config.seed)
    device = _resolve_device(config.device)
    output_dir = Path(config.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_root = Path(config.dataset_path)
    labels_path = dataset_root / "train_val" / "labels.csv"
    frame = pd.read_csv(labels_path)
    frame = frame.dropna(subset=[config.target_label_column]).reset_index(drop=True)
    frame["subject_id"] = make_subject_id(frame)
    labels, label_mapping = encode_ordinal_labels(frame[config.target_label_column])
    groups = make_group_id(frame, config.group_by)

    history_builder = HistoryFeatureBuilder.from_labels_frame(frame)
    history_scores, history_levels = history_builder.transform(frame)
    if tabular_features is None:
        tabular_features = history_scores

    audio, video, clip_mask, fused, audio_dim, video_dim = load_or_build_multimodal(
        dataset_root,
        "train_val",
        frame,
        config.audio_feature_name,
        config.video_feature_name,
        use_cache=config.feature_cache,
    )

    fold_indices = build_group_folds(labels, groups, config.num_folds, config.seed)
    oof_logits = np.zeros((len(frame), len(label_mapping)), dtype=np.float32)
    oof_predictions = np.full(len(frame), -1, dtype=np.int64)
    fold_metrics: list[dict[str, Any]] = []
    fold_states: list[dict[str, Any]] = []

    for fold_id, (train_idx, val_idx) in enumerate(fold_indices, start=1):
        fold_dir = output_dir / f"fold_{fold_id}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        mm_scaler = _fit_scaler(fused[train_idx], clip_mask[train_idx])
        tab_scaler = _fit_scaler(tabular_features[train_idx])

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

        train_tab = _apply_scaler(tabular_features[train_idx], tab_scaler)
        val_tab = _apply_scaler(tabular_features[val_idx], tab_scaler)

        train_dataset = _IndexedDataset(train_base, np.arange(len(train_idx)), train_tab)
        val_dataset = _IndexedDataset(val_base, np.arange(len(val_idx)), val_tab)

        def collate_train(items):
            local_indices = np.arange(len(items))
            return _collate_with_tabular(items, train_tab, local_indices)

        def collate_val(items):
            local_indices = np.arange(len(items))
            return _collate_with_tabular(items, val_tab, local_indices)

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            collate_fn=collate_train,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=collate_val,
        )

        use_history = config.use_history and not config.multimodal_only
        model = model_factory(
            audio_dim,
            video_dim,
            history_scores.shape[1],
            history_levels.shape[1],
            len(label_mapping),
        ).to(device)

        criterion = build_criterion(
            len(label_mapping),
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
                optimizer.zero_grad(set_to_none=True)
                logits = _forward_model(model, batch)
                loss = criterion(logits, batch["labels"])
                loss.backward()
                optimizer.step()

            _, val_logits = _predict_logits(model, val_loader, device)
            val_labels = labels[val_idx]
            val_metrics = classification_metrics(val_labels, val_logits.argmax(axis=1))
            score = _metric_value(val_metrics, config.checkpoint_metric)

            if score > best_metric:
                best_metric = score
                best_epoch = epoch
                epochs_without_improvement = 0
                best_state = {
                    "model": model.state_dict(),
                    "mm_scaler_mean": mm_scaler[0].tolist(),
                    "mm_scaler_std": mm_scaler[1].tolist(),
                    "tab_scaler_mean": tab_scaler[0].tolist(),
                    "tab_scaler_std": tab_scaler[1].tolist(),
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
        val_labels, val_logits = _predict_logits(model, val_loader, device)
        bias = np.zeros(len(label_mapping), dtype=np.float32)
        calibrated_metrics = best_state["metrics"]
        if config.calibrate_bias:
            bias, calibrated_metrics = search_class_bias(val_logits, val_labels)
            best_state["class_bias"] = bias.tolist()
            best_state["metrics_calibrated"] = calibrated_metrics
            torch.save(best_state, fold_dir / f"best_{config.checkpoint_metric}.pt")

        calibrated_logits = apply_class_bias(val_logits, bias)
        predictions = calibrated_logits.argmax(axis=1)
        oof_logits[val_idx] = calibrated_logits
        oof_predictions[val_idx] = predictions

        fold_record = {
            "fold": fold_id,
            "best_epoch": best_epoch,
            "model_kind": model_kind,
            **best_state["metrics"],
            **{f"calibrated_{k}": v for k, v in calibrated_metrics.items()},
        }
        fold_metrics.append(fold_record)
        fold_states.append(best_state)
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
    for class_index in range(len(label_mapping)):
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
    summary = {
        "model_kind": model_kind,
        "config": asdict(config),
        "label_mapping": label_mapping,
        "feature_dims": {
            "audio_dim": audio_dim,
            "video_dim": video_dim,
            "history_score_dim": int(history_scores.shape[1]),
            "history_level_slots": int(history_levels.shape[1]),
            "tabular_dim": int(tabular_features.shape[1]),
        },
        "fold_metrics_mean": metrics_df.mean(numeric_only=True).to_dict(),
        "overall_oof_metrics": overall,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    _write_test_predictions(
        config,
        dataset_root,
        output_dir,
        frame,
        fold_states,
        label_mapping,
        model_factory,
        audio_dim,
        video_dim,
        history_builder,
        device,
        model_kind,
    )
    return summary


def _write_test_predictions(
    config: TrainConfig,
    dataset_root: Path,
    output_dir: Path,
    train_frame: pd.DataFrame,
    fold_states: list[dict[str, Any]],
    label_mapping: dict[str, int],
    model_factory: Callable,
    audio_dim: int,
    video_dim: int,
    history_builder: HistoryFeatureBuilder,
    device: torch.device,
    model_kind: str,
) -> None:
    test_path = dataset_root / "test" / "subjects.csv"
    if not test_path.exists() or not fold_states:
        return

    test_frame = pd.read_csv(test_path)
    if test_frame.empty:
        return
    test_frame["subject_id"] = make_subject_id(test_frame)
    history_scores, history_levels = history_builder.transform(test_frame)
    tabular_features = history_scores

    audio, video, clip_mask, fused, _, _ = load_or_build_multimodal(
        dataset_root,
        "test",
        test_frame,
        config.audio_feature_name,
        config.video_feature_name,
        use_cache=config.feature_cache,
    )

    all_probs = []
    num_classes = len(label_mapping)
    for state in fold_states:
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
        scaled_tab = _apply_scaler(tabular_features, tab_scaler)

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
        indexed = _IndexedDataset(dataset, np.arange(len(test_frame)), scaled_tab)

        def collate_fn(items):
            return _collate_with_tabular(items, scaled_tab, np.arange(len(items)))

        loader = DataLoader(indexed, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
        model = model_factory(
            audio_dim,
            video_dim,
            history_scores.shape[1],
            history_levels.shape[1],
            num_classes,
        ).to(device)
        model.load_state_dict(state["model"])
        _, logits = _predict_logits(model, loader, device)
        bias = np.asarray(state.get("class_bias", np.zeros(num_classes)), dtype=np.float32)
        logits = apply_class_bias(logits, bias)
        probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
        all_probs.append(probs)

    ensemble = np.mean(np.stack(all_probs, axis=0), axis=0)
    label_by_index = {index: label for label, index in label_mapping.items()}
    pred_index = ensemble.argmax(axis=1)
    output = test_frame[["anon_school", "anon_class", "anon_person", "subject_id"]].copy()
    output["label"] = [label_by_index[int(i)] for i in pred_index]
    for class_index in range(num_classes):
        output[f"prob_class_{class_index}"] = ensemble[:, class_index]
    output.to_csv(output_dir / "test_predictions.csv", index=False, encoding="utf-8")
