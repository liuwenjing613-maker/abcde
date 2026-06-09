from __future__ import annotations

import json
import math
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from txy.constants import INDEX_TO_LEVEL, TARGET_LABEL_COLUMN
from txy.data.feature_io import load_or_build_multimodal, make_subject_id
from txy.data.group_split import build_group_folds, make_group_id
from txy.data.history_features import HistoryFeatureBuilder
from txy.data.labels import NUM_CLASSES, encode_ordinal_labels
from txy.data.logit_io import build_train_subject_order, load_oof_aligned
from txy.data.longitudinal_dataset import LongitudinalPersonDataset
from txy.ensemble.class_wise import blend_logits
from txy.models.stagewise_v4 import StageWiseV4Model
from txy.training.calibration import apply_class_bias, search_class_bias
from txy.training.losses import OrdinalLoss, build_criterion, kl_distillation_loss
from txy.training.metrics import classification_metrics, format_classification_report
from txy.training.trainer import (
    _apply_multimodal_scalers,
    _fit_scaler,
    _metric_value,
    _move_batch,
    _resolve_device,
    _set_seed,
)
from txy.training.trainer_v3 import _IndexedV3Dataset, _collate_v3
from txy.training.trainer_v4 import _predict_v4


@dataclass
class TrainV41Config:
    dataset_path: str
    output_dir: str
    audio_feature_name: str = "audio_wavlm_base"
    video_feature_name: str = "video_dinov2_small"
    target_label_column: str = TARGET_LABEL_COLUMN
    hidden_dim: int = 256
    temporal_hidden_dim: int = 192
    dropout: float = 0.3
    learning_rate: float = 3e-4
    weight_decay: float = 5e-4
    class_weight_power: float = 0.75
    label_smoothing: float = 0.0
    batch_size: int = 32
    epochs: int = 40
    patience: int = 8
    warmup_epochs: int = 1
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
    kd_temperature: float = 2.0
    kd_weight: float = 0.5
    ordinal_weight: float = 0.2
    ce_weight: float = 1.0
    loss_mode: str = "ce_kd_ordinal"  # ce_only | ce_kd | ce_ordinal | ce_kd_ordinal
    use_balanced_sampler: bool = True
    teacher_w_v3: float = 0.5
    teacher_w_baseline: float = 0.3
    teacher_w_tabular: float = 0.2
    baseline_oof: str = "artifacts/baseline_ordinal/oof_predictions.csv"
    v3_oof: str = "artifacts/residual_v3/oof_predictions.csv"
    tabular_oof: str = "artifacts/history_tabular/oof_predictions.csv"


def _build_ensemble_teacher(
    subject_ids: list[str],
    config: TrainV41Config,
) -> np.ndarray:
    _, baseline = load_oof_aligned(Path(config.baseline_oof), subject_ids)
    _, v3 = load_oof_aligned(Path(config.v3_oof), subject_ids)
    tab_path = Path(config.tabular_oof)
    tab_map = tab_path.parent / "label_mapping.json"
    _, tabular = load_oof_aligned(tab_path, subject_ids, tab_map)
    weights = [config.teacher_w_baseline, config.teacher_w_v3, config.teacher_w_tabular]
    return blend_logits([baseline, v3, tabular], weights)


def _make_balanced_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(np.float64)
    counts = np.where(counts == 0, 1.0, counts)
    sample_weights = 1.0 / counts[labels]
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(labels),
        replacement=True,
    )


def _v41_loss(
    outputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    teacher_logits: torch.Tensor,
    ce_criterion: nn.Module,
    ordinal_criterion: OrdinalLoss,
    config: TrainV41Config,
) -> torch.Tensor:
    total = torch.zeros(1, device=labels.device)
    mode = config.loss_mode

    if mode in ("ce_only", "ce_kd", "ce_ordinal", "ce_kd_ordinal"):
        total = total + config.ce_weight * ce_criterion(outputs["logits_mm"], labels)

    if mode in ("ce_kd", "ce_kd_ordinal"):
        total = total + config.kd_weight * kl_distillation_loss(
            outputs["logits_mm"], teacher_logits, config.kd_temperature
        )

    if mode in ("ce_ordinal", "ce_kd_ordinal"):
        total = total + config.ordinal_weight * ordinal_criterion(outputs["ordinal_logits"], labels)

    return total.squeeze()


def _apply_warmup_lr(optimizer: torch.optim.Optimizer, epoch: int, base_lr: float, warmup_epochs: int) -> None:
    if warmup_epochs <= 0 or epoch > warmup_epochs:
        lr = base_lr
    else:
        lr = base_lr * (epoch / warmup_epochs)
    for group in optimizer.param_groups:
        group["lr"] = lr


def train_stagewise_v41(config: TrainV41Config) -> dict[str, Any]:
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
    subject_ids = frame["subject_id"].astype(str).tolist()

    teacher_logits_all = _build_ensemble_teacher(subject_ids, config)
    np.savez_compressed(
        output_dir / "ensemble_teacher_logits.npz",
        subject_ids=np.asarray(subject_ids, dtype=object),
        logits=teacher_logits_all,
    )

    history_builder = HistoryFeatureBuilder.from_labels_frame(frame)
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
        val_dist = {INDEX_TO_LEVEL[int(k)]: int(v) for k, v in pd.Series(labels[val_idx]).value_counts().sort_index().items()}
        print(json.dumps({"fold": fold_id, "val_class_dist": val_dist}, ensure_ascii=False))

        train_teacher = teacher_logits_all[train_idx]
        val_teacher = teacher_logits_all[val_idx]

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

        train_dataset = _IndexedV3Dataset(train_base, np.arange(len(train_idx)), train_teacher)
        val_dataset = _IndexedV3Dataset(val_base, np.arange(len(val_idx)), val_teacher)

        sampler = _make_balanced_sampler(labels[train_idx]) if config.use_balanced_sampler else None
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=sampler is None,
            sampler=sampler,
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

        model = StageWiseV4Model(
            audio_dim=audio_dim,
            video_dim=video_dim,
            num_classes=NUM_CLASSES,
            hidden_dim=config.hidden_dim,
            temporal_hidden_dim=config.temporal_hidden_dim,
            dropout=config.dropout,
        ).to(device)

        ce_criterion = build_criterion(
            NUM_CLASSES,
            torch.from_numpy(labels[train_idx]),
            config.class_weight_power,
            config.label_smoothing,
            device,
        )
        ordinal_criterion = OrdinalLoss(NUM_CLASSES).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

        best_state = None
        best_metric = -math.inf
        best_epoch = 0
        epochs_without_improvement = 0

        for epoch in range(1, config.epochs + 1):
            _apply_warmup_lr(optimizer, epoch, config.learning_rate, config.warmup_epochs)
            model.train()
            for batch in train_loader:
                batch = _move_batch(batch, device)
                optimizer.zero_grad(set_to_none=True)
                outputs = model(batch["audio"], batch["video"], batch["clip_mask"])
                loss = _v41_loss(
                    outputs,
                    batch["labels"],
                    batch["logits_tab"],
                    ce_criterion,
                    ordinal_criterion,
                    config,
                )
                loss.backward()
                optimizer.step()

            _, val_logits = _predict_v4(model, val_loader, device)
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
        _, val_logits_mm = _predict_v4(model, val_loader, device)
        bias = np.zeros(NUM_CLASSES, dtype=np.float32)
        calibrated_metrics = classification_metrics(labels[val_idx], val_logits_mm.argmax(axis=1))
        val_logits_for_oof = val_logits_mm
        if config.calibrate_bias:
            bias, calibrated_metrics = search_class_bias(val_logits_mm, labels[val_idx])
            best_state["class_bias"] = bias.tolist()
            best_state["class_bias_shrink"] = (0.5 * bias).tolist()
            best_state["metrics_calibrated"] = calibrated_metrics
            torch.save(best_state, fold_dir / f"best_{config.checkpoint_metric}.pt")
            val_logits_for_oof = apply_class_bias(val_logits_mm, bias)

        oof_logits[val_idx] = val_logits_for_oof
        oof_predictions[val_idx] = val_logits_for_oof.argmax(axis=1)

        fold_record = {
            "fold": fold_id,
            "best_epoch": best_epoch,
            "model_kind": "stagewise_v41",
            **best_state["metrics"],
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

    metrics_df.to_csv(output_dir / "fold_metrics.csv", index=False, encoding="utf-8")
    oof_df.to_csv(output_dir / "oof_predictions.csv", index=False, encoding="utf-8")
    (output_dir / "label_mapping.json").write_text(json.dumps(label_mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "train_config.json").write_text(json.dumps(asdict(config), ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "classification_report.txt").write_text(
        format_classification_report(labels, oof_predictions, class_names),
        encoding="utf-8",
    )

    _write_test_predictions_v41(config, dataset_root, output_dir, fold_states, history_builder, device, audio_dim, video_dim)

    summary = {
        "model_kind": "stagewise_v41",
        "config": asdict(config),
        "overall_oof_metrics": overall,
        "fold_metrics_mean": metrics_df.mean(numeric_only=True).to_dict(),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _write_test_predictions_v41(
    config: TrainV41Config,
    dataset_root: Path,
    output_dir: Path,
    fold_states: list[dict[str, Any]],
    history_builder: HistoryFeatureBuilder,
    device: torch.device,
    audio_dim: int,
    video_dim: int,
) -> None:
    test_path = dataset_root / "test" / "subjects.csv"
    if not test_path.exists() or not fold_states:
        return

    test_frame = pd.read_csv(test_path)
    test_frame["subject_id"] = make_subject_id(test_frame)
    history_scores, history_levels = history_builder.transform(test_frame)
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
    dummy = np.zeros((len(test_frame), NUM_CLASSES), dtype=np.float32)
    all_probs = []

    for fold_dir, state in zip(fold_dirs, fold_states):
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
        indexed = _IndexedV3Dataset(dataset, np.arange(len(test_frame)), dummy)
        loader = DataLoader(indexed, batch_size=config.batch_size, shuffle=False, collate_fn=_collate_v3)
        model = StageWiseV4Model(
            audio_dim=audio_dim,
            video_dim=video_dim,
            num_classes=NUM_CLASSES,
            hidden_dim=config.hidden_dim,
            temporal_hidden_dim=config.temporal_hidden_dim,
            dropout=config.dropout,
        ).to(device)
        model.load_state_dict(state["model"])
        _, logits = _predict_v4(model, loader, device)
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
    output[["anon_school", "anon_class", "anon_person", "label"] + [f"prob_class_{i}" for i in range(NUM_CLASSES)]].to_csv(
        output_dir / "test_predictions_submission.csv", index=False, encoding="utf-8"
    )
