"""
Trial 3: Augment SSL features with audio_basic + video_basic pooled features.

audio_basic: 74-dim handcrafted acoustic features (rms, zcr, centroid, etc.)
video_basic: 13-dim handcrafted visual features (brightness, blur, motion)

These are concatenated with the main SSL features per clip as early fusion.
The architecture is the DeepResidualModel from Trial 2 (current best).
"""

from __future__ import annotations

import copy, json, math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

from ccac.baselines.anxiety_baseline import (
    STAGES, CLIP_TYPES, BaselineConfig,
    _set_seed, _resolve_device, _resolve_num_workers,
    _encode_labels, _build_folds, _fit_scaler, _apply_scaler,
    _class_weights, _classification_metrics,
    _is_release_dataset, _load_release_train_val,
)
from ccac.baselines.dass_baseline import (
    DASSConfig, FocalLoss, DASSDataset, _extract_dass_features,
    _build_multi_features, _load_multi_train_val,
    _calibrate_thresholds, _apply_thresholds,
)
from ccac.experiments.deep_residual import DeepResidualModel


@dataclass(frozen=True)
class BasicFeaturesConfig:
    dass_scheme: str = "scores_das"
    focal_gamma: float = 1.0
    num_heads: int = 4
    num_residual_blocks: int = 3
    calibrate_thresholds: bool = True
    use_audio_basic: bool = True
    use_video_basic: bool = True


def train_basic_features(
    baseline_config: BaselineConfig,
    basic_config: BasicFeaturesConfig | None = None,
) -> dict[str, Any]:
    if basic_config is None:
        basic_config = BasicFeaturesConfig()

    _set_seed(baseline_config.seed)
    device = _resolve_device(baseline_config.device)
    output_dir = Path(baseline_config.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = Path(baseline_config.dataset_path)

    # Build multi-feature list: SSL + basic features
    audio_features = [baseline_config.audio_feature_name]
    if basic_config.use_audio_basic and "audio_basic" != baseline_config.audio_feature_name:
        audio_features.append("audio_basic")
    video_features = [baseline_config.video_feature_name]
    if basic_config.use_video_basic and "video_basic" != baseline_config.video_feature_name:
        video_features.append("video_basic")

    print(f"Audio features: {audio_features}")
    print(f"Video features: {video_features}")

    if _is_release_dataset(dataset_path):
        frame, _, _, label_mapping, _ = _load_release_train_val(baseline_config, dataset_path)
        labels = frame["_label_index"].to_numpy(dtype=np.int64)
        av_features, clip_mask, input_dim = _load_multi_train_val(
            dataset_path, frame, audio_features, video_features,
            baseline_config.target_label_column, baseline_config.feature_cache,
        )
    else:
        frame = pd.read_csv(baseline_config.dataset_path)
        frame = frame.dropna(subset=[baseline_config.target_label_column]).reset_index(drop=True)
        labels, label_mapping = _encode_labels(frame[baseline_config.target_label_column])
        # Fallback to single feature
        from ccac.baselines.anxiety_baseline import BaselineFeatureBuilder
        builder = BaselineFeatureBuilder(
            baseline_config.audio_feature_name, baseline_config.video_feature_name
        ).fit(frame)
        av_features, clip_mask = builder.transform(frame)
        input_dim = builder.input_dim

    print(f"Input dim: {input_dim}")

    dass_features = _extract_dass_features(frame, basic_config.dass_scheme)
    dass_input_dim = dass_features.shape[1]

    fold_indices = _build_folds(labels, baseline_config.num_folds, baseline_config.seed)

    oof_probabilities = np.zeros((len(frame), len(label_mapping)), dtype=np.float32)
    oof_predictions = np.full(len(frame), fill_value=-1, dtype=np.int64)
    metrics: list[dict[str, Any]] = []
    fold_states: list[dict[str, Any]] = []

    for fold_id, (train_idx, val_idx) in enumerate(fold_indices, start=1):
        fold_dir = output_dir / f"fold_{fold_id}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        workers = _resolve_num_workers(baseline_config.num_workers)

        scaler = _fit_scaler(av_features[train_idx], clip_mask[train_idx])
        train_av = _apply_scaler(av_features[train_idx], scaler)
        val_av = _apply_scaler(av_features[val_idx], scaler)

        dass_mean = dass_features[train_idx].mean(axis=0)
        dass_std = dass_features[train_idx].std(axis=0)
        dass_std = np.where(dass_std < 1e-6, 1.0, dass_std)
        train_dass = (dass_features[train_idx] - dass_mean) / dass_std
        val_dass = (dass_features[val_idx] - dass_mean) / dass_std

        train_dataset = DASSDataset(train_av, clip_mask[train_idx], train_dass, labels[train_idx])
        val_dataset = DASSDataset(val_av, clip_mask[val_idx], val_dass, labels[val_idx])

        train_loader = DataLoader(train_dataset, batch_size=baseline_config.batch_size,
                                  shuffle=True, num_workers=workers)
        val_loader = DataLoader(val_dataset, batch_size=baseline_config.batch_size,
                                shuffle=False, num_workers=workers)

        model = DeepResidualModel(
            input_dim=input_dim,
            num_classes=len(label_mapping),
            hidden_dim=baseline_config.hidden_dim,
            num_heads=basic_config.num_heads,
            num_residual_blocks=basic_config.num_residual_blocks,
            dropout=baseline_config.dropout,
            dass_config=DASSConfig(dass_scheme=basic_config.dass_scheme),
        ).to(device)

        class_weights = _class_weights(labels[train_idx], len(label_mapping), baseline_config.class_weight_power)
        if class_weights is not None:
            class_weights = class_weights.to(device)

        criterion = FocalLoss(gamma=basic_config.focal_gamma, alpha=class_weights,
                              label_smoothing=baseline_config.label_smoothing) \
            if basic_config.focal_gamma > 0 else \
            nn.CrossEntropyLoss(weight=class_weights, label_smoothing=baseline_config.label_smoothing)

        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=baseline_config.learning_rate,
                                      weight_decay=baseline_config.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6)

        best_state = None
        best_metric = -math.inf
        best_epoch = 0
        epochs_without_improvement = 0

        for epoch in range(1, baseline_config.epochs + 1):
            _train_one_epoch(model, train_loader, optimizer, criterion, device)
            scheduler.step()
            val_metrics, probabilities = _evaluate(model, val_loader, criterion, device, len(label_mapping))
            macro_f1 = float(val_metrics["macro_f1"])
            if macro_f1 > best_metric:
                best_metric = macro_f1
                best_epoch = epoch
                epochs_without_improvement = 0
                best_state = {
                    "model": copy.deepcopy(model.state_dict()),
                    "scaler_mean": scaler[0].tolist(),
                    "scaler_std": scaler[1].tolist(),
                    "dass_mean": dass_mean.tolist(),
                    "dass_std": dass_std.tolist(),
                    "epoch": epoch, "metrics": val_metrics,
                }
                torch.save(best_state, fold_dir / "best_model.pt")
                np.save(fold_dir / "val_probabilities.npy", probabilities)
            else:
                epochs_without_improvement += 1
            if epochs_without_improvement >= baseline_config.patience:
                break

        if best_state is None:
            raise RuntimeError(f"Fold {fold_id} failed")

        model.load_state_dict(best_state["model"])
        val_metrics, probabilities = _evaluate(model, val_loader, criterion, device, len(label_mapping))
        oof_probabilities[val_idx] = probabilities
        oof_predictions[val_idx] = probabilities.argmax(axis=1)
        fold_metric = {"fold": fold_id, "best_epoch": best_epoch, **val_metrics}
        metrics.append(fold_metric)
        fold_states.append(best_state)
        (fold_dir / "metrics.json").write_text(json.dumps(fold_metric, ensure_ascii=False, indent=2), encoding="utf-8")

    # OOF evaluation
    label_by_index = {i: label for label, i in label_mapping.items()}
    overall_metrics = _classification_metrics(labels, oof_predictions)
    metrics_df = pd.DataFrame(metrics)

    oof_df = pd.DataFrame({
        baseline_config.subject_id_column: frame[baseline_config.subject_id_column].astype(str),
        "true_label": frame[baseline_config.target_label_column].astype(str),
        "pred_label": np.asarray([label_by_index[int(i)] for i in oof_predictions], dtype=object),
    })
    for ci in range(len(label_mapping)):
        oof_df[f"prob_class_{ci}"] = oof_probabilities[:, ci]

    metrics_df.to_csv(output_dir / "fold_metrics.csv", index=False, encoding="utf-8")
    oof_df.to_csv(output_dir / "oof_predictions.csv", index=False, encoding="utf-8")
    (output_dir / "label_mapping.json").write_text(json.dumps(label_mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "baseline_config.json").write_text(json.dumps(asdict(baseline_config), ensure_ascii=False, indent=2), encoding="utf-8")

    if basic_config.calibrate_thresholds:
        cal_thr, cal_met = _calibrate_thresholds(oof_probabilities, labels, len(label_mapping))
        (output_dir / "calibrated_thresholds.json").write_text(
            json.dumps({"thresholds": cal_thr.tolist(), "metrics": cal_met}, ensure_ascii=False, indent=2), encoding="utf-8")
        oof_preds_cal = _apply_thresholds(oof_probabilities, cal_thr)
        overall_metrics = _classification_metrics(labels, oof_preds_cal)
        (output_dir / "classification_report_calibrated.txt").write_text(
            classification_report(labels, oof_preds_cal,
                target_names=[l for l, _ in sorted(label_mapping.items(), key=lambda x: x[1])],
                zero_division=0), encoding="utf-8")

    (output_dir / "classification_report.txt").write_text(
        classification_report(labels, oof_predictions,
            target_names=[l for l, _ in sorted(label_mapping.items(), key=lambda x: x[1])],
            zero_division=0), encoding="utf-8")

    summary = {
        "audio_features": audio_features, "video_features": video_features,
        "feature_input_dim": input_dim, "dass_input_dim": dass_input_dim,
        "label_mapping": label_mapping,
        "fold_metrics_mean": metrics_df.mean(numeric_only=True).to_dict(),
        "overall_oof_metrics": overall_metrics,
        "calibrated_macro_f1": float(overall_metrics["macro_f1"]),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return {"feature_input_dim": input_dim, "dass_input_dim": dass_input_dim,
            "label_mapping": label_mapping, "fold_metrics": metrics,
            "overall_oof_metrics": overall_metrics}


def _train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    for batch in dataloader:
        batch_av, batch_mask, batch_dass, batch_labels = batch[:4]
        batch_av = batch_av.to(device); batch_mask = batch_mask.to(device)
        batch_dass = batch_dass.to(device); batch_labels = batch_labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_av, batch_mask, batch_dass)
        loss = criterion(logits, batch_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()


def _evaluate(model, dataloader, criterion, device, num_classes):
    model.eval()
    losses, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch in dataloader:
            batch_av, batch_mask, batch_dass, batch_labels = batch[:4]
            batch_av = batch_av.to(device); batch_mask = batch_mask.to(device)
            batch_dass = batch_dass.to(device); batch_labels = batch_labels.to(device)
            logits = model(batch_av, batch_mask, batch_dass)
            loss = criterion(logits, batch_labels)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            losses.append(float(loss.item()))
            all_labels.append(batch_labels.cpu().numpy())
            all_probs.append(probs)
    labels = np.concatenate(all_labels, axis=0)
    probs = np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0, num_classes), dtype=np.float32)
    preds = probs.argmax(axis=1) if len(probs) else np.zeros(0, dtype=np.int64)
    m = _classification_metrics(labels, preds)
    m["loss"] = float(np.mean(losses)) if losses else 0.0
    return m, probs
