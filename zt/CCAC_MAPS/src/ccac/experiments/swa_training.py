"""
Trial 21: Stochastic Weight Averaging (SWA) for imbalanced anxiety prediction.

SWA averages model weights along the training trajectory with a cyclical or
constant learning rate schedule. This smooths optimization and improves
generalization, particularly for minority classes with noisy gradients.

Combined with DeepResidualModel + Focal Loss (current best setup).

Reference: Izmailov et al., "Averaging Weights Leads to Wider Optima and
Better Generalization", UAI 2018.
"""

from __future__ import annotations

import copy
import json
import math
import os
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
    STAGES, CLIP_TYPES, BaselineConfig, BaselineFeatureBuilder,
    _set_seed, _resolve_device, _resolve_num_workers,
    _encode_labels, _build_folds, _fit_scaler, _apply_scaler,
    _class_weights, _classification_metrics,
    _is_release_dataset, _load_release_train_val,
)
from ccac.baselines.dass_baseline import (
    DASSConfig, FocalLoss, DASSDataset, _extract_dass_features,
    _calibrate_thresholds, _apply_thresholds,
)
from ccac.experiments.deep_residual import DeepResidualModel


# ---------------------------------------------------------------------------
# SWA Utilities
# ---------------------------------------------------------------------------

class SWAModel(nn.Module):
    """Wraps a model and maintains a running average of its weights."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.n_averaged = 0
        self.average_weights: dict[str, torch.Tensor] | None = None

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def update_average(self) -> None:
        """Accumulate current weights into the running average."""
        if self.average_weights is None:
            self.average_weights = {
                name: param.data.clone().detach()
                for name, param in self.model.named_parameters()
            }
            self.n_averaged = 1
        else:
            for name, param in self.model.named_parameters():
                self.average_weights[name] = (
                    self.average_weights[name] * self.n_averaged + param.data.clone().detach()
                ) / (self.n_averaged + 1)
            self.n_averaged += 1

    def apply_average(self) -> None:
        """Replace model weights with the running average."""
        if self.average_weights is None:
            return
        for name, param in self.model.named_parameters():
            param.data.copy_(self.average_weights[name])

    def restore_snapshot(self, snapshot: dict[str, torch.Tensor]) -> None:
        """Restore model weights from a snapshot."""
        for name, param in self.model.named_parameters():
            param.data.copy_(snapshot[name])

    def snapshot(self) -> dict[str, torch.Tensor]:
        """Create a snapshot of current model weights."""
        return {name: param.data.clone() for name, param in self.model.named_parameters()}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SWAConfig:
    dass_scheme: str = "scores_das"
    focal_gamma: float = 1.0
    num_heads: int = 4
    num_residual_blocks: int = 3
    calibrate_thresholds: bool = True
    swa_start_epoch: int = 40
    """Epoch at which to start collecting SWA averages."""
    swa_frequency: int = 1
    """Average every N epochs."""
    swa_lr: float = 1e-4
    """Learning rate during SWA collection phase (usually lower)."""


# ---------------------------------------------------------------------------
# Training Entry Point
# ---------------------------------------------------------------------------

def train_swa(
    baseline_config: BaselineConfig,
    swa_config: SWAConfig | None = None,
) -> dict[str, Any]:
    if swa_config is None:
        swa_config = SWAConfig()

    _set_seed(baseline_config.seed)
    device = _resolve_device(baseline_config.device)
    output_dir = Path(baseline_config.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = Path(baseline_config.dataset_path)

    if _is_release_dataset(dataset_path):
        frame, av_features, clip_mask, label_mapping, input_dim = _load_release_train_val(
            baseline_config, dataset_path
        )
        labels = frame["_label_index"].to_numpy(dtype=np.int64)
    else:
        frame = pd.read_csv(baseline_config.dataset_path)
        frame = frame.dropna(subset=[baseline_config.target_label_column]).reset_index(drop=True)
        labels, label_mapping = _encode_labels(frame[baseline_config.target_label_column])
        builder = BaselineFeatureBuilder(
            baseline_config.audio_feature_name, baseline_config.video_feature_name
        ).fit(frame)
        av_features, clip_mask = builder.transform(frame)
        input_dim = builder.input_dim

    dass_features = _extract_dass_features(frame, swa_config.dass_scheme)
    dass_input_dim = dass_features.shape[1]
    num_classes = len(label_mapping)

    fold_indices = _build_folds(labels, baseline_config.num_folds, baseline_config.seed)

    oof_probabilities = np.zeros((len(frame), num_classes), dtype=np.float32)
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
            num_classes=num_classes,
            hidden_dim=baseline_config.hidden_dim,
            num_heads=swa_config.num_heads,
            num_residual_blocks=swa_config.num_residual_blocks,
            dropout=baseline_config.dropout,
            dass_config=DASSConfig(dass_scheme=swa_config.dass_scheme),
        ).to(device)

        swa = SWAModel(model)

        class_weights = _class_weights(labels[train_idx], num_classes, baseline_config.class_weight_power)
        if class_weights is not None:
            class_weights = class_weights.to(device)

        if swa_config.focal_gamma > 0:
            criterion = FocalLoss(gamma=swa_config.focal_gamma, alpha=class_weights,
                                  label_smoothing=baseline_config.label_smoothing)
        else:
            criterion = nn.CrossEntropyLoss(weight=class_weights,
                                            label_smoothing=baseline_config.label_smoothing)

        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=baseline_config.learning_rate,
                                      weight_decay=baseline_config.weight_decay)

        # Cosine annealing for pre-SWA phase, constant for SWA phase
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=swa_config.swa_lr,
        )

        best_state = None
        best_metric = -math.inf
        best_epoch = 0
        epochs_without_improvement = 0
        swa_enabled = False
        best_model_snapshot = None  # Keep best model snapshot alongside SWA

        for epoch in range(1, baseline_config.epochs + 1):
            _train_one_epoch(model, train_loader, optimizer, criterion, device)
            scheduler.step()

            # Evaluate with current weights
            val_metrics, probabilities = _evaluate(model, val_loader, device, num_classes)
            macro_f1 = float(val_metrics["macro_f1"])

            # Track best single-point model
            if macro_f1 > best_metric:
                best_metric = macro_f1
                best_epoch = epoch
                epochs_without_improvement = 0
                best_model_snapshot = {
                    name: param.data.clone() for name, param in model.named_parameters()
                }
                best_state = {
                    "model": copy.deepcopy(model.state_dict()),
                    "scaler_mean": scaler[0].tolist(),
                    "scaler_std": scaler[1].tolist(),
                    "dass_mean": dass_mean.tolist(),
                    "dass_std": dass_std.tolist(),
                    "epoch": epoch,
                    "metrics": val_metrics,
                    "swa": False,
                }
            else:
                epochs_without_improvement += 1

            # SWA collection phase
            if epoch >= swa_config.swa_start_epoch:
                swa_enabled = True
                if (epoch - swa_config.swa_start_epoch) % swa_config.swa_frequency == 0:
                    swa.update_average()

            if epochs_without_improvement >= baseline_config.patience:
                break

        # After training, compare SWA with best single-point
        if swa_enabled and swa.n_averaged > 0:
            swa.apply_average()
            swa_val_metrics, swa_probabilities = _evaluate(model, val_loader, device, num_classes)
            swa_mf1 = float(swa_val_metrics["macro_f1"])

            if swa_mf1 > best_metric:
                print(f"  Fold {fold_id}: SWA ({swa_mf1:.4f}) > Best ({best_metric:.4f})")
                best_metric = swa_mf1
                best_state = {
                    "model": copy.deepcopy(model.state_dict()),
                    "scaler_mean": scaler[0].tolist(),
                    "scaler_std": scaler[1].tolist(),
                    "dass_mean": dass_mean.tolist(),
                    "dass_std": dass_std.tolist(),
                    "epoch": best_epoch,
                    "metrics": swa_val_metrics,
                    "swa": True,
                }
            else:
                print(f"  Fold {fold_id}: Best ({best_metric:.4f}) > SWA ({swa_mf1:.4f})")
                # Restore best single-point
                if best_model_snapshot is not None:
                    swa.restore_snapshot(best_model_snapshot)

        if best_state is None:
            raise RuntimeError(f"Fold {fold_id} failed")

        # Save best state
        torch.save(best_state, fold_dir / "best_model.pt")
        if best_state.get("swa"):
            # Re-evaluate with SWA weights for OOF
            val_metrics, probabilities = _evaluate(model, val_loader, device, num_classes)
        else:
            model.load_state_dict(best_state["model"])
            val_metrics, probabilities = _evaluate(model, val_loader, device, num_classes)

        predictions = probabilities.argmax(axis=1)
        oof_probabilities[val_idx] = probabilities
        oof_predictions[val_idx] = predictions
        fold_metric = {"fold": fold_id, "best_epoch": best_epoch, "swa": best_state.get("swa", False), **val_metrics}
        metrics.append(fold_metric)
        fold_states.append(best_state)
        np.save(fold_dir / "val_probabilities.npy", probabilities)
        (fold_dir / "metrics.json").write_text(
            json.dumps(fold_metric, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # OOF evaluation
    label_by_index = {i: label for label, i in label_mapping.items()}
    overall_metrics = _classification_metrics(labels, oof_predictions)
    metrics_df = pd.DataFrame(metrics)

    oof_df = pd.DataFrame({
        baseline_config.subject_id_column: frame[baseline_config.subject_id_column].astype(str),
        "true_label": frame[baseline_config.target_label_column].astype(str),
        "pred_label": np.asarray([label_by_index[int(i)] for i in oof_predictions], dtype=object),
    })
    for ci in range(num_classes):
        oof_df[f"prob_class_{ci}"] = oof_probabilities[:, ci]

    metrics_df.to_csv(output_dir / "fold_metrics.csv", index=False, encoding="utf-8")
    oof_df.to_csv(output_dir / "oof_predictions.csv", index=False, encoding="utf-8")
    (output_dir / "label_mapping.json").write_text(
        json.dumps(label_mapping, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "baseline_config.json").write_text(
        json.dumps(asdict(baseline_config), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "swa_config.json").write_text(
        json.dumps(asdict(swa_config), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if swa_config.calibrate_thresholds:
        cal_thr, cal_met = _calibrate_thresholds(oof_probabilities, labels, num_classes)
        (output_dir / "calibrated_thresholds.json").write_text(
            json.dumps({"thresholds": cal_thr.tolist(), "metrics": cal_met}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        oof_preds_cal = _apply_thresholds(oof_probabilities, cal_thr)
        overall_metrics = _classification_metrics(labels, oof_preds_cal)
        (output_dir / "classification_report_calibrated.txt").write_text(
            classification_report(
                labels, oof_preds_cal,
                target_names=[l for l, _ in sorted(label_mapping.items(), key=lambda x: x[1])],
                zero_division=0,
            ), encoding="utf-8",
        )

    (output_dir / "classification_report.txt").write_text(
        classification_report(
            labels, oof_predictions,
            target_names=[l for l, _ in sorted(label_mapping.items(), key=lambda x: x[1])],
            zero_division=0,
        ), encoding="utf-8",
    )

    summary = {
        "config": asdict(baseline_config),
        "swa_config": asdict(swa_config),
        "feature_input_dim": input_dim,
        "dass_input_dim": dass_input_dim,
        "label_mapping": label_mapping,
        "fold_metrics_mean": metrics_df.mean(numeric_only=True).to_dict(),
        "overall_oof_metrics": overall_metrics,
        "calibrated_macro_f1": float(overall_metrics["macro_f1"]),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return {
        "feature_input_dim": input_dim,
        "dass_input_dim": dass_input_dim,
        "label_mapping": label_mapping,
        "fold_metrics": metrics,
        "overall_oof_metrics": overall_metrics,
    }


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


def _evaluate(model, dataloader, device, num_classes):
    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for batch in dataloader:
            batch_av, batch_mask, batch_dass, batch_labels = batch[:4]
            batch_av = batch_av.to(device); batch_mask = batch_mask.to(device)
            batch_dass = batch_dass.to(device); batch_labels = batch_labels.to(device)
            logits = model(batch_av, batch_mask, batch_dass)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_labels.append(batch_labels.cpu().numpy())
            all_probs.append(probs)
    labels = np.concatenate(all_labels, axis=0)
    probs = np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0, num_classes), dtype=np.float32)
    preds = probs.argmax(axis=1) if len(probs) else np.zeros(0, dtype=np.int64)
    m = _classification_metrics(labels, preds)
    m["loss"] = 0.0
    return m, probs
