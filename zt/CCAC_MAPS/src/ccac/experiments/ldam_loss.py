"""
Trial 20: LDAM (Label-Distribution-Aware Margin) Loss for imbalanced classification.

LDAM enforces larger margins for minority classes, providing theoretical guarantees
for long-tailed recognition. Combined with Deferred Re-weighting (DRW) which applies
class re-weighting only after an initial phase of unweighted training.

Reference: Cao et al., "Learning Imbalanced Datasets with Label-Distribution-Aware
Margin Loss", NeurIPS 2019.

Architecture: DeepResidualModel (current best) with LDAM loss replacing Focal Loss.
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
import torch.nn.functional as F
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
    DASSConfig, DASSDataset, _extract_dass_features,
    _calibrate_thresholds, _apply_thresholds,
)
from ccac.experiments.deep_residual import DeepResidualModel


# ---------------------------------------------------------------------------
# LDAM Loss
# ---------------------------------------------------------------------------

class LDAMLoss(nn.Module):
    """Label-Distribution-Aware Margin Loss.

    Adds a class-dependent margin to the logits:
        margin_j = C / n_j^{1/4}

    where n_j is the number of training samples in class j.
    The margin is larger for minority classes, forcing the model to learn
    more discriminative features for them.

    Can be combined with class re-weighting (DRW strategy).
    """

    def __init__(
        self,
        cls_num_list: list[int],
        max_m: float = 0.5,
        s: float = 30.0,
        weight: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
    ):
        """
        Parameters
        ----------
        cls_num_list : list[int]
            Number of samples per class in the training set.
        max_m : float
            Maximum margin. Higher values push minority class boundaries further.
        s : float
            Scaling factor for logits (temperature-like). Typically 30.
        weight : Tensor or None
            Per-class re-weighting (used in DRW phase).
        label_smoothing : float
            Label smoothing factor.
        """
        super().__init__()
        m_list = torch.tensor(
            [0.0 if n == 0 else max_m / (float(n) ** 0.25) for n in cls_num_list],
            dtype=torch.float32,
        )
        # Normalize so that max margin = max_m
        m_list_max = m_list.max()
        if m_list_max > 0:
            m_list = m_list / m_list_max * max_m
        self.register_buffer("m_list", m_list)
        self.s = s
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits : (N, C) raw logits
        targets : (N,) class indices
        """
        num_classes = logits.size(-1)

        # Apply class-dependent margin
        # Move m_list to same device as input for correct indexing
        m_list = self.m_list.to(logits.device)
        batch_m = m_list[targets]  # (N,)
        # Subtract margin only from the correct class logit
        margin = torch.zeros_like(logits)
        margin.scatter_(1, targets.unsqueeze(1), batch_m.unsqueeze(1))
        adjusted_logits = logits - margin

        # Label smoothing
        if self.label_smoothing > 0:
            smooth = self.label_smoothing / (num_classes - 1)
            target_one_hot = torch.full_like(adjusted_logits, smooth)
            target_one_hot.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
        else:
            target_one_hot = torch.zeros_like(adjusted_logits)
            target_one_hot.scatter_(1, targets.unsqueeze(1), 1.0)

        # Compute loss with scaling
        log_probs = torch.log_softmax(adjusted_logits * self.s, dim=-1)
        loss = -(target_one_hot * log_probs).sum(dim=-1)

        # Apply class re-weighting if specified
        if self.weight is not None:
            weight_per_sample = self.weight.to(logits.device)[targets]
            loss = weight_per_sample * loss

        return loss.mean()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LDAMConfig:
    dass_scheme: str = "scores_das"
    ldam_max_m: float = 0.5
    """Maximum margin for LDAM loss. Larger = more margin for minority classes."""
    ldam_scale: float = 30.0
    """Logit scaling factor. Typically 30 for normalized features."""
    drw_epoch: int = 50
    """Epoch after which to apply class re-weighting (DRW).
    Set to 0 to always use re-weighting, or > epochs to never use it."""
    num_heads: int = 4
    num_residual_blocks: int = 3
    calibrate_thresholds: bool = True


# ---------------------------------------------------------------------------
# Training Entry Point
# ---------------------------------------------------------------------------

def train_ldam(
    baseline_config: BaselineConfig,
    ldam_config: LDAMConfig | None = None,
) -> dict[str, Any]:
    if ldam_config is None:
        ldam_config = LDAMConfig()

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

    dass_features = _extract_dass_features(frame, ldam_config.dass_scheme)
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
            num_heads=ldam_config.num_heads,
            num_residual_blocks=ldam_config.num_residual_blocks,
            dropout=baseline_config.dropout,
            dass_config=DASSConfig(dass_scheme=ldam_config.dass_scheme),
        ).to(device)

        # Compute per-class sample counts for LDAM
        train_labels_fold = labels[train_idx]
        cls_num_list = [int((train_labels_fold == c).sum()) for c in range(num_classes)]

        # Class weights for DRW phase
        class_weights = _class_weights(train_labels_fold, num_classes, baseline_config.class_weight_power)
        if class_weights is not None:
            class_weights = class_weights.to(device)

        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=baseline_config.learning_rate,
                                      weight_decay=baseline_config.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6,
        )

        best_state = None
        best_metric = -math.inf
        best_epoch = 0
        epochs_without_improvement = 0

        for epoch in range(1, baseline_config.epochs + 1):
            # DRW: apply class re-weighting only after drw_epoch
            use_weight = class_weights if epoch > ldam_config.drw_epoch else None
            criterion = LDAMLoss(
                cls_num_list=cls_num_list,
                max_m=ldam_config.ldam_max_m,
                s=ldam_config.ldam_scale,
                weight=use_weight,
                label_smoothing=baseline_config.label_smoothing,
            )
            _train_one_epoch(model, train_loader, optimizer, criterion, device)
            scheduler.step()
            val_metrics, probabilities = _evaluate(model, val_loader, device, num_classes)
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
                    "epoch": epoch,
                    "metrics": val_metrics,
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
        val_metrics, probabilities = _evaluate(model, val_loader, device, num_classes)
        predictions = probabilities.argmax(axis=1)
        oof_probabilities[val_idx] = probabilities
        oof_predictions[val_idx] = predictions
        fold_metric = {"fold": fold_id, "best_epoch": best_epoch, **val_metrics}
        metrics.append(fold_metric)
        fold_states.append(best_state)
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
    (output_dir / "ldam_config.json").write_text(
        json.dumps(asdict(ldam_config), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if ldam_config.calibrate_thresholds:
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
        "ldam_config": asdict(ldam_config),
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
    losses, all_labels, all_probs = [], [], []
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
