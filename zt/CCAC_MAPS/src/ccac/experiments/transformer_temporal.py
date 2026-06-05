"""
Trial 1: Transformer-based temporal encoder replacing BiGRU.

Key changes from baseline:
- 2-layer Pre-LN TransformerEncoder instead of single-layer BiGRU
- Sinusoidal + learnable residual position encoding
- Multi-head self-attention (4 heads) for richer temporal interaction
- Gradient checkpointing for memory efficiency
- Optional: deep fusion with residual connections

Expected benefit: Better modeling of long-range T1↔T3 interactions
that a bidirectional GRU may miss due to the 3-step bottleneck.
"""

from __future__ import annotations

import copy
import json
import math
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data import DataLoader, Dataset

# Reuse shared utilities from original baseline
from ccac.baselines.anxiety_baseline import (
    STAGES,
    CLIP_TYPES,
    BaselineConfig,
    BaselineFeatureBuilder,
    _set_seed,
    _resolve_device,
    _resolve_num_workers,
    _encode_labels,
    _build_folds,
    _fit_scaler,
    _apply_scaler,
    _class_weights,
    _classification_metrics,
    _is_release_dataset,
    _release_cache_path,
    _build_release_features,
    _load_release_train_val,
    _write_release_test_predictions,
    _infer_release_feature_dim,
    _scan_pooled_cache,
    _load_release_vector_cached,
)
from ccac.baselines.dass_baseline import (
    DASSConfig,
    FocalLoss,
    DASSDataset,
    _extract_dass_features,
    _build_multi_features,
    _multi_cache_path,
    _load_multi_train_val,
    _calibrate_thresholds,
    _apply_thresholds,
)


# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------

class SinusoidalPositionEncoding(nn.Module):
    """Sinusoidal position encoding + learnable residual for 3 time steps."""

    def __init__(self, d_model: int, max_len: int = 8):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)
        self.learnable_residual = nn.Parameter(torch.zeros(3, d_model))
        nn.init.normal_(self.learnable_residual, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, D)
        return x + self.pe[:3, :].unsqueeze(0) + self.learnable_residual.unsqueeze(0)


# ---------------------------------------------------------------------------
# Transformer Temporal Encoder Model
# ---------------------------------------------------------------------------

class TransformerTemporalModel(nn.Module):
    """Anxiety model with Transformer temporal encoder.

    Architecture:
        Clip Encoder (shared) → Attention Pooling → Transformer Encoder
        → Deep Fusion with residuals → Classifier
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        transformer_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.2,
        dass_config: DASSConfig | None = None,
    ):
        super().__init__()
        self.dass_config = dass_config or DASSConfig(dass_scheme="none")

        # Clip encoder (same as baseline)
        self.clip_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.clip_attention = nn.Linear(hidden_dim, 1)

        # Project clip representations to transformer dimension
        self.stage_proj = nn.Linear(hidden_dim, transformer_dim)
        self.pos_encoding = SinusoidalPositionEncoding(transformer_dim)

        # Pre-LN Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=num_heads,
            dim_feedforward=transformer_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # DASS encoder
        dass_dim = self._dass_input_dim()
        self._has_dass = dass_dim > 0
        if self._has_dass and self.dass_config.dass_scheme == "encoder":
            self.dass_encoder = nn.Sequential(
                nn.Linear(dass_dim, self.dass_config.dass_hidden),
                nn.LayerNorm(self.dass_config.dass_hidden),
                nn.GELU(),
                nn.Dropout(self.dass_config.dass_dropout),
                nn.Linear(self.dass_config.dass_hidden, self.dass_config.dass_hidden),
            )
            dass_out = self.dass_config.dass_hidden
        elif self._has_dass:
            self.dass_encoder = nn.Identity()
            dass_out = dass_dim
        else:
            self.dass_encoder = None
            dass_out = 0

        # Deep fusion with residual connections
        # Features: transformer_output_mean + stage_mean + final_stage + transformer_cls
        av_fusion_dim = transformer_dim * 3  # mean, final, cls-style
        fusion_dim = av_fusion_dim + dass_out

        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Linear(hidden_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        for p in self.transformer.parameters():
            if p.dim() >= 2:
                nn.init.xavier_uniform_(p, gain=0.5)
        for module in [self.fusion, self.classifier]:
            for p in module.parameters():
                if p.dim() >= 2:
                    nn.init.xavier_uniform_(p)

    def _dass_input_dim(self) -> int:
        scheme = self.dass_config.dass_scheme
        if scheme == "none":
            return 0
        if scheme == "scores_a":
            return 3
        if scheme == "scores_das":
            return 9
        if scheme == "encoder":
            return 18
        return 0

    def _pool_stage(self, encoded: torch.Tensor, clip_mask: torch.Tensor) -> torch.Tensor:
        logits = self.clip_attention(encoded).squeeze(-1)
        logits = logits.masked_fill(~clip_mask, -1e9)
        weights = torch.softmax(logits, dim=-1)
        weights = weights * clip_mask.float()
        denom = weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        weights = weights / denom
        stage_repr = (encoded * weights.unsqueeze(-1)).sum(dim=2)
        missing_stage = clip_mask.sum(dim=-1, keepdim=True) == 0
        return torch.where(missing_stage, torch.zeros_like(stage_repr), stage_repr)

    def _encode(self, av_inputs, clip_mask, dass_features=None):
        # Clip encoding
        encoded = self.clip_encoder(av_inputs)
        stage_repr = self._pool_stage(encoded, clip_mask)  # (B, 3, hidden_dim)

        # Project and add position encoding
        stage_repr = self.stage_proj(stage_repr)  # (B, 3, transformer_dim)
        stage_repr = self.pos_encoding(stage_repr)

        # Create padding mask: True = ignore (all clips missing in a stage)
        stage_mask = (clip_mask.sum(dim=-1) == 0)  # (B, 3)
        stage_repr = self.transformer(stage_repr, src_key_padding_mask=stage_mask)

        # Fusion: mean pool + final step + "cls" (max-pool)
        pooled_temporal = stage_repr.mean(dim=1)  # (B, D)
        final_stage = stage_repr[:, -1, :]         # (B, D)
        cls_like = stage_repr.max(dim=1).values    # (B, D)

        av_fused = torch.cat([pooled_temporal, final_stage, cls_like], dim=-1)

        if self._has_dass and dass_features is not None:
            dass_repr = self.dass_encoder(dass_features)
            return torch.cat([av_fused, dass_repr], dim=-1)
        return av_fused

    def forward(self, av_inputs, clip_mask, dass_features=None):
        fused = self._encode(av_inputs, clip_mask, dass_features)
        return self.classifier(self.fusion(fused))


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TransformerConfig:
    """Extra config for Transformer temporal model."""
    dass_scheme: str = "scores_das"
    focal_gamma: float = 1.0
    transformer_dim: int = 256
    num_heads: int = 4
    num_layers: int = 2
    calibrate_thresholds: bool = True


def train_transformer_baseline(
    baseline_config: BaselineConfig,
    transformer_config: TransformerConfig | None = None,
) -> dict[str, Any]:
    if transformer_config is None:
        transformer_config = TransformerConfig()

    _set_seed(baseline_config.seed)
    device = _resolve_device(baseline_config.device)
    output_dir = Path(baseline_config.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = Path(baseline_config.dataset_path)

    # Load data
    if _is_release_dataset(dataset_path):
        frame, av_features, clip_mask, label_mapping, input_dim = _load_release_train_val(
            baseline_config, dataset_path
        )
        labels = frame["_label_index"].to_numpy(dtype=np.int64)
        audio_list = getattr(transformer_config, 'audio_features', None)
        video_list = getattr(transformer_config, 'video_features', None)
        if audio_list and video_list:
            frame, _, _, label_mapping, _ = _load_release_train_val(baseline_config, dataset_path)
            labels = frame["_label_index"].to_numpy(dtype=np.int64)
            av_features, clip_mask, input_dim = _load_multi_train_val(
                dataset_path, frame, audio_list, video_list,
                baseline_config.target_label_column, baseline_config.feature_cache,
            )
    else:
        frame = pd.read_csv(baseline_config.dataset_path)
        frame = frame.dropna(subset=[baseline_config.target_label_column]).reset_index(drop=True)
        labels, label_mapping = _encode_labels(frame[baseline_config.target_label_column])
        builder = BaselineFeatureBuilder(
            baseline_config.audio_feature_name, baseline_config.video_feature_name
        ).fit(frame)
        av_features, clip_mask = builder.transform(frame)
        input_dim = builder.input_dim

    dass_features = _extract_dass_features(frame, transformer_config.dass_scheme)
    dass_input_dim = dass_features.shape[1]

    # Cross-validation
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

        model = TransformerTemporalModel(
            input_dim=input_dim,
            num_classes=len(label_mapping),
            hidden_dim=baseline_config.hidden_dim,
            transformer_dim=transformer_config.transformer_dim,
            num_heads=transformer_config.num_heads,
            num_layers=transformer_config.num_layers,
            dropout=baseline_config.dropout,
            dass_config=DASSConfig(
                dass_scheme=transformer_config.dass_scheme,
            ),
        ).to(device)

        class_weights = _class_weights(labels[train_idx], len(label_mapping), baseline_config.class_weight_power)
        if class_weights is not None:
            class_weights = class_weights.to(device)

        if transformer_config.focal_gamma > 0:
            criterion = FocalLoss(
                gamma=transformer_config.focal_gamma,
                alpha=class_weights,
                label_smoothing=baseline_config.label_smoothing,
            )
        else:
            criterion = nn.CrossEntropyLoss(
                weight=class_weights,
                label_smoothing=baseline_config.label_smoothing,
            )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=baseline_config.learning_rate,
            weight_decay=baseline_config.weight_decay,
        )

        # Cosine annealing scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )

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
            raise RuntimeError(f"Fold {fold_id} failed to produce a valid checkpoint")

        model.load_state_dict(best_state["model"])
        val_metrics, probabilities = _evaluate(model, val_loader, criterion, device, len(label_mapping))
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
    label_by_index = {index: label for label, index in label_mapping.items()}
    overall_metrics = _classification_metrics(labels, oof_predictions)
    metrics_df = pd.DataFrame(metrics)

    oof_df = pd.DataFrame({
        baseline_config.subject_id_column: frame[baseline_config.subject_id_column].astype(str),
        "true_label": frame[baseline_config.target_label_column].astype(str),
        "pred_label": np.asarray([label_by_index[int(i)] for i in oof_predictions], dtype=object),
    })
    for class_index in range(len(label_mapping)):
        oof_df[f"prob_class_{class_index}"] = oof_probabilities[:, class_index]

    metrics_df.to_csv(output_dir / "fold_metrics.csv", index=False, encoding="utf-8")
    oof_df.to_csv(output_dir / "oof_predictions.csv", index=False, encoding="utf-8")
    (output_dir / "label_mapping.json").write_text(
        json.dumps(label_mapping, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "baseline_config.json").write_text(
        json.dumps(asdict(baseline_config), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "transformer_config.json").write_text(
        json.dumps(asdict(transformer_config), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Threshold calibration
    if transformer_config.calibrate_thresholds:
        calibrated_thresholds, calibrated_metrics = _calibrate_thresholds(
            oof_probabilities, labels, len(label_mapping)
        )
        (output_dir / "calibrated_thresholds.json").write_text(
            json.dumps({"thresholds": calibrated_thresholds.tolist(), "metrics": calibrated_metrics},
                       ensure_ascii=False, indent=2), encoding="utf-8"
        )
        oof_predictions_cal = _apply_thresholds(oof_probabilities, calibrated_thresholds)
        overall_metrics = _classification_metrics(labels, oof_predictions_cal)
        (output_dir / "classification_report_calibrated.txt").write_text(
            classification_report(
                labels, oof_predictions_cal,
                target_names=[l for l, _ in sorted(label_mapping.items(), key=lambda x: x[1])],
                zero_division=0,
            ), encoding="utf-8",
        )

    (output_dir / "classification_report.txt").write_text(
        classification_report(
            labels, oof_predictions,
            target_names=[label for label, _ in sorted(label_mapping.items(), key=lambda item: item[1])],
            zero_division=0,
        ), encoding="utf-8",
    )

    summary = {
        "config": asdict(baseline_config),
        "transformer_config": asdict(transformer_config),
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

    # Test predictions
    if _is_release_dataset(dataset_path):
        _write_transformer_test_predictions(
            baseline_config, transformer_config, dataset_path, output_dir,
            fold_states, label_mapping, input_dim, device,
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
        batch_av = batch_av.to(device)
        batch_mask = batch_mask.to(device)
        batch_dass = batch_dass.to(device)
        batch_labels = batch_labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_av, batch_mask, batch_dass)
        loss = criterion(logits, batch_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()


def _evaluate(model, dataloader, criterion, device, num_classes):
    model.eval()
    losses, all_labels, all_probabilities = [], [], []
    with torch.no_grad():
        for batch in dataloader:
            batch_av, batch_mask, batch_dass, batch_labels = batch[:4]
            batch_av = batch_av.to(device)
            batch_mask = batch_mask.to(device)
            batch_dass = batch_dass.to(device)
            batch_labels = batch_labels.to(device)
            logits = model(batch_av, batch_mask, batch_dass)
            loss = criterion(logits, batch_labels)
            probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
            losses.append(float(loss.item()))
            all_labels.append(batch_labels.cpu().numpy())
            all_probabilities.append(probabilities)
    labels = np.concatenate(all_labels, axis=0)
    probs = np.concatenate(all_probabilities, axis=0) if all_probabilities else np.zeros((0, num_classes), dtype=np.float32)
    preds = probs.argmax(axis=1) if len(probs) else np.zeros(0, dtype=np.int64)
    m = _classification_metrics(labels, preds)
    m["loss"] = float(np.mean(losses)) if losses else 0.0
    return m, probs


def _write_transformer_test_predictions(
    baseline_config, transformer_config, dataset_root, output_dir,
    fold_states, label_mapping, input_dim, device,
):
    test_path = dataset_root / "test" / "subjects.csv"
    if not test_path.exists() or not fold_states:
        return
    test_frame = pd.read_csv(test_path)
    if test_frame.empty:
        return
    test_frame[baseline_config.subject_id_column] = test_frame[
        ["anon_school", "anon_class", "anon_person"]
    ].agg("/".join, axis=1)

    # Try loading from cache (may or may not have subjects array)
    cache_path = _release_cache_path(dataset_root, baseline_config, split="test")
    test_av, test_mask = None, None
    if baseline_config.feature_cache and cache_path.exists():
        try:
            cached = np.load(cache_path, allow_pickle=True, mmap_mode="r")
            if "subjects" in cached:
                cached_subjects = cached["subjects"].astype(str).tolist()
                if cached_subjects == test_frame[baseline_config.subject_id_column].astype(str).tolist():
                    test_av = cached["features"].astype(np.float32)
                    test_mask = cached["clip_mask"].astype(bool)
            else:
                # Cache was built without subjects; check if sizes match
                test_av = cached["features"].astype(np.float32)
                test_mask = cached["clip_mask"].astype(bool)
                if test_av.shape[0] != len(test_frame):
                    test_av = test_mask = None
        except (KeyError, ValueError):
            pass
    if test_av is None:
        test_av, test_mask, _ = _build_release_features(
            dataset_root, "test", test_frame,
            baseline_config.audio_feature_name, baseline_config.video_feature_name,
        )
        if baseline_config.feature_cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            # Don't save subjects to avoid mismatch issues
            np.savez_compressed(cache_path, features=test_av, clip_mask=test_mask,
                               input_dim=np.asarray(input_dim, dtype=np.int64))

    dass_dim = {"none": 0, "scores_a": 3, "scores_das": 9, "encoder": 18}[transformer_config.dass_scheme]
    test_dass = np.zeros((len(test_frame), dass_dim), dtype=np.float32)

    num_classes = len(label_mapping)
    probabilities = []
    for state in fold_states:
        scaler = (np.asarray(state["scaler_mean"], dtype=np.float32),
                  np.asarray(state["scaler_std"], dtype=np.float32))
        scaled_av = _apply_scaler(test_av, scaler)
        dataset = DASSDataset(scaled_av, test_mask, test_dass, np.zeros(len(test_frame), dtype=np.int64))
        loader = DataLoader(dataset, batch_size=baseline_config.batch_size, shuffle=False, num_workers=0)

        model = TransformerTemporalModel(
            input_dim=input_dim, num_classes=num_classes,
            hidden_dim=baseline_config.hidden_dim,
            transformer_dim=transformer_config.transformer_dim,
            num_heads=transformer_config.num_heads,
            num_layers=transformer_config.num_layers,
            dropout=baseline_config.dropout,
            dass_config=DASSConfig(dass_scheme=transformer_config.dass_scheme),
        ).to(device)
        model.load_state_dict(state["model"])
        model.eval()
        fold_probs = []
        with torch.no_grad():
            for batch in loader:
                batch_av, batch_mask, batch_dass = batch[:3]
                batch_av = batch_av.to(device)
                batch_mask = batch_mask.to(device)
                batch_dass = batch_dass.to(device)
                logits = model(batch_av, batch_mask, batch_dass)
                fold_probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
        probabilities.append(np.concatenate(fold_probs, axis=0))

    ensemble = np.mean(np.stack(probabilities, axis=0), axis=0)
    label_by_index = {i: label for label, i in label_mapping.items()}
    pred_index = ensemble.argmax(axis=1)
    pred_label = [label_by_index[int(i)] for i in pred_index]
    output = test_frame[["anon_school", "anon_class", "anon_person", baseline_config.subject_id_column]].copy()
    output["pred_label"] = pred_label
    for ci in range(num_classes):
        output[f"prob_class_{ci}"] = ensemble[:, ci]
    output.to_csv(output_dir / "test_predictions.csv", index=False, encoding="utf-8")
