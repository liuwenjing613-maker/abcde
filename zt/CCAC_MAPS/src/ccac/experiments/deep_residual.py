"""
Trial 2: Deep residual architecture with cross-stage attention.

Key innovations:
1. Cross-stage multi-head attention — T1/T2/T3 attend to each other
2. Stage difference features (ΔT2-T1, ΔT3-T2, ΔT3-T1) for explicit trajectory
3. Residual fusion blocks with squeeze-and-excitation
4. Deeper classifier with residual skip connections
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
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader

from ccac.baselines.anxiety_baseline import (
    STAGES, CLIP_TYPES, BaselineConfig, BaselineFeatureBuilder,
    _set_seed, _resolve_device, _resolve_num_workers,
    _encode_labels, _build_folds, _fit_scaler, _apply_scaler,
    _class_weights, _classification_metrics,
    _is_release_dataset, _release_cache_path,
    _build_release_features, _load_release_train_val,
)
from ccac.baselines.dass_baseline import (
    DASSConfig, FocalLoss, DASSDataset, _extract_dass_features,
    _load_multi_train_val, _calibrate_thresholds, _apply_thresholds,
)


# ---------------------------------------------------------------------------
# Squeeze-and-Excitation Block
# ---------------------------------------------------------------------------

class SEBlock(nn.Module):
    """Channel-wise squeeze-and-excitation."""
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.GELU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C) or (B, T, C)
        if x.dim() == 2:
            w = self.se(x.unsqueeze(-1))  # (B, C)
            return x * w
        w = self.se(x.transpose(1, 2)).unsqueeze(1)  # (B, 1, C)
        return x * w


# ---------------------------------------------------------------------------
# Residual Fusion Block
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """Pre-LN residual block with SE."""
    def __init__(self, dim: int, dropout: float = 0.2, expansion: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion, dim),
            nn.Dropout(dropout),
        )
        self.se = SEBlock(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.se(self.net(self.norm(x)))


# ---------------------------------------------------------------------------
# Cross-Stage Attention
# ---------------------------------------------------------------------------

class CrossStageAttention(nn.Module):
    """Multi-head cross-attention between temporal stages.

    Each stage (T1, T2, T3) attends to all other stages, capturing
    inter-stage relationships explicitly.
    """

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"

        self.qkv = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, 3, D), mask: (B, 3) — True = pad (missing stage)
        shortcut = x
        x = self.norm(x)
        B, T, D = x.shape

        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each: (B, T, H, D/H)

        q = q.permute(0, 2, 1, 3)  # (B, H, T, D/H)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale  # (B, H, T, T)

        if mask is not None:
            # mask: (B, T), True = ignore
            attn_mask = mask.unsqueeze(1).unsqueeze(2).expand(-1, self.num_heads, T, -1)
            attn = attn.masked_fill(attn_mask, -1e9)

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = attn @ v  # (B, H, T, D/H)
        out = out.permute(0, 2, 1, 3).reshape(B, T, D)
        out = self.out_proj(out)
        return shortcut + self.dropout(out)


# ---------------------------------------------------------------------------
# Deep Residual Model
# ---------------------------------------------------------------------------

class DeepResidualModel(nn.Module):
    """Deep model with cross-stage attention and residual fusion.

    Architecture:
        Clip Encoder → Attention Pooling → Cross-Stage Attention
        → Stage Diff Features → Residual Blocks → Classifier
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_residual_blocks: int = 3,
        dropout: float = 0.2,
        dass_config: DASSConfig | None = None,
    ):
        super().__init__()
        self.dass_config = dass_config or DASSConfig(dass_scheme="none")

        # Clip encoder
        self.clip_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.clip_attention = nn.Linear(hidden_dim, 1)

        # Stage position encoding
        self.stage_position = nn.Parameter(torch.zeros(3, hidden_dim))
        nn.init.normal_(self.stage_position, mean=0.0, std=0.02)

        # Cross-stage attention
        self.cross_stage_attn = CrossStageAttention(hidden_dim, num_heads, dropout)

        # Stage difference projector
        # Compute: ΔT2-T1, ΔT3-T2, ΔT3-T1 → 3 diff vectors
        self.diff_proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

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

        # Fusion: mean_stage + final_stage + diff_features + dass
        fusion_dim = hidden_dim * 2 + hidden_dim + dass_out  # 256*3 + dass

        # Residual fusion blocks
        self.fusion_blocks = nn.Sequential(*[
            ResidualBlock(fusion_dim, dropout) for _ in range(num_residual_blocks)
        ])

        # Classifier with residual
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if 'classifier' in name and p.dim() >= 2:
                nn.init.xavier_uniform_(p, gain=0.5)
            elif 'fusion' in name and p.dim() >= 2:
                nn.init.xavier_uniform_(p)

    def _dass_input_dim(self) -> int:
        scheme = self.dass_config.dass_scheme
        if scheme == "none": return 0
        if scheme == "scores_a": return 3
        if scheme == "scores_das": return 9
        if scheme == "encoder": return 18
        return 0

    def _pool_stage(self, encoded, clip_mask):
        logits = self.clip_attention(encoded).squeeze(-1)
        logits = logits.masked_fill(~clip_mask, -1e9)
        weights = torch.softmax(logits, dim=-1)
        weights = weights * clip_mask.float()
        denom = weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        weights = weights / denom
        stage_repr = (encoded * weights.unsqueeze(-1)).sum(dim=2)
        missing_stage = clip_mask.sum(dim=-1, keepdim=True) == 0
        return torch.where(missing_stage, torch.zeros_like(stage_repr), stage_repr)

    def _compute_stage_diffs(self, stage_repr):
        # stage_repr: (B, 3, D)
        diff_21 = stage_repr[:, 1, :] - stage_repr[:, 0, :]  # T2 - T1
        diff_32 = stage_repr[:, 2, :] - stage_repr[:, 1, :]  # T3 - T2
        diff_31 = stage_repr[:, 2, :] - stage_repr[:, 0, :]  # T3 - T1
        diffs = torch.cat([diff_21, diff_32, diff_31], dim=-1)  # (B, 3*D)
        return self.diff_proj(diffs)  # (B, D)

    def _encode(self, av_inputs, clip_mask, dass_features=None):
        encoded = self.clip_encoder(av_inputs)
        stage_repr = self._pool_stage(encoded, clip_mask)
        stage_repr = stage_repr + self.stage_position.unsqueeze(0)

        # Cross-stage attention
        stage_mask = (clip_mask.sum(dim=-1) == 0)  # (B, 3)
        stage_repr = self.cross_stage_attn(stage_repr, stage_mask)

        # Pool and compute diffs
        pooled_stage = stage_repr.mean(dim=1)       # (B, D)
        final_stage = stage_repr[:, -1, :]           # (B, D)
        diff_features = self._compute_stage_diffs(stage_repr)  # (B, D)

        fused = torch.cat([pooled_stage, final_stage, diff_features], dim=-1)

        if self._has_dass and dass_features is not None:
            dass_repr = self.dass_encoder(dass_features)
            fused = torch.cat([fused, dass_repr], dim=-1)

        return fused

    def forward(self, av_inputs, clip_mask, dass_features=None):
        fused = self._encode(av_inputs, clip_mask, dass_features)
        fused = self.fusion_blocks(fused)
        return self.classifier(fused)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DeepResidualConfig:
    dass_scheme: str = "scores_das"
    focal_gamma: float = 1.0
    num_heads: int = 4
    num_residual_blocks: int = 3
    calibrate_thresholds: bool = True


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_deep_residual(
    baseline_config: BaselineConfig,
    deep_config: DeepResidualConfig | None = None,
) -> dict[str, Any]:
    if deep_config is None:
        deep_config = DeepResidualConfig()

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

    dass_features = _extract_dass_features(frame, deep_config.dass_scheme)
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
            num_heads=deep_config.num_heads,
            num_residual_blocks=deep_config.num_residual_blocks,
            dropout=baseline_config.dropout,
            dass_config=DASSConfig(dass_scheme=deep_config.dass_scheme),
        ).to(device)

        class_weights = _class_weights(labels[train_idx], len(label_mapping), baseline_config.class_weight_power)
        if class_weights is not None:
            class_weights = class_weights.to(device)

        if deep_config.focal_gamma > 0:
            criterion = FocalLoss(gamma=deep_config.focal_gamma, alpha=class_weights,
                                  label_smoothing=baseline_config.label_smoothing)
        else:
            criterion = nn.CrossEntropyLoss(weight=class_weights,
                                            label_smoothing=baseline_config.label_smoothing)

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
            _train_one_epoch(model, train_loader, optimizer, criterion, device)
            scheduler.step()
            val_metrics, probabilities = _evaluate(model, val_loader, criterion, device, len(label_mapping))
            selection_metric = "robust_score"
            selection_score = float(val_metrics[selection_metric])
            if selection_score > best_metric:
                best_metric = selection_score
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
                    "selection_metric": selection_metric,
                    "selection_score": selection_score,
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
        predictions = probabilities.argmax(axis=1)
        oof_probabilities[val_idx] = probabilities
        oof_predictions[val_idx] = predictions
        fold_metric = {
            "fold": fold_id,
            "best_epoch": best_epoch,
            "selection_metric": best_state.get("selection_metric", "robust_score"),
            "selection_score": best_metric,
            **val_metrics,
        }
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
    for ci in range(len(label_mapping)):
        oof_df[f"prob_class_{ci}"] = oof_probabilities[:, ci]

    metrics_df.to_csv(output_dir / "fold_metrics.csv", index=False, encoding="utf-8")
    oof_df.to_csv(output_dir / "oof_predictions.csv", index=False, encoding="utf-8")
    (output_dir / "label_mapping.json").write_text(
        json.dumps(label_mapping, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "baseline_config.json").write_text(
        json.dumps(asdict(baseline_config), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "deep_config.json").write_text(
        json.dumps(asdict(deep_config), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Threshold calibration
    if deep_config.calibrate_thresholds:
        cal_thr, cal_met = _calibrate_thresholds(oof_probabilities, labels, len(label_mapping))
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
        "deep_config": asdict(deep_config),
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
    losses, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch in dataloader:
            batch_av, batch_mask, batch_dass, batch_labels = batch[:4]
            batch_av = batch_av.to(device)
            batch_mask = batch_mask.to(device)
            batch_dass = batch_dass.to(device)
            batch_labels = batch_labels.to(device)
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
