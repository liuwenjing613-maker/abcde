"""
ND-13: Temporal Convolutional Network (TCN) for stage-level modeling.

TCNs use dilated causal convolutions to capture temporal dependencies.
Unlike GRU (recurrent) or Transformer (self-attention), TCNs:
  - Have a fixed-size receptive field via dilation
  - Process all time steps in parallel (faster than GRU)
  - Are more parameter-efficient than Transformers
  - Capture local temporal patterns well (good for 3-step sequences)

Architecture:
    Clip Encoder → Attention Pooling → TCN (dilated convs)
    → Fusion (mean + final + diff) → Classifier
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
import torch.nn.functional as F
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
    _calibrate_thresholds, _apply_thresholds,
)


# ---------------------------------------------------------------------------
# TCN Components
# ---------------------------------------------------------------------------

class CausalConv1d(nn.Module):
    """Causal 1D convolution with optional dilation."""
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1, dropout=0.2):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=self.padding,
                              dilation=dilation, groups=1)
        self.norm = nn.LayerNorm(out_ch)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, C, T)
        out = self.conv(x)  # (B, C_out, T + padding)
        out = out[:, :, :x.size(2)]  # causal: only use past
        out = out.permute(0, 2, 1)  # (B, T, C_out)
        out = self.norm(out)
        out = F.gelu(out)
        out = self.dropout(out)
        return out.permute(0, 2, 1)  # (B, C_out, T)


class TCNBlock(nn.Module):
    """TCN residual block with dilated causal convs."""
    def __init__(self, channels, kernel_size=3, dilation=1, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size,
                               padding=(kernel_size - 1) * dilation,
                               dilation=dilation)
        self.norm1 = nn.LayerNorm(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size,
                               padding=(kernel_size - 1) * dilation,
                               dilation=dilation)
        self.norm2 = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(channels, channels, 1) if channels != channels else nn.Identity()

    def forward(self, x):
        # x: (B, C, T)
        residual = x
        out = self.conv1(x)[:, :, :x.size(2)]
        out = out.permute(0, 2, 1)
        out = F.gelu(self.norm1(out))
        out = self.dropout(out)
        out = out.permute(0, 2, 1)

        out = self.conv2(out)[:, :, :x.size(2)]
        out = out.permute(0, 2, 1)
        out = self.norm2(out)
        out = self.dropout(out)
        out = out.permute(0, 2, 1)

        return F.gelu(out + residual)


class BidirectionalTCN(nn.Module):
    """Bidirectional TCN — two TCNs, one forward, one backward.

    This gives the model access to both past and future context at each step,
    analogous to a bidirectional GRU.
    """

    def __init__(self, input_dim, hidden_dim, num_layers=3, kernel_size=3,
                 dropout=0.2):
        super().__init__()
        self.input_proj = nn.Conv1d(input_dim, hidden_dim, 1)

        dilations = [2 ** i for i in range(num_layers)]
        self.forward_blocks = nn.ModuleList([
            TCNBlock(hidden_dim, kernel_size, d, dropout) for d in dilations
        ])
        self.backward_blocks = nn.ModuleList([
            TCNBlock(hidden_dim, kernel_size, d, dropout) for d in dilations
        ])

        self.output_dim = hidden_dim * 2  # forward + backward

    def forward(self, x):
        # x: (B, 3, D) — 3 time steps, D features
        x = x.permute(0, 2, 1)  # (B, D, 3)
        x = self.input_proj(x)  # (B, H, 3)

        # Forward pass
        fwd = x
        for block in self.forward_blocks:
            fwd = block(fwd)

        # Backward pass (reverse time dimension)
        bwd = torch.flip(x, [2])
        for block in self.backward_blocks:
            bwd = block(bwd)
        bwd = torch.flip(bwd, [2])

        # Concatenate forward + backward
        out = torch.cat([fwd, bwd], dim=1)  # (B, 2H, 3)
        return out.permute(0, 2, 1)  # (B, 3, 2H)


# ---------------------------------------------------------------------------
# TCN Model
# ---------------------------------------------------------------------------

class TCNTemporalModel(nn.Module):
    """Anxiety prediction with TCN temporal encoder.

    Architecture:
        Clip Encoder → Attention Pooling → +PosEmb
        → Bidirectional TCN (dilated causal convs)
        → Fusion (mean + final + diff) → Residual Blocks → Classifier
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        tcn_layers: int = 3,
        kernel_size: int = 3,
        num_residual_blocks: int = 2,
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

        # Position encoding
        self.stage_position = nn.Parameter(torch.zeros(3, hidden_dim))
        nn.init.normal_(self.stage_position, mean=0.0, std=0.02)

        # Bidirectional TCN
        self.tcn = BidirectionalTCN(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=tcn_layers,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        tcn_output_dim = self.tcn.output_dim  # hidden_dim * 2

        # Stage difference features
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

        # Fusion: TCN mean + TCN final + TCN max + stage_diffs + dass
        fusion_dim = tcn_output_dim * 3 + hidden_dim + dass_out

        # Residual fusion blocks
        from ccac.experiments.deep_residual import ResidualBlock
        self.fusion_blocks = nn.Sequential(*[
            ResidualBlock(fusion_dim, dropout) for _ in range(num_residual_blocks)
        ])

        # Classifier
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
        diff_21 = stage_repr[:, 1, :] - stage_repr[:, 0, :]
        diff_32 = stage_repr[:, 2, :] - stage_repr[:, 1, :]
        diff_31 = stage_repr[:, 2, :] - stage_repr[:, 0, :]
        diffs = torch.cat([diff_21, diff_32, diff_31], dim=-1)
        return self.diff_proj(diffs)

    def _encode(self, av_inputs, clip_mask, dass_features=None):
        encoded = self.clip_encoder(av_inputs)
        stage_repr = self._pool_stage(encoded, clip_mask)
        stage_repr = stage_repr + self.stage_position.unsqueeze(0)

        # TCN processing
        tcn_out = self.tcn(stage_repr)  # (B, 3, 2H)

        # Multi-granularity pooling
        tcn_mean = tcn_out.mean(dim=1)     # (B, 2H)
        tcn_final = tcn_out[:, -1, :]       # (B, 2H)
        tcn_max = tcn_out.max(dim=1).values  # (B, 2H)

        # Stage difference features (from pre-TCN clip representations)
        diff_features = self._compute_stage_diffs(stage_repr)

        fused = torch.cat([tcn_mean, tcn_final, tcn_max, diff_features], dim=-1)

        if self._has_dass and dass_features is not None:
            dass_repr = self.dass_encoder(dass_features)
            fused = torch.cat([fused, dass_repr], dim=-1)

        return fused

    def forward(self, av_inputs, clip_mask, dass_features=None):
        fused = self._encode(av_inputs, clip_mask, dass_features)
        fused = self.fusion_blocks(fused)
        return self.classifier(fused)


# ---------------------------------------------------------------------------
# Config and training
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TCNConfig:
    dass_scheme: str = "none"
    focal_gamma: float = 2.0
    tcn_layers: int = 3
    kernel_size: int = 3
    num_residual_blocks: int = 2
    calibrate_thresholds: bool = True


def train_tcn(
    baseline_config: BaselineConfig,
    tcn_config: TCNConfig | None = None,
) -> dict[str, Any]:
    if tcn_config is None:
        tcn_config = TCNConfig()

    _set_seed(baseline_config.seed)
    device = _resolve_device(baseline_config.device)
    output_dir = Path(baseline_config.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = Path(baseline_config.dataset_path)

    if _is_release_dataset(dataset_path):
        frame, av_features, clip_mask, label_mapping, input_dim = \
            _load_release_train_val(baseline_config, dataset_path)
        labels = frame["_label_index"].to_numpy(dtype=np.int64)
    else:
        raise RuntimeError("Only release dataset supported")

    num_classes = len(label_mapping)
    dass_features = _extract_dass_features(frame, tcn_config.dass_scheme)

    fold_indices = _build_folds(labels, baseline_config.num_folds, baseline_config.seed)

    oof_probabilities = np.zeros((len(frame), num_classes), dtype=np.float32)
    oof_predictions = np.full(len(frame), -1, dtype=np.int64)
    metrics = []
    fold_states = []

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

        train_ds = DASSDataset(train_av, clip_mask[train_idx], train_dass, labels[train_idx])
        val_ds = DASSDataset(val_av, clip_mask[val_idx], val_dass, labels[val_idx])

        train_loader = DataLoader(train_ds, batch_size=baseline_config.batch_size,
                                  shuffle=True, num_workers=workers)
        val_loader = DataLoader(val_ds, batch_size=baseline_config.batch_size,
                               shuffle=False, num_workers=workers)

        model = TCNTemporalModel(
            input_dim=input_dim, num_classes=num_classes,
            hidden_dim=baseline_config.hidden_dim,
            tcn_layers=tcn_config.tcn_layers,
            kernel_size=tcn_config.kernel_size,
            num_residual_blocks=tcn_config.num_residual_blocks,
            dropout=baseline_config.dropout,
            dass_config=DASSConfig(dass_scheme=tcn_config.dass_scheme),
        ).to(device)

        cw = _class_weights(labels[train_idx], num_classes, baseline_config.class_weight_power)
        if cw is not None:
            cw = cw.to(device)

        criterion = FocalLoss(gamma=tcn_config.focal_gamma, alpha=cw) \
            if tcn_config.focal_gamma > 0 else \
            nn.CrossEntropyLoss(weight=cw, label_smoothing=baseline_config.label_smoothing)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=baseline_config.learning_rate,
            weight_decay=baseline_config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6,
        )

        best_state = None
        best_metric = -math.inf
        best_epoch = 0
        epochs_no_improve = 0

        for epoch in range(1, baseline_config.epochs + 1):
            _train_epoch(model, train_loader, optimizer, criterion, device)
            scheduler.step()
            val_m, val_probs = _evaluate(model, val_loader, criterion, device, num_classes)
            mf1 = float(val_m["macro_f1"])
            if mf1 > best_metric:
                best_metric = mf1
                best_epoch = epoch
                epochs_no_improve = 0
                best_state = {
                    "model": copy.deepcopy(model.state_dict()),
                    "scaler_mean": scaler[0].tolist(),
                    "scaler_std": scaler[1].tolist(),
                    "dass_mean": dass_mean.tolist(),
                    "dass_std": dass_std.tolist(),
                    "epoch": epoch, "metrics": val_m,
                }
                torch.save(best_state, fold_dir / "best_model.pt")
                np.save(fold_dir / "val_probabilities.npy", val_probs)
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= baseline_config.patience:
                break

        if best_state is None:
            raise RuntimeError(f"Fold {fold_id} failed")

        model.load_state_dict(best_state["model"])
        val_m, val_probs = _evaluate(model, val_loader, criterion, device, num_classes)
        oof_probabilities[val_idx] = val_probs
        oof_predictions[val_idx] = val_probs.argmax(axis=1)
        fold_metric = {"fold": fold_id, "best_epoch": best_epoch, **val_m}
        metrics.append(fold_metric)
        fold_states.append(best_state)
        (fold_dir / "metrics.json").write_text(
            json.dumps(fold_metric, ensure_ascii=False, indent=2))
        print(f"  Fold {fold_id}: MF1={val_m['macro_f1']:.4f} Acc={val_m['accuracy']:.4f}")

    # OOF
    label_by_idx = {i: l for l, i in label_mapping.items()}
    overall = _classification_metrics(labels, oof_predictions)
    mdf = pd.DataFrame(metrics)

    oof_df = pd.DataFrame({
        baseline_config.subject_id_column: frame[baseline_config.subject_id_column].astype(str),
        "true_label": frame[baseline_config.target_label_column].astype(str),
        "pred_label": [label_by_idx[int(i)] for i in oof_predictions],
    })
    for ci in range(num_classes):
        oof_df[f"prob_class_{ci}"] = oof_probabilities[:, ci]

    mdf.to_csv(output_dir / "fold_metrics.csv", index=False)
    oof_df.to_csv(output_dir / "oof_predictions.csv", index=False)
    (output_dir / "label_mapping.json").write_text(json.dumps(label_mapping, ensure_ascii=False, indent=2))
    (output_dir / "baseline_config.json").write_text(json.dumps(asdict(baseline_config), ensure_ascii=False, indent=2))
    (output_dir / "tcn_config.json").write_text(json.dumps(asdict(tcn_config), ensure_ascii=False, indent=2))

    if tcn_config.calibrate_thresholds:
        cal_thr, cal_met = _calibrate_thresholds(oof_probabilities, labels, num_classes)
        (output_dir / "calibrated_thresholds.json").write_text(
            json.dumps({"thresholds": cal_thr.tolist(), "metrics": cal_met}, ensure_ascii=False, indent=2))
        oof_cal = _apply_thresholds(oof_probabilities, cal_thr)
        overall = _classification_metrics(labels, oof_cal)
        (output_dir / "classification_report_calibrated.txt").write_text(
            classification_report(labels, oof_cal,
                target_names=[l for l, _ in sorted(label_mapping.items(), key=lambda x: x[1])],
                zero_division=0), encoding="utf-8")

    (output_dir / "classification_report.txt").write_text(
        classification_report(labels, oof_predictions,
            target_names=[l for l, _ in sorted(label_mapping.items(), key=lambda x: x[1])],
            zero_division=0), encoding="utf-8")

    summary = {
        "experiment": "ND-13: TCN Temporal Encoder",
        "tcn_config": asdict(tcn_config),
        "feature_input_dim": input_dim,
        "fold_metrics_mean": mdf.mean(numeric_only=True).to_dict(),
        "overall_oof_metrics": overall,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    print(f"\nND-13 OOF: MF1={overall['macro_f1']:.4f} Acc={overall['accuracy']:.4f}")
    return {"feature_input_dim": input_dim, "label_mapping": label_mapping,
            "fold_metrics": metrics, "overall_oof_metrics": overall}


def _train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    for av, mask, dass, lbl in loader:
        av, mask, dass, lbl = av.to(device), mask.to(device), dass.to(device), lbl.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(av, mask, dass), lbl)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()


def _evaluate(model, loader, criterion, device, num_classes):
    model.eval()
    all_labels, all_probs, losses = [], [], []
    with torch.no_grad():
        for av, mask, dass, lbl in loader:
            av, mask, dass, lbl = av.to(device), mask.to(device), dass.to(device), lbl.to(device)
            logits = model(av, mask, dass)
            loss = criterion(logits, lbl)
            probs = torch.softmax(logits, -1).cpu().numpy()
            losses.append(float(loss.item()))
            all_labels.append(lbl.cpu().numpy())
            all_probs.append(probs)
    labels = np.concatenate(all_labels)
    probs = np.concatenate(all_probs) if all_probs else np.zeros((0, num_classes), dtype=np.float32)
    preds = probs.argmax(axis=1) if len(probs) else np.zeros(0, dtype=np.int64)
    m = _classification_metrics(labels, preds)
    m["loss"] = float(np.mean(losses)) if losses else 0.0
    return m, probs
