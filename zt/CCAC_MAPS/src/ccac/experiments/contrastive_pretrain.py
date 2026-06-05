"""
Trial 7: Contrastive pretraining on stage representations.

Approach:
1. Use the DeepResidualModel encoder (clip→stage→cross-attn→fused)
2. Add a contrastive projection head for each stage representation
3. InfoNCE loss: T1/T2/T3 from the same subject are mutual positives
4. Pretrain → finetune with classification loss
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
from sklearn.metrics import classification_report, f1_score
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
# Contrastive Model
# ---------------------------------------------------------------------------

class ContrastiveEncoder(nn.Module):
    """Stage encoder with contrastive projection head.

    Shared backbone: clip encoder + attention pooling + cross-stage attention
    Contrastive head: projects each stage to embedding space for InfoNCE
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 4,
        proj_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        # Shared backbone
        self.clip_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.clip_attention = nn.Linear(hidden_dim, 1)
        self.stage_position = nn.Parameter(torch.zeros(3, hidden_dim))
        nn.init.normal_(self.stage_position, mean=0.0, std=0.02)

        # Cross-stage attention (simplified for contrastive pretraining)
        self.cross_stage_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(hidden_dim)

        # Contrastive projection head
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, proj_dim),
        )

    def get_stage_repr(self, av_inputs, clip_mask):
        """Get stage representations (B, 3, hidden_dim)."""
        encoded = self.clip_encoder(av_inputs)
        # Attention pooling
        logits = self.clip_attention(encoded).squeeze(-1)
        logits = logits.masked_fill(~clip_mask, -1e9)
        weights = torch.softmax(logits, dim=-1)
        weights = weights * clip_mask.float()
        denom = weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        weights = weights / denom
        stage_repr = (encoded * weights.unsqueeze(-1)).sum(dim=2)
        missing_stage = clip_mask.sum(dim=-1, keepdim=True) == 0
        stage_repr = torch.where(missing_stage, torch.zeros_like(stage_repr), stage_repr)
        stage_repr = stage_repr + self.stage_position.unsqueeze(0)

        # Cross-stage attention (simplified)
        stage_repr_normed = self.attn_norm(stage_repr)
        attn_out, _ = self.cross_stage_attn(stage_repr_normed, stage_repr_normed, stage_repr_normed)
        stage_repr = stage_repr + F.dropout(attn_out, p=0.1, training=self.training)

        return stage_repr  # (B, 3, hidden_dim)

    def forward(self, av_inputs, clip_mask):
        """Get projected stage embeddings for contrastive loss."""
        stage_repr = self.get_stage_repr(av_inputs, clip_mask)  # (B, 3, H)
        proj = self.projection(stage_repr)  # (B, 3, proj_dim)
        return F.normalize(proj, dim=-1)


def contrastive_loss(embeddings: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """InfoNCE loss on stage embeddings.

    embeddings: (B, 3, D) — normalized
    Each subject's 3 stages are mutual positives.
    Other subjects' stages are negatives.
    """
    B, T, D = embeddings.shape
    # All embeddings: (B*T, D)
    all_emb = embeddings.reshape(B * T, D)
    # Similarity matrix: (B*T, B*T)
    sim = all_emb @ all_emb.T / temperature

    # Positive mask: same subject, different stage
    pos_mask = torch.zeros(B * T, B * T, device=embeddings.device)
    for t1 in range(T):
        for t2 in range(T):
            if t1 != t2:
                # Stage t1 of subject i → Stage t2 of subject i
                for b in range(B):
                    pos_mask[b * T + t1, b * T + t2] = 1.0

    # Self-contrast mask (ignore diagonal)
    self_mask = torch.eye(B * T, device=embeddings.device)

    # Numerator: sum over positives
    pos_sim = (sim * pos_mask).sum(dim=-1)  # (B*T,)
    pos_count = pos_mask.sum(dim=-1)  # (B*T,)

    # Denominator: sum over all except self
    neg_mask = 1.0 - self_mask
    neg_sim = (torch.exp(sim) * neg_mask).sum(dim=-1)  # (B*T,)

    # Loss per embedding
    loss_per_sample = -pos_sim / pos_count.clamp_min(1) + torch.log(neg_sim.clamp_min(1e-8))
    loss = loss_per_sample.mean()

    return loss


# ---------------------------------------------------------------------------
# Full model: Contrastive encoder + classifier
# ---------------------------------------------------------------------------

class ContrastivePretrainedModel(nn.Module):
    """Classification model built on contrastively pretrained encoder."""

    def __init__(self, encoder: ContrastiveEncoder, num_classes: int,
                 hidden_dim: int = 256, dropout: float = 0.2,
                 dass_config: DASSConfig | None = None):
        super().__init__()
        self.dass_config = dass_config or DASSConfig(dass_scheme="none")
        self.encoder = encoder

        # DASS encoder
        dass_dim = self._dass_input_dim()
        self._has_dass = dass_dim > 0
        if self._has_dass and self.dass_config.dass_scheme == "encoder":
            self.dass_encoder = nn.Sequential(
                nn.Linear(dass_dim, self.dass_config.dass_hidden),
                nn.LayerNorm(self.dass_config.dass_hidden), nn.GELU(),
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

        # Stage diff features
        self.diff_proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout),
        )

        fusion_dim = hidden_dim * 2 + hidden_dim + dass_out
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def _dass_input_dim(self) -> int:
        scheme = self.dass_config.dass_scheme
        return {"none": 0, "scores_a": 3, "scores_das": 9, "encoder": 18}.get(scheme, 0)

    def forward(self, av_inputs, clip_mask, dass_features=None):
        stage_repr = self.encoder.get_stage_repr(av_inputs, clip_mask)
        pooled = stage_repr.mean(dim=1)
        final = stage_repr[:, -1, :]

        # Stage diffs
        d21 = stage_repr[:, 1, :] - stage_repr[:, 0, :]
        d32 = stage_repr[:, 2, :] - stage_repr[:, 1, :]
        d31 = stage_repr[:, 2, :] - stage_repr[:, 0, :]
        diffs = self.diff_proj(torch.cat([d21, d32, d31], dim=-1))

        fused = torch.cat([pooled, final, diffs], dim=-1)
        if self._has_dass and dass_features is not None:
            dass_repr = self.dass_encoder(dass_features)
            fused = torch.cat([fused, dass_repr], dim=-1)

        return self.classifier(fused)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ContrastiveConfig:
    dass_scheme: str = "scores_das"
    focal_gamma: float = 1.0
    contrastive_epochs: int = 20
    contrastive_lr: float = 1e-3
    contrastive_temperature: float = 0.07
    proj_dim: int = 128
    calibrate_thresholds: bool = True


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_contrastive(
    baseline_config: BaselineConfig,
    contrastive_config: ContrastiveConfig | None = None,
) -> dict[str, Any]:
    if contrastive_config is None:
        contrastive_config = ContrastiveConfig()

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
        from ccac.baselines.anxiety_baseline import BaselineFeatureBuilder
        builder = BaselineFeatureBuilder(
            baseline_config.audio_feature_name, baseline_config.video_feature_name
        ).fit(frame)
        av_features, clip_mask = builder.transform(frame)
        input_dim = builder.input_dim

    dass_features = _extract_dass_features(frame, contrastive_config.dass_scheme)

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

        # ---- Phase 1: Contrastive pretraining ----
        print(f"Fold {fold_id} - Phase 1: Contrastive pretraining ({contrastive_config.contrastive_epochs} epochs)")
        encoder = ContrastiveEncoder(
            input_dim=input_dim, hidden_dim=baseline_config.hidden_dim,
            num_heads=4, proj_dim=contrastive_config.proj_dim,
            dropout=baseline_config.dropout,
        ).to(device)

        opt_contrastive = torch.optim.AdamW(
            encoder.parameters(),
            lr=contrastive_config.contrastive_lr,
            weight_decay=baseline_config.weight_decay,
        )

        for epoch in range(1, contrastive_config.contrastive_epochs + 1):
            encoder.train()
            total_loss = 0.0
            for batch in train_loader:
                batch_av, batch_mask = batch[0].to(device), batch[1].to(device)
                opt_contrastive.zero_grad(set_to_none=True)
                embeddings = encoder(batch_av, batch_mask)
                loss = contrastive_loss(embeddings, contrastive_config.contrastive_temperature)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
                opt_contrastive.step()
                total_loss += loss.item()
            if epoch % 5 == 0:
                print(f"  Contrastive epoch {epoch}/{contrastive_config.contrastive_epochs}, loss={total_loss/len(train_loader):.4f}")

        # ---- Phase 2: Classification finetuning ----
        print(f"Fold {fold_id} - Phase 2: Classification finetuning")
        model = ContrastivePretrainedModel(
            encoder=encoder, num_classes=len(label_mapping),
            hidden_dim=baseline_config.hidden_dim, dropout=baseline_config.dropout,
            dass_config=DASSConfig(dass_scheme=contrastive_config.dass_scheme),
        ).to(device)

        class_weights = _class_weights(labels[train_idx], len(label_mapping), baseline_config.class_weight_power)
        if class_weights is not None:
            class_weights = class_weights.to(device)

        criterion = FocalLoss(gamma=contrastive_config.focal_gamma, alpha=class_weights,
                              label_smoothing=baseline_config.label_smoothing)
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

    if contrastive_config.calibrate_thresholds:
        cal_thr, cal_met = _calibrate_thresholds(oof_probabilities, labels, len(label_mapping))
        (output_dir / "calibrated_thresholds.json").write_text(
            json.dumps({"thresholds": cal_thr.tolist(), "metrics": cal_met}, ensure_ascii=False, indent=2), encoding="utf-8")
        oof_preds_cal = _apply_thresholds(oof_probabilities, cal_thr)
        overall_metrics = _classification_metrics(labels, oof_preds_cal)
        (output_dir / "classification_report_calibrated.txt").write_text(
            classification_report(labels, oof_preds_cal,
                target_names=[l for l, _ in sorted(label_mapping.items(), key=lambda x: x[1])], zero_division=0),
            encoding="utf-8")

    (output_dir / "classification_report.txt").write_text(
        classification_report(labels, oof_predictions,
            target_names=[l for l, _ in sorted(label_mapping.items(), key=lambda x: x[1])], zero_division=0),
        encoding="utf-8")

    summary = {
        "feature_input_dim": input_dim,
        "fold_metrics_mean": metrics_df.mean(numeric_only=True).to_dict(),
        "overall_oof_metrics": overall_metrics,
        "calibrated_macro_f1": float(overall_metrics["macro_f1"]),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return {"fold_metrics": metrics, "overall_oof_metrics": overall_metrics}


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
