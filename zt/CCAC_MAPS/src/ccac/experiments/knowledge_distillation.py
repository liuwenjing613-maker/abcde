"""
ND-11: Knowledge distillation from DASS teacher to No-DASS student.

Problem: DASS features give MF1 0.36+ OOF but test set has no DASS → collapse to 0.09.
Solution: Train DASS teacher → use soft labels to train AV-only student via KL divergence.

Loss = alpha * KL(teacher_probs || student_probs) + (1-alpha) * FocalLoss(student, labels)

The teacher's OOF predictions (from 5-fold CV) are used as soft targets.
Each subject's teacher probability is from the fold where it was held out,
so the teacher hasn't seen that subject during training → unbiased soft labels.
"""

from __future__ import annotations

import copy, json, math, os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, Dataset

from ccac.baselines.anxiety_baseline import (
    STAGES, CLIP_TYPES, BaselineConfig,
    _set_seed, _resolve_device, _resolve_num_workers,
    _encode_labels, _build_folds, _fit_scaler, _apply_scaler,
    _class_weights, _classification_metrics, _extended_metrics,
    _is_release_dataset, _load_release_train_val,
)
from ccac.baselines.dass_baseline import (
    DASSConfig, FocalLoss, _extract_dass_features,
    _calibrate_thresholds, _apply_thresholds,
)
from ccac.experiments.deep_residual import DeepResidualModel


# ---------------------------------------------------------------------------
# Indexed dataset — returns original index for teacher-prob lookup
# ---------------------------------------------------------------------------

class IndexedDASSDataset(Dataset):
    """Like DASSDataset but also returns the original subject index."""

    def __init__(self, av_features, clip_mask, dass_features, labels, indices):
        self.av_features = torch.from_numpy(av_features).float()
        self.clip_mask = torch.from_numpy(clip_mask).bool()
        self.dass_features = torch.from_numpy(dass_features).float()
        self.labels = torch.from_numpy(labels).long()
        self.indices = torch.from_numpy(indices).long()

    def __len__(self):
        return int(self.labels.shape[0])

    def __getitem__(self, idx):
        return (self.av_features[idx], self.clip_mask[idx],
                self.dass_features[idx], self.labels[idx],
                self.indices[idx])


# ---------------------------------------------------------------------------
# Distillation config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DistillationConfig:
    teacher_dass_scheme: str = "scores_das"
    teacher_focal_gamma: float = 1.0
    alpha: float = 0.9
    temperature: float = 3.0
    student_focal_gamma: float = 2.0
    num_heads: int = 4
    num_residual_blocks: int = 3
    calibrate_thresholds: bool = True


# ---------------------------------------------------------------------------
# Distillation loss
# ---------------------------------------------------------------------------

class DistillationLoss(nn.Module):
    """Combined KL distillation + Focal hard-label loss."""

    def __init__(self, alpha=0.9, temperature=3.0, focal_gamma=2.0, class_weights=None):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.focal = FocalLoss(gamma=focal_gamma, alpha=class_weights)

    def forward(self, student_logits, teacher_probs, hard_labels):
        # Soften both distributions with temperature
        student_log_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
        with torch.no_grad():
            teacher_soft = teacher_probs ** (1.0 / self.temperature)
            teacher_soft = teacher_soft / teacher_soft.sum(dim=-1, keepdim=True)

        kl = F.kl_div(student_log_soft, teacher_soft, reduction="batchmean",
                      log_target=False) * (self.temperature ** 2)
        hard = self.focal(student_logits, hard_labels)
        return self.alpha * kl + (1.0 - self.alpha) * hard


# ---------------------------------------------------------------------------
# Main training
# ---------------------------------------------------------------------------

def train_distillation(
    baseline_config: BaselineConfig,
    distill_config: DistillationConfig | None = None,
    teacher_oof_path: str | None = None,
) -> dict[str, Any]:
    if distill_config is None:
        distill_config = DistillationConfig()

    _set_seed(baseline_config.seed)
    device = _resolve_device(baseline_config.device)
    output_dir = Path(baseline_config.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = Path(baseline_config.dataset_path)

    # --- Load data ---
    if _is_release_dataset(dataset_path):
        frame, av_features, clip_mask, label_mapping, input_dim = \
            _load_release_train_val(baseline_config, dataset_path)
        labels = frame["_label_index"].to_numpy(dtype=np.int64)
    else:
        raise RuntimeError("Only release dataset supported")

    num_classes = len(label_mapping)

    # --- Load teacher OOF ---
    if teacher_oof_path is None:
        teacher_oof_path = os.environ.get("TEACHER_OOF_PATH", "")
    teacher_path = Path(teacher_oof_path)
    if not teacher_path.exists():
        # Try default locations
        for candidate in [
            dataset_path.parent / "artifacts" / "dass" / "focal_g1" / "oof_predictions.csv",
            dataset_path.parent / "artifacts" / "exp" / "deep_residual" / "oof_predictions.csv",
        ]:
            if candidate.exists():
                teacher_path = candidate
                break

    if not teacher_path.exists():
        raise FileNotFoundError(
            f"Teacher OOF not found at {teacher_path}. "
            "Train teacher first or provide --teacher-oof path."
        )

    print(f"Teacher OOF: {teacher_path}")
    teacher_df = pd.read_csv(teacher_path)
    prob_cols = [c for c in teacher_df.columns if c.startswith("prob_class_")]
    teacher_probs_all = teacher_df[prob_cols].to_numpy(dtype=np.float32)
    print(f"Teacher OOF shape: {teacher_probs_all.shape}, classes: {len(prob_cols)}")

    # --- No DASS for student ---
    dass_features = np.zeros((len(frame), 0), dtype=np.float32)

    # --- Cross-validation ---
    fold_indices = _build_folds(labels, baseline_config.num_folds, baseline_config.seed)
    subject_indices = np.arange(len(frame), dtype=np.int64)

    oof_probabilities = np.zeros((len(frame), num_classes), dtype=np.float32)
    oof_predictions = np.full(len(frame), -1, dtype=np.int64)
    metrics: list[dict[str, Any]] = []
    fold_states: list[dict[str, Any]] = []

    for fold_id, (train_idx, val_idx) in enumerate(fold_indices, start=1):
        fold_dir = output_dir / f"fold_{fold_id}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        workers = _resolve_num_workers(baseline_config.num_workers)

        # Scale
        scaler = _fit_scaler(av_features[train_idx], clip_mask[train_idx])
        train_av = _apply_scaler(av_features[train_idx], scaler)
        val_av = _apply_scaler(av_features[val_idx], scaler)

        train_ds = IndexedDASSDataset(
            train_av, clip_mask[train_idx], dass_features[train_idx],
            labels[train_idx], subject_indices[train_idx],
        )
        val_ds = IndexedDASSDataset(
            val_av, clip_mask[val_idx], dass_features[val_idx],
            labels[val_idx], subject_indices[val_idx],
        )

        train_loader = DataLoader(train_ds, batch_size=baseline_config.batch_size,
                                  shuffle=True, num_workers=workers)
        val_loader = DataLoader(val_ds, batch_size=baseline_config.batch_size,
                               shuffle=False, num_workers=workers)

        # Model
        model = DeepResidualModel(
            input_dim=input_dim, num_classes=num_classes,
            hidden_dim=baseline_config.hidden_dim,
            num_heads=distill_config.num_heads,
            num_residual_blocks=distill_config.num_residual_blocks,
            dropout=baseline_config.dropout,
            dass_config=DASSConfig(dass_scheme="none"),
        ).to(device)

        cw = _class_weights(labels[train_idx], num_classes, baseline_config.class_weight_power)
        if cw is not None:
            cw = cw.to(device)

        criterion = DistillationLoss(
            alpha=distill_config.alpha,
            temperature=distill_config.temperature,
            focal_gamma=distill_config.student_focal_gamma,
            class_weights=cw,
        )

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
            _train_epoch_distill(
                model, train_loader, optimizer, criterion, device,
                teacher_probs_all,
            )
            scheduler.step()
            val_m, val_probs = _eval_distill(
                model, val_loader, device, num_classes,
                teacher_probs_all, criterion,
            )
            selection_metric = "robust_score"
            selection_score = float(val_m[selection_metric])
            if selection_score > best_metric:
                best_metric = selection_score
                best_epoch = epoch
                epochs_no_improve = 0
                best_state = {
                    "model": copy.deepcopy(model.state_dict()),
                    "scaler_mean": scaler[0].tolist(),
                    "scaler_std": scaler[1].tolist(),
                    "epoch": epoch, "metrics": val_m,
                    "selection_metric": selection_metric,
                    "selection_score": selection_score,
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
        val_m, val_probs = _eval_distill(
            model, val_loader, device, num_classes,
            teacher_probs_all, criterion,
        )
        oof_probabilities[val_idx] = val_probs
        oof_predictions[val_idx] = val_probs.argmax(axis=1)
        fold_metric = {
            "fold": fold_id,
            "best_epoch": best_epoch,
            "selection_metric": best_state.get("selection_metric", "robust_score"),
            "selection_score": best_metric,
            **val_m,
        }
        metrics.append(fold_metric)
        fold_states.append(best_state)
        (fold_dir / "metrics.json").write_text(
            json.dumps(fold_metric, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  Fold {fold_id}: MF1={val_m['macro_f1']:.4f} Acc={val_m['accuracy']:.4f}")

    # --- OOF ---
    label_by_idx = {i: l for l, i in label_mapping.items()}
    overall = _extended_metrics(labels, oof_predictions, oof_probabilities, num_classes)
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
    (output_dir / "label_mapping.json").write_text(
        json.dumps(label_mapping, ensure_ascii=False, indent=2))
    (output_dir / "baseline_config.json").write_text(
        json.dumps(asdict(baseline_config), ensure_ascii=False, indent=2))
    (output_dir / "distill_config.json").write_text(
        json.dumps(asdict(distill_config), ensure_ascii=False, indent=2))

    if distill_config.calibrate_thresholds:
        cal_thr, cal_met = _calibrate_thresholds(oof_probabilities, labels, num_classes)
        (output_dir / "calibrated_thresholds.json").write_text(
            json.dumps({"thresholds": cal_thr.tolist(), "metrics": cal_met},
                       ensure_ascii=False, indent=2))
        oof_cal = _apply_thresholds(oof_probabilities, cal_thr)
        overall = _extended_metrics(labels, oof_cal, oof_probabilities, num_classes)
        (output_dir / "classification_report_calibrated.txt").write_text(
            classification_report(labels, oof_cal,
                target_names=[l for l, _ in sorted(label_mapping.items(), key=lambda x: x[1])],
                zero_division=0), encoding="utf-8")

    (output_dir / "classification_report.txt").write_text(
        classification_report(labels, oof_predictions,
            target_names=[l for l, _ in sorted(label_mapping.items(), key=lambda x: x[1])],
            zero_division=0), encoding="utf-8")

    summary = {
        "experiment": "ND-11: Knowledge Distillation",
        "teacher_oof": str(teacher_path),
        "distill_config": asdict(distill_config),
        "feature_input_dim": input_dim,
        "fold_metrics_mean": mdf.mean(numeric_only=True).to_dict(),
        "overall_oof_metrics": overall,
        "calibrated_macro_f1": float(overall["macro_f1"]),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2))

    print(f"\nND-11 OOF: MF1={overall['macro_f1']:.4f} Acc={overall['accuracy']:.4f}")
    return {"feature_input_dim": input_dim, "label_mapping": label_mapping,
            "fold_metrics": metrics, "overall_oof_metrics": overall}


# ---------------------------------------------------------------------------
# Training / eval helpers
# ---------------------------------------------------------------------------

def _train_epoch_distill(model, loader, optimizer, criterion, device, teacher_all):
    """Train one epoch. teacher_all is (N, C) numpy array indexed by original subject idx."""
    model.train()
    for batch_av, batch_mask, batch_dass, batch_labels, batch_idx in loader:
        batch_av = batch_av.to(device)
        batch_mask = batch_mask.to(device)
        batch_labels = batch_labels.to(device)
        batch_idx = batch_idx.cpu().numpy()

        # Look up teacher probs for these subjects
        teacher_batch = torch.from_numpy(teacher_all[batch_idx]).float().to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_av, batch_mask, batch_dass.to(device))
        loss = criterion(logits, teacher_batch, batch_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()


def _eval_distill(model, loader, device, num_classes, teacher_all=None, criterion=None):
    model.eval()
    all_labels, all_probs, losses = [], [], []
    with torch.no_grad():
        for batch_av, batch_mask, batch_dass, batch_labels, batch_idx in loader:
            batch_av = batch_av.to(device)
            batch_mask = batch_mask.to(device)
            batch_labels = batch_labels.to(device)
            logits = model(batch_av, batch_mask, batch_dass.to(device))
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_labels.append(batch_labels.cpu().numpy())
            all_probs.append(probs)

            if criterion is not None and teacher_all is not None:
                batch_idx_np = batch_idx.cpu().numpy()
                teacher_batch = torch.from_numpy(teacher_all[batch_idx_np]).float().to(device)
                loss = criterion(logits, teacher_batch, batch_labels)
                losses.append(float(loss.item()))

    labels = np.concatenate(all_labels)
    probs = np.concatenate(all_probs) if all_probs else np.zeros((0, num_classes), dtype=np.float32)
    preds = probs.argmax(axis=1) if len(probs) else np.zeros(0, dtype=np.int64)
    m = _extended_metrics(labels, preds, probs, num_classes)
    m["loss"] = float(np.mean(losses)) if losses else 0.0
    return m, probs
