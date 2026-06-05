#!/usr/bin/env python3
"""ND-15: Knowledge Distillation with Transformer student architecture.

Same as ND-11 but uses TransformerTemporalModel as the student instead of
DeepResidualModel. The Transformer's self-attention may capture different
temporal patterns that complement the distillation signal.

Usage:
    PYTHONPATH=src python scripts/exp_nd15_distill_transformer.py \
        --dataset-path datasets \
        --output-dir artifacts/exp/nd15_distill_transformer \
        --teacher-oof artifacts/dass/focal_g1/oof_predictions.csv \
        --device cuda
"""

from __future__ import annotations

import argparse, copy, json, math, sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, f1_score, accuracy_score
from torch.utils.data import DataLoader, Dataset

from ccac.baselines.anxiety_baseline import (
    BaselineConfig,
    _set_seed, _resolve_device, _resolve_num_workers,
    _build_folds, _fit_scaler, _apply_scaler,
    _class_weights, _classification_metrics,
    _is_release_dataset, _load_release_train_val,
)
from ccac.baselines.dass_baseline import (
    DASSConfig, FocalLoss, _calibrate_thresholds, _apply_thresholds,
)
from ccac.experiments.transformer_temporal import TransformerTemporalModel


# ---------------------------------------------------------------------------
# Indexed dataset
# ---------------------------------------------------------------------------

class IndexedDataset(Dataset):
    def __init__(self, av, mask, dass, labels, indices):
        self.av = torch.from_numpy(av).float()
        self.mask = torch.from_numpy(mask).bool()
        self.dass = torch.from_numpy(dass).float()
        self.labels = torch.from_numpy(labels).long()
        self.indices = torch.from_numpy(indices).long()

    def __len__(self):
        return int(self.labels.shape[0])

    def __getitem__(self, idx):
        return (self.av[idx], self.mask[idx], self.dass[idx],
                self.labels[idx], self.indices[idx])


# ---------------------------------------------------------------------------
# Distillation loss
# ---------------------------------------------------------------------------

class DistillLoss(nn.Module):
    def __init__(self, alpha=0.9, T=3.0, focal_gamma=2.0, class_weights=None):
        super().__init__()
        self.alpha = alpha
        self.T = T
        self.focal = FocalLoss(gamma=focal_gamma, alpha=class_weights)

    def forward(self, student_logits, teacher_probs, hard_labels):
        student_log_soft = F.log_softmax(student_logits / self.T, dim=-1)
        with torch.no_grad():
            teacher_soft = teacher_probs ** (1.0 / self.T)
            teacher_soft = teacher_soft / teacher_soft.sum(dim=-1, keepdim=True)
        kl = F.kl_div(student_log_soft, teacher_soft, reduction="batchmean",
                      log_target=False) * (self.T ** 2)
        hard = self.focal(student_logits, hard_labels)
        return self.alpha * kl + (1.0 - self.alpha) * hard


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="ND-15: Transformer student distillation")
    p.add_argument("--dataset-path", default="datasets")
    p.add_argument("--output-dir", default="artifacts/exp/nd15_distill_transformer")
    p.add_argument("--teacher-oof", required=True)
    p.add_argument("--audio-feature-name", default="audio_wavlm_base")
    p.add_argument("--video-feature-name", default="video_clip_base")
    p.add_argument("--device", default="cuda")
    p.add_argument("--alpha", type=float, default=0.9)
    p.add_argument("--temperature", type=float, default=3.0)
    p.add_argument("--student-focal-gamma", type=float, default=2.0)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--transformer-dim", type=int, default=256)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--num-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-calibrate", action="store_true")
    args = p.parse_args()

    _set_seed(args.seed)
    device = _resolve_device(args.device)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = Path(args.dataset_path)

    bc = BaselineConfig(
        dataset_path=str(dataset_path), output_dir=str(output_dir),
        audio_feature_name=args.audio_feature_name,
        video_feature_name=args.video_feature_name,
        hidden_dim=args.hidden_dim, dropout=args.dropout,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size, epochs=args.epochs,
        patience=args.patience, num_folds=args.num_folds,
        seed=args.seed, device=args.device,
    )

    # Load data
    if _is_release_dataset(dataset_path):
        frame, av_features, clip_mask, label_mapping, input_dim = \
            _load_release_train_val(bc, dataset_path)
        labels = frame["_label_index"].to_numpy(dtype=np.int64)
    else:
        print("Only release dataset supported")
        sys.exit(1)

    num_classes = len(label_mapping)
    print(f"Subjects: {len(frame)}, Input dim: {input_dim}, Classes: {num_classes}")

    # Load teacher OOF
    teacher_df = pd.read_csv(args.teacher_oof)
    prob_cols = [c for c in teacher_df.columns if c.startswith("prob_class_")]
    teacher_all = teacher_df[prob_cols].to_numpy(dtype=np.float32)
    print(f"Teacher OOF: {teacher_all.shape}")

    # No DASS for student
    dass_features = np.zeros((len(frame), 0), dtype=np.float32)
    subject_indices = np.arange(len(frame), dtype=np.int64)

    # CV
    fold_indices = _build_folds(labels, args.num_folds, args.seed)
    oof_probs = np.zeros((len(frame), num_classes), dtype=np.float32)
    oof_preds = np.full(len(frame), -1, dtype=np.int64)
    metrics = []

    for fold_id, (train_idx, val_idx) in enumerate(fold_indices, start=1):
        fold_dir = output_dir / f"fold_{fold_id}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        workers = _resolve_num_workers(0)

        scaler = _fit_scaler(av_features[train_idx], clip_mask[train_idx])
        train_av = _apply_scaler(av_features[train_idx], scaler)
        val_av = _apply_scaler(av_features[val_idx], scaler)

        train_ds = IndexedDataset(train_av, clip_mask[train_idx], dass_features[train_idx],
                                  labels[train_idx], subject_indices[train_idx])
        val_ds = IndexedDataset(val_av, clip_mask[val_idx], dass_features[val_idx],
                                labels[val_idx], subject_indices[val_idx])

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=workers)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=workers)

        # Transformer student
        model = TransformerTemporalModel(
            input_dim=input_dim, num_classes=num_classes,
            hidden_dim=args.hidden_dim,
            transformer_dim=args.transformer_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            dropout=args.dropout,
            dass_config=DASSConfig(dass_scheme="none"),
        ).to(device)

        cw = _class_weights(labels[train_idx], num_classes, 1.0)
        if cw is not None:
            cw = cw.to(device)

        criterion = DistillLoss(
            alpha=args.alpha, T=args.temperature,
            focal_gamma=args.student_focal_gamma, class_weights=cw,
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6)

        best_mf1 = -math.inf
        best_state = None
        best_probs = None
        patience_ct = 0

        for epoch in range(1, args.epochs + 1):
            model.train()
            for av, mask, dass, lbl, idx in train_loader:
                av, mask, lbl = av.to(device), mask.to(device), lbl.to(device)
                idx_np = idx.cpu().numpy()
                teacher_batch = torch.from_numpy(teacher_all[idx_np]).float().to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(av, mask, dass.to(device))
                loss = criterion(logits, teacher_batch, lbl)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

            model.eval()
            vprobs_list, vlbls_list = [], []
            with torch.no_grad():
                for av, mask, dass, lbl, idx in val_loader:
                    av, mask, lbl = av.to(device), mask.to(device), lbl.to(device)
                    logits = model(av, mask, dass.to(device))
                    vprobs_list.append(torch.softmax(logits, -1).cpu().numpy())
                    vlbls_list.append(lbl.cpu().numpy())
            vprob = np.concatenate(vprobs_list)
            vlbl = np.concatenate(vlbls_list)
            mf1 = float(f1_score(vlbl, vprob.argmax(1),
                                    average="macro", zero_division=0))

            if mf1 > best_mf1:
                best_mf1 = mf1
                patience_ct = 0
                best_state = {
                    "model": copy.deepcopy(model.state_dict()),
                    "scaler_mean": scaler[0].tolist(),
                    "scaler_std": scaler[1].tolist(),
                    "epoch": epoch,
                }
                best_probs = vprob
            else:
                patience_ct += 1
            if patience_ct >= args.patience:
                break

        model.load_state_dict(best_state["model"])
        oof_probs[val_idx] = best_probs
        oof_preds[val_idx] = best_probs.argmax(axis=1)
        fold_mf1 = float(f1_score(
            labels[val_idx], oof_preds[val_idx], average="macro", zero_division=0))
        fold_acc = float(accuracy_score(labels[val_idx], oof_preds[val_idx]))
        print(f"  Fold {fold_id}: MF1={fold_mf1:.4f} Acc={fold_acc:.4f} (best_epoch={best_state['epoch']})")
        metrics.append({"fold": fold_id, "best_epoch": best_state["epoch"],
                        "macro_f1": fold_mf1, "accuracy": fold_acc})
        torch.save(best_state, fold_dir / "best_model.pt")

    # OOF
    overall_mf1 = float(f1_score(
        labels, oof_preds, average="macro", zero_division=0))
    overall_acc = float(accuracy_score(labels, oof_preds))
    print(f"\nND-15 OOF: MF1={overall_mf1:.4f} Acc={overall_acc:.4f}")

    label_by_idx = {i: l for l, i in label_mapping.items()}
    print(classification_report(labels, oof_preds,
          target_names=[l for l, _ in sorted(label_mapping.items(), key=lambda x: x[1])],
          zero_division=0))

    # Save
    oof_df = pd.DataFrame({
        "subject_id": frame["subject_id"].astype(str),
        "true_label": frame["t4_anxiety_level"].astype(str),
        "pred_label": [label_by_idx[int(i)] for i in oof_preds],
    })
    for ci in range(num_classes):
        oof_df[f"prob_class_{ci}"] = oof_probs[:, ci]
    oof_df.to_csv(output_dir / "oof_predictions.csv", index=False)
    pd.DataFrame(metrics).to_csv(output_dir / "fold_metrics.csv", index=False)
    (output_dir / "label_mapping.json").write_text(json.dumps(label_mapping, ensure_ascii=False, indent=2))

    # Calibrate
    calibrate = not args.no_calibrate
    if calibrate:
        cal_thr, cal_met = _calibrate_thresholds(oof_probs, labels, num_classes)
        (output_dir / "calibrated_thresholds.json").write_text(
            json.dumps({"thresholds": cal_thr.tolist(), "metrics": cal_met}, ensure_ascii=False, indent=2))
        oof_cal = _apply_thresholds(oof_probs, cal_thr)
        overall_mf1 = float(f1_score(
            labels, oof_cal, average="macro", zero_division=0))
        (output_dir / "classification_report_calibrated.txt").write_text(
            classification_report(labels, oof_cal,
                target_names=[l for l, _ in sorted(label_mapping.items(), key=lambda x: x[1])],
                zero_division=0), encoding="utf-8")

    (output_dir / "classification_report.txt").write_text(
        classification_report(labels, oof_preds,
            target_names=[l for l, _ in sorted(label_mapping.items(), key=lambda x: x[1])],
            zero_division=0), encoding="utf-8")

    summary = {
        "experiment": "ND-15: Transformer student distillation",
        "teacher_oof": args.teacher_oof,
        "alpha": args.alpha, "temperature": args.temperature,
        "input_dim": input_dim,
        "overall_oof_mf1": overall_mf1,
        "overall_oof_acc": overall_acc,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nFinal MF1: {overall_mf1:.4f}")
    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()
