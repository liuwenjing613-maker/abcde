#!/usr/bin/env python3
"""ND-16: Multi-teacher knowledge distillation.

Uses MULTIPLE DASS teachers (different architectures) to provide diverse soft
targets for the no-DASS student. The student learns from the consensus.

Teachers:
  - Teacher 1: DeepResidualModel + scores_das + Focal γ=1.0 (OOF MF1 0.363)
  - Teacher 2: TransformerTemporalModel + scores_das + Focal γ=1.0 (OOF MF1 ~0.35)

The student averages the KL divergences against each teacher's soft labels.

Usage:
    # First ensure both teacher OOFs exist:
    # Teacher 1: artifacts/dass/focal_g1/oof_predictions.csv
    # Teacher 2: Train with: PYTHONPATH=src python scripts/exp_transformer.py \
    #   --dass-scheme scores_das --focal-gamma 1.0 --output-dir artifacts/exp/teacher_transformer

    PYTHONPATH=src python scripts/exp_nd16_multiteacher.py \
        --dataset-path datasets \
        --output-dir artifacts/exp/nd16_multiteacher \
        --teacher-oofs artifacts/dass/focal_g1/oof_predictions.csv \
        --device cuda
"""

from __future__ import annotations

import argparse, copy, json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, f1_score, accuracy_score

from ccac.baselines.anxiety_baseline import (
    BaselineConfig, _set_seed, _resolve_device, _resolve_num_workers,
    _build_folds, _fit_scaler, _apply_scaler,
    _class_weights, _classification_metrics,
    _is_release_dataset, _load_release_train_val, _extended_metrics,
)
from ccac.baselines.dass_baseline import (
    DASSConfig, FocalLoss, _calibrate_thresholds, _apply_thresholds,
)
from ccac.experiments.deep_residual import DeepResidualModel


class IndexedDataset(Dataset):
    def __init__(self, av, mask, dass, labels, indices):
        self.av = torch.from_numpy(av).float()
        self.mask = torch.from_numpy(mask).bool()
        self.dass = torch.from_numpy(dass).float()
        self.labels = torch.from_numpy(labels).long()
        self.indices = torch.from_numpy(indices).long()
    def __len__(self): return int(self.labels.shape[0])
    def __getitem__(self, idx): return (self.av[idx], self.mask[idx], self.dass[idx], self.labels[idx], self.indices[idx])


class MultiTeacherDistillLoss(nn.Module):
    """Distillation loss with multiple teachers.
    Loss = alpha * mean(KL(teacher_i || student)) * T^2 + (1-alpha) * Focal(student, labels)
    """
    def __init__(self, alpha=0.9, T=3.0, focal_gamma=2.0, class_weights=None):
        super().__init__()
        self.alpha = alpha
        self.T = T
        self.focal = FocalLoss(gamma=focal_gamma, alpha=class_weights)

    def forward(self, student_logits, teacher_probs_list, hard_labels):
        # Average KL across all teachers
        student_log_soft = F.log_softmax(student_logits / self.T, dim=-1)
        kl_total = 0.0
        for teacher_probs in teacher_probs_list:
            with torch.no_grad():
                teacher_soft = teacher_probs ** (1.0 / self.T)
                teacher_soft = teacher_soft / teacher_soft.sum(dim=-1, keepdim=True)
            kl_total += F.kl_div(student_log_soft, teacher_soft, reduction="batchmean", log_target=False)
        kl_loss = kl_total / len(teacher_probs_list) * (self.T ** 2)

        hard_loss = self.focal(student_logits, hard_labels)
        return self.alpha * kl_loss + (1.0 - self.alpha) * hard_loss


def main():
    p = argparse.ArgumentParser(description="ND-16: Multi-teacher distillation")
    p.add_argument("--dataset-path", default="datasets")
    p.add_argument("--output-dir", default="artifacts/exp/nd16_multiteacher")
    p.add_argument("--teacher-oofs", nargs="+", required=True,
                   help="Paths to teacher OOF CSV files (space-separated)")
    p.add_argument("--audio-feature-name", default="audio_wavlm_base")
    p.add_argument("--video-feature-name", default="video_clip_base")
    p.add_argument("--device", default="cuda")
    p.add_argument("--alpha", type=float, default=0.9)
    p.add_argument("--temperature", type=float, default=3.0)
    p.add_argument("--student-focal-gamma", type=float, default=2.0)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--num-residual-blocks", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=12)
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

    if _is_release_dataset(dataset_path):
        frame, av_features, clip_mask, label_mapping, input_dim = \
            _load_release_train_val(bc, dataset_path)
        labels = frame["_label_index"].to_numpy(dtype=np.int64)
    else:
        print("Only release dataset supported"); sys.exit(1)

    num_classes = len(label_mapping)
    print(f"Subjects: {len(frame)}, Input: {input_dim}")

    # Load all teacher OOFs
    teacher_all_list = []
    for i, tpath in enumerate(args.teacher_oofs):
        tdf = pd.read_csv(tpath)
        prob_cols = [c for c in tdf.columns if c.startswith("prob_class_")]
        tprobs = tdf[prob_cols].to_numpy(dtype=np.float32)
        print(f"Teacher {i+1} ({tpath}): {tprobs.shape}")
        teacher_all_list.append(tprobs)

    dass_features = np.zeros((len(frame), 0), dtype=np.float32)
    subject_indices = np.arange(len(frame), dtype=np.int64)

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

        model = DeepResidualModel(
            input_dim=input_dim, num_classes=num_classes,
            hidden_dim=args.hidden_dim, num_heads=args.num_heads,
            num_residual_blocks=args.num_residual_blocks,
            dropout=args.dropout, dass_config=DASSConfig(dass_scheme="none"),
        ).to(device)

        cw = _class_weights(labels[train_idx], num_classes, 1.0)
        if cw is not None: cw = cw.to(device)

        criterion = MultiTeacherDistillLoss(
            alpha=args.alpha, T=args.temperature,
            focal_gamma=args.student_focal_gamma, class_weights=cw,
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6)

        best_score, best_state, best_probs, patience_ct = -math.inf, None, None, 0

        for epoch in range(1, args.epochs + 1):
            model.train()
            for av, mask, dass, lbl, idx in train_loader:
                av, mask, lbl = av.to(device), mask.to(device), lbl.to(device)
                idx_np = idx.cpu().numpy()
                teacher_batches = [torch.from_numpy(t[idx_np]).float().to(device) for t in teacher_all_list]
                optimizer.zero_grad(set_to_none=True)
                logits = model(av, mask, dass.to(device))
                loss = criterion(logits, teacher_batches, lbl)
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
            vpred = vprob.argmax(1)
            val_metrics = _extended_metrics(vlbl, vpred, vprob, num_classes)
            selection_metric = "robust_score"
            selection_score = float(val_metrics[selection_metric])

            if selection_score > best_score:
                best_score = selection_score; patience_ct = 0
                best_state = {
                    "model": copy.deepcopy(model.state_dict()),
                    "scaler_mean": scaler[0].tolist(),
                    "scaler_std": scaler[1].tolist(),
                    "epoch": epoch,
                    "metrics": val_metrics,
                    "selection_metric": selection_metric,
                    "selection_score": selection_score,
                }
                best_probs = vprob
            else:
                patience_ct += 1
            if patience_ct >= args.patience: break

        model.load_state_dict(best_state["model"])
        oof_probs[val_idx] = best_probs
        oof_preds[val_idx] = best_probs.argmax(1)
        fold_metrics = _extended_metrics(labels[val_idx], oof_preds[val_idx], best_probs, num_classes)
        print(
            f"  Fold {fold_id}: MF1={fold_metrics['macro_f1']:.4f} "
            f"AUC={fold_metrics['macro_auc']:.4f} "
            f"Robust={fold_metrics['robust_score']:.4f} "
            f"Acc={fold_metrics['accuracy']:.4f}"
        )
        metrics.append({
            "fold": fold_id,
            "best_epoch": best_state.get("epoch"),
            "selection_metric": best_state.get("selection_metric", "robust_score"),
            "selection_score": best_state.get("selection_score"),
            **fold_metrics,
        })
        torch.save(best_state, fold_dir / "best_model.pt")

    overall = _extended_metrics(labels, oof_preds, oof_probs, num_classes)
    print(
        f"\nND-16 OOF: MF1={overall['macro_f1']:.4f} "
        f"AUC={overall['macro_auc']:.4f} "
        f"Robust={overall['robust_score']:.4f} "
        f"Acc={overall['accuracy']:.4f}"
    )

    label_by_idx = {i: l for l, i in label_mapping.items()}
    print(classification_report(labels, oof_preds,
          target_names=[l for l, _ in sorted(label_mapping.items(), key=lambda x: x[1])], zero_division=0))

    oof_df = pd.DataFrame({
        "subject_id": frame["subject_id"].astype(str),
        "true_label": frame["t4_anxiety_level"].astype(str),
        "pred_label": [label_by_idx[int(i)] for i in oof_preds],
    })
    for ci in range(num_classes): oof_df[f"prob_class_{ci}"] = oof_probs[:, ci]
    oof_df.to_csv(output_dir / "oof_predictions.csv", index=False)
    pd.DataFrame(metrics).to_csv(output_dir / "fold_metrics.csv", index=False)
    (output_dir / "label_mapping.json").write_text(json.dumps(label_mapping, ensure_ascii=False, indent=2))

    if not args.no_calibrate:
        cal_thr, cal_met = _calibrate_thresholds(oof_probs, labels, num_classes)
        oof_cal = _apply_thresholds(oof_probs, cal_thr)
        overall = _extended_metrics(labels, oof_cal, oof_probs, num_classes)
        (output_dir / "classification_report_calibrated.txt").write_text(
            classification_report(labels, oof_cal,
                target_names=[l for l, _ in sorted(label_mapping.items(), key=lambda x: x[1])],
                zero_division=0), encoding="utf-8")

    (output_dir / "classification_report.txt").write_text(
        classification_report(labels, oof_preds,
            target_names=[l for l, _ in sorted(label_mapping.items(), key=lambda x: x[1])],
            zero_division=0), encoding="utf-8")

    summary = {"experiment": "ND-16: Multi-teacher distillation",
               "teachers": args.teacher_oofs, "alpha": args.alpha, "temperature": args.temperature,
               "selection_metric": "robust_score",
               "overall_oof_metrics": overall,
               "overall_mf1": overall["macro_f1"],
               "overall_acc": overall["accuracy"],
               "overall_macro_auc": overall["macro_auc"],
               "overall_robust_score": overall["robust_score"]}
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Final robust_score: {overall['robust_score']:.4f}")


if __name__ == "__main__":
    main()
