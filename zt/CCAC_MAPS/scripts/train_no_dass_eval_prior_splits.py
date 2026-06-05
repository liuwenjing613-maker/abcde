#!/usr/bin/env python3
"""Train no-DASS DeepResidual models on eval-prior sampled validation splits."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ccac.baselines.anxiety_baseline import (  # noqa: E402
    BaselineConfig,
    _apply_scaler,
    _class_weights,
    _extended_metrics,
    _fit_scaler,
    _load_release_train_val,
    _resolve_device,
    _resolve_num_workers,
    _set_seed,
)
from ccac.baselines.dass_baseline import DASSConfig, DASSDataset, FocalLoss  # noqa: E402
from ccac.experiments.deep_residual import DeepResidualModel  # noqa: E402
from ccac.metrics import SEVERITY_RANK_BY_INDEX  # noqa: E402


PUBLIC_SUPPORT_BY_INDEX = np.asarray([292, 15, 46, 14, 15], dtype=np.float64)


class OrdinalAwareLoss(torch.nn.Module):
    """Focal/CE loss plus expected-severity distance penalty.

    Class IDs are competition IDs, while severity ranks are:
    1=正常 < 2=轻度 < 0=中度 < 3=重度 < 4=非常严重.
    """

    def __init__(
        self,
        base_loss: torch.nn.Module,
        severity_ranks: np.ndarray,
        ordinal_weight: float,
    ) -> None:
        super().__init__()
        self.base_loss = base_loss
        self.ordinal_weight = float(ordinal_weight)
        rank_values = torch.as_tensor(severity_ranks, dtype=torch.float32)
        self.register_buffer("rank_values", rank_values)
        self.rank_scale = float(max(1, int(rank_values.max().item())))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        base = self.base_loss(logits, targets)
        if self.ordinal_weight <= 0:
            return base
        probs = torch.softmax(logits, dim=-1)
        expected_rank = probs @ self.rank_values
        target_rank = self.rank_values[targets]
        ordinal = torch.square((expected_rank - target_rank) / self.rank_scale).mean()
        return base + self.ordinal_weight * ordinal


def _load_split_indices(split_dir: Path, fold_id: int) -> tuple[np.ndarray, pd.DataFrame]:
    train_csv = split_dir / f"fold_{fold_id}_train.csv"
    val_csv = split_dir / f"fold_{fold_id}_val.csv"
    if not train_csv.exists() or not val_csv.exists():
        raise FileNotFoundError(f"missing split files for fold {fold_id}: {train_csv}, {val_csv}")
    train_frame = pd.read_csv(train_csv)
    val_frame = pd.read_csv(val_csv)
    if "source_row" not in train_frame.columns or "source_row" not in val_frame.columns:
        raise ValueError("split CSVs must contain source_row")
    train_idx = train_frame["source_row"].to_numpy(dtype=np.int64)
    return train_idx, val_frame


def _make_model(input_dim: int, num_classes: int, device: torch.device) -> DeepResidualModel:
    return DeepResidualModel(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=256,
        num_heads=4,
        num_residual_blocks=3,
        dropout=0.2,
        dass_config=DASSConfig(dass_scheme="none"),
    ).to(device)


def _resample_train_indices(
    train_idx: np.ndarray,
    labels: np.ndarray,
    mode: str,
    rng: np.random.Generator,
    num_classes: int,
    multiplier: float,
) -> np.ndarray:
    if mode == "none":
        return train_idx

    n_samples = max(num_classes, int(round(len(train_idx) * multiplier)))
    if mode == "eval_prior":
        target = PUBLIC_SUPPORT_BY_INDEX[:num_classes].copy()
        target = target / target.sum()
    elif mode == "balanced":
        target = np.full(num_classes, 1.0 / num_classes, dtype=np.float64)
    else:
        raise ValueError(f"unsupported train resample mode: {mode}")

    train_labels = labels[train_idx]
    sampled_parts = []
    remaining = n_samples
    for class_idx in range(num_classes):
        class_positions = np.where(train_labels == class_idx)[0]
        if len(class_positions) == 0:
            continue
        if class_idx == num_classes - 1:
            count = remaining
        else:
            count = max(1, int(round(n_samples * target[class_idx])))
            remaining -= count
        sampled_positions = rng.choice(class_positions, size=count, replace=True)
        sampled_parts.append(train_idx[sampled_positions])
    sampled = np.concatenate(sampled_parts)
    rng.shuffle(sampled)
    return sampled.astype(np.int64)


def _predict(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    probs = []
    with torch.no_grad():
        for av, mask, dass, _ in loader:
            logits = model(av.to(device), mask.to(device), dass.to(device))
            probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(probs, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", default="datasets")
    parser.add_argument("--split-dir", default="artifacts/exp/eval_prior_val_splits/exact_public_with_replacement")
    parser.add_argument("--output-dir", default="artifacts/exp/no_dass_eval_prior_splits")
    parser.add_argument("--audio-feature-name", default="audio_wavlm_base")
    parser.add_argument("--video-feature-name", default="video_clip_base")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--torch-num-threads", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--class-weight-power", type=float, default=1.0)
    parser.add_argument("--selection-metric", choices=["macro_f1", "robust_score", "macro_auc", "qwk"], default="macro_f1")
    parser.add_argument("--train-resample-prior", choices=["none", "eval_prior", "balanced"], default="none")
    parser.add_argument("--train-sample-multiplier", type=float, default=1.0)
    parser.add_argument("--ordinal-loss-weight", type=float, default=0.0)
    args = parser.parse_args()

    _set_seed(args.seed)
    if args.torch_num_threads > 0:
        torch.set_num_threads(args.torch_num_threads)
        torch.set_num_interop_threads(max(1, min(2, args.torch_num_threads)))
    device = _resolve_device(args.device)
    dataset_path = Path(args.dataset_path)
    split_dir = Path(args.split_dir)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}, Output: {output_dir}")
    print(f"Split dir: {split_dir}")

    config = BaselineConfig(
        dataset_path=str(dataset_path.resolve()),
        output_dir=str(output_dir),
        audio_feature_name=args.audio_feature_name,
        video_feature_name=args.video_feature_name,
    )
    frame, av_features, clip_mask, label_mapping, input_dim = _load_release_train_val(config, dataset_path)
    labels = frame["_label_index"].to_numpy(dtype=np.int64)
    num_classes = len(label_mapping)
    label_by_idx = {idx: label for label, idx in label_mapping.items()}
    dass_features = np.zeros((len(frame), 0), dtype=np.float32)
    print(f"Loaded {len(frame)} subjects, input_dim={input_dim}, classes={num_classes}")
    print(f"Label mapping: {label_mapping}")

    metrics = []
    eval_rows = []
    all_eval_probs = []
    all_eval_labels = []
    workers = _resolve_num_workers(args.num_workers)

    for fold_id in range(1, args.num_folds + 1):
        fold_dir = output_dir / f"fold_{fold_id}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        train_idx, val_frame = _load_split_indices(split_dir, fold_id)
        val_idx = val_frame["source_row"].to_numpy(dtype=np.int64)
        rng = np.random.default_rng(args.seed + fold_id - 1)
        fit_idx = train_idx
        train_sample_idx = _resample_train_indices(
            train_idx,
            labels,
            args.train_resample_prior,
            rng,
            num_classes,
            args.train_sample_multiplier,
        )
        print(
            f"Fold {fold_id}/{args.num_folds}: "
            f"train_unique={len(train_idx)} val_rows={len(val_idx)} "
            f"val_unique={len(np.unique(val_idx))} train_rows={len(train_sample_idx)} "
            f"resample={args.train_resample_prior}",
            flush=True,
        )

        scaler = _fit_scaler(av_features[fit_idx], clip_mask[fit_idx])
        train_av = _apply_scaler(av_features[train_sample_idx], scaler)
        val_av = _apply_scaler(av_features[val_idx], scaler)

        train_ds = DASSDataset(
            train_av,
            clip_mask[train_sample_idx],
            dass_features[train_sample_idx],
            labels[train_sample_idx],
        )
        val_ds = DASSDataset(val_av, clip_mask[val_idx], dass_features[val_idx], labels[val_idx])
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=workers)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=workers)

        model = _make_model(input_dim, num_classes, device)
        class_weight = _class_weights(labels[train_idx], num_classes, args.class_weight_power)
        if class_weight is not None:
            class_weight = class_weight.to(device)
        base_criterion = FocalLoss(gamma=args.focal_gamma, alpha=class_weight)
        severity_ranks = SEVERITY_RANK_BY_INDEX[:num_classes]
        criterion = OrdinalAwareLoss(base_criterion, severity_ranks, args.ordinal_loss_weight).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )

        best_state = None
        best_probs = None
        best_score = -math.inf
        patience_count = 0

        for epoch in range(1, args.epochs + 1):
            model.train()
            for av, mask, dass, target in train_loader:
                av, mask, dass, target = av.to(device), mask.to(device), dass.to(device), target.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(av, mask, dass), target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

            probs = _predict(model, val_loader, device)
            preds = probs.argmax(axis=1)
            val_metrics = _extended_metrics(labels[val_idx], preds, probs, num_classes)
            selection_metric = args.selection_metric
            selection_score = float(val_metrics[selection_metric])

            if selection_score > best_score:
                best_score = selection_score
                patience_count = 0
                best_probs = probs
                best_state = {
                    "model": copy.deepcopy(model.state_dict()),
                    "scaler_mean": scaler[0].tolist(),
                    "scaler_std": scaler[1].tolist(),
                    "epoch": epoch,
                    "metrics": val_metrics,
                    "selection_metric": selection_metric,
                    "selection_score": selection_score,
                    "split_dir": str(split_dir),
                    "val_source_rows": val_idx.tolist(),
                }
            else:
                patience_count += 1

            if patience_count >= args.patience:
                break

        if best_state is None or best_probs is None:
            raise RuntimeError(f"fold {fold_id} failed to produce a checkpoint")

        model.load_state_dict(best_state["model"])
        preds = best_probs.argmax(axis=1)
        fold_metrics = _extended_metrics(labels[val_idx], preds, best_probs, num_classes)
        print(
            f"  best_epoch={best_state['epoch']} "
            f"MF1={fold_metrics['macro_f1']:.4f} "
            f"AUC={fold_metrics['macro_auc']:.4f} "
            f"Robust={fold_metrics['robust_score']:.4f} "
            f"Acc={fold_metrics['accuracy']:.4f}",
            flush=True,
        )

        torch.save(best_state, fold_dir / "best_model.pt")
        np.save(fold_dir / "val_probabilities.npy", best_probs)

        metrics.append({
            "fold": fold_id,
            "train_unique": int(len(train_idx)),
            "train_rows": int(len(train_sample_idx)),
            "val_rows": int(len(val_idx)),
            "val_unique": int(len(np.unique(val_idx))),
            "best_epoch": best_state["epoch"],
            "selection_metric": best_state["selection_metric"],
            "selection_score": best_state["selection_score"],
            **fold_metrics,
        })

        fold_eval = val_frame.copy()
        fold_eval["fold"] = fold_id
        fold_eval["true_label"] = [label_by_idx[int(i)] for i in labels[val_idx]]
        fold_eval["pred_label"] = [label_by_idx[int(i)] for i in preds]
        for class_idx in range(num_classes):
            fold_eval[f"prob_class_{class_idx}"] = best_probs[:, class_idx]
        eval_rows.append(fold_eval)
        all_eval_probs.append(best_probs)
        all_eval_labels.append(labels[val_idx])

    eval_df = pd.concat(eval_rows, ignore_index=True)
    eval_probs = np.concatenate(all_eval_probs, axis=0)
    eval_labels = np.concatenate(all_eval_labels, axis=0)
    eval_preds = eval_probs.argmax(axis=1)
    overall = _extended_metrics(eval_labels, eval_preds, eval_probs, num_classes)
    metrics_df = pd.DataFrame(metrics)

    eval_df.to_csv(output_dir / "eval_prior_val_predictions.csv", index=False, encoding="utf-8")
    metrics_df.to_csv(output_dir / "fold_metrics.csv", index=False, encoding="utf-8")
    (output_dir / "label_mapping.json").write_text(
        json.dumps(label_mapping, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "experiment": "no-DASS training with eval-prior validation splits",
                "split_dir": str(split_dir),
                "selection_metric": args.selection_metric,
                "overall_eval_prior_metrics": overall,
                "fold_metrics_mean": metrics_df.mean(numeric_only=True).to_dict(),
                "input_dim": input_dim,
                "label_mapping": label_mapping,
                "focal_gamma": args.focal_gamma,
                "class_weight_power": args.class_weight_power,
                "ordinal_loss_weight": args.ordinal_loss_weight,
                "train_resample_prior": args.train_resample_prior,
                "train_sample_multiplier": args.train_sample_multiplier,
                "num_eval_rows": int(len(eval_labels)),
                "num_unique_eval_subjects_across_rows": int(eval_df["source_row"].nunique()),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "classification_report.txt").write_text(
        classification_report(
            eval_labels,
            eval_preds,
            target_names=[label for label, _ in sorted(label_mapping.items(), key=lambda item: item[1])],
            zero_division=0,
        ),
        encoding="utf-8",
    )
    print(
        f"\nEval-prior validation: MF1={overall['macro_f1']:.4f} "
        f"AUC={overall['macro_auc']:.4f} "
        f"Robust={overall['robust_score']:.4f} "
        f"Acc={overall['accuracy']:.4f}"
    )
    print(f"Done. Saved to {output_dir}")


if __name__ == "__main__":
    main()
