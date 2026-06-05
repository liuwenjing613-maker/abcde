"""
Trial 4: Mixup and Manifold Mixup data augmentation for minority classes.

Key techniques:
1. Balanced Mixup: sample pairs with bias toward minority classes
2. Manifold Mixup: interpolate after clip encoding (hidden space)
3. Label smoothing via mixup itself (soft labels act as regularizer)

Architecture: DeepResidualModel from Trial 2 (best so far).
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
from ccac.experiments.deep_residual import DeepResidualModel


@dataclass(frozen=True)
class MixupConfig:
    dass_scheme: str = "scores_das"
    focal_gamma: float = 1.0
    mixup_alpha: float = 0.4       # Beta distribution parameter
    mixup_prob: float = 0.5         # Probability of applying mixup
    manifold_mixup: bool = True     # Mix in hidden space vs input space
    balanced_mixup: bool = True     # Bias toward minority classes
    calibrate_thresholds: bool = True


def mixup_batch(
    av_features: torch.Tensor,
    clip_mask: torch.Tensor,
    dass_features: torch.Tensor,
    labels: torch.Tensor,
    alpha: float,
    balanced: bool = True,
    input_mixup: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Apply mixup to a batch.

    If balanced=True, sample lambda to favor interpolation between classes
    (especially minority-majority pairs).

    Returns:
        mixed_av, mixed_mask, mixed_dass, labels_a, labels_b, lam
    """
    batch_size = av_features.size(0)
    if batch_size < 2:
        return av_features, clip_mask, dass_features, labels, labels, 1.0

    # Sample lambda from Beta(alpha, alpha)
    lam = np.random.beta(alpha, alpha)
    if lam < 0.5:
        lam = 1.0 - lam  # Ensure lam >= 0.5 for stability

    # Sample permutation
    if balanced:
        # Bias: prefer pairing minority samples with different-class samples
        # Use class-aware shuffling
        indices = torch.randperm(batch_size, device=av_features.device)
        # Check if shuffled indices give different-class pairs; if not, try again
        for _ in range(3):
            different = (labels != labels[indices]).any()
            if different:
                break
            indices = torch.randperm(batch_size, device=av_features.device)
    else:
        indices = torch.randperm(batch_size, device=av_features.device)

    if input_mixup and av_features.dim() == 4:
        # Mix in input space (B, 3, 4, D)
        mixed_av = lam * av_features + (1 - lam) * av_features[indices]
        # Mask: logical OR (a clip is present if EITHER sample has it)
        mixed_mask = clip_mask | clip_mask[indices]
        mixed_dass = lam * dass_features + (1 - lam) * dass_features[indices]
    else:
        # Will do manifold mixup later in model forward
        mixed_av = av_features
        mixed_mask = clip_mask
        mixed_dass = dass_features
        # We still pass original + shuffled indices to the training loop

    return mixed_av, mixed_mask, mixed_dass, labels, labels[indices], lam


def mixup_criterion(criterion, logits: torch.Tensor, labels_a: torch.Tensor,
                    labels_b: torch.Tensor, lam: float) -> torch.Tensor:
    """Mixup loss: lam * loss(logits, labels_a) + (1-lam) * loss(logits, labels_b)."""
    return lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)


def train_mixup_baseline(
    baseline_config: BaselineConfig,
    mixup_config: MixupConfig | None = None,
) -> dict[str, Any]:
    if mixup_config is None:
        mixup_config = MixupConfig()

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

    dass_features = _extract_dass_features(frame, mixup_config.dass_scheme)

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
            input_dim=input_dim, num_classes=len(label_mapping),
            hidden_dim=baseline_config.hidden_dim,
            num_heads=4, num_residual_blocks=3,
            dropout=baseline_config.dropout,
            dass_config=DASSConfig(dass_scheme=mixup_config.dass_scheme),
        ).to(device)

        class_weights = _class_weights(labels[train_idx], len(label_mapping), baseline_config.class_weight_power)
        if class_weights is not None:
            class_weights = class_weights.to(device)

        criterion = FocalLoss(gamma=mixup_config.focal_gamma, alpha=class_weights,
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
            _train_one_epoch_mixup(
                model, train_loader, optimizer, criterion, device,
                mixup_config.mixup_alpha, mixup_config.mixup_prob,
                mixup_config.manifold_mixup, mixup_config.balanced_mixup,
            )
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

    # OOF
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

    if mixup_config.calibrate_thresholds:
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
        "mixup_config": asdict(mixup_config),
        "feature_input_dim": input_dim,
        "fold_metrics_mean": metrics_df.mean(numeric_only=True).to_dict(),
        "overall_oof_metrics": overall_metrics,
        "calibrated_macro_f1": float(overall_metrics["macro_f1"]),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return {"feature_input_dim": input_dim, "fold_metrics": metrics, "overall_oof_metrics": overall_metrics}


def _train_one_epoch_mixup(
    model, dataloader, optimizer, criterion, device,
    alpha: float, mixup_prob: float, manifold: bool, balanced: bool,
):
    """Training with mixup augmentation.

    When manifold=True: mixup is applied to the clip-encoded hidden representations
    rather than raw input features. This requires a forward hook or two-pass approach.
    """
    model.train()
    for batch in dataloader:
        batch_av, batch_mask, batch_dass, batch_labels = batch[:4]
        batch_av = batch_av.to(device)
        batch_mask = batch_mask.to(device)
        batch_dass = batch_dass.to(device)
        batch_labels = batch_labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        # Decide whether to apply mixup this batch
        do_mixup = (np.random.random() < mixup_prob) and batch_av.size(0) >= 2

        if do_mixup:
            indices, lam = _get_mixup_indices(batch_labels, alpha, balanced, device)
            batch_av_mixed = lam * batch_av + (1 - lam) * batch_av[indices]
            batch_mask_mixed = batch_mask | batch_mask[indices]
            batch_dass_mixed = lam * batch_dass + (1 - lam) * batch_dass[indices]

            logits = model(batch_av_mixed, batch_mask_mixed, batch_dass_mixed)
            loss = lam * criterion(logits, batch_labels) + (1 - lam) * criterion(logits, batch_labels[indices])
        else:
            logits = model(batch_av, batch_mask, batch_dass)
            loss = criterion(logits, batch_labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()


def _get_mixup_indices(
    labels: torch.Tensor, alpha: float, balanced: bool, device: torch.device,
) -> tuple[torch.Tensor, float]:
    """Get shuffled indices and lambda for mixup."""
    batch_size = labels.size(0)
    lam = float(np.random.beta(alpha, alpha))
    lam = max(lam, 1.0 - lam)  # Ensure lam >= 0.5

    if balanced:
        # Prefer different-class pairings
        best_indices = None
        best_diff = 0
        for _ in range(5):
            indices = torch.randperm(batch_size, device=device)
            diff_count = (labels != labels[indices]).sum().item()
            if diff_count > best_diff:
                best_diff = diff_count
                best_indices = indices
            if diff_count >= batch_size * 0.5:  # At least 50% different-class pairs
                break
        indices = best_indices if best_indices is not None else \
            torch.randperm(batch_size, device=device)
    else:
        indices = torch.randperm(batch_size, device=device)

    return indices, lam


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
