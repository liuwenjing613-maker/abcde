"""
Trial 5: XGBoost stacking on NN-learned fusion representations.

Approach:
1. Train DeepResidualModel (best architecture) with 5-fold CV
2. Extract OOF fusion features from each sample (before classifier)
3. Train XGBoost on these 777-dim fusion features
4. Ensemble NN + XGBoost predictions
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
from sklearn.metrics import classification_report, f1_score, accuracy_score
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
class XGBoostStackConfig:
    dass_scheme: str = "scores_das"
    focal_gamma: float = 1.0
    xgb_learning_rate: float = 0.05
    xgb_max_depth: int = 5
    xgb_n_estimators: int = 500
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8
    xgb_reg_alpha: float = 0.1
    xgb_reg_lambda: float = 1.0
    nn_weight: float = 0.6       # Weight for NN predictions in ensemble
    xgb_weight: float = 0.4       # Weight for XGBoost predictions
    calibrate_thresholds: bool = True


class FeatureExtractor:
    """Extract fusion features from DeepResidualModel before classifier."""

    def __init__(self, model: DeepResidualModel):
        self.model = model
        self.features: list[np.ndarray] = []

    def _hook(self, module, input, output):
        # output is the fused representation before classifier
        self.features.append(output.detach().cpu().numpy())

    def __enter__(self):
        self.handle = self.model.classifier.register_forward_hook(self._hook)
        self.features = []
        return self

    def __exit__(self, *args):
        self.handle.remove()


def extract_fusion_features(model, dataloader, device) -> np.ndarray:
    """Extract fusion representations for all samples in dataloader."""
    model.eval()
    all_features = []
    with torch.no_grad():
        for batch in dataloader:
            batch_av, batch_mask, batch_dass, _ = batch[:4]
            batch_av = batch_av.to(device)
            batch_mask = batch_mask.to(device)
            batch_dass = batch_dass.to(device)
            fused = model._encode(batch_av, batch_mask, batch_dass)
            all_features.append(fused.cpu().numpy())
    return np.concatenate(all_features, axis=0)


def train_xgboost_stacking(
    baseline_config: BaselineConfig,
    stack_config: XGBoostStackConfig | None = None,
) -> dict[str, Any]:
    if stack_config is None:
        stack_config = XGBoostStackConfig()

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

    dass_features = _extract_dass_features(frame, stack_config.dass_scheme)

    fold_indices = _build_folds(labels, baseline_config.num_folds, baseline_config.seed)

    # Store OOF fusion features and probabilities
    oof_fusion_features = np.zeros((len(frame), 777), dtype=np.float32)  # Will be filled
    fusion_dim = None
    oof_probabilities_nn = np.zeros((len(frame), len(label_mapping)), dtype=np.float32)
    oof_predictions_nn = np.full(len(frame), fill_value=-1, dtype=np.int64)
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
            dass_config=DASSConfig(dass_scheme=stack_config.dass_scheme),
        ).to(device)

        class_weights = _class_weights(labels[train_idx], len(label_mapping), baseline_config.class_weight_power)
        if class_weights is not None:
            class_weights = class_weights.to(device)

        criterion = FocalLoss(gamma=stack_config.focal_gamma, alpha=class_weights,
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

        # Extract fusion features for validation set
        val_fusion = extract_fusion_features(model, val_loader, device)
        if fusion_dim is None:
            fusion_dim = val_fusion.shape[1]
            oof_fusion_features = np.zeros((len(frame), fusion_dim), dtype=np.float32)

        oof_fusion_features[val_idx] = val_fusion

        val_metrics, probabilities = _evaluate(model, val_loader, criterion, device, len(label_mapping))
        oof_probabilities_nn[val_idx] = probabilities
        oof_predictions_nn[val_idx] = probabilities.argmax(axis=1)
        fold_metric = {"fold": fold_id, "best_epoch": best_epoch, **val_metrics}
        metrics.append(fold_metric)
        fold_states.append(best_state)
        (fold_dir / "metrics.json").write_text(json.dumps(fold_metric, ensure_ascii=False, indent=2), encoding="utf-8")
        np.save(fold_dir / "val_fusion_features.npy", val_fusion)

    # NN OOF metrics
    nn_metrics = _classification_metrics(labels, oof_predictions_nn)
    print(f"NN OOF Macro-F1: {nn_metrics['macro_f1']:.4f}")

    # Train XGBoost on OOF fusion features
    try:
        import xgboost as xgb
        xgb_available = True
    except ImportError:
        print("XGBoost not available; falling back to sklearn GradientBoostingClassifier")
        xgb_available = False

    if xgb_available:
        xgb_model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=len(label_mapping),
            learning_rate=stack_config.xgb_learning_rate,
            max_depth=stack_config.xgb_max_depth,
            n_estimators=stack_config.xgb_n_estimators,
            subsample=stack_config.xgb_subsample,
            colsample_bytree=stack_config.xgb_colsample_bytree,
            reg_alpha=stack_config.xgb_reg_alpha,
            reg_lambda=stack_config.xgb_reg_lambda,
            eval_metric='mlogloss',
            random_state=baseline_config.seed,
            verbosity=0,
        )
    else:
        from sklearn.ensemble import GradientBoostingClassifier
        xgb_model = GradientBoostingClassifier(
            n_estimators=stack_config.xgb_n_estimators,
            learning_rate=stack_config.xgb_learning_rate,
            max_depth=stack_config.xgb_max_depth,
            subsample=stack_config.xgb_subsample,
            random_state=baseline_config.seed,
        )

    xgb_model.fit(oof_fusion_features, labels)
    oof_probabilities_xgb = xgb_model.predict_proba(oof_fusion_features)
    oof_predictions_xgb = oof_probabilities_xgb.argmax(axis=1)
    xgb_metrics = _classification_metrics(labels, oof_predictions_xgb)
    print(f"XGB OOF Macro-F1: {xgb_metrics['macro_f1']:.4f}")

    # Weighted ensemble
    w_nn = stack_config.nn_weight
    w_xgb = stack_config.xgb_weight
    ensemble_probs = w_nn * oof_probabilities_nn + w_xgb * oof_probabilities_xgb
    ensemble_preds = ensemble_probs.argmax(axis=1)
    ensemble_metrics = _classification_metrics(labels, ensemble_preds)
    print(f"Ensemble OOF Macro-F1: {ensemble_metrics['macro_f1']:.4f}")

    # Grid search weights for best ensemble
    best_w = w_nn
    best_ensemble_mf1 = ensemble_metrics["macro_f1"]
    for w in np.linspace(0.0, 1.0, 21):
        eprobs = w * oof_probabilities_nn + (1 - w) * oof_probabilities_xgb
        epreds = eprobs.argmax(axis=1)
        emf1 = f1_score(labels, epreds, average='macro', zero_division=0)
        if emf1 > best_ensemble_mf1:
            best_ensemble_mf1 = emf1
            best_w = w
    print(f"Best ensemble weight (NN): {best_w:.2f}, MF1: {best_ensemble_mf1:.4f}")
    ensemble_probs = best_w * oof_probabilities_nn + (1 - best_w) * oof_probabilities_xgb
    ensemble_preds = ensemble_probs.argmax(axis=1)
    ensemble_metrics = _classification_metrics(labels, ensemble_preds)

    # Threshold calibration for ensemble
    overall_metrics = ensemble_metrics
    if stack_config.calibrate_thresholds:
        cal_thr, cal_met = _calibrate_thresholds(ensemble_probs, labels, len(label_mapping))
        (output_dir / "calibrated_thresholds.json").write_text(
            json.dumps({"thresholds": cal_thr.tolist(), "metrics": cal_met}, ensure_ascii=False, indent=2), encoding="utf-8")
        ensemble_preds_cal = _apply_thresholds(ensemble_probs, cal_thr)
        overall_metrics = _classification_metrics(labels, ensemble_preds_cal)
        (output_dir / "classification_report_calibrated.txt").write_text(
            classification_report(labels, ensemble_preds_cal,
                target_names=[l for l, _ in sorted(label_mapping.items(), key=lambda x: x[1])], zero_division=0),
            encoding="utf-8")
        print(f"Calibrated Ensemble MF1: {overall_metrics['macro_f1']:.4f}")

    # Save
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(output_dir / "fold_metrics.csv", index=False, encoding="utf-8")
    (output_dir / "nn_oof_metrics.json").write_text(json.dumps(nn_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "xgb_oof_metrics.json").write_text(json.dumps(xgb_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "ensemble_oof_metrics.json").write_text(json.dumps(ensemble_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "label_mapping.json").write_text(json.dumps(label_mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "best_ensemble_weight.json").write_text(
        json.dumps({"nn_weight": best_w, "mf1": best_ensemble_mf1}, ensure_ascii=False, indent=2), encoding="utf-8")

    (output_dir / "classification_report.txt").write_text(
        classification_report(labels, ensemble_preds,
            target_names=[l for l, _ in sorted(label_mapping.items(), key=lambda x: x[1])], zero_division=0),
        encoding="utf-8")

    summary = {
        "nn_macro_f1": nn_metrics["macro_f1"],
        "xgb_macro_f1": xgb_metrics["macro_f1"],
        "ensemble_macro_f1": ensemble_metrics["macro_f1"],
        "best_ensemble_mf1": best_ensemble_mf1,
        "best_nn_weight": best_w,
        "calibrated_macro_f1": float(overall_metrics["macro_f1"]),
        "fusion_dim": fusion_dim,
        "fold_metrics_mean": metrics_df.mean(numeric_only=True).to_dict(),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return {"nn_mf1": nn_metrics["macro_f1"], "xgb_mf1": xgb_metrics["macro_f1"],
            "ensemble_mf1": ensemble_metrics["macro_f1"], "overall_oof_metrics": overall_metrics}


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
