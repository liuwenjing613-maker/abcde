#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from txy.constants import TARGET_LABEL_COLUMN
from txy.data.feature_io import load_or_build_multimodal, make_subject_id
from txy.data.group_split import build_group_folds, make_group_id
from txy.data.history_features import HistoryFeatureBuilder
from txy.data.longitudinal_dataset import LongitudinalPersonDataset, collate_person_batch
from txy.models.ordinal import OrdinalAnxietyHead
from txy.models.stagewise import StageWiseLongitudinalModel
from txy.training.calibration import apply_class_bias, search_class_bias
from txy.training.losses import OrdinalLoss
from txy.training.metrics import classification_metrics, format_classification_report
from txy.training.trainer import TrainConfig, _apply_scaler, _fit_scaler, _resolve_device, _set_seed


class OrdinalStageWiseModel(torch.nn.Module):
    def __init__(self, stagewise: StageWiseLongitudinalModel, num_classes: int):
        super().__init__()
        self.stagewise = stagewise
        fusion_dim = stagewise.classifier[1].in_features
        self.ordinal_head = OrdinalAnxietyHead(fusion_dim, num_thresholds=num_classes - 1)

    def encode(self, audio, video, clip_mask, history_scores, history_levels):
        mm_repr = self.stagewise.encode_multimodal(audio, video, clip_mask)
        if self.stagewise.use_history and self.stagewise.history_encoder is not None:
            hist_repr = self.stagewise.history_encoder(history_scores, history_levels)
            return torch.cat([mm_repr, hist_repr], dim=-1)
        return mm_repr

    def forward(self, audio, video, clip_mask, history_scores, history_levels):
        threshold_logits = self.ordinal_head(self.encode(audio, video, clip_mask, history_scores, history_levels))
        return OrdinalAnxietyHead.thresholds_to_logits(threshold_logits)

    def threshold_logits(self, audio, video, clip_mask, history_scores, history_levels):
        return self.ordinal_head(self.encode(audio, video, clip_mask, history_scores, history_levels))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_labels, all_logits = [], []
    for pb in loader:
        logits = model(
            pb.audio.to(device),
            pb.video.to(device),
            pb.clip_mask.to(device),
            pb.history_scores.to(device),
            pb.history_levels.to(device),
        )
        all_labels.append(pb.labels.cpu().numpy())
        all_logits.append(logits.cpu().numpy())
    return np.concatenate(all_labels), np.concatenate(all_logits)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ordinal StageWise model (Experiment 5)")
    parser.add_argument("--dataset-path", type=str, default="/home/adodas/dataset_ccac")
    parser.add_argument("--output-dir", type=str, default="artifacts/ordinal")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = TrainConfig(
        dataset_path=str(Path(args.dataset_path).resolve()),
        output_dir=str(Path(args.output_dir).resolve()),
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_folds=args.num_folds,
        seed=args.seed,
    )
    _set_seed(config.seed)
    device = _resolve_device(config.device)
    output_dir = Path(config.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_root = Path(config.dataset_path)
    frame = pd.read_csv(dataset_root / "train_val" / "labels.csv")
    frame = frame.dropna(subset=[TARGET_LABEL_COLUMN]).reset_index(drop=True)
    groups = make_group_id(frame, config.group_by)
    labels_series = frame[TARGET_LABEL_COLUMN].astype(str).str.strip()
    label_mapping = {label: index for index, label in enumerate(sorted(labels_series.unique()))}
    y = labels_series.map(label_mapping).to_numpy(dtype=np.int64)

    history_builder = HistoryFeatureBuilder.from_labels_frame(frame)
    history_features, history_levels = history_builder.transform(frame)
    audio, video, clip_mask, fused, audio_dim, video_dim = load_or_build_multimodal(
        dataset_root, "train_val", frame, config.audio_feature_name, config.video_feature_name
    )

    fold_indices = build_group_folds(y, groups, config.num_folds, config.seed)
    oof_logits = np.zeros((len(frame), len(label_mapping)), dtype=np.float32)
    oof_predictions = np.full(len(frame), -1, dtype=np.int64)
    fold_metrics = []

    for fold_id, (train_idx, val_idx) in enumerate(fold_indices, start=1):
        fold_dir = output_dir / f"fold_{fold_id}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        mm_scaler = _fit_scaler(fused[train_idx], clip_mask[train_idx])

        train_ds = LongitudinalPersonDataset(
            _apply_scaler(audio[train_idx], mm_scaler),
            _apply_scaler(video[train_idx], mm_scaler),
            _apply_scaler(fused[train_idx], mm_scaler),
            clip_mask[train_idx],
            history_features[train_idx],
            history_levels[train_idx],
            y[train_idx],
            make_subject_id(frame).astype(str).iloc[train_idx].tolist(),
            stage_drop_prob=config.stage_drop_prob,
            clip_drop_prob=config.clip_drop_prob,
            feature_noise_std=config.feature_noise_std,
            train=True,
        )
        val_ds = LongitudinalPersonDataset(
            _apply_scaler(audio[val_idx], mm_scaler),
            _apply_scaler(video[val_idx], mm_scaler),
            _apply_scaler(fused[val_idx], mm_scaler),
            clip_mask[val_idx],
            history_features[val_idx],
            history_levels[val_idx],
            y[val_idx],
            make_subject_id(frame).astype(str).iloc[val_idx].tolist(),
            train=False,
        )
        train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, collate_fn=collate_person_batch)
        val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, collate_fn=collate_person_batch)

        stagewise = StageWiseLongitudinalModel(
            audio_dim, video_dim, history_features.shape[1], history_levels.shape[1], len(label_mapping)
        )
        model = OrdinalStageWiseModel(stagewise, len(label_mapping)).to(device)
        criterion = OrdinalLoss(len(label_mapping))
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

        best_metric = -math.inf
        best_state = None
        stale = 0
        for epoch in range(1, config.epochs + 1):
            model.train()
            for pb in train_loader:
                audio_b = pb.audio.to(device)
                video_b = pb.video.to(device)
                mask_b = pb.clip_mask.to(device)
                hs = pb.history_scores.to(device)
                hl = pb.history_levels.to(device)
                labels_b = pb.labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                thresh = model.threshold_logits(audio_b, video_b, mask_b, hs, hl)
                loss = criterion(thresh, labels_b)
                loss.backward()
                optimizer.step()

            val_labels, val_logits = evaluate(model, val_loader, device)
            metrics = classification_metrics(val_labels, val_logits.argmax(axis=1))
            if metrics["macro_f1"] > best_metric:
                best_metric = metrics["macro_f1"]
                best_state = {"model": model.state_dict(), "metrics": metrics}
                stale = 0
                torch.save(best_state, fold_dir / "best_macro_f1.pt")
            else:
                stale += 1
            if stale >= config.patience:
                break

        model.load_state_dict(best_state["model"])
        val_labels, val_logits = evaluate(model, val_loader, device)
        bias, cal_metrics = search_class_bias(val_logits, val_labels)
        cal_logits = apply_class_bias(val_logits, bias)
        preds = cal_logits.argmax(axis=1)
        oof_logits[val_idx] = cal_logits
        oof_predictions[val_idx] = preds
        fold_metrics.append({"fold": fold_id, **best_state["metrics"], **{f"calibrated_{k}": v for k, v in cal_metrics.items()}})

    label_by_index = {index: label for label, index in label_mapping.items()}
    class_names = [label for label, _ in sorted(label_mapping.items(), key=lambda x: x[1])]
    overall = classification_metrics(y, oof_predictions)
    pd.DataFrame(fold_metrics).to_csv(output_dir / "fold_metrics.csv", index=False)
    (output_dir / "classification_report.txt").write_text(
        format_classification_report(y, oof_predictions, class_names), encoding="utf-8"
    )
    summary = {"overall_oof_metrics": overall, "fold_metrics": fold_metrics}
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
