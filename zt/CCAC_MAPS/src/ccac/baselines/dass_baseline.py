"""
DASS-history-aware baseline for CCAC MAPS.

Extends the original anxiety baseline by feeding T1/T2/T3 DASS scores
as additional inputs. Three schemes are supported:

  - "scores_a"  : 3 anxiety scores only  [t1_anxiety_score, t2_anxiety_score, t3_anxiety_score]
  - "scores_das" : 9 DASS scores (depression + anxiety + stress × 3 timepoints)
  - "encoder"    : full 18-dim DASS history (9 scores + 9 levels) through an MLP encoder
"""

from __future__ import annotations

import json
import math
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import KFold, StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset

# Re-use shared utilities from the original baseline
from ccac.baselines.anxiety_baseline import (
    STAGES,
    CLIP_TYPES,
    BaselineConfig,
    BaselineFeatureBuilder,
    LongitudinalAnxietyModel,
    _set_seed,
    _resolve_device,
    _resolve_num_workers,
    _encode_labels,
    _build_folds,
    _fit_scaler,
    _apply_scaler,
    _class_weights,
    _classification_metrics,
    _extended_metrics,
    _is_release_dataset,
    _release_cache_path,
    _build_release_features,
    _load_release_train_val,
    _write_release_test_predictions,
    _infer_release_feature_dim,
    _load_release_vector,
    _read_feature_vector,
    _scan_pooled_cache,
    _load_release_vector_cached,
)

# ---------------------------------------------------------------------------
# DASS column definitions
# ---------------------------------------------------------------------------

# Raw scores (continuous, 0–21+)
DASS_SCORE_COLS: dict[str, list[str]] = {
    "depression": ["t1_depression_score", "t2_depression_score", "t3_depression_score"],
    "anxiety":    ["t1_anxiety_score",    "t2_anxiety_score",    "t3_anxiety_score"],
    "stress":     ["t1_stress_score",     "t2_stress_score",     "t3_stress_score"],
}

# Severity levels (categorical: 正常/轻度/中度/重度/非常严重)
DASS_LEVEL_COLS: dict[str, list[str]] = {
    "depression": ["t1_depression_level", "t2_depression_level", "t3_depression_level"],
    "anxiety":    ["t1_anxiety_level",    "t2_anxiety_level",    "t3_anxiety_level"],
    "stress":     ["t1_stress_level",     "t2_stress_level",     "t3_stress_level"],
}

# Ordinal encoding: 正常=0, 轻度=1, 中度=2, 重度=3, 非常严重=4
LEVEL_TO_INT: dict[str, int] = {"正常": 0, "轻度": 1, "中度": 2, "重度": 3, "非常严重": 4}

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DASSConfig:
    """Extra configuration for DASS-history integration."""

    dass_scheme: str = "scores_a"
    """Which DASS features to use:
    - "none"     : no DASS features (identical to original baseline)
    - "scores_a" : 3 anxiety scores only
    - "scores_das": 9 scores (depression + anxiety + stress × 3 timepoints)
    - "encoder"  : 18-dim (9 scores + 9 levels) through a small MLP encoder
    """

    dass_hidden: int = 64
    """Hidden dimension of the DASS MLP encoder (only used when scheme="encoder")."""

    dass_dropout: float = 0.1
    """Dropout inside the DASS MLP encoder."""

    focal_gamma: float = 0.0
    """Focal loss gamma. 0 = standard cross-entropy. 2.0 is a good starting point.
    Higher gamma focuses the loss more on hard (misclassified) examples."""

    audio_features: list[str] | None = None
    """List of audio feature names for early fusion. If None, uses baseline_config.audio_feature_name."""

    video_features: list[str] | None = None
    """List of video feature names for early fusion. If None, uses baseline_config.video_feature_name."""

    # -- Multi-task auxiliary loss --
    aux_loss_weight: float = 0.0
    """Weight for T1/T2/T3 auxiliary anxiety prediction loss. 0 = disabled.
    Suggested: 0.2-0.5. The model learns to predict anxiety at all timepoints
    simultaneously, improving the shared encoder."""

    # -- Over-sampling --
    oversample: bool = False
    """If True, use balanced batch sampling so each batch has roughly equal
    per-class representation. Helps minority classes get more gradient signal."""

    # -- Threshold calibration --
    calibrate_thresholds: bool = False
    """If True, after OOF evaluation, search per-class decision thresholds
    to maximize macro-F1. Applied to OOF and test predictions."""


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------


class FocalLoss(nn.Module):
    """Focal Loss for imbalanced classification.

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    Parameters
    ----------
    gamma : float
        Focusing parameter. γ=0 reduces to standard CE.
        Higher γ down-weights easy examples more aggressively.
    alpha : Tensor or None
        Optional per-class weights (same semantics as CE weight).
    label_smoothing : float
        Label smoothing factor (0.0 = no smoothing).
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(-1)

        # Label smoothing
        if self.label_smoothing > 0:
            smooth = self.label_smoothing / (num_classes - 1)
            target_one_hot = torch.full_like(logits, smooth)
            target_one_hot.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
        else:
            target_one_hot = torch.zeros_like(logits)
            target_one_hot.scatter_(1, targets.unsqueeze(1), 1.0)

        log_probs = torch.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)

        # Focal weight: (1 - p_t)^gamma
        pt = (target_one_hot * probs).sum(dim=-1)
        focal_weight = (1.0 - pt).pow(self.gamma)

        ce = -(target_one_hot * log_probs).sum(dim=-1)
        loss = focal_weight * ce

        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[targets]
            loss = alpha_t * loss

        return loss.mean()


# ---------------------------------------------------------------------------
# DASS feature extraction
# ---------------------------------------------------------------------------


def _extract_dass_features(frame: pd.DataFrame, scheme: str) -> np.ndarray:
    """Extract DASS history features from the labels dataframe.

    Returns float32 array of shape (N, dass_dim).
    """
    if scheme == "none":
        return np.zeros((len(frame), 0), dtype=np.float32)

    if scheme == "scores_a":
        cols = DASS_SCORE_COLS["anxiety"]  # 3 cols
        values = frame[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        return values.to_numpy(dtype=np.float32)

    if scheme == "scores_das":
        cols = []
        for dim in ["depression", "anxiety", "stress"]:
            cols.extend(DASS_SCORE_COLS[dim])  # 9 cols
        values = frame[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        return values.to_numpy(dtype=np.float32)

    if scheme == "encoder":
        features: list[np.ndarray] = []
        # Scores (9)
        for dim in ["depression", "anxiety", "stress"]:
            for col in DASS_SCORE_COLS[dim]:
                vals = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)
                features.append(vals.to_numpy(dtype=np.float32).reshape(-1, 1))
        # Levels (9) encoded as ordinal integers
        for dim in ["depression", "anxiety", "stress"]:
            for col in DASS_LEVEL_COLS[dim]:
                vals = frame[col].map(LEVEL_TO_INT).fillna(0).astype(np.float32)
                features.append(vals.to_numpy(dtype=np.float32).reshape(-1, 1))
        return np.concatenate(features, axis=1)  # (N, 18)

    raise ValueError(f"Unknown dass_scheme: {scheme}")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class DASSAnxietyModel(nn.Module):
    """Longitudinal anxiety model with an optional DASS-history encoder branch.

    Architecture
    ------------
    Audio-Video branch (same as original):
        Clip Encoder → Attention Pooling → BiGRU → fusion features [896]

    DASS branch (if enabled):
        DASS scores/levels → optional MLP encoder → dass_repr [dass_hidden]

    Final:
        Concat(audio_video_fusion, dass_repr) → Classifier → 5 classes
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        temporal_hidden_dim: int = 192,
        dropout: float = 0.2,
        dass_config: DASSConfig | None = None,
    ):
        super().__init__()
        self.dass_config = dass_config or DASSConfig(dass_scheme="none")

        # --- Original audio-video backbone ---
        self.clip_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.clip_attention = nn.Linear(hidden_dim, 1)
        self.stage_position = nn.Parameter(torch.zeros(len(STAGES), hidden_dim))
        nn.init.normal_(self.stage_position, mean=0.0, std=0.02)
        self.temporal_encoder = nn.GRU(
            input_size=hidden_dim,
            hidden_size=temporal_hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # --- DASS encoder branch ---
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

        # --- Fusion ---
        av_fusion_dim = hidden_dim * 2 + temporal_hidden_dim * 2  # 896
        fusion_dim = av_fusion_dim + dass_out
        self.fusion_dim = fusion_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        # --- Auxiliary heads for T1/T2/T3 anxiety prediction ---
        self.aux_loss_weight = self.dass_config.aux_loss_weight
        if self.aux_loss_weight > 0:
            self.aux_head_t1 = nn.Sequential(
                nn.LayerNorm(fusion_dim),
                nn.Linear(fusion_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, num_classes),
            )
            self.aux_head_t2 = nn.Sequential(
                nn.LayerNorm(fusion_dim),
                nn.Linear(fusion_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, num_classes),
            )
            self.aux_head_t3 = nn.Sequential(
                nn.LayerNorm(fusion_dim),
                nn.Linear(fusion_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, num_classes),
            )

    def _dass_input_dim(self) -> int:
        scheme = self.dass_config.dass_scheme
        if scheme == "none":
            return 0
        if scheme == "scores_a":
            return 3
        if scheme == "scores_das":
            return 9
        if scheme == "encoder":
            return 18
        return 0

    def _encode(self, av_inputs, clip_mask, dass_features=None) -> torch.Tensor:
        """Return the fused representation before the classifier."""
        encoded = self.clip_encoder(av_inputs)
        stage_repr = self._pool_stage(encoded, clip_mask)
        stage_repr = stage_repr + self.stage_position.unsqueeze(0)
        temporal_out, _ = self.temporal_encoder(stage_repr)
        pooled_temporal = temporal_out.mean(dim=1)
        pooled_stage = stage_repr.mean(dim=1)
        final_stage = stage_repr[:, -1, :]
        av_fused = torch.cat([pooled_temporal, pooled_stage, final_stage], dim=-1)
        if self._has_dass and dass_features is not None:
            dass_repr = self.dass_encoder(dass_features)
            return torch.cat([av_fused, dass_repr], dim=-1)
        return av_fused

    def forward(
        self,
        av_inputs: torch.Tensor,
        clip_mask: torch.Tensor,
        dass_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        fused = self._encode(av_inputs, clip_mask, dass_features)
        return self.classifier(fused)

    def forward_with_aux(
        self,
        av_inputs: torch.Tensor,
        clip_mask: torch.Tensor,
        dass_features: torch.Tensor | None = None,
    ):
        """Forward pass returning (t4_logits, t1_logits, t2_logits, t3_logits).
        Aux heads return None when aux_loss_weight == 0."""
        fused = self._encode(av_inputs, clip_mask, dass_features)
        t4 = self.classifier(fused)
        if self.aux_loss_weight > 0:
            return t4, self.aux_head_t1(fused), self.aux_head_t2(fused), self.aux_head_t3(fused)
        return t4, None, None, None

    def _pool_stage(self, encoded: torch.Tensor, clip_mask: torch.Tensor) -> torch.Tensor:
        logits = self.clip_attention(encoded).squeeze(-1)
        logits = logits.masked_fill(~clip_mask, -1e9)
        weights = torch.softmax(logits, dim=-1)
        weights = weights * clip_mask.float()
        denom = weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        weights = weights / denom
        stage_repr = (encoded * weights.unsqueeze(-1)).sum(dim=2)
        missing_stage = clip_mask.sum(dim=-1, keepdim=True) == 0
        return torch.where(missing_stage, torch.zeros_like(stage_repr), stage_repr)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class DASSDataset(Dataset):
    """Dataset that bundles AV features + DASS features + labels (+ optional aux labels)."""

    def __init__(
        self,
        av_features: np.ndarray,
        clip_mask: np.ndarray,
        dass_features: np.ndarray,
        labels: np.ndarray,
        aux_labels: np.ndarray | None = None,
    ):
        self.av_features = torch.from_numpy(av_features).float()
        self.clip_mask = torch.from_numpy(clip_mask).bool()
        self.dass_features = torch.from_numpy(dass_features).float()
        self.labels = torch.from_numpy(labels).long()
        self.has_aux = aux_labels is not None
        if self.has_aux:
            self.aux_labels = torch.from_numpy(aux_labels).long()

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int):
        if self.has_aux:
            return (
                self.av_features[index], self.clip_mask[index],
                self.dass_features[index], self.labels[index],
                self.aux_labels[index],
            )
        return (
            self.av_features[index], self.clip_mask[index],
            self.dass_features[index], self.labels[index],
        )


# ---------------------------------------------------------------------------
# Multi-feature loading (early fusion of multiple audio/video features)
# ---------------------------------------------------------------------------


def _build_multi_features(
    dataset_root: Path,
    split: str,
    frame: pd.DataFrame,
    audio_features: list[str],
    video_features: list[str],
) -> tuple[np.ndarray, np.ndarray, int]:
    """Build feature tensor by concatenating multiple audio + video features per clip."""
    audio_dims = {
        name: _infer_release_feature_dim(dataset_root, split, name, prefer_npy=True)
        for name in audio_features
    }
    video_dims = {
        name: _infer_release_feature_dim(dataset_root, split, name, prefer_npy=True)
        for name in video_features
    }
    total_audio_dim = sum(audio_dims.values())
    total_video_dim = sum(video_dims.values())
    total_dim = total_audio_dim + total_video_dim

    # Pre-scan all pooled files
    audio_caches = {name: _scan_pooled_cache(dataset_root / split / name) for name in audio_features}
    video_caches = {name: _scan_pooled_cache(dataset_root / split / name) for name in video_features}

    features = np.zeros((len(frame), len(STAGES), len(CLIP_TYPES), total_dim), dtype=np.float32)
    clip_mask = np.zeros((len(frame), len(STAGES), len(CLIP_TYPES)), dtype=bool)

    # Build task list
    tasks: list[tuple[int, int, int, tuple[str, str, str], str, str]] = []
    for row_index, row in enumerate(frame.itertuples(index=False)):
        subject_parts = (row.anon_school, row.anon_class, row.anon_person)
        for stage_index, stage in enumerate(STAGES):
            for clip_index, clip_type in enumerate(CLIP_TYPES):
                tasks.append((row_index, stage_index, clip_index, subject_parts, stage, clip_type))

    max_workers = min(32, (os.cpu_count() or 4) * 2)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:

        def _load_task(task):
            row_index, stage_index, clip_index, subject_parts, stage, clip_type = task
            audio_vecs = []
            audio_offset = 0
            for name in audio_features:
                vec = _load_release_vector_cached(
                    dataset_root / split / name, audio_caches[name],
                    subject_parts, stage, clip_type, audio_dims[name],
                )
                audio_vecs.append(vec)
            video_vecs = []
            for name in video_features:
                vec = _load_release_vector_cached(
                    dataset_root / split / name, video_caches[name],
                    subject_parts, stage, clip_type, video_dims[name],
                )
                video_vecs.append(vec)
            return row_index, stage_index, clip_index, audio_vecs, video_vecs

        for row_i, stage_i, clip_i, audio_vecs, video_vecs in pool.map(_load_task, tasks):
            a_offset = 0
            for vec in audio_vecs:
                features[row_i, stage_i, clip_i, a_offset:a_offset + len(vec)] = vec
                a_offset += len(vec)
            v_offset = total_audio_dim
            for vec in video_vecs:
                features[row_i, stage_i, clip_i, v_offset:v_offset + len(vec)] = vec
                v_offset += len(vec)
            any_valid = any(np.isfinite(v).any() for v in audio_vecs + video_vecs)
            clip_mask[row_i, stage_i, clip_i] = bool(any_valid)

    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0), clip_mask, total_dim


def _multi_cache_path(dataset_root: Path, audio_features: list[str], video_features: list[str],
                      target_column: str, split: str) -> Path:
    a_key = "+".join(sorted(audio_features))
    v_key = "+".join(sorted(video_features))
    name = f"{a_key}__{v_key}__{target_column}.npz"
    return dataset_root / "metadata" / "baseline_cache" / split / name


def _load_multi_train_val(
    dataset_root: Path,
    frame: pd.DataFrame,
    audio_features: list[str],
    video_features: list[str],
    target_column: str,
    feature_cache: bool,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Load or build multi-feature tensor for train_val."""
    cache_path = _multi_cache_path(dataset_root, audio_features, video_features, target_column, "train_val")
    if feature_cache and cache_path.exists():
        cached = np.load(cache_path, allow_pickle=True, mmap_mode="r")
        return (
            cached["features"].astype(np.float32),
            cached["clip_mask"].astype(bool),
            int(cached["input_dim"]),
        )

    features, clip_mask, input_dim = _build_multi_features(
        dataset_root, "train_val", frame, audio_features, video_features,
    )
    if feature_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            features=features,
            clip_mask=clip_mask,
            input_dim=np.asarray(input_dim, dtype=np.int64),
        )
    return features, clip_mask, input_dim


# ---------------------------------------------------------------------------
# Threshold calibration
# ---------------------------------------------------------------------------


def _calibrate_thresholds(
    probabilities: np.ndarray, labels: np.ndarray, num_classes: int,
) -> tuple[np.ndarray, dict]:
    """Search per-class thresholds to maximize macro-F1 on OOF predictions,
    with a hard constraint that no class may have zero recall.

    Returns (thresholds, metrics). thresholds shape: (num_classes,).
    metrics includes legacy scores + new macro_auc, qwk, robust_score.
    """
    best_thresholds = np.full(num_classes, 0.5, dtype=np.float32)
    for c in range(num_classes):
        best_f1 = 0.0
        best_t = 0.5
        for t in np.linspace(0.05, 0.95, 19):
            preds = probabilities.argmax(axis=1).copy()
            # Boost class c: if p[c] > t, predict c
            preds[probabilities[:, c] > t] = c
            # Constraint: require non-zero recall on all classes
            recalls = [
                float(f1_score(labels == k, preds == k, zero_division=0.0))
                > 0.0
                for k in range(num_classes)
            ]
            if not all(recalls):
                continue  # reject collapsed solutions
            f1 = f1_score(labels, preds, average="macro", zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        # If all thresholds cause collapse for this class, keep default 0.5
        best_thresholds[c] = best_t

    calibrated_preds = _apply_thresholds(probabilities, best_thresholds)

    # If calibration still collapsed a class, fall back to argmax
    all_nonzero = all(
        f1_score(labels == c, calibrated_preds == c, zero_division=0.0) > 0.0
        for c in range(num_classes)
    )
    if not all_nonzero:
        calibrated_preds = probabilities.argmax(axis=1)

    # Extended metrics including new framework
    metrics = _extended_metrics(labels, calibrated_preds, probabilities, num_classes)
    return best_thresholds, metrics


def _apply_thresholds(probabilities: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Apply per-class thresholds to probabilities. Returns integer predictions."""
    preds = probabilities.argmax(axis=1).copy()
    for c, t in enumerate(thresholds):
        preds[probabilities[:, c] > t] = c
    return preds


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------


def train_dass_baseline(
    baseline_config: BaselineConfig,
    dass_config: DASSConfig | None = None,
) -> dict[str, Any]:
    """Train the DASS-augmented baseline.

    Parameters
    ----------
    baseline_config : BaselineConfig
        Standard baseline configuration (audio/video features, device, etc.).
    dass_config : DASSConfig or None
        DASS integration settings. If None, defaults to scheme="scores_a".
    """
    if dass_config is None:
        dass_config = DASSConfig(dass_scheme="scores_a")

    _set_seed(baseline_config.seed)
    device = _resolve_device(baseline_config.device)
    output_dir = Path(baseline_config.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = Path(baseline_config.dataset_path)

    # --- Load data ---
    use_multi = (
        dass_config.audio_features is not None
        and dass_config.video_features is not None
    )
    if _is_release_dataset(dataset_path):
        if use_multi:
            audio_list = dass_config.audio_features
            video_list = dass_config.video_features
            frame, _, _, label_mapping, _ = _load_release_train_val(
                baseline_config, dataset_path
            )
            labels = frame["_label_index"].to_numpy(dtype=np.int64)
            av_features, clip_mask, input_dim = _load_multi_train_val(
                dataset_path, frame, audio_list, video_list,
                baseline_config.target_label_column,
                baseline_config.feature_cache,
            )
        else:
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

    # --- Extract DASS features ---
    dass_features = _extract_dass_features(frame, dass_config.dass_scheme)
    dass_input_dim = dass_features.shape[1]

    # --- Extract T1/T2/T3 anxiety labels for multi-task aux loss ---
    aux_labels: np.ndarray | None = None
    if dass_config.aux_loss_weight > 0:
        aux_labels_raw = frame[["t1_anxiety_level", "t2_anxiety_level", "t3_anxiety_level"]]
        aux_label_mapping = {}  # reuse same mapping for consistency
        aux_encoded = np.zeros((len(frame), 3), dtype=np.int64)
        for i, col in enumerate(["t1_anxiety_level", "t2_anxiety_level", "t3_anxiety_level"]):
            vals = frame[col].astype(str).str.strip()
            unique = sorted(vals.unique().tolist())
            for label in unique:
                if label not in aux_label_mapping:
                    aux_label_mapping[label] = len(aux_label_mapping)
            aux_encoded[:, i] = np.array([aux_label_mapping.get(v, 0) for v in vals])
        aux_labels = aux_encoded  # (N, 3)
        aux_num_classes = len(aux_label_mapping)
    else:
        aux_num_classes = len(label_mapping)

    # --- Cross-validation ---
    fold_indices = _build_folds(labels, baseline_config.num_folds, baseline_config.seed)

    oof_probabilities = np.zeros((len(frame), len(label_mapping)), dtype=np.float32)
    oof_predictions = np.full(len(frame), fill_value=-1, dtype=np.int64)
    metrics: list[dict[str, Any]] = []
    fold_states: list[dict[str, Any]] = []

    for fold_id, (train_idx, val_idx) in enumerate(fold_indices, start=1):
        fold_dir = output_dir / f"fold_{fold_id}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        workers = _resolve_num_workers(baseline_config.num_workers)

        # Scale AV features
        scaler = _fit_scaler(av_features[train_idx], clip_mask[train_idx])
        train_av = _apply_scaler(av_features[train_idx], scaler)
        val_av = _apply_scaler(av_features[val_idx], scaler)

        # Scale DASS features
        dass_mean = dass_features[train_idx].mean(axis=0)
        dass_std = dass_features[train_idx].std(axis=0)
        dass_std = np.where(dass_std < 1e-6, 1.0, dass_std)
        train_dass = (dass_features[train_idx] - dass_mean) / dass_std
        val_dass = (dass_features[val_idx] - dass_mean) / dass_std

        train_aux = aux_labels[train_idx] if aux_labels is not None else None
        val_aux = aux_labels[val_idx] if aux_labels is not None else None
        train_dataset = DASSDataset(train_av, clip_mask[train_idx], train_dass, labels[train_idx], train_aux)
        val_dataset = DASSDataset(val_av, clip_mask[val_idx], val_dass, labels[val_idx], val_aux)

        # Over-sampling: weighted sampler for balanced batches
        train_sampler = None
        if dass_config.oversample:
            train_lbls = labels[train_idx]
            class_counts = np.bincount(train_lbls, minlength=len(label_mapping))
            class_weights_samp = 1.0 / np.maximum(class_counts, 1)
            sample_weights = class_weights_samp[train_lbls]
            sample_weights = sample_weights / sample_weights.sum()
            train_sampler = torch.utils.data.WeightedRandomSampler(
                weights=torch.from_numpy(sample_weights).float(),
                num_samples=len(train_lbls),
                replacement=True,
            )

        train_loader = DataLoader(
            train_dataset, batch_size=baseline_config.batch_size,
            sampler=train_sampler, shuffle=(train_sampler is None),
            num_workers=workers,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=baseline_config.batch_size, shuffle=False, num_workers=workers
        )

        model = DASSAnxietyModel(
            input_dim=input_dim,
            num_classes=len(label_mapping),
            hidden_dim=baseline_config.hidden_dim,
            temporal_hidden_dim=baseline_config.temporal_hidden_dim,
            dropout=baseline_config.dropout,
            dass_config=dass_config,
        ).to(device)

        class_weights = _class_weights(
            labels[train_idx], len(label_mapping), baseline_config.class_weight_power
        )
        if class_weights is not None:
            class_weights = class_weights.to(device)

        def _make_criterion(cw):
            if dass_config.focal_gamma > 0:
                return FocalLoss(gamma=dass_config.focal_gamma, alpha=cw,
                                 label_smoothing=baseline_config.label_smoothing)
            return nn.CrossEntropyLoss(weight=cw, label_smoothing=baseline_config.label_smoothing)

        criterion = _make_criterion(class_weights)

        # Auxiliary loss for T1/T2/T3 (same criterion, no class weights on aux)
        aux_criterion = None
        if dass_config.aux_loss_weight > 0:
            aux_criterion = _make_criterion(None)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=baseline_config.learning_rate,
            weight_decay=baseline_config.weight_decay,
        )

        best_state = None
        best_metric = -math.inf
        best_epoch = 0
        epochs_without_improvement = 0

        for epoch in range(1, baseline_config.epochs + 1):
            _train_one_epoch_dass(model, train_loader, optimizer, criterion, device,
                                  aux_criterion, None, dass_config.aux_loss_weight,
                                  len(label_mapping))
            val_metrics, probabilities = _evaluate_dass(
                model, val_loader, criterion, device, len(label_mapping)
            )
            selection_metric = "robust_score"
            selection_score = float(val_metrics[selection_metric])
            if selection_score > best_metric:
                best_metric = selection_score
                best_epoch = epoch
                epochs_without_improvement = 0
                best_state = {
                    "model": model.state_dict(),
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
            raise RuntimeError(f"fold {fold_id} failed to produce a valid checkpoint")

        model.load_state_dict(best_state["model"])
        val_metrics, probabilities = _evaluate_dass(
            model, val_loader, criterion, device, len(label_mapping)
        )
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

    # --- OOF evaluation ---
    label_by_index = {index: label for label, index in label_mapping.items()}
    oof_labels = np.asarray([label_by_index[int(i)] for i in oof_predictions], dtype=object)
    overall_metrics = _extended_metrics(labels, oof_predictions, oof_probabilities, len(label_mapping))
    metrics_df = pd.DataFrame(metrics)
    oof_df = pd.DataFrame({
        baseline_config.subject_id_column: frame[baseline_config.subject_id_column].astype(str),
        "true_label": frame[baseline_config.target_label_column].astype(str),
        "pred_label": oof_labels,
    })
    for class_index in range(len(label_mapping)):
        oof_df[f"prob_class_{class_index}"] = oof_probabilities[:, class_index]

    metrics_df.to_csv(output_dir / "fold_metrics.csv", index=False, encoding="utf-8")
    oof_df.to_csv(output_dir / "oof_predictions.csv", index=False, encoding="utf-8")
    (output_dir / "label_mapping.json").write_text(
        json.dumps(label_mapping, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "baseline_config.json").write_text(
        json.dumps(asdict(baseline_config), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "dass_config.json").write_text(
        json.dumps(asdict(dass_config), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps({
            "config": asdict(baseline_config),
            "dass_config": asdict(dass_config),
            "feature_input_dim": input_dim,
            "dass_input_dim": dass_input_dim,
            "label_mapping": label_mapping,
            "fold_metrics_mean": metrics_df.mean(numeric_only=True).to_dict(),
            "overall_oof_metrics": overall_metrics,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "classification_report.txt").write_text(
        classification_report(
            labels, oof_predictions,
            target_names=[label for label, _ in sorted(label_mapping.items(), key=lambda x: x[1])],
            zero_division=0,
        ),
        encoding="utf-8",
    )

    # --- Threshold calibration ---
    if dass_config.calibrate_thresholds:
        calibrated_thresholds, calibrated_metrics = _calibrate_thresholds(
            oof_probabilities, labels, len(label_mapping)
        )
        (output_dir / "calibrated_thresholds.json").write_text(
            json.dumps({"thresholds": calibrated_thresholds.tolist(), "metrics": calibrated_metrics},
                       ensure_ascii=False, indent=2), encoding="utf-8"
        )
        # Override with calibrated predictions
        oof_predictions = _apply_thresholds(oof_probabilities, calibrated_thresholds)
        overall_metrics = _extended_metrics(labels, oof_predictions, oof_probabilities, len(label_mapping))
        (output_dir / "classification_report_calibrated.txt").write_text(
            classification_report(
                labels, oof_predictions,
                target_names=[l for l, _ in sorted(label_mapping.items(), key=lambda x: x[1])],
                zero_division=0,
            ), encoding="utf-8",
        )

    # --- Test predictions (using only AV features; DASS unavailable for test) ---
    if _is_release_dataset(dataset_path):
        _write_dass_test_predictions(
            baseline_config, dass_config, dataset_path, output_dir,
            fold_states, label_mapping, input_dim, device,
        )

    return {
        "feature_input_dim": input_dim,
        "dass_input_dim": dass_input_dim,
        "label_mapping": label_mapping,
        "fold_metrics": metrics,
        "overall_oof_metrics": overall_metrics,
    }


# ---------------------------------------------------------------------------
# Per-epoch helpers
# ---------------------------------------------------------------------------


def _train_one_epoch_dass(model, dataloader, optimizer, criterion, device,
                          aux_criterion=None, aux_labels_all=None, aux_weight=0.0,
                          num_classes=5):
    model.train()
    for batch in dataloader:
        has_aux = len(batch) == 5  # (av, mask, dass, labels, aux_labels)
        if has_aux:
            batch_av, batch_mask, batch_dass, batch_labels, batch_aux = batch
            batch_aux = batch_aux.to(device)
        else:
            batch_av, batch_mask, batch_dass, batch_labels = batch
        batch_av = batch_av.to(device)
        batch_mask = batch_mask.to(device)
        batch_dass = batch_dass.to(device)
        batch_labels = batch_labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        if has_aux and aux_weight > 0:
            t4_logits, t1_logits, t2_logits, t3_logits = model.forward_with_aux(
                batch_av, batch_mask, batch_dass)
            loss = criterion(t4_logits, batch_labels)
            loss = loss + aux_weight * (
                aux_criterion(t1_logits, batch_aux[:, 0]) +
                aux_criterion(t2_logits, batch_aux[:, 1]) +
                aux_criterion(t3_logits, batch_aux[:, 2])
            ) / 3.0
        else:
            logits = model(batch_av, batch_mask, batch_dass)
            loss = criterion(logits, batch_labels)

        loss.backward()
        optimizer.step()


def _evaluate_dass(model, dataloader, criterion, device, num_classes):
    model.eval()
    losses, all_labels, all_probabilities = [], [], []
    with torch.no_grad():
        for batch in dataloader:
            batch_av, batch_mask, batch_dass, batch_labels = batch[:4]
            batch_av = batch_av.to(device)
            batch_mask = batch_mask.to(device)
            batch_dass = batch_dass.to(device)
            batch_labels = batch_labels.to(device)
            logits = model(batch_av, batch_mask, batch_dass)
            loss = criterion(logits, batch_labels)
            probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
            losses.append(float(loss.item()))
            all_labels.append(batch_labels.cpu().numpy())
            all_probabilities.append(probabilities)
    labels = np.concatenate(all_labels, axis=0)
    probs = np.concatenate(all_probabilities, axis=0) if all_probabilities else np.zeros((0, num_classes), dtype=np.float32)
    preds = probs.argmax(axis=1) if len(probs) else np.zeros(0, dtype=np.int64)
    metrics = _classification_metrics(labels, preds)
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    return metrics, probs


# ---------------------------------------------------------------------------
# Test predictions (AV-only; DASS not available for test subjects)
# ---------------------------------------------------------------------------


def _write_dass_test_predictions(
    baseline_config, dass_config, dataset_root, output_dir,
    fold_states, label_mapping, input_dim, device,
):
    test_path = dataset_root / "test" / "subjects.csv"
    if not test_path.exists() or not fold_states:
        return
    test_frame = pd.read_csv(test_path)
    if test_frame.empty:
        return
    test_frame[baseline_config.subject_id_column] = test_frame[
        ["anon_school", "anon_class", "anon_person"]
    ].agg("/".join, axis=1)

    use_multi = (
        dass_config.audio_features is not None
        and dass_config.video_features is not None
    )
    if use_multi:
        multi_cache = _multi_cache_path(
            dataset_root, dass_config.audio_features, dass_config.video_features,
            baseline_config.target_label_column, "test",
        )
        if baseline_config.feature_cache and multi_cache.exists():
            cached = np.load(multi_cache, allow_pickle=True, mmap_mode="r")
            test_av = cached["features"].astype(np.float32)
            test_mask = cached["clip_mask"].astype(bool)
            actual_input_dim = int(cached["input_dim"])
        else:
            test_av, test_mask, actual_input_dim = _build_multi_features(
                dataset_root, "test", test_frame,
                dass_config.audio_features, dass_config.video_features,
            )
            if baseline_config.feature_cache:
                multi_cache.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    multi_cache,
                    features=test_av, clip_mask=test_mask,
                    input_dim=np.asarray(actual_input_dim, dtype=np.int64),
                )
    else:
        cache_path = _release_cache_path(dataset_root, baseline_config, split="test")
        if baseline_config.feature_cache and cache_path.exists():
            cached = np.load(cache_path, allow_pickle=True, mmap_mode="r")
            cached_subjects = cached["subjects"].astype(str).tolist()
            if cached_subjects == test_frame[baseline_config.subject_id_column].astype(str).tolist():
                test_av = cached["features"].astype(np.float32)
                test_mask = cached["clip_mask"].astype(bool)
            else:
                test_av, test_mask, _ = _build_release_features(
                    dataset_root, "test", test_frame,
                    baseline_config.audio_feature_name, baseline_config.video_feature_name,
                )
        else:
            test_av, test_mask, _ = _build_release_features(
                dataset_root, "test", test_frame,
                baseline_config.audio_feature_name, baseline_config.video_feature_name,
            )
            if baseline_config.feature_cache:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    cache_path,
                    subjects=test_frame[baseline_config.subject_id_column].astype(str).to_numpy(dtype=object),
                    features=test_av, clip_mask=test_mask,
                    input_dim=np.asarray(input_dim, dtype=np.int64),
                )

    # Test subjects have no DASS history → use zeros
    dass_dim = {"none": 0, "scores_a": 3, "scores_das": 9, "encoder": 18}[dass_config.dass_scheme]
    test_dass = np.zeros((len(test_frame), dass_dim), dtype=np.float32)

    num_classes = len(label_mapping)
    probabilities = []
    for state in fold_states:
        scaler = (np.asarray(state["scaler_mean"], dtype=np.float32),
                  np.asarray(state["scaler_std"], dtype=np.float32))
        scaled_av = _apply_scaler(test_av, scaler)
        dataset = DASSDataset(scaled_av, test_mask, test_dass, np.zeros(len(test_frame), dtype=np.int64))
        loader = DataLoader(dataset, batch_size=baseline_config.batch_size, shuffle=False, num_workers=0)

        model = DASSAnxietyModel(
            input_dim=input_dim, num_classes=num_classes,
            hidden_dim=baseline_config.hidden_dim,
            temporal_hidden_dim=baseline_config.temporal_hidden_dim,
            dropout=baseline_config.dropout, dass_config=dass_config,
        ).to(device)
        model.load_state_dict(state["model"])
        model.eval()
        fold_probs = []
        with torch.no_grad():
            for batch in loader:
                batch_av, batch_mask, batch_dass = batch[:3]
                batch_av = batch_av.to(device)
                batch_mask = batch_mask.to(device)
                batch_dass = batch_dass.to(device)
                logits = model(batch_av, batch_mask, batch_dass)
                fold_probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
        probabilities.append(np.concatenate(fold_probs, axis=0))

    ensemble = np.mean(np.stack(probabilities, axis=0), axis=0)
    label_by_index = {i: label for label, i in label_mapping.items()}
    pred_index = ensemble.argmax(axis=1)
    pred_label = [label_by_index[int(i)] for i in pred_index]
    output = test_frame[["anon_school", "anon_class", "anon_person", baseline_config.subject_id_column]].copy()
    output["pred_label"] = pred_label
    for ci in range(num_classes):
        output[f"prob_class_{ci}"] = ensemble[:, ci]
    output.to_csv(output_dir / "test_predictions.csv", index=False, encoding="utf-8")
