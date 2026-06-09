from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from txy.constants import CLIP_TYPES, STAGES


def make_subject_id(frame: pd.DataFrame) -> pd.Series:
    return frame[["anon_school", "anon_class", "anon_person"]].agg("/".join, axis=1)


def infer_feature_dim(dataset_root: Path, split: str, feature_name: str) -> int:
    feature_root = dataset_root / split / feature_name
    if not feature_root.exists():
        raise ValueError(f"feature not found: {feature_root}")
    candidate = next(feature_root.rglob("pooled.npy"), None)
    if candidate is None:
        candidate = next(feature_root.rglob("pooled.json"), None)
    if candidate is None:
        raise ValueError(f"no pooled feature files under {feature_root}")
    return int(load_pooled_vector(candidate).shape[0])


def load_pooled_vector(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.asarray(np.load(path), dtype=np.float32).reshape(-1)
    values = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(values, dict):
        items = [values[key] for key in sorted(values)]
    elif isinstance(values, list):
        items = values
    else:
        raise ValueError(f"unsupported pooled feature JSON: {path}")
    return np.asarray([float(item) for item in items], dtype=np.float32).reshape(-1)


def _load_release_vector(
    dataset_root: Path,
    split: str,
    feature_name: str,
    subject_parts: tuple[str, str, str],
    stage: str,
    clip_type: str,
    target_dim: int,
) -> np.ndarray:
    base = (
        dataset_root
        / split
        / feature_name
        / subject_parts[0]
        / subject_parts[1]
        / subject_parts[2]
        / stage
        / clip_type
    )
    npy_path = base / "pooled.npy"
    json_path = base / "pooled.json"
    if npy_path.exists():
        values = load_pooled_vector(npy_path)
    elif json_path.exists():
        values = load_pooled_vector(json_path)
    else:
        values = np.zeros(target_dim, dtype=np.float32)
    vector = np.zeros(target_dim, dtype=np.float32)
    length = min(target_dim, values.shape[0])
    vector[:length] = values[:length]
    return vector


def build_multimodal_features(
    dataset_root: Path,
    split: str,
    frame: pd.DataFrame,
    audio_feature_name: str,
    video_feature_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Return audio [N,3,4,Da], video [N,3,4,Dv], clip_mask, fused [N,3,4,Da+Dv]."""
    audio_dim = infer_feature_dim(dataset_root, split, audio_feature_name)
    video_dim = infer_feature_dim(dataset_root, split, video_feature_name)
    n = len(frame)
    audio = np.zeros((n, len(STAGES), len(CLIP_TYPES), audio_dim), dtype=np.float32)
    video = np.zeros((n, len(STAGES), len(CLIP_TYPES), video_dim), dtype=np.float32)
    clip_mask = np.zeros((n, len(STAGES), len(CLIP_TYPES)), dtype=bool)

    for row_index, row in enumerate(frame.itertuples(index=False)):
        subject_parts = (row.anon_school, row.anon_class, row.anon_person)
        for stage_index, stage in enumerate(STAGES):
            for clip_index, clip_type in enumerate(CLIP_TYPES):
                a = _load_release_vector(
                    dataset_root, split, audio_feature_name, subject_parts, stage, clip_type, audio_dim
                )
                v = _load_release_vector(
                    dataset_root, split, video_feature_name, subject_parts, stage, clip_type, video_dim
                )
                audio[row_index, stage_index, clip_index] = a
                video[row_index, stage_index, clip_index] = v
                clip_mask[row_index, stage_index, clip_index] = bool(
                    np.isfinite(a).any() or np.isfinite(v).any()
                )

    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    video = np.nan_to_num(video, nan=0.0, posinf=0.0, neginf=0.0)
    fused = np.concatenate([audio, video], axis=-1)
    return audio, video, clip_mask, fused, audio_dim, video_dim


def cache_path(dataset_root: Path, split: str, audio_name: str, video_name: str) -> Path:
    name = f"{audio_name}__{video_name}.npz"
    return dataset_root / "metadata" / "txy_cache" / split / name


def load_or_build_multimodal(
    dataset_root: Path,
    split: str,
    frame: pd.DataFrame,
    audio_feature_name: str,
    video_feature_name: str,
    use_cache: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    path = cache_path(dataset_root, split, audio_feature_name, video_feature_name)
    subjects = make_subject_id(frame).astype(str).tolist()
    if use_cache and path.exists():
        cached = np.load(path, allow_pickle=True)
        if cached["subjects"].astype(str).tolist() == subjects:
            return (
                cached["audio"].astype(np.float32),
                cached["video"].astype(np.float32),
                cached["clip_mask"].astype(bool),
                cached["fused"].astype(np.float32),
                int(cached["audio_dim"]),
                int(cached["video_dim"]),
            )

    audio, video, clip_mask, fused, audio_dim, video_dim = build_multimodal_features(
        dataset_root, split, frame, audio_feature_name, video_feature_name
    )
    if use_cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            subjects=np.asarray(subjects, dtype=object),
            audio=audio,
            video=video,
            clip_mask=clip_mask,
            fused=fused,
            audio_dim=np.asarray(audio_dim, dtype=np.int64),
            video_dim=np.asarray(video_dim, dtype=np.int64),
        )
    return audio, video, clip_mask, fused, audio_dim, video_dim
