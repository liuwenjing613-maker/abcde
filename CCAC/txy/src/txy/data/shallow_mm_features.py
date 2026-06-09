from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from txy.constants import CLIP_TYPES, STAGES


@dataclass
class ShallowMMConfig:
    audio_pca_dim: int = 128
    video_pca_dim: int = 128
    seed: int = 42


class ShallowMultimodalFeatureBuilder:
    """Participant-level summary stats from [N, 3 stages, 4 clips, D] tensors."""

    def __init__(self, audio_pca: PCA | None = None, video_pca: PCA | None = None):
        self.audio_pca = audio_pca
        self.video_pca = video_pca
        self.feature_names: list[str] = []

    def _stage_stats(self, tensor: np.ndarray, mask: np.ndarray, prefix: str) -> dict[str, float]:
        # tensor [n_clips, D], mask [n_clips]
        feats: dict[str, float] = {}
        if not mask.any():
            for name in ["mean", "std", "max"]:
                feats[f"{prefix}_{name}"] = 0.0
            return feats
        valid = tensor[mask]
        feats[f"{prefix}_mean"] = float(valid.mean())
        feats[f"{prefix}_std"] = float(valid.std())
        feats[f"{prefix}_max"] = float(valid.max())
        return feats

    def transform(
        self,
        audio: np.ndarray,
        video: np.ndarray,
        clip_mask: np.ndarray,
        fit_pca: bool = False,
        config: ShallowMMConfig | None = None,
    ) -> np.ndarray:
        config = config or ShallowMMConfig()
        n = audio.shape[0]
        rows: list[list[float]] = []
        names: list[str] | None = None

        audio_flat = audio.reshape(n, -1, audio.shape[-1])
        video_flat = video.reshape(n, -1, video.shape[-1])
        mask_flat = clip_mask.reshape(n, -1)

        if fit_pca:
            valid_audio = audio_flat[mask_flat]
            valid_video = video_flat[mask_flat]
            self.audio_pca = PCA(
                n_components=min(config.audio_pca_dim, valid_audio.shape[1], max(1, valid_audio.shape[0] - 1)),
                random_state=config.seed,
            )
            self.video_pca = PCA(
                n_components=min(config.video_pca_dim, valid_video.shape[1], max(1, valid_video.shape[0] - 1)),
                random_state=config.seed,
            )
            self.audio_pca.fit(valid_audio)
            self.video_pca.fit(valid_video)

        assert self.audio_pca is not None and self.video_pca is not None

        for i in range(n):
            feats: dict[str, float] = {}
            a = audio[i]
            v = video[i]
            m = clip_mask[i]

            feats["clip_missing_count"] = float((~m).sum())
            feats.update(self._stage_stats(a.reshape(-1, a.shape[-1]), m.reshape(-1), "audio_all"))
            feats.update(self._stage_stats(v.reshape(-1, v.shape[-1]), m.reshape(-1), "video_all"))

            for stage_idx, stage in enumerate(STAGES):
                feats.update(self._stage_stats(a[stage_idx], m[stage_idx], f"audio_{stage.lower()}"))
                feats.update(self._stage_stats(v[stage_idx], m[stage_idx], f"video_{stage.lower()}"))

            # T3 vs T1/T2 deltas (mean pooled per stage)
            def stage_mean(tensor, mask_stage):
                if not mask_stage.any():
                    return 0.0
                return float(tensor[mask_stage].mean())

            a_means = [stage_mean(a[s], m[s]) for s in range(len(STAGES))]
            v_means = [stage_mean(v[s], m[s]) for s in range(len(STAGES))]
            feats["audio_t3_t1"] = a_means[2] - a_means[0]
            feats["audio_t3_t2"] = a_means[2] - a_means[1]
            feats["video_t3_t1"] = v_means[2] - v_means[0]
            feats["video_t3_t2"] = v_means[2] - v_means[1]
            feats["audio_slope"] = (a_means[2] - a_means[0]) / 2.0
            feats["video_slope"] = (v_means[2] - v_means[0]) / 2.0

            # PCA projections on valid clips
            a_valid = audio_flat[i][mask_flat[i]]
            v_valid = video_flat[i][mask_flat[i]]
            if a_valid.shape[0] == 0:
                a_pca = np.zeros(self.audio_pca.n_components_, dtype=np.float32)
                v_pca = np.zeros(self.video_pca.n_components_, dtype=np.float32)
            else:
                a_pca = self.audio_pca.transform(a_valid.mean(axis=0, keepdims=True)).reshape(-1)
                v_pca = self.video_pca.transform(v_valid.mean(axis=0, keepdims=True)).reshape(-1)

            for j, val in enumerate(a_pca):
                feats[f"audio_pca_{j:03d}"] = float(val)
            for j, val in enumerate(v_pca):
                feats[f"video_pca_{j:03d}"] = float(val)

            if names is None:
                names = list(feats.keys())
                self.feature_names = names
            rows.append([feats[k] for k in names])

        return np.asarray(rows, dtype=np.float32)
