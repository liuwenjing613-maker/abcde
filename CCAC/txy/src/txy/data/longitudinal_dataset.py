from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class PersonBatch:
    audio: torch.Tensor
    video: torch.Tensor
    fused: torch.Tensor
    clip_mask: torch.Tensor
    history_scores: torch.Tensor
    history_levels: torch.Tensor
    labels: torch.Tensor
    subject_ids: list[str]


class LongitudinalPersonDataset(Dataset):
    """Person-level dataset: stages x clips multimodal + history DASS."""

    def __init__(
        self,
        audio: np.ndarray,
        video: np.ndarray,
        fused: np.ndarray,
        clip_mask: np.ndarray,
        history_scores: np.ndarray,
        history_levels: np.ndarray,
        labels: np.ndarray,
        subject_ids: list[str],
        stage_drop_prob: float = 0.0,
        clip_drop_prob: float = 0.0,
        feature_noise_std: float = 0.0,
        train: bool = False,
    ):
        self.audio = torch.from_numpy(audio).float()
        self.video = torch.from_numpy(video).float()
        self.fused = torch.from_numpy(fused).float()
        self.clip_mask = torch.from_numpy(clip_mask).bool()
        self.history_scores = torch.from_numpy(history_scores).float()
        self.history_levels = torch.from_numpy(history_levels).long()
        self.labels = torch.from_numpy(labels).long()
        self.subject_ids = subject_ids
        self.stage_drop_prob = stage_drop_prob
        self.clip_drop_prob = clip_drop_prob
        self.feature_noise_std = feature_noise_std
        self.train = train

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        audio = self.audio[index].clone()
        video = self.video[index].clone()
        fused = self.fused[index].clone()
        clip_mask = self.clip_mask[index].clone()

        if self.train:
            if self.stage_drop_prob > 0:
                for stage_idx in range(clip_mask.shape[0]):
                    if torch.rand(1).item() < self.stage_drop_prob:
                        clip_mask[stage_idx] = False
                        fused[stage_idx] = 0.0
                        audio[stage_idx] = 0.0
                        video[stage_idx] = 0.0
            if self.clip_drop_prob > 0:
                for stage_idx in range(clip_mask.shape[0]):
                    for clip_idx in range(clip_mask.shape[1]):
                        if clip_mask[stage_idx, clip_idx] and torch.rand(1).item() < self.clip_drop_prob:
                            clip_mask[stage_idx, clip_idx] = False
                            fused[stage_idx, clip_idx] = 0.0
                            audio[stage_idx, clip_idx] = 0.0
                            video[stage_idx, clip_idx] = 0.0
            if self.feature_noise_std > 0:
                noise = torch.randn_like(fused) * self.feature_noise_std
                fused = fused + noise * clip_mask.unsqueeze(-1).float()
                audio = audio + torch.randn_like(audio) * self.feature_noise_std * clip_mask.unsqueeze(-1).float()
                video = video + torch.randn_like(video) * self.feature_noise_std * clip_mask.unsqueeze(-1).float()

        return {
            "audio": audio,
            "video": video,
            "fused": fused,
            "clip_mask": clip_mask,
            "history_scores": self.history_scores[index],
            "history_levels": self.history_levels[index],
            "label": self.labels[index],
            "subject_id": self.subject_ids[index],
        }


def collate_person_batch(items: list[dict]) -> PersonBatch:
    return PersonBatch(
        audio=torch.stack([item["audio"] for item in items], dim=0),
        video=torch.stack([item["video"] for item in items], dim=0),
        fused=torch.stack([item["fused"] for item in items], dim=0),
        clip_mask=torch.stack([item["clip_mask"] for item in items], dim=0),
        history_scores=torch.stack([item["history_scores"] for item in items], dim=0),
        history_levels=torch.stack([item["history_levels"] for item in items], dim=0),
        labels=torch.stack([item["label"] for item in items], dim=0),
        subject_ids=[str(item["subject_id"]) for item in items],
    )
