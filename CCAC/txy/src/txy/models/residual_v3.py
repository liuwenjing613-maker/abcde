from __future__ import annotations

import torch
from torch import nn

from txy.models.stagewise import StageWiseLongitudinalModel


class ResidualV3Model(nn.Module):
    """
    Fixed external tabular logits + trainable multimodal branch.

    history available: logits_final = logits_tab + alpha_with_history * logits_mm
    history missing:   logits_final = logits_mm
    """

    def __init__(
        self,
        audio_dim: int,
        video_dim: int,
        num_classes: int = 5,
        hidden_dim: int = 256,
        temporal_hidden_dim: int = 192,
        dropout: float = 0.2,
        alpha_with_history: float = 0.25,
        alpha_missing_history: float = 1.0,
    ):
        super().__init__()
        self.alpha_with_history = alpha_with_history
        self.alpha_missing_history = alpha_missing_history
        self.stagewise = StageWiseLongitudinalModel(
            audio_dim=audio_dim,
            video_dim=video_dim,
            history_score_dim=1,
            history_level_slots=0,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            temporal_hidden_dim=temporal_hidden_dim,
            dropout=dropout,
            use_history=False,
        )

    def forward_mm(self, audio: torch.Tensor, video: torch.Tensor, clip_mask: torch.Tensor) -> torch.Tensor:
        return self.stagewise(audio, video, clip_mask, None, None)

    def fuse(
        self,
        logits_tab: torch.Tensor,
        logits_mm: torch.Tensor,
        history_available: torch.Tensor,
    ) -> torch.Tensor:
        alpha = torch.where(
            history_available,
            torch.full_like(history_available, self.alpha_with_history, dtype=logits_mm.dtype),
            torch.full_like(history_available, self.alpha_missing_history, dtype=logits_mm.dtype),
        ).unsqueeze(-1)
        fused_with_history = logits_tab + alpha * logits_mm
        return torch.where(history_available.unsqueeze(-1), fused_with_history, logits_mm)

    def forward(
        self,
        audio: torch.Tensor,
        video: torch.Tensor,
        clip_mask: torch.Tensor,
        logits_tab: torch.Tensor,
        history_available: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        logits_mm = self.forward_mm(audio, video, clip_mask)
        logits_final = self.fuse(logits_tab, logits_mm, history_available)
        return {
            "logits_final": logits_final,
            "logits_mm": logits_mm,
            "logits_tab": logits_tab,
        }
