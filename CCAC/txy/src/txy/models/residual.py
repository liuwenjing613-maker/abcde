from __future__ import annotations

import torch
from torch import nn

from txy.models.stagewise import StageWiseLongitudinalModel


class ResidualFusionModel(nn.Module):
    """
    logits_final = logits_tabular + alpha * logits_multimodal

    Migrated from AdoDAS anchor + motion residual fusion.
    """

    def __init__(
        self,
        tabular_dim: int,
        stagewise_model: StageWiseLongitudinalModel,
        num_classes: int,
        alpha: float = 0.25,
        tabular_hidden: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.alpha = alpha
        self.stagewise = stagewise_model
        self.tabular_head = nn.Sequential(
            nn.LayerNorm(tabular_dim),
            nn.Linear(tabular_dim, tabular_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(tabular_hidden, num_classes),
        )

    def forward(
        self,
        tabular_features: torch.Tensor,
        audio: torch.Tensor,
        video: torch.Tensor,
        clip_mask: torch.Tensor,
        history_scores: torch.Tensor,
        history_levels: torch.Tensor,
    ) -> torch.Tensor:
        logits_tab = self.tabular_head(tabular_features)
        logits_mm = self.stagewise(audio, video, clip_mask, history_scores, history_levels)
        return logits_tab + self.alpha * logits_mm
