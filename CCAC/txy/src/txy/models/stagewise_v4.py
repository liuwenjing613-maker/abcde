from __future__ import annotations

import torch
from torch import nn

from txy.models.ordinal import OrdinalAnxietyHead
from txy.models.stagewise import StageWiseLongitudinalModel


class StageWiseV4Model(nn.Module):
    """
    Multimodal-only student with ordinal auxiliary head.

    Training uses tabular teacher logits via distillation (outside this module).
    Inference uses logits_mm (+ optional class bias) only.
    """

    def __init__(
        self,
        audio_dim: int,
        video_dim: int,
        num_classes: int = 5,
        hidden_dim: int = 256,
        temporal_hidden_dim: int = 192,
        dropout: float = 0.2,
    ):
        super().__init__()
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
        fusion_dim = hidden_dim * 2 + temporal_hidden_dim * 2
        self.ordinal_head = OrdinalAnxietyHead(fusion_dim, num_thresholds=num_classes - 1, dropout=dropout)

    def forward(self, audio: torch.Tensor, video: torch.Tensor, clip_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        mm_repr = self.stagewise.encode_multimodal(audio, video, clip_mask)
        logits_mm = self.stagewise.classifier(mm_repr)
        ordinal_logits = self.ordinal_head(mm_repr)
        return {
            "logits_mm": logits_mm,
            "ordinal_logits": ordinal_logits,
        }
