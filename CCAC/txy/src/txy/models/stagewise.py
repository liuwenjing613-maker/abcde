from __future__ import annotations

import torch
from torch import nn

from txy.constants import STAGES
from txy.models.adapters import FeatureAdapter, GatedFusion
from txy.models.history_encoder import HistoryEncoder


class ClipAttentionPool(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, clip_reprs: torch.Tensor, clip_mask: torch.Tensor) -> torch.Tensor:
        logits = self.score(clip_reprs).squeeze(-1)
        logits = logits.masked_fill(~clip_mask, -1e9)
        weights = torch.softmax(logits, dim=-1)
        weights = weights * clip_mask.float()
        denom = weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        weights = weights / denom
        pooled = (clip_reprs * weights.unsqueeze(-1)).sum(dim=-2)
        missing = clip_mask.sum(dim=-1, keepdim=True) == 0
        return torch.where(missing, torch.zeros_like(pooled), pooled)


class StageWiseLongitudinalModel(nn.Module):
    """
    AdoDAS LabelWiseSession -> CCAC StageWiseLongitudinal migration.

    clip attention within each stage + temporal encoder across T1/T2/T3.
    """

    def __init__(
        self,
        audio_dim: int,
        video_dim: int,
        history_score_dim: int,
        history_level_slots: int,
        num_classes: int,
        hidden_dim: int = 256,
        temporal_hidden_dim: int = 192,
        dropout: float = 0.2,
        use_history: bool = True,
        history_hidden_dim: int = 128,
    ):
        super().__init__()
        self.use_history = use_history
        self.audio_adapter = FeatureAdapter(audio_dim, hidden_dim, dropout)
        self.video_adapter = FeatureAdapter(video_dim, hidden_dim, dropout)
        self.gated_fusion = GatedFusion(hidden_dim)
        self.clip_pool = ClipAttentionPool(hidden_dim)
        self.stage_position = nn.Parameter(torch.zeros(len(STAGES), hidden_dim))
        nn.init.normal_(self.stage_position, mean=0.0, std=0.02)

        self.temporal_encoder = nn.GRU(
            input_size=hidden_dim,
            hidden_size=temporal_hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        if use_history:
            self.history_encoder = HistoryEncoder(
                score_dim=history_score_dim,
                n_level_slots=history_level_slots,
                hidden_dim=history_hidden_dim,
                dropout=dropout,
            )
            fusion_dim = hidden_dim * 2 + temporal_hidden_dim * 2 + history_hidden_dim
        else:
            self.history_encoder = None
            fusion_dim = hidden_dim * 2 + temporal_hidden_dim * 2

        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def encode_multimodal(self, audio: torch.Tensor, video: torch.Tensor, clip_mask: torch.Tensor) -> torch.Tensor:
        # audio/video: [B, 3 stages, 4 clips, D]
        encoded_a = self.audio_adapter(audio)
        encoded_v = self.video_adapter(video)
        clip_reprs = self.gated_fusion(encoded_a, encoded_v)
        stage_repr = self.clip_pool(clip_reprs, clip_mask)
        stage_repr = stage_repr + self.stage_position.unsqueeze(0)
        temporal_out, _ = self.temporal_encoder(stage_repr)
        pooled_temporal = temporal_out.mean(dim=1)
        pooled_stage = stage_repr.mean(dim=1)
        final_stage = stage_repr[:, -1, :]
        return torch.cat([pooled_temporal, pooled_stage, final_stage], dim=-1)

    def forward(
        self,
        audio: torch.Tensor,
        video: torch.Tensor,
        clip_mask: torch.Tensor,
        history_scores: torch.Tensor | None = None,
        history_levels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mm_repr = self.encode_multimodal(audio, video, clip_mask)
        if self.use_history and self.history_encoder is not None:
            hist_repr = self.history_encoder(history_scores, history_levels)
            fused = torch.cat([mm_repr, hist_repr], dim=-1)
        else:
            fused = mm_repr
        return self.classifier(fused)
