from __future__ import annotations

import torch
from torch import nn


class HistoryEncoder(nn.Module):
    """Encode T1/T2/T3 DASS scores and level embeddings."""

    def __init__(
        self,
        score_dim: int,
        n_level_slots: int,
        level_vocab_size: int = 5,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.level_embed = nn.Embedding(level_vocab_size, 16)
        level_in = n_level_slots * 16
        self.score_mlp = nn.Sequential(
            nn.Linear(score_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.level_mlp = nn.Sequential(
            nn.Linear(level_in, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        ) if n_level_slots > 0 else None
        fusion_in = hidden_dim + (hidden_dim // 2 if n_level_slots > 0 else 0)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, history_scores: torch.Tensor, history_levels: torch.Tensor) -> torch.Tensor:
        score_emb = self.score_mlp(history_scores)
        if self.level_mlp is not None and history_levels.numel() > 0:
            level_emb = self.level_embed(history_levels.clamp(min=0, max=4))
            level_flat = level_emb.reshape(history_levels.shape[0], -1)
            level_repr = self.level_mlp(level_flat)
            fused = torch.cat([score_emb, level_repr], dim=-1)
        else:
            fused = score_emb
        return self.fusion(fused)
