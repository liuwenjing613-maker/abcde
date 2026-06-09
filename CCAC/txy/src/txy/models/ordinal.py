from __future__ import annotations

import torch
from torch import nn


class OrdinalAnxietyHead(nn.Module):
    """CORAL-style ordinal head: 4 binary thresholds for 5 anxiety levels."""

    def __init__(self, d_in: int, num_thresholds: int = 4, dropout: float = 0.2):
        super().__init__()
        self.num_thresholds = num_thresholds
        self.backbone = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, d_in // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.threshold_logits = nn.Linear(d_in // 2, num_thresholds)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(features)
        return self.threshold_logits(hidden)

    @staticmethod
    def thresholds_to_class_probs(threshold_logits: torch.Tensor) -> torch.Tensor:
        """Convert threshold logits to 5-class probabilities."""
        probs = torch.sigmoid(threshold_logits)
        batch_size = probs.shape[0]
        num_classes = probs.shape[1] + 1
        class_probs = torch.zeros(batch_size, num_classes, device=probs.device, dtype=probs.dtype)
        class_probs[:, 0] = 1.0 - probs[:, 0]
        for class_idx in range(1, num_classes - 1):
            class_probs[:, class_idx] = probs[:, class_idx - 1] - probs[:, class_idx]
        class_probs[:, -1] = probs[:, -1]
        class_probs = class_probs.clamp_min(0.0)
        denom = class_probs.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return class_probs / denom

    @staticmethod
    def thresholds_to_logits(threshold_logits: torch.Tensor) -> torch.Tensor:
        class_probs = OrdinalAnxietyHead.thresholds_to_class_probs(threshold_logits)
        return torch.log(class_probs.clamp_min(1e-6))
