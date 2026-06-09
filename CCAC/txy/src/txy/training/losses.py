from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def build_criterion(
    num_classes: int,
    labels: torch.Tensor | None = None,
    class_weight_power: float = 1.0,
    label_smoothing: float = 0.0,
    device: torch.device | None = None,
) -> nn.Module:
    weight = None
    if class_weight_power > 0 and labels is not None:
        counts = torch.bincount(labels, minlength=num_classes).float()
        counts = torch.where(counts == 0, torch.ones_like(counts), counts)
        weight = counts.sum() / (num_classes * counts)
        weight = torch.pow(weight, class_weight_power)
        weight = weight / weight.mean()
        if device is not None:
            weight = weight.to(device)
    return nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)


class OrdinalLoss(nn.Module):
    """BCE on cumulative ordinal targets for 5-class anxiety levels."""

    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.num_thresholds = num_classes - 1
        self.criterion = nn.BCEWithLogitsLoss()

    @staticmethod
    def labels_to_targets(labels: torch.Tensor, num_thresholds: int) -> torch.Tensor:
        targets = torch.zeros(labels.shape[0], num_thresholds, device=labels.device, dtype=torch.float32)
        for threshold_idx in range(num_thresholds):
            targets[:, threshold_idx] = (labels > threshold_idx).float()
        return targets

    def forward(self, threshold_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        targets = self.labels_to_targets(labels, self.num_thresholds)
        return self.criterion(threshold_logits, targets)


def kl_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    """KL(softmax(teacher/T) || softmax(student/T)) * T^2."""
    temp = max(float(temperature), 1e-6)
    student_log_probs = F.log_softmax(student_logits / temp, dim=-1)
    teacher_probs = F.softmax(teacher_logits.detach() / temp, dim=-1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temp * temp)
