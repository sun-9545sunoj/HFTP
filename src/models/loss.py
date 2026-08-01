"""
Custom Loss Functions for Imbalanced Directional Classification in LOB Mid-Price Modeling.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss (Lin et al., 2017) tailored for class-imbalanced LOB direction prediction.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        """
        Args:
            alpha: Class weights tensor of shape (num_classes,).
            gamma: Focusing parameter gamma >= 0 (default: 2.0).
            reduction: Loss reduction mode ('mean', 'sum', 'none').
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits tensor of shape (batch_size, num_classes).
            targets: Class indices tensor of shape (batch_size,).

        Returns:
            Scalar loss tensor.
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        p_t = torch.exp(-ce_loss)
        focal_loss = ((1.0 - p_t) ** self.gamma) * ce_loss

        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class ClassWeightedCrossEntropy(nn.Module):
    """Class-imbalance aware Cross-Entropy Loss with automatic inverse-frequency weighting option."""

    def __init__(
        self,
        weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.05,
    ) -> None:
        """
        Args:
            weights: Optional tensor of class weights.
            label_smoothing: Label smoothing factor (default: 0.05).
        """
        super().__init__()
        self.weights = weights
        self.label_smoothing = label_smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        weights = self.weights.to(inputs.device) if self.weights is not None else None
        return F.cross_entropy(
            inputs,
            targets,
            weight=weights,
            label_smoothing=self.label_smoothing,
        )
