"""Loss functions for knowledge distillation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    """Combined hard-label BCE + soft-label distillation loss.

    L_total = alpha * L_hard + beta * L_soft

    L_hard = BCE(logits, hard_labels)
    L_soft = BCE(sigmoid(logits / T), teacher_probs)
    """

    def __init__(
        self,
        alpha_hard: float = 0.7,
        beta_soft: float = 0.3,
        temperature: float = 1.5,
    ):
        super().__init__()
        self.alpha = alpha_hard
        self.beta = beta_soft
        self.temperature = temperature

    def forward(
        self,
        logits: torch.Tensor,
        hard_labels: torch.Tensor,
        teacher_probs: torch.Tensor,
    ) -> dict:
        """Compute combined distillation loss.

        Args:
            logits: student model raw logits (B,)
            hard_labels: binary labels 0/1 (B,)
            teacher_probs: teacher soft probabilities (B,)

        Returns:
            dict with total_loss, hard_loss, soft_loss
        """
        # Hard label loss
        loss_hard = F.binary_cross_entropy_with_logits(logits, hard_labels)

        # Soft label loss (temperature-scaled)
        # Use logits form to be AMP-safe: BCE_with_logits(logits/T, teacher_probs)
        scaled_logits = logits / self.temperature
        loss_soft = F.binary_cross_entropy_with_logits(scaled_logits, teacher_probs)

        # Combined
        total = self.alpha * loss_hard + self.beta * loss_soft

        return {
            "total_loss": total,
            "hard_loss": loss_hard,
            "soft_loss": loss_soft,
        }
