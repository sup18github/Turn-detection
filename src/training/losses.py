import torch
import torch.nn as nn

class WeightedTurnLoss(nn.Module):
    """
    Weighted Binary Cross Entropy Loss designed to penalize Premature END predictions.
    False Early-End (predicting END = 1 when ground truth is CONTINUE = 0) is treated
    as a high-severity error in voice AI systems.
    """
    def __init__(self, pos_weight: float = 2.0, false_end_penalty: float = 1.5):
        super().__init__()
        self.pos_weight = pos_weight
        self.false_end_penalty = false_end_penalty
        self.bce = nn.BCELoss(reduction='none')

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = self.bce(preds, targets)
        # Apply asymmetric penalty: when target == 0 (CONTINUE) and pred is high (False END)
        weights = torch.ones_like(targets)
        weights[targets == 1] = self.pos_weight
        # Heavy penalty for False END (target == 0 but pred > 0.5)
        false_end_mask = (targets == 0) & (preds > 0.5)
        weights[false_end_mask] *= self.false_end_penalty

        weighted_loss = loss * weights
        return torch.mean(weighted_loss)
