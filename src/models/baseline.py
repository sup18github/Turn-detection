import torch
import torch.nn as nn
import numpy as np

class PauseThresholdClassifier:
    """
    Baseline 0: Fixed Silence Threshold Rule-Based Model
    If silence duration > threshold_ms -> END (1) else CONTINUE (0)
    """
    def __init__(self, threshold_ms: float = 700.0):
        self.threshold_ms = threshold_ms

    def predict(self, current_silence_ms: float) -> int:
        return 1 if current_silence_ms >= self.threshold_ms else 0

    def predict_prob(self, current_silence_ms: float) -> float:
        # Smooth sigmoid around threshold
        diff = (current_silence_ms - self.threshold_ms) / 100.0
        return float(1.0 / (1.0 + np.exp(-diff)))


class VADPauseMLP(nn.Module):
    """
    Baseline 1: VAD + Pause Features MLP Classifier
    Inputs 6 pause/timing features -> outputs P(END)
    """
    def __init__(self, in_features: int = 6, hidden_dims: list = [32, 16], dropout: float = 0.1):
        super().__init__()
        layers = []
        prev_dim = in_features
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
