import torch
import torch.nn as nn

class AcousticMLP(nn.Module):
    """
    Baseline 2: Acoustic MLP Classifier
    Takes 32-dimensional acoustic feature vector (MFCCs, RMS, spectral centroid, rolloff, ZCR, energy slope)
    Outputs P(END).
    """
    def __init__(self, in_features: int = 32, hidden_dims: list = [128, 64, 32], dropout: float = 0.2):
        super().__init__()
        layers = []
        prev_dim = in_features
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
            return self.net(x).squeeze(0).squeeze(-1)
        return self.net(x).squeeze(-1)

