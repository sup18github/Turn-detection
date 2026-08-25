import torch
import torch.nn as nn

class MeanPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, feature_dim)
        return torch.mean(x, dim=1)


class Conv1DTemporal(nn.Module):
    def __init__(self, in_channels: int = 384, out_channels: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, feature_dim) -> transpose to (batch_size, feature_dim, seq_len)
        x = x.transpose(1, 2)
        out = self.conv(x).squeeze(-1)
        return out


class GRUTemporal(nn.Module):
    def __init__(self, input_dim: int = 384, hidden_dim: int = 64, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, feature_dim)
        out, h_n = self.gru(x)
        # Return final hidden state
        return h_n[-1]
