import torch
import torch.nn as nn
from src.models.temporal import MeanPooling, GRUTemporal, Conv1DTemporal

class WhisperTurnClassifier(nn.Module):
    """
    Baseline 3: Whisper Tiny Encoder Classifier
    Uses Whisper Tiny audio encoder (384-dim hidden representations)
    pooled via Mean/GRU/Conv1D and passed to MLP classification head.
    """
    def __init__(
        self,
        whisper_encoder: nn.Module = None,
        feature_dim: int = 384,
        temporal_type: str = "mean",
        hidden_dims: list = [128, 64],
        dropout: float = 0.2,
        freeze_encoder: bool = True
    ):
        super().__init__()
        self.whisper_encoder = whisper_encoder
        self.freeze_encoder = freeze_encoder

        if self.whisper_encoder is not None and freeze_encoder:
            for param in self.whisper_encoder.parameters():
                param.requires_grad = False

        if temporal_type == "gru":
            self.temporal = GRUTemporal(input_dim=feature_dim, hidden_dim=64)
            temporal_out_dim = 64
        elif temporal_type == "conv":
            self.temporal = Conv1DTemporal(in_channels=feature_dim, out_channels=128)
            temporal_out_dim = 128
        else:
            self.temporal = MeanPooling()
            temporal_out_dim = feature_dim

        layers = []
        prev_dim = temporal_out_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        self.classifier = nn.Sequential(*layers)

    def forward(self, encoder_features: torch.Tensor, pause_features: torch.Tensor = None) -> torch.Tensor:
        # encoder_features shape: (batch_size, seq_len, 384)
        # pause_features is accepted but ignored (used only for API compatibility with hybrid seq mode)
        seq_rep = self.temporal(encoder_features)
        prob = self.classifier(seq_rep).squeeze(-1)
        return prob
