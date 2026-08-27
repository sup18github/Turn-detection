import torch
import torch.nn as nn
from src.models.temporal import GRUTemporal, MeanPooling, Conv1DTemporal

class HybridTurnModel(nn.Module):
    """
    Proposed Hybrid Model Architecture:
    Combines:
    1. Whisper Tiny Encoder audio representation (384-dim)
    2. Temporal sequence modeling (GRU / Conv1D)
    3. Explicit pause & acoustic VAD timing features (6-dim)
    4. Feature Fusion MLP Head -> P(END)
    """
    def __init__(
        self,
        whisper_encoder: nn.Module = None,
        audio_feature_dim: int = 384,
        pause_feature_dim: int = 6,
        temporal_type: str = "gru",
        gru_hidden_dim: int = 64,
        fusion_hidden_dims: list = [128, 64],
        dropout: float = 0.2,
        freeze_encoder: bool = True
    ):
        super().__init__()
        self.whisper_encoder = whisper_encoder

        if self.whisper_encoder is not None and freeze_encoder:
            for param in self.whisper_encoder.parameters():
                param.requires_grad = False

        if temporal_type == "gru":
            self.temporal = GRUTemporal(input_dim=audio_feature_dim, hidden_dim=gru_hidden_dim)
            audio_out_dim = gru_hidden_dim
        elif temporal_type == "conv":
            self.temporal = Conv1DTemporal(in_channels=audio_feature_dim, out_channels=128)
            audio_out_dim = 128
        else:
            self.temporal = MeanPooling()
            audio_out_dim = audio_feature_dim

        fusion_in_dim = audio_out_dim + pause_feature_dim

        layers = []
        prev_dim = fusion_in_dim
        for h in fusion_hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        self.fusion_mlp = nn.Sequential(*layers)

    def forward(self, encoder_features: torch.Tensor, pause_features: torch.Tensor) -> torch.Tensor:
        unbatched = False
        if encoder_features.dim() == 2:
            encoder_features = encoder_features.unsqueeze(0)
            unbatched = True
        if pause_features.dim() == 1:
            pause_features = pause_features.unsqueeze(0)

        # encoder_features: (batch_size, seq_len, 384)
        # pause_features:   (batch_size, 6)
        audio_rep = self.temporal(encoder_features)
        fused = torch.cat([audio_rep, pause_features], dim=-1)
        prob = self.fusion_mlp(fused).squeeze(-1)
        if unbatched:
            return prob.squeeze(0)
        return prob

