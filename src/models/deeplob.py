"""
DeepLOB & Spatial-Temporal Transformer PyTorch Architectures.

Based on Zhang et al. (2019): "DeepLOB: Deep Convolutional Neural Networks for Limit Order Books".
Combines 2D Convolutional layers, Inception/ResNet spatial feature modules, and
LSTM / Multi-Head Self-Attention temporal layers for high-frequency mid-price direction forecasting.
"""

import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class InceptionModule(nn.Module):
    """Multi-scale spatial-temporal Inception block for LOB feature extraction."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        branch_channels = out_channels // 4

        self.branch1x1 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1),
            nn.BatchNorm2d(branch_channels),
            nn.LeakyReLU(0.01),
        )

        self.branch3x1 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1),
            nn.BatchNorm2d(branch_channels),
            nn.LeakyReLU(0.01),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(branch_channels),
            nn.LeakyReLU(0.01),
        )

        self.branch5x1 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1),
            nn.BatchNorm2d(branch_channels),
            nn.LeakyReLU(0.01),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=(5, 1), padding=(2, 0)),
            nn.BatchNorm2d(branch_channels),
            nn.LeakyReLU(0.01),
        )

        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 1), stride=1, padding=(1, 0)),
            nn.Conv2d(in_channels, branch_channels, kernel_size=1),
            nn.BatchNorm2d(branch_channels),
            nn.LeakyReLU(0.01),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1 = self.branch1x1(x)
        out2 = self.branch3x1(x)
        out3 = self.branch5x1(x)
        out4 = self.branch_pool(x)
        return torch.cat([out1, out2, out3, out4], dim=1)


class DeepLOB(nn.Module):
    """
    DeepLOB PyTorch Neural Network Architecture.

    Input shape: (Batch_Size, 1, Sequence_Length T, Num_Features F)
    Default T = 100, F = 40 (10 levels of ask/bid prices and volumes).
    Output shape: (Batch_Size, Num_Classes) where Num_Classes = 3 (DOWN: 0, STATIONARY: 1, UP: 2).
    """

    def __init__(
        self,
        in_features: int = 40,
        seq_len: int = 100,
        num_classes: int = 3,
        lstm_hidden_size: int = 64,
        lstm_num_layers: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.seq_len = seq_len
        self.num_classes = num_classes

        # Spatial Convolutional Stage 1 (extract relationships across adjacent levels)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 2), stride=(1, 2)),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.01),
            nn.Conv2d(16, 16, kernel_size=(4, 1), padding=(1, 0)),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.01),
            nn.Conv2d(16, 16, kernel_size=(4, 1), padding=(1, 0)),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.01),
        )
        self.drop1 = nn.Dropout2d(dropout * 0.5)  # Light spatial dropout after stage 1

        # Spatial Convolutional Stage 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(1, 2), stride=(1, 2)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 32, kernel_size=(4, 1), padding=(1, 0)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 32, kernel_size=(4, 1), padding=(1, 0)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.01),
        )
        self.drop2 = nn.Dropout2d(dropout * 0.5)  # Light spatial dropout after stage 2

        # Spatial Convolutional Stage 3
        conv3_in_width = in_features // 4
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, conv3_in_width)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01),
        )

        # Inception Multi-Scale Spatial-Temporal Blocks
        self.inception1 = InceptionModule(64, 64)
        self.inception2 = InceptionModule(64, 64)
        self.drop_inception = nn.Dropout2d(dropout)  # Heavier dropout after inception

        # Temporal Sequence Processing (LSTM)
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0.0,
        )

        # Fully Connected Classification Head
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, T, F) or (B, 1, T, F).

        Returns:
            Logits of shape (B, num_classes).
        """
        if x.dim() == 3:
            # Reshape (B, T, F) to (B, 1, T, F)
            x = x.unsqueeze(1)

        # Spatial Conv Stages with L2-friendly dropout
        x = self.drop1(self.conv1(x))
        x = self.drop2(self.conv2(x))
        x = self.conv3(x)  # Shape: (B, 64, T, 1)

        # Inception Blocks
        x = self.inception1(x)
        x = self.drop_inception(self.inception2(x))  # Shape: (B, 64, T, 1)

        # Prepare for LSTM: (B, 64, T, 1) -> (B, T, 64)
        x = x.squeeze(-1).permute(0, 2, 1)

        # LSTM Temporal Dynamics
        lstm_out, _ = self.lstm(x)  # (B, T, lstm_hidden_size)

        # Take last time step
        last_step = lstm_out[:, -1, :]

        # Classification Head
        logits = self.fc(last_step)
        return logits


class SpatialTemporalTransformerLOB(nn.Module):
    """
    Spatial-Temporal Multi-Head Attention Transformer for LOB forecasting.
    Replaces RNNs with multi-head self-attention mechanisms.
    """

    def __init__(
        self,
        in_features: int = 40,
        seq_len: int = 100,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.1,
        pooling: str = "mean",
    ) -> None:
        super().__init__()
        self.pooling = pooling
        self.input_projection = nn.Linear(in_features, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, seq_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.squeeze(1)  # (B, 1, T, F) -> (B, T, F)

        h = self.input_projection(x) + self.pos_encoder[:, : x.size(1), :]
        out = self.transformer_encoder(h)
        # "last": final time step (most recent tick); "mean": average over time
        pooled = out[:, -1, :] if self.pooling == "last" else out.mean(dim=1)
        return self.classifier(pooled)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer architectures."""

    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x shape: (batch_size, seq_len, d_model)"""
        return x + self.pe[:, :x.size(1), :]


class ResNetBlock1D(nn.Module):
    """1D Residual Block for spatial feature extraction over LOB sequences."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)
        return out


class TransLOB(nn.Module):
    """
    TransLOB: ResNet-1D Spatial Extractor + Transformer Encoder for LOB forecasting.
    Replaces LSTM sequential processing with parallel self-attention.
    """

    def __init__(self, in_features: int = 61, d_model: int = 64, num_heads: int = 4, num_layers: int = 3, dropout: float = 0.2, pooling: str = "mean"):
        super().__init__()
        self.pooling = pooling
        self.resnet = nn.Sequential(
            ResNetBlock1D(in_features, d_model, dropout),
            ResNetBlock1D(d_model, d_model, dropout)
        )

        self.pos_encoder = PositionalEncoding(d_model=d_model)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        self.fc1 = nn.Linear(d_model, 32)
        self.relu = nn.ReLU()
        self.dropout_fc = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 3)

    def forward(self, x):
        x = x.squeeze(1)
        x = x.permute(0, 2, 1)
        x = self.resnet(x)
        x = x.permute(0, 2, 1)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x[:, -1, :] if self.pooling == "last" else x.mean(dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout_fc(x)
        logits = self.fc2(x)
        return logits
