"""
Codabench-Compatible Model for Neural Forecasting.

This model is designed to be compatible with the Codabench evaluation pipeline.
It loads pre-trained weights and provides predict() and load() methods.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import math
from typing import Tuple, Optional


device = "cuda" if torch.cuda.is_available() else "cpu"


class InstanceNorm(nn.Module):
    """Instance normalization for handling session drift."""

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True) + self.eps
        return (x - mean) / std


class TemporalConvBlock(nn.Module):
    """Temporal convolution block with dilated convolutions."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2

        self.conv1 = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )
        self.conv2 = nn.Conv1d(
            out_channels, out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )
        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)

        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_conv = x.transpose(1, 2)
        residual = self.residual(x_conv)

        out = self.conv1(x_conv)
        out = out.transpose(1, 2)
        out = self.norm1(out)
        out = F.gelu(out)
        out = self.dropout(out)

        out = out.transpose(1, 2)
        out = self.conv2(out)

        out = out + residual
        out = out.transpose(1, 2)
        out = self.norm2(out)
        out = F.gelu(out)

        return out


class ChannelAttention(nn.Module):
    """Channel (spatial) attention for electrode dependencies."""

    def __init__(self, n_channels: int, d_model: int, n_heads: int = 4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, D = x.shape
        x_flat = x.reshape(B * T, C, D)

        Q = self.q_proj(x_flat)
        K = self.k_proj(x_flat)
        V = self.v_proj(x_flat)

        Q = Q.view(B * T, C, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(B * T, C, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(B * T, C, self.n_heads, self.head_dim).transpose(1, 2)

        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).reshape(B * T, C, D)
        out = self.out_proj(out)

        out = x_flat + self.dropout(out)
        out = self.norm(out)

        return out.view(B, T, C, D)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with pre-norm."""

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm, attn_mask=mask)
        x = x + self.dropout(attn_out)

        x_norm = self.norm2(x)
        ff_out = self.ff(x_norm)
        x = x + ff_out

        return x


class NeuralForecaster(nn.Module):
    """Neural Forecasting Model for μECoG signals."""

    def __init__(
        self,
        n_channels: int,
        n_features: int = 9,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.1,
        use_channel_attention: bool = True,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_features = n_features
        self.d_model = d_model
        self.use_channel_attention = use_channel_attention

        self.instance_norm = InstanceNorm()
        self.input_proj = nn.Linear(n_features, d_model)

        self.temporal_conv = nn.Sequential(
            TemporalConvBlock(d_model, d_model, kernel_size=3, dilation=1, dropout=dropout),
            TemporalConvBlock(d_model, d_model, kernel_size=3, dilation=2, dropout=dropout),
        )

        if use_channel_attention:
            self.channel_attn = ChannelAttention(n_channels, d_model, n_heads)

        self.pos_encoding = PositionalEncoding(d_model, max_len=30)

        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 10),
        )

        self.input_decoder = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, C, F = x.shape

        x = self.instance_norm(x)
        x = self.input_proj(x)

        if self.use_channel_attention:
            x = self.channel_attn(x)

        x = x.permute(0, 2, 1, 3)
        x = x.reshape(B * C, T, self.d_model)

        x = self.temporal_conv(x)
        x = self.pos_encoding(x)

        for layer in self.transformer_layers:
            x = layer(x)

        x = x.view(B, C, T, self.d_model)

        x_pooled = x.mean(dim=2)

        pred_future = self.decoder(x_pooled)
        pred_future = pred_future.transpose(1, 2)

        x_time = x.permute(0, 2, 1, 3)
        pred_input = self.input_decoder(x_time).squeeze(-1)

        return pred_future, pred_input


class Model(nn.Module):
    """
    Codabench-compatible wrapper for NeuralForecaster.

    Provides the interface expected by ingestion.py:
    - __init__(monkey_name): Initialize model for specific monkey
    - load(): Load pre-trained weights
    - predict(X): Make predictions
    """

    def __init__(self, monkey_name: str = ""):
        super().__init__()
        self.monkey_name = monkey_name

        if self.monkey_name == 'beignet':
            self.n_channels = 89
        elif self.monkey_name == 'affi':
            self.n_channels = 239
        else:
            raise ValueError(f'No such a monkey: {self.monkey_name}')

        # Initialize the neural forecaster
        # Must match training configuration: d_model=96, n_layers=2
        self.forecaster = NeuralForecaster(
            n_channels=self.n_channels,
            n_features=9,
            d_model=96,
            n_heads=4,
            n_layers=2,
            d_ff=256,
            dropout=0.15,
            use_channel_attention=True,
        )

        # Load normalization statistics
        self.norm_stats = None

    def load(self):
        """Load pre-trained model weights."""
        base = os.path.dirname(__file__)

        # Load model weights
        if self.monkey_name == 'beignet':
            model_path = os.path.join(base, "model_beignet.pth")
            stats_path = os.path.join(base, "norm_stats_beignet.npz")
        elif self.monkey_name == 'affi':
            model_path = os.path.join(base, "model_affi.pth")
            stats_path = os.path.join(base, "norm_stats_affi.npz")
        else:
            raise ValueError(f'No such a monkey: {self.monkey_name}')

        # Load model state
        state_dict = torch.load(
            model_path,
            map_location=torch.device(device),
            weights_only=True,
        )
        self.forecaster.load_state_dict(state_dict)
        self.forecaster.to(device)
        self.forecaster.eval()

        # Load normalization statistics
        if os.path.exists(stats_path):
            stats = np.load(stats_path)
            self.norm_stats = {
                'mean': stats['mean'],
                'std': stats['std'],
            }

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        """Apply normalization."""
        if self.norm_stats is not None:
            mean = self.norm_stats['mean'][0, 0]
            std = self.norm_stats['std'][0, 0]
            return (x - mean) / std
        return x

    def _denormalize(self, x: np.ndarray) -> np.ndarray:
        """Remove normalization."""
        if self.norm_stats is not None:
            mean = self.norm_stats['mean'][0, 0, :, 0]
            std = self.norm_stats['std'][0, 0, :, 0]
            return x * std + mean
        return x

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Input array of shape (N, 20, C, 9)
               First 10 timesteps have data, last 10 are masked

        Returns:
            predictions: Array of shape (N, 20, C)
                        Contains input values for t=0:10 and predictions for t=10:20
        """
        # Input shape: (N, 20, C, F)
        N, T, C, F = X.shape

        # Get the first 10 timesteps as input
        input_data = X[:, :10, :, :]  # (N, 10, C, F)

        # Normalize
        input_data = self._normalize(input_data)

        # Convert to tensor
        input_tensor = torch.tensor(input_data, dtype=torch.float32, device=device)

        # Make predictions
        predictions = []
        self.forecaster.eval()

        with torch.no_grad():
            # Process in batches to handle large inputs
            batch_size = 64
            for i in range(0, N, batch_size):
                batch = input_tensor[i:i+batch_size]

                # Forward pass
                pred_future, pred_input = self.forecaster(batch)

                # pred_future: (B, 10, C) - future predictions
                # pred_input: (B, 10, C) - reconstructed input

                # Combine input reconstruction and future prediction
                full_pred = torch.cat([pred_input, pred_future], dim=1)  # (B, 20, C)
                predictions.append(full_pred.cpu().numpy())

        # Concatenate all predictions
        predictions = np.concatenate(predictions, axis=0)

        # Denormalize
        predictions = self._denormalize(predictions)

        return predictions
