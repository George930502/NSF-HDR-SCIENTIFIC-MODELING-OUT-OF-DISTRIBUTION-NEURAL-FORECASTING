"""
Codabench-Compatible Model for Neural Forecasting.

Improved version with:
1. Per-sample robust normalization (handles session drift)
2. Linear extrapolation baseline + residual learning
3. Compact architecture (71K parameters)
4. Focus on temporal dynamics (features 1-8 are not predictive)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import math
from typing import Tuple, Optional


device = "cuda" if torch.cuda.is_available() else "cpu"


class RobustNorm(nn.Module):
    """Robust per-sample normalization using median and IQR."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        if x.dim() == 4:
            B, T, C, F = x.shape
            x_flat = x.view(B, -1)
        else:
            B, T, C = x.shape
            x_flat = x.view(B, -1)

        median = x_flat.median(dim=1, keepdim=True)[0]
        q75 = torch.quantile(x_flat, 0.75, dim=1, keepdim=True)
        q25 = torch.quantile(x_flat, 0.25, dim=1, keepdim=True)
        iqr = (q75 - q25).clamp(min=self.eps)

        x_norm = (x_flat - median) / iqr

        if x.dim() == 4:
            x_norm = x_norm.view(B, T, C, F)
        else:
            x_norm = x_norm.view(B, T, C)

        return x_norm, {'median': median, 'iqr': iqr}

    def denormalize(self, x: torch.Tensor, stats: dict) -> torch.Tensor:
        B = x.shape[0]
        median = stats['median'].view(B, 1, 1)
        iqr = stats['iqr'].view(B, 1, 1)
        return x * iqr + median


class LinearExtrapolation(nn.Module):
    """Linear extrapolation baseline."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        t = torch.arange(T, device=x.device, dtype=x.dtype)
        t = t.view(1, T, 1).expand(B, T, C)

        t_mean = t.mean(dim=1, keepdim=True)
        x_mean = x.mean(dim=1, keepdim=True)

        numerator = ((t - t_mean) * (x - x_mean)).sum(dim=1, keepdim=True)
        denominator = ((t - t_mean) ** 2).sum(dim=1, keepdim=True) + 1e-6
        slope = numerator / denominator

        intercept = x_mean - slope * t_mean

        t_future = torch.arange(T, 2*T, device=x.device, dtype=x.dtype)
        t_future = t_future.view(1, T, 1).expand(B, T, C)

        extrapolated = slope * t_future + intercept
        return extrapolated


class ResidualBlock(nn.Module):
    """Simple residual block with 1D convolution."""

    def __init__(self, channels: int, kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = x.transpose(1, 2)
        x = self.norm1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)
        x = self.conv2(x)
        x = x.transpose(1, 2)
        x = self.norm2(x)
        return x + residual


class ImprovedForecaster(nn.Module):
    """Improved neural forecaster."""

    def __init__(
        self,
        n_channels: int,
        d_model: int = 64,
        n_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.d_model = d_model

        self.robust_norm = RobustNorm()
        self.linear_extrap = LinearExtrapolation()
        self.input_proj = nn.Linear(1, d_model)

        self.temporal_blocks = nn.ModuleList([
            ResidualBlock(d_model, kernel_size=3, dropout=dropout)
            for _ in range(n_layers)
        ])

        self.time_attn = nn.MultiheadAttention(
            d_model, num_heads=4, dropout=dropout, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(d_model)

        self.residual_predictor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 10),
        )

        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        B, T, C, F = x.shape
        x_target = x[:, :, :, 0]
        x_norm, norm_stats = self.robust_norm(x_target)
        baseline = self.linear_extrap(x_norm)

        x_flat = x_norm.permute(0, 2, 1).reshape(B * C, T, 1)
        x_proj = self.input_proj(x_flat)

        for block in self.temporal_blocks:
            x_proj = block(x_proj)

        x_attn, _ = self.time_attn(x_proj, x_proj, x_proj)
        x_proj = self.attn_norm(x_proj + x_attn)

        x_pooled = x_proj.mean(dim=1)
        residuals = self.residual_predictor(x_pooled)
        residuals = residuals.view(B, C, 10).permute(0, 2, 1)

        future_pred = baseline + self.residual_scale * residuals
        input_recon = x_norm

        return future_pred, input_recon, norm_stats


class Model(nn.Module):
    """Codabench-compatible wrapper."""

    def __init__(self, monkey_name: str = ""):
        super().__init__()
        self.monkey_name = monkey_name

        if self.monkey_name == 'beignet':
            self.n_channels = 89
        elif self.monkey_name == 'affi':
            self.n_channels = 239
        else:
            raise ValueError(f'No such a monkey: {self.monkey_name}')

        self.forecaster = ImprovedForecaster(
            n_channels=self.n_channels,
            d_model=64,
            n_layers=2,
            dropout=0.2,
        )

    def load(self):
        """Load pre-trained model weights."""
        base = os.path.dirname(__file__)

        if self.monkey_name == 'beignet':
            model_path = os.path.join(base, "improved_model_beignet.pth")
        elif self.monkey_name == 'affi':
            model_path = os.path.join(base, "improved_model_affi.pth")
        else:
            raise ValueError(f'No such a monkey: {self.monkey_name}')

        state_dict = torch.load(
            model_path,
            map_location=torch.device(device),
            weights_only=True,
        )
        self.forecaster.load_state_dict(state_dict)
        self.forecaster.to(device)
        self.forecaster.eval()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Input array of shape (N, 20, C, 9)

        Returns:
            predictions: Array of shape (N, 20, C)
        """
        N, T, C, F = X.shape
        input_data = X[:, :10, :, :].astype(np.float32)
        input_tensor = torch.tensor(input_data, dtype=torch.float32, device=device)

        self.forecaster.eval()
        predictions = []

        with torch.no_grad():
            batch_size = 64
            for i in range(0, N, batch_size):
                batch = input_tensor[i:i+batch_size]
                future_pred, input_recon, norm_stats = self.forecaster(batch)

                future_denorm = self.forecaster.robust_norm.denormalize(
                    future_pred, norm_stats
                )
                input_denorm = self.forecaster.robust_norm.denormalize(
                    input_recon, norm_stats
                )

                full_pred = torch.cat([input_denorm, future_denorm], dim=1)
                predictions.append(full_pred.cpu().numpy())

        predictions = np.concatenate(predictions, axis=0)
        return predictions
