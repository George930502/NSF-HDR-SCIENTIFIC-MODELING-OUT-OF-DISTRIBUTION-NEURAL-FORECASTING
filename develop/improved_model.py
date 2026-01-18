"""
Improved Neural Forecasting Model.

Key improvements based on dataset analysis:
1. Per-sample robust normalization (handles session drift)
2. Focus on temporal dynamics (features 1-8 are NOT predictive)
3. Residual learning from linear extrapolation baseline
4. Huber loss for robustness to outliers
5. Smaller model to prevent overfitting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Tuple, Optional


class RobustNorm(nn.Module):
    """
    Robust per-sample normalization using median and IQR.
    This handles session drift and outliers better than z-score.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Normalize using median and IQR (interquartile range).

        Args:
            x: Input tensor (B, T, C, F) or (B, T, C)

        Returns:
            Normalized tensor and stats for denormalization
        """
        # Flatten spatial-temporal dims for robust statistics
        if x.dim() == 4:
            B, T, C, F = x.shape
            x_flat = x.view(B, -1)
        else:
            B, T, C = x.shape
            x_flat = x.view(B, -1)

        # Compute median and IQR per sample
        median = x_flat.median(dim=1, keepdim=True)[0]
        q75 = torch.quantile(x_flat, 0.75, dim=1, keepdim=True)
        q25 = torch.quantile(x_flat, 0.25, dim=1, keepdim=True)
        iqr = (q75 - q25).clamp(min=self.eps)

        # Normalize
        x_norm = (x_flat - median) / iqr

        # Reshape back
        if x.dim() == 4:
            x_norm = x_norm.view(B, T, C, F)
        else:
            x_norm = x_norm.view(B, T, C)

        return x_norm, {'median': median, 'iqr': iqr}

    def denormalize(self, x: torch.Tensor, stats: dict) -> torch.Tensor:
        """Reverse the normalization."""
        B = x.shape[0]
        median = stats['median'].view(B, 1, 1)
        iqr = stats['iqr'].view(B, 1, 1)
        return x * iqr + median


class LinearExtrapolation(nn.Module):
    """
    Linear extrapolation baseline.
    Given high autocorrelation (r=0.97), linear extrapolation is a strong baseline.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extrapolate future values using linear trend.

        Args:
            x: Input tensor (B, 10, C) - last 10 known timesteps

        Returns:
            Extrapolated values (B, 10, C) - next 10 timesteps
        """
        B, T, C = x.shape

        # Use last few timesteps to estimate linear trend
        # y = slope * t + intercept
        t = torch.arange(T, device=x.device, dtype=x.dtype)  # [0, 1, ..., 9]
        t = t.view(1, T, 1).expand(B, T, C)

        # Compute slope using simple linear regression
        t_mean = t.mean(dim=1, keepdim=True)  # (B, 1, C)
        x_mean = x.mean(dim=1, keepdim=True)  # (B, 1, C)

        numerator = ((t - t_mean) * (x - x_mean)).sum(dim=1, keepdim=True)
        denominator = ((t - t_mean) ** 2).sum(dim=1, keepdim=True) + 1e-6
        slope = numerator / denominator  # (B, 1, C)

        intercept = x_mean - slope * t_mean  # (B, 1, C)

        # Extrapolate to future timesteps [10, 11, ..., 19]
        t_future = torch.arange(T, 2*T, device=x.device, dtype=x.dtype)
        t_future = t_future.view(1, T, 1).expand(B, T, C)

        extrapolated = slope * t_future + intercept  # (B, 10, C)

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
        # x: (B*C, T, D)
        residual = x

        # Conv expects (B, C, T)
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
    """
    Improved neural forecaster with:
    1. Per-sample robust normalization
    2. Linear extrapolation baseline
    3. Residual learning for corrections
    4. Compact architecture to prevent overfitting
    """

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

        # Robust normalization
        self.robust_norm = RobustNorm()

        # Linear baseline
        self.linear_extrap = LinearExtrapolation()

        # Input projection (only use target feature 0)
        self.input_proj = nn.Linear(1, d_model)

        # Temporal encoder - simple but effective
        self.temporal_blocks = nn.ModuleList([
            ResidualBlock(d_model, kernel_size=3, dropout=dropout)
            for _ in range(n_layers)
        ])

        # Attention over time
        self.time_attn = nn.MultiheadAttention(
            d_model, num_heads=4, dropout=dropout, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(d_model)

        # Residual predictor (learns correction to linear baseline)
        self.residual_predictor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 10),  # Predict 10 future residuals
        )

        # Scale factor for residual (learnable, initialized small)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Forward pass.

        Args:
            x: Input (B, 10, C, F) - first 10 timesteps with all features

        Returns:
            future_pred: (B, 10, C) - predicted next 10 timesteps
            input_recon: (B, 10, C) - reconstructed input
            norm_stats: normalization statistics for denormalization
        """
        B, T, C, F = x.shape

        # Extract only feature 0 (target) - other features are not predictive!
        x_target = x[:, :, :, 0]  # (B, 10, C)

        # Robust per-sample normalization
        x_norm, norm_stats = self.robust_norm(x_target)  # (B, 10, C)

        # Linear extrapolation baseline
        baseline = self.linear_extrap(x_norm)  # (B, 10, C)

        # Process each channel independently but in parallel
        x_flat = x_norm.permute(0, 2, 1).reshape(B * C, T, 1)  # (B*C, 10, 1)

        # Project to model dimension
        x_proj = self.input_proj(x_flat)  # (B*C, 10, d_model)

        # Apply temporal blocks
        for block in self.temporal_blocks:
            x_proj = block(x_proj)

        # Self-attention over time
        x_attn, _ = self.time_attn(x_proj, x_proj, x_proj)
        x_proj = self.attn_norm(x_proj + x_attn)

        # Pool over time and predict residuals
        x_pooled = x_proj.mean(dim=1)  # (B*C, d_model)
        residuals = self.residual_predictor(x_pooled)  # (B*C, 10)

        # Reshape residuals
        residuals = residuals.view(B, C, 10).permute(0, 2, 1)  # (B, 10, C)

        # Final prediction = baseline + scaled residual
        future_pred = baseline + self.residual_scale * residuals

        # Input reconstruction (just return normalized input for now)
        input_recon = x_norm

        return future_pred, input_recon, norm_stats


class ImprovedModel(nn.Module):
    """
    Codabench-compatible wrapper for ImprovedForecaster.
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

        # Initialize the forecaster
        self.forecaster = ImprovedForecaster(
            n_channels=self.n_channels,
            d_model=64,
            n_layers=2,
            dropout=0.2,
        )

        # Per-channel statistics for global normalization
        self.global_mean = None
        self.global_std = None

    def set_global_stats(self, mean: np.ndarray, std: np.ndarray):
        """Set global normalization statistics from training data."""
        self.global_mean = mean
        self.global_std = std

    def load(self):
        """Load pre-trained model weights."""
        import os
        base = os.path.dirname(__file__)

        if self.monkey_name == 'beignet':
            model_path = os.path.join(base, "improved_model_beignet.pth")
            stats_path = os.path.join(base, "improved_stats_beignet.npz")
        elif self.monkey_name == 'affi':
            model_path = os.path.join(base, "improved_model_affi.pth")
            stats_path = os.path.join(base, "improved_stats_affi.npz")
        else:
            raise ValueError(f'No such a monkey: {self.monkey_name}')

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load model state
        state_dict = torch.load(
            model_path,
            map_location=torch.device(device),
            weights_only=True,
        )
        self.forecaster.load_state_dict(state_dict)
        self.forecaster.to(device)
        self.forecaster.eval()

        # Load global statistics
        if os.path.exists(stats_path):
            stats = np.load(stats_path)
            self.global_mean = stats['mean']
            self.global_std = stats['std']

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Input array of shape (N, 20, C, 9)
               First 10 timesteps have data, last 10 are masked

        Returns:
            predictions: Array of shape (N, 20, C)
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        N, T, C, F = X.shape

        # Get the first 10 timesteps as input
        input_data = X[:, :10, :, :].astype(np.float32)

        # Convert to tensor
        input_tensor = torch.tensor(input_data, dtype=torch.float32, device=device)

        # Make predictions
        self.forecaster.eval()
        predictions = []

        with torch.no_grad():
            batch_size = 64
            for i in range(0, N, batch_size):
                batch = input_tensor[i:i+batch_size]

                # Forward pass
                future_pred, input_recon, norm_stats = self.forecaster(batch)

                # Denormalize predictions
                future_denorm = self.forecaster.robust_norm.denormalize(
                    future_pred, norm_stats
                )
                input_denorm = self.forecaster.robust_norm.denormalize(
                    input_recon, norm_stats
                )

                # Combine input and future predictions
                full_pred = torch.cat([input_denorm, future_denorm], dim=1)
                predictions.append(full_pred.cpu().numpy())

        predictions = np.concatenate(predictions, axis=0)
        return predictions


# Model parameter count check
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick test
    model = ImprovedForecaster(n_channels=239, d_model=64, n_layers=2)
    print(f"Model parameters: {count_parameters(model):,}")

    # Test forward pass
    x = torch.randn(4, 10, 239, 9)
    future, recon, stats = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Future prediction shape: {future.shape}")
    print(f"Reconstruction shape: {recon.shape}")
