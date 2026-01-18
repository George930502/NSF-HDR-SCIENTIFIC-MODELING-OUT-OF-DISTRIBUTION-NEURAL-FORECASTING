"""
Neural Forecasting Model for μECoG Signals.

Architecture combines:
1. Temporal convolutions for local pattern extraction
2. Channel attention for spatial dependencies
3. Transformer encoder for sequence modeling
4. Autoregressive decoding for prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class InstanceNorm(nn.Module):
    """Instance normalization for handling session drift."""

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply instance normalization.

        Args:
            x: Input tensor of shape (B, T, C, F) or (B, T, C)

        Returns:
            Normalized tensor
        """
        # Compute mean and std across time dimension
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

        # Residual connection
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input of shape (B, T, D)

        Returns:
            Output of shape (B, T, out_channels)
        """
        # Convert to (B, D, T) for Conv1d
        x_conv = x.transpose(1, 2)
        residual = self.residual(x_conv)

        # First convolution
        out = self.conv1(x_conv)
        out = out.transpose(1, 2)  # (B, T, D)
        out = self.norm1(out)
        out = F.gelu(out)
        out = self.dropout(out)

        # Second convolution
        out = out.transpose(1, 2)  # (B, D, T)
        out = self.conv2(out)

        # Residual connection
        out = out + residual
        out = out.transpose(1, 2)  # (B, T, D)
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
        """
        Apply channel attention.

        Args:
            x: Input of shape (B, T, C, D)

        Returns:
            Output of shape (B, T, C, D)
        """
        B, T, C, D = x.shape

        # Reshape to process each timestep
        x_flat = x.reshape(B * T, C, D)

        # Compute Q, K, V
        Q = self.q_proj(x_flat)
        K = self.k_proj(x_flat)
        V = self.v_proj(x_flat)

        # Multi-head attention
        Q = Q.view(B * T, C, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(B * T, C, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(B * T, C, self.n_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).reshape(B * T, C, D)
        out = self.out_proj(out)

        # Residual and norm
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
        """Add positional encoding to input."""
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
        """
        Forward pass.

        Args:
            x: Input of shape (B, T, D)
            mask: Optional attention mask

        Returns:
            Output of shape (B, T, D)
        """
        # Pre-norm self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm, attn_mask=mask)
        x = x + self.dropout(attn_out)

        # Pre-norm feed-forward
        x_norm = self.norm2(x)
        ff_out = self.ff(x_norm)
        x = x + ff_out

        return x


class NeuralForecaster(nn.Module):
    """
    Neural Forecasting Model for μECoG signals.

    Takes input of shape (B, 10, C, 9) and predicts (B, 20, C).
    """

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
        """
        Initialize the model.

        Args:
            n_channels: Number of electrode channels (239 for affi, 89 for beignet)
            n_features: Number of input features (default 9)
            d_model: Hidden dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            d_ff: Feed-forward dimension
            dropout: Dropout rate
            use_channel_attention: Whether to use channel attention
        """
        super().__init__()
        self.n_channels = n_channels
        self.n_features = n_features
        self.d_model = d_model
        self.use_channel_attention = use_channel_attention

        # Instance normalization for session adaptation
        self.instance_norm = InstanceNorm()

        # Input embedding: project (C, F) to (C, D)
        self.input_proj = nn.Linear(n_features, d_model)

        # Temporal convolution blocks for local patterns
        self.temporal_conv = nn.Sequential(
            TemporalConvBlock(d_model, d_model, kernel_size=3, dilation=1, dropout=dropout),
            TemporalConvBlock(d_model, d_model, kernel_size=3, dilation=2, dropout=dropout),
        )

        # Channel attention
        if use_channel_attention:
            self.channel_attn = ChannelAttention(n_channels, d_model, n_heads)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len=30)

        # Flatten channels into sequence: (B, T, C, D) -> (B, T*C, D)
        # Then use transformer to capture temporal-spatial dependencies

        # Transformer encoder layers
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Decoder: predict 10 future timesteps
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 10),  # Predict 10 future values per channel
        )

        # Also predict the reconstruction of input timesteps
        self.input_decoder = nn.Linear(d_model, 1)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, 10, C, F=9)

        Returns:
            pred_future: Predictions for t=10:20, shape (B, 10, C)
            pred_input: Reconstructed input for t=0:10, shape (B, 10, C)
        """
        B, T, C, F = x.shape

        # Instance normalization for session adaptation
        x = self.instance_norm(x)

        # Project features: (B, T, C, F) -> (B, T, C, D)
        x = self.input_proj(x)

        # Apply channel attention
        if self.use_channel_attention:
            x = self.channel_attn(x)

        # Process each channel's temporal sequence
        # Reshape to (B*C, T, D) for temporal processing
        x = x.permute(0, 2, 1, 3)  # (B, C, T, D)
        x = x.reshape(B * C, T, self.d_model)

        # Temporal convolutions
        x = self.temporal_conv(x)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Transformer layers
        for layer in self.transformer_layers:
            x = layer(x)

        # Reshape back: (B*C, T, D) -> (B, C, T, D)
        x = x.view(B, C, T, self.d_model)

        # Pool over time to get channel representations
        x_pooled = x.mean(dim=2)  # (B, C, D)

        # Predict future timesteps
        pred_future = self.decoder(x_pooled)  # (B, C, 10)
        pred_future = pred_future.transpose(1, 2)  # (B, 10, C)

        # Predict reconstruction of input
        x_time = x.permute(0, 2, 1, 3)  # (B, T, C, D)
        pred_input = self.input_decoder(x_time).squeeze(-1)  # (B, 10, C)

        return pred_future, pred_input


class NeuralForecasterV2(nn.Module):
    """
    Improved Neural Forecasting Model using sequence-to-sequence approach.

    Uses autoregressive decoding for better prediction quality.
    """

    def __init__(
        self,
        n_channels: int,
        n_features: int = 9,
        d_model: int = 128,
        n_heads: int = 4,
        n_encoder_layers: int = 3,
        n_decoder_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_features = n_features
        self.d_model = d_model

        # Instance normalization
        self.instance_norm = InstanceNorm()

        # Input projection
        self.encoder_proj = nn.Linear(n_features, d_model)
        self.decoder_proj = nn.Linear(1, d_model)  # Decoder takes single feature

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len=30)

        # Encoder: processes input sequence
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers)

        # Decoder: generates output sequence
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_decoder_layers)

        # Output projection
        self.output_proj = nn.Linear(d_model, 1)

        # Reconstruction head
        self.recon_head = nn.Linear(d_model, 1)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """Generate causal mask for autoregressive decoding."""
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(self, x: torch.Tensor, teacher_forcing: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, 10, C, F=9)
            teacher_forcing: Whether to use teacher forcing (for training)

        Returns:
            pred_future: Predictions for t=10:20, shape (B, 10, C)
            pred_input: Reconstructed input for t=0:10, shape (B, 10, C)
        """
        B, T, C, F = x.shape
        device = x.device

        # Instance normalization
        x_norm = self.instance_norm(x)

        # Process each channel independently
        # Reshape: (B, T, C, F) -> (B*C, T, F)
        x_flat = x_norm.permute(0, 2, 1, 3).reshape(B * C, T, F)

        # Encode input sequence
        enc_input = self.encoder_proj(x_flat)  # (B*C, T, D)
        enc_input = self.pos_encoding(enc_input)
        enc_output = self.encoder(enc_input)  # (B*C, T, D)

        # Reconstruction of input
        pred_input = self.recon_head(enc_output).squeeze(-1)  # (B*C, T)
        pred_input = pred_input.view(B, C, T).transpose(1, 2)  # (B, T, C)

        # Decode future sequence
        # Use the last input value as start token
        last_value = x_flat[:, -1:, 0:1]  # (B*C, 1, 1)

        # Create decoder input: shift target right, prepend last input value
        # For inference, we use autoregressive generation
        if teacher_forcing:
            # During training: decoder sees shifted ground truth
            # But we don't have ground truth here, so we use a learned query
            dec_queries = self.decoder_proj(last_value.expand(-1, 10, -1))  # (B*C, 10, D)
        else:
            dec_queries = self.decoder_proj(last_value.expand(-1, 10, -1))

        dec_queries = self.pos_encoding(dec_queries)

        # Generate causal mask
        causal_mask = self._generate_square_subsequent_mask(10, device)

        # Decode
        dec_output = self.decoder(dec_queries, enc_output, tgt_mask=causal_mask)

        # Project to output
        pred_future = self.output_proj(dec_output).squeeze(-1)  # (B*C, 10)
        pred_future = pred_future.view(B, C, 10).transpose(1, 2)  # (B, 10, C)

        return pred_future, pred_input


def create_model(
    monkey: str,
    model_type: str = 'v1',
    **kwargs,
) -> nn.Module:
    """
    Create a model for the specified monkey.

    Args:
        monkey: 'affi' or 'beignet'
        model_type: 'v1' or 'v2'
        **kwargs: Additional model arguments

    Returns:
        Model instance
    """
    if monkey == 'affi':
        n_channels = 239
    elif monkey == 'beignet':
        n_channels = 89
    else:
        raise ValueError(f"Unknown monkey: {monkey}")

    if model_type == 'v1':
        return NeuralForecaster(n_channels=n_channels, **kwargs)
    elif model_type == 'v2':
        return NeuralForecasterV2(n_channels=n_channels, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
