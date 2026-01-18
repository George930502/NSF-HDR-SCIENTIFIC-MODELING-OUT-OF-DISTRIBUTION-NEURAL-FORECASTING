"""
Data utilities for Neural Forecasting Challenge.
Handles data loading, preprocessing, and dataset creation.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, Optional, Dict, List


class NeuralForecastDataset(Dataset):
    """
    Dataset for neural signal forecasting.

    Data format: (N, T=20, C, F=9)
    - N: number of samples
    - T: 20 timesteps (first 10 input, last 10 target)
    - C: number of channels (239 for affi, 89 for beignet)
    - F: 9 features (feature 0 is target, 1-8 are frequency bands)
    """

    def __init__(
        self,
        data: np.ndarray,
        normalize: bool = True,
        normalization_stats: Optional[Dict] = None,
        augment: bool = False,
    ):
        """
        Initialize dataset.

        Args:
            data: Neural data with shape (N, T, C, F)
            normalize: Whether to apply normalization
            normalization_stats: Pre-computed statistics for normalization
            augment: Whether to apply data augmentation
        """
        self.data = data.astype(np.float32)
        self.normalize = normalize
        self.augment = augment

        # Compute or use provided normalization statistics
        if normalize:
            if normalization_stats is not None:
                self.stats = normalization_stats
            else:
                self.stats = self._compute_normalization_stats()
        else:
            self.stats = None

    def _compute_normalization_stats(self) -> Dict:
        """Compute per-channel normalization statistics."""
        # Compute stats across samples and timesteps
        # Shape: (N, T, C, F) -> compute mean/std per (C, F)
        mean = self.data.mean(axis=(0, 1), keepdims=True)  # (1, 1, C, F)
        std = self.data.std(axis=(0, 1), keepdims=True)   # (1, 1, C, F)
        std = np.clip(std, 1e-6, None)  # Prevent division by zero

        return {'mean': mean, 'std': std}

    def get_normalization_stats(self) -> Dict:
        """Return normalization statistics for inference."""
        return self.stats

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample.

        Returns:
            input_data: Shape (10, C, F) - first 10 timesteps, all features
            target_data: Shape (10, C) - last 10 timesteps, feature 0 only
        """
        sample = self.data[idx].copy()  # (T=20, C, F)

        # Apply normalization
        if self.normalize and self.stats is not None:
            sample = (sample - self.stats['mean'][0, 0]) / self.stats['std'][0, 0]

        # Apply augmentation during training
        if self.augment:
            sample = self._augment(sample)

        # Split into input and target
        input_data = sample[:10]  # (10, C, F)
        target_data = sample[10:, :, 0]  # (10, C) - only feature 0

        # Also include the first 10 timesteps of target for reconstruction loss
        input_target = sample[:10, :, 0]  # (10, C)

        return (
            torch.from_numpy(input_data),
            torch.from_numpy(target_data),
            torch.from_numpy(input_target),
        )

    def _augment(self, sample: np.ndarray) -> np.ndarray:
        """Apply data augmentation."""
        # Gaussian noise augmentation
        if np.random.random() < 0.3:
            noise = np.random.randn(*sample.shape).astype(np.float32) * 0.01
            sample = sample + noise

        # Channel dropout
        if np.random.random() < 0.1:
            n_channels = sample.shape[1]
            dropout_mask = np.random.rand(n_channels) > 0.05
            sample[:, ~dropout_mask, :] = 0

        return sample.astype(np.float32)


class InferenceDataset(Dataset):
    """Dataset for inference (no target separation)."""

    def __init__(
        self,
        data: np.ndarray,
        normalization_stats: Dict,
    ):
        """
        Initialize inference dataset.

        Args:
            data: Neural data with shape (N, T, C, F)
            normalization_stats: Pre-computed normalization statistics
        """
        self.data = data.astype(np.float32)
        self.stats = normalization_stats

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get a sample for inference.

        Returns:
            input_data: Shape (T=20, C, F)
        """
        sample = self.data[idx].copy()

        # Apply normalization
        if self.stats is not None:
            sample = (sample - self.stats['mean'][0, 0]) / self.stats['std'][0, 0]

        return torch.from_numpy(sample)


def load_training_data(data_dir: Path, monkey: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and combine training data for a specific monkey.

    Args:
        data_dir: Path to dataset directory
        monkey: 'affi' or 'beignet'

    Returns:
        train_data: Combined training data
        val_data: Validation data (split from main training set)
    """
    train_dir = data_dir / "train"

    if monkey == 'affi':
        files = [
            "train_data_affi.npz",
            "train_data_affi_2024-03-20_private.npz",
        ]
    elif monkey == 'beignet':
        files = [
            "train_data_beignet.npz",
            "train_data_beignet_2022-06-01_private.npz",
            "train_data_beignet_2022-06-02_private.npz",
        ]
    else:
        raise ValueError(f"Unknown monkey: {monkey}")

    # Load all files
    all_data = []
    for filename in files:
        filepath = train_dir / filename
        if filepath.exists():
            data = np.load(filepath)['arr_0']
            all_data.append(data)
            print(f"Loaded {filename}: shape = {data.shape}")
        else:
            print(f"Warning: {filename} not found")

    # Combine all data
    combined_data = np.concatenate(all_data, axis=0)
    print(f"Combined training data for {monkey}: shape = {combined_data.shape}")

    # Split into train and validation (90% / 10%)
    n_samples = len(combined_data)
    n_val = max(1, int(n_samples * 0.1))

    # Shuffle before splitting
    indices = np.random.permutation(n_samples)
    train_indices = indices[n_val:]
    val_indices = indices[:n_val]

    train_data = combined_data[train_indices]
    val_data = combined_data[val_indices]

    print(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}")

    return train_data, val_data


def create_dataloaders(
    train_data: np.ndarray,
    val_data: np.ndarray,
    batch_size: int = 32,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, Dict]:
    """
    Create training and validation dataloaders.

    Args:
        train_data: Training data array
        val_data: Validation data array
        batch_size: Batch size
        num_workers: Number of data loading workers

    Returns:
        train_loader: Training dataloader
        val_loader: Validation dataloader
        normalization_stats: Statistics for inference
    """
    # Create training dataset (compute normalization from training data)
    train_dataset = NeuralForecastDataset(
        train_data,
        normalize=True,
        augment=True,
    )

    # Get normalization stats from training dataset
    norm_stats = train_dataset.get_normalization_stats()

    # Create validation dataset with training normalization
    val_dataset = NeuralForecastDataset(
        val_data,
        normalize=True,
        normalization_stats=norm_stats,
        augment=False,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, norm_stats


def denormalize(
    data: np.ndarray,
    stats: Dict,
    feature_idx: int = 0,
) -> np.ndarray:
    """
    Denormalize predictions back to original scale.

    Args:
        data: Normalized data
        stats: Normalization statistics
        feature_idx: Which feature index to use for denormalization

    Returns:
        Denormalized data
    """
    mean = stats['mean'][0, 0, :, feature_idx]
    std = stats['std'][0, 0, :, feature_idx]

    return data * std + mean
