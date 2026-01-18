"""
Improved Training Script for Neural Forecasting.

Key improvements:
1. Huber loss (robust to outliers)
2. Combined training on all datasets
3. Per-sample normalization
4. Strong regularization
5. Early stopping with proper validation
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from tqdm import tqdm

from improved_model import ImprovedForecaster, count_parameters


class RobustDataset(Dataset):
    """
    Dataset with robust preprocessing.

    Key improvements:
    - No global normalization (per-sample normalization is done in model)
    - Outlier clipping
    - All data sources combined
    """

    def __init__(
        self,
        data: np.ndarray,
        clip_percentile: float = 99.5,
        augment: bool = False,
    ):
        """
        Initialize dataset.

        Args:
            data: Neural data with shape (N, T, C, F)
            clip_percentile: Percentile for outlier clipping
            augment: Whether to apply data augmentation
        """
        self.data = data.astype(np.float32)
        self.augment = augment

        # Clip outliers per sample
        if clip_percentile < 100:
            for i in range(len(self.data)):
                low = np.percentile(self.data[i], 100 - clip_percentile)
                high = np.percentile(self.data[i], clip_percentile)
                self.data[i] = np.clip(self.data[i], low, high)

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

        # Apply augmentation during training
        if self.augment:
            sample = self._augment(sample)

        # Split into input and target
        input_data = sample[:10]  # (10, C, F)
        target_data = sample[10:, :, 0]  # (10, C) - only feature 0

        return (
            torch.from_numpy(input_data),
            torch.from_numpy(target_data),
        )

    def _augment(self, sample: np.ndarray) -> np.ndarray:
        """Apply data augmentation."""
        # Gaussian noise (very small)
        if np.random.random() < 0.3:
            # Scale noise to be relative to sample std
            sample_std = sample.std()
            noise = np.random.randn(*sample.shape).astype(np.float32) * 0.02 * sample_std
            sample = sample + noise

        # Temporal jitter (shift by 1 timestep with 10% probability)
        if np.random.random() < 0.1:
            shift = np.random.choice([-1, 1])
            if shift > 0:
                sample = np.concatenate([sample[shift:], sample[-shift:]], axis=0)
            else:
                sample = np.concatenate([sample[:shift], sample[:-shift]], axis=0)

        # Scale augmentation (small multiplicative noise)
        if np.random.random() < 0.2:
            scale = 1.0 + np.random.uniform(-0.05, 0.05)
            sample = sample * scale

        return sample.astype(np.float32)


def load_all_data(data_dir: Path, monkey: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and combine ALL training data for a specific monkey.
    This includes both main and private datasets.
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
    print(f"Combined data for {monkey}: shape = {combined_data.shape}")

    # Split with stratification across sources
    # Use 90% train, 10% validation
    n_samples = len(combined_data)
    n_val = max(1, int(n_samples * 0.1))

    # Shuffle
    indices = np.random.permutation(n_samples)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    train_data = combined_data[train_indices]
    val_data = combined_data[val_indices]

    print(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}")
    return train_data, val_data


def compute_global_stats(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute global statistics for reference."""
    # Only for feature 0 (target)
    target = data[:, :, :, 0]  # (N, T, C)
    mean = target.mean(axis=(0, 1))  # (C,)
    std = target.std(axis=(0, 1))  # (C,)
    std = np.clip(std, 1e-6, None)
    return mean, std


class HuberLoss(nn.Module):
    """Huber loss - more robust to outliers than MSE."""

    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = delta

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return nn.functional.huber_loss(pred, target, delta=self.delta)


class ImprovedTrainer:
    """Trainer for the improved model."""

    def __init__(
        self,
        model: ImprovedForecaster,
        device: str = "cuda",
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        huber_delta: float = 1.0,
    ):
        self.model = model.to(device)
        self.device = device

        # Huber loss for robustness
        self.criterion = HuberLoss(delta=huber_delta)

        # Optimizer with weight decay
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        # Cosine annealing scheduler
        self.scheduler = None  # Set later based on epochs

        # History
        self.history = {'train_loss': [], 'val_loss': [], 'val_mse': []}

    def set_scheduler(self, epochs: int, warmup_epochs: int = 2):
        """Set up learning rate scheduler."""
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=epochs // 2,
            T_mult=1,
            eta_min=1e-6,
        )

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0

        for batch in train_loader:
            input_data, target_data = batch
            input_data = input_data.to(self.device)
            target_data = target_data.to(self.device)

            # Forward pass
            future_pred, input_recon, norm_stats = self.model(input_data)

            # Get normalized target for loss computation
            target_norm, _ = self.model.robust_norm(target_data)

            # Main loss: Huber loss on future prediction
            loss = self.criterion(future_pred, target_norm)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(train_loader)

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        total_mse = 0.0
        n_samples = 0

        for batch in val_loader:
            input_data, target_data = batch
            input_data = input_data.to(self.device)
            target_data = target_data.to(self.device)

            # Forward pass
            future_pred, input_recon, norm_stats = self.model(input_data)

            # Get normalized target
            target_norm, _ = self.model.robust_norm(target_data)

            # Huber loss
            loss = self.criterion(future_pred, target_norm)
            total_loss += loss.item()

            # Denormalize for MSE calculation
            future_denorm = self.model.robust_norm.denormalize(future_pred, norm_stats)

            # MSE in original scale
            mse = ((future_denorm - target_data) ** 2).mean().item()
            total_mse += mse * len(input_data)
            n_samples += len(input_data)

        val_loss = total_loss / len(val_loader)
        val_mse = total_mse / n_samples

        return val_loss, val_mse

    def save_checkpoint(self, path: str, epoch: int, val_loss: float):
        """Save model checkpoint."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'history': self.history,
        }, path)


def train_model(
    monkey: str,
    data_dir: Path,
    output_dir: Path,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    patience: int = 10,
    d_model: int = 64,
    n_layers: int = 2,
    dropout: float = 0.2,
):
    """Train the improved model for one monkey."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"Training improved model for {monkey}")
    print(f"{'='*60}")
    print(f"Using device: {device}")

    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Load data
    train_data, val_data = load_all_data(data_dir, monkey)

    # Compute global stats for reference
    global_mean, global_std = compute_global_stats(train_data)

    # Get number of channels
    n_channels = train_data.shape[2]
    print(f"Number of channels: {n_channels}")

    # Create datasets
    train_dataset = RobustDataset(train_data, clip_percentile=99.5, augment=True)
    val_dataset = RobustDataset(val_data, clip_percentile=99.5, augment=False)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # Create model
    model = ImprovedForecaster(
        n_channels=n_channels,
        d_model=d_model,
        n_layers=n_layers,
        dropout=dropout,
    )
    print(f"Model parameters: {count_parameters(model):,}")

    # Create trainer
    trainer = ImprovedTrainer(
        model=model,
        device=device,
        lr=lr,
        weight_decay=1e-4,
        huber_delta=1.0,
    )
    trainer.set_scheduler(epochs)

    # Training loop
    best_val_loss = float('inf')
    best_epoch = 0
    no_improve = 0

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    pbar = tqdm(range(1, epochs + 1), desc="Training")
    for epoch in pbar:
        # Train
        train_loss = trainer.train_epoch(train_loader)

        # Validate
        val_loss, val_mse = trainer.validate(val_loader)

        # Update scheduler
        if trainer.scheduler is not None:
            trainer.scheduler.step()

        # Record history
        trainer.history['train_loss'].append(train_loss)
        trainer.history['val_loss'].append(val_loss)
        trainer.history['val_mse'].append(val_mse)

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            no_improve = 0

            # Save best model
            trainer.save_checkpoint(
                str(checkpoint_dir / f"improved_{monkey}_best.pt"),
                epoch,
                val_loss,
            )
            pbar.set_postfix({
                'loss': f'{train_loss:.4f}',
                'val_loss': f'{val_loss:.4f}',
                'val_mse': f'{val_mse:.2f}',
                'best': True,
            })
        else:
            no_improve += 1
            pbar.set_postfix({
                'loss': f'{train_loss:.4f}',
                'val_loss': f'{val_loss:.4f}',
                'val_mse': f'{val_mse:.2f}',
            })

        # Early stopping
        if no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    print(f"\nTraining completed. Best val_loss: {best_val_loss:.6f} at epoch {best_epoch}")

    # Load best model and save for submission
    checkpoint = torch.load(str(checkpoint_dir / f"improved_{monkey}_best.pt"))
    model.load_state_dict(checkpoint['model_state_dict'])

    # Save model weights
    submission_dir = output_dir / "submission"
    submission_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), str(submission_dir / f"improved_model_{monkey}.pth"))

    # Save global stats
    np.savez(
        str(submission_dir / f"improved_stats_{monkey}.npz"),
        mean=global_mean,
        std=global_std,
    )

    # Save history
    with open(str(output_dir / f"improved_history_{monkey}.json"), 'w') as f:
        json.dump(trainer.history, f, indent=2)

    print(f"Saved model to {submission_dir / f'improved_model_{monkey}.pth'}")
    return best_val_loss


def main():
    parser = argparse.ArgumentParser(description="Train Improved Neural Forecasting Model")
    parser.add_argument("--monkey", type=str, choices=['affi', 'beignet', 'both'],
                       default='both', help="Which monkey to train")
    parser.add_argument("--data_dir", type=str, default="../dataset",
                       help="Path to dataset directory")
    parser.add_argument("--output_dir", type=str, default="./output_improved",
                       help="Path to save outputs")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--d_model", type=int, default=64, help="Model dimension")
    parser.add_argument("--n_layers", type=int, default=2, help="Number of layers")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")

    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    monkeys = ['affi', 'beignet'] if args.monkey == 'both' else [args.monkey]

    for monkey in monkeys:
        train_model(
            monkey=monkey,
            data_dir=data_dir,
            output_dir=output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=args.patience,
            d_model=args.d_model,
            n_layers=args.n_layers,
            dropout=args.dropout,
        )

    print("\nTraining completed for all monkeys!")


if __name__ == "__main__":
    main()
