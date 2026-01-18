"""
Training script for Neural Forecasting Model.
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils import load_training_data, create_dataloaders, denormalize
from neural_model import create_model, NeuralForecaster, NeuralForecasterV2


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Trainer:
    """Training manager for neural forecasting model."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs: int = 100,
        patience: int = 15,
        checkpoint_dir: Path = Path("checkpoints"),
        model_name: str = "model",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.max_epochs = max_epochs
        self.patience = patience
        self.checkpoint_dir = checkpoint_dir
        self.model_name = model_name

        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Loss function
        self.criterion = nn.MSELoss()

        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max_epochs,
            eta_min=1e-6,
        )

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'best_val_loss': float('inf'),
            'best_epoch': 0,
        }

    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for batch in pbar:
            input_data, target_data, input_target = batch

            # Move to device
            input_data = input_data.to(self.device)
            target_data = target_data.to(self.device)
            input_target = input_target.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            pred_future, pred_input = self.model(input_data)

            # Compute loss
            future_loss = self.criterion(pred_future, target_data)
            recon_loss = self.criterion(pred_input, input_target)

            # Combined loss (prioritize future prediction)
            loss = future_loss + 0.3 * recon_loss

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            pbar.set_postfix({'loss': f'{loss.item():.6f}'})

        return total_loss / n_batches

    @torch.no_grad()
    def validate(self) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        total_future_loss = 0.0
        n_batches = 0

        for batch in self.val_loader:
            input_data, target_data, input_target = batch

            # Move to device
            input_data = input_data.to(self.device)
            target_data = target_data.to(self.device)
            input_target = input_target.to(self.device)

            # Forward pass
            pred_future, pred_input = self.model(input_data)

            # Compute loss
            future_loss = self.criterion(pred_future, target_data)
            recon_loss = self.criterion(pred_input, input_target)
            loss = future_loss + 0.3 * recon_loss

            total_loss += loss.item()
            total_future_loss += future_loss.item()
            n_batches += 1

        return total_loss / n_batches, total_future_loss / n_batches

    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'history': self.history,
        }

        # Save latest checkpoint
        checkpoint_path = self.checkpoint_dir / f"{self.model_name}_latest.pt"
        torch.save(checkpoint, checkpoint_path)

        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / f"{self.model_name}_best.pt"
            torch.save(checkpoint, best_path)
            print(f"  Saved best model with val_loss: {val_loss:.6f}")

    def load_checkpoint(self, path: Path) -> int:
        """Load checkpoint and return starting epoch."""
        if not path.exists():
            return 0

        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']

        return checkpoint['epoch'] + 1

    def train(self, resume: bool = False) -> Dict:
        """
        Run the full training loop.

        Args:
            resume: Whether to resume from latest checkpoint

        Returns:
            Training history
        """
        start_epoch = 0
        if resume:
            resume_path = self.checkpoint_dir / f"{self.model_name}_latest.pt"
            start_epoch = self.load_checkpoint(resume_path)
            if start_epoch > 0:
                print(f"Resumed training from epoch {start_epoch}")

        no_improve_count = 0

        for epoch in range(start_epoch, self.max_epochs):
            epoch_start = time.time()

            # Training
            train_loss = self.train_epoch()

            # Validation
            val_loss, val_future_loss = self.validate()

            # Update scheduler
            self.scheduler.step()

            # Track history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)

            # Check for improvement
            is_best = val_loss < self.history['best_val_loss']
            if is_best:
                self.history['best_val_loss'] = val_loss
                self.history['best_epoch'] = epoch
                no_improve_count = 0
            else:
                no_improve_count += 1

            # Save checkpoint
            self.save_checkpoint(epoch, val_loss, is_best)

            # Print progress
            epoch_time = time.time() - epoch_start
            lr = self.optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{self.max_epochs} ({epoch_time:.1f}s) - "
                  f"train_loss: {train_loss:.6f}, val_loss: {val_loss:.6f}, "
                  f"val_future_mse: {val_future_loss:.6f}, lr: {lr:.2e}")

            # Early stopping
            if no_improve_count >= self.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        print(f"\nTraining completed. Best val_loss: {self.history['best_val_loss']:.6f} "
              f"at epoch {self.history['best_epoch']+1}")

        return self.history


def train_model(
    monkey: str,
    data_dir: Path,
    output_dir: Path,
    model_type: str = 'v1',
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    max_epochs: int = 100,
    patience: int = 15,
    seed: int = 42,
    resume: bool = False,
    **model_kwargs,
) -> Tuple[nn.Module, Dict]:
    """
    Train a model for a specific monkey.

    Args:
        monkey: 'affi' or 'beignet'
        data_dir: Path to dataset directory
        output_dir: Path to save outputs
        model_type: Model architecture type
        batch_size: Training batch size
        learning_rate: Initial learning rate
        max_epochs: Maximum training epochs
        patience: Early stopping patience
        seed: Random seed
        resume: Whether to resume training
        **model_kwargs: Additional model arguments

    Returns:
        Trained model and normalization stats
    """
    print(f"\n{'='*60}")
    print(f"Training model for {monkey}")
    print(f"{'='*60}")

    # Set seed
    set_seed(seed)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    train_data, val_data = load_training_data(data_dir, monkey)

    # Create dataloaders
    train_loader, val_loader, norm_stats = create_dataloaders(
        train_data, val_data,
        batch_size=batch_size,
    )

    # Create model
    model = create_model(monkey, model_type=model_type, **model_kwargs)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        patience=patience,
        checkpoint_dir=output_dir / "checkpoints",
        model_name=f"model_{monkey}",
    )

    # Train
    history = trainer.train(resume=resume)

    # Load best model
    best_checkpoint = output_dir / "checkpoints" / f"model_{monkey}_best.pt"
    checkpoint = torch.load(best_checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Save normalization stats
    stats_path = output_dir / f"norm_stats_{monkey}.npz"
    np.savez(
        stats_path,
        mean=norm_stats['mean'],
        std=norm_stats['std'],
    )
    print(f"Saved normalization stats to {stats_path}")

    # Save training history
    history_path = output_dir / f"history_{monkey}.json"
    with open(history_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        history_json = {k: v if not isinstance(v, np.ndarray) else v.tolist()
                       for k, v in history.items()}
        json.dump(history_json, f, indent=2)
    print(f"Saved training history to {history_path}")

    return model, norm_stats


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description="Train Neural Forecasting Model")
    parser.add_argument("--monkey", type=str, choices=['affi', 'beignet', 'both'],
                       default='both', help="Which monkey to train for")
    parser.add_argument("--data_dir", type=str, default="../dataset",
                       help="Path to dataset directory")
    parser.add_argument("--output_dir", type=str, default="./output",
                       help="Path to save outputs")
    parser.add_argument("--model_type", type=str, choices=['v1', 'v2'], default='v1',
                       help="Model architecture type")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=100, help="Max epochs")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resume", action="store_true", help="Resume training")
    parser.add_argument("--d_model", type=int, default=128, help="Model hidden dimension")
    parser.add_argument("--n_layers", type=int, default=3, help="Number of transformer layers")
    parser.add_argument("--n_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")

    args = parser.parse_args()

    # Convert paths
    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Model kwargs
    model_kwargs = {
        'd_model': args.d_model,
        'n_heads': args.n_heads,
        'dropout': args.dropout,
    }

    if args.model_type == 'v1':
        model_kwargs['n_layers'] = args.n_layers
    else:
        model_kwargs['n_encoder_layers'] = args.n_layers
        model_kwargs['n_decoder_layers'] = max(1, args.n_layers - 1)

    # Train models
    monkeys = ['affi', 'beignet'] if args.monkey == 'both' else [args.monkey]

    for monkey in monkeys:
        model, norm_stats = train_model(
            monkey=monkey,
            data_dir=data_dir,
            output_dir=output_dir,
            model_type=args.model_type,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            max_epochs=args.epochs,
            patience=args.patience,
            seed=args.seed,
            resume=args.resume,
            **model_kwargs,
        )

        # Save final model for submission
        submission_dir = output_dir / "submission"
        submission_dir.mkdir(parents=True, exist_ok=True)

        # Save model state dict
        model_path = submission_dir / f"model_{monkey}.pth"
        torch.save(model.state_dict(), model_path)
        print(f"Saved model weights to {model_path}")

    print("\nTraining completed for all monkeys!")


if __name__ == "__main__":
    main()
