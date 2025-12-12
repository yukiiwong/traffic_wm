"""
Main Training Script for World Model

Handles training loop, validation, checkpointing, and logging.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import argparse
from typing import Dict, Optional

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.dataset import get_dataloader
from src.models.world_model import WorldModel
from src.training.losses import WorldModelLoss
from src.utils.logger import setup_logger
from src.utils.common import set_seed, count_parameters


class WorldModelTrainer:
    """Trainer for the world model."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        loss_fn: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[optim.lr_scheduler._LRScheduler],
        device: torch.device,
        checkpoint_dir: str = './checkpoints',
        log_dir: str = './logs'
    ):
        """
        Initialize trainer.

        Args:
            model: World model
            train_loader: Training data loader
            val_loader: Validation data loader
            loss_fn: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            device: Device to train on
            checkpoint_dir: Directory for checkpoints
            log_dir: Directory for logs
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.logger = setup_logger('trainer', log_dir)
        self.global_step = 0
        self.epoch = 0

        self.logger.info(f"Model parameters: {count_parameters(model):,}")

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        total_losses = {
            'total': 0.0,
            'reconstruction': 0.0,
            'prediction': 0.0,
            'existence': 0.0
        }

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")

        for batch_idx, batch in enumerate(pbar):
            # Move to device
            states = batch['states'].to(self.device)  # [B, T, K, F]
            masks = batch['masks'].to(self.device)    # [B, T, K]

            # Forward pass
            predictions = self.model(states, masks)

            # Compute loss
            targets = {'states': states, 'masks': masks}
            losses = self.loss_fn(predictions, targets)

            # Backward pass
            self.optimizer.zero_grad()
            losses['total'].backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Accumulate losses
            for key in total_losses.keys():
                total_losses[key] += losses[key].item()

            # Update progress bar
            pbar.set_postfix({
                'loss': losses['total'].item(),
                'recon': losses['reconstruction'].item(),
                'pred': losses['prediction'].item()
            })

            self.global_step += 1

        # Average losses
        num_batches = len(self.train_loader)
        for key in total_losses.keys():
            total_losses[key] /= num_batches

        return total_losses

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        if self.val_loader is None:
            return {}

        self.model.eval()

        total_losses = {
            'total': 0.0,
            'reconstruction': 0.0,
            'prediction': 0.0,
            'existence': 0.0
        }

        for batch in tqdm(self.val_loader, desc="Validation"):
            states = batch['states'].to(self.device)
            masks = batch['masks'].to(self.device)

            # Forward pass
            predictions = self.model(states, masks)

            # Compute loss
            targets = {'states': states, 'masks': masks}
            losses = self.loss_fn(predictions, targets)

            # Accumulate
            for key in total_losses.keys():
                total_losses[key] += losses[key].item()

        # Average
        num_batches = len(self.val_loader)
        for key in total_losses.keys():
            total_losses[key] /= num_batches

        return total_losses

    def save_checkpoint(self, name: str = 'checkpoint.pt'):
        """Save model checkpoint with normalization stats."""
        # Get normalization stats from training dataset
        train_mean = None
        train_std = None
        if hasattr(self.train_loader.dataset, 'mean') and hasattr(self.train_loader.dataset, 'std'):
            train_mean = self.train_loader.dataset.mean.cpu().numpy().tolist()
            train_std = self.train_loader.dataset.std.cpu().numpy().tolist()

        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            # ✅ 新增：保存归一化统计
            'normalization': {
                'mean': train_mean,
                'std': train_std
            }
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        save_path = self.checkpoint_dir / name
        torch.save(checkpoint, save_path)
        self.logger.info(f"Saved checkpoint to {save_path}")
        if train_mean is not None:
            self.logger.info(f"  ✅ Saved normalization stats (mean[:2]={train_mean[:2]})")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']

        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.logger.info(f"Loaded checkpoint from {path}")

    def train(self, n_epochs: int):
        """Main training loop."""
        self.logger.info(f"Starting training for {n_epochs} epochs")

        best_val_loss = float('inf')

        for epoch in range(self.epoch, n_epochs):
            self.epoch = epoch

            # Train
            train_losses = self.train_epoch()

            # Log training losses
            self.logger.info(
                f"Epoch {epoch} - Train Loss: {train_losses['total']:.4f} "
                f"(Recon: {train_losses['reconstruction']:.4f}, "
                f"Pred: {train_losses['prediction']:.4f}, "
                f"Exist: {train_losses['existence']:.4f})"
            )

            # Validate
            if self.val_loader is not None:
                val_losses = self.validate()
                self.logger.info(
                    f"Epoch {epoch} - Val Loss: {val_losses['total']:.4f} "
                    f"(Recon: {val_losses['reconstruction']:.4f}, "
                    f"Pred: {val_losses['prediction']:.4f}, "
                    f"Exist: {val_losses['existence']:.4f})"
                )

                # Save best model
                if val_losses['total'] < best_val_loss:
                    best_val_loss = val_losses['total']
                    self.save_checkpoint('best_model.pt')

            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()

            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt')

        self.logger.info("Training completed!")


def main(args):
    """Main training entry point."""
    print("===== Experiment Config =====")
    print(f"train_data: {args.train_data}")
    print(f"val_data:   {args.val_data}")
    print(f"input_dim:  {args.input_dim}")
    print(f"latent_dim: {args.latent_dim}")
    print(f"dynamics:   {args.dynamics_type}")
    print(f"batch_size: {args.batch_size}")
    print(f"n_epochs:   {args.n_epochs}")
    print(f"lr:         {args.learning_rate}")
    print("=============================")
    # Set seed
    set_seed(args.seed)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")


    # Data loaders
    train_loader = get_dataloader(
        data_path=args.train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        normalize=True
    )

    val_loader = None
    if args.val_data:
        val_loader = get_dataloader(
            data_path=args.val_data,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            normalize=True
        )

    # Model
    model = WorldModel(
        input_dim=args.input_dim,
        max_agents=args.max_agents,
        latent_dim=args.latent_dim,
        dynamics_type=args.dynamics_type
    )

    # Loss
    loss_fn = WorldModelLoss(
        recon_weight=args.recon_weight,
        pred_weight=args.pred_weight,
        existence_weight=args.existence_weight
    )

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.n_epochs,
        eta_min=1e-6
    )

    # Trainer
    trainer = WorldModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir
    )

    # Load checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    trainer.train(args.n_epochs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train World Model')

    # Data
    parser.add_argument('--train_data', type=str, required=True,
                        help='Path to training data (episodes.npz)')
    parser.add_argument('--val_data', type=str, default=None,
                        help='Path to validation data')

    # Model
    parser.add_argument('--input_dim', type=int, default=10,
                        help='Number of features per agent')
    parser.add_argument('--max_agents', type=int, default=50,
                        help='Maximum number of agents')
    parser.add_argument('--latent_dim', type=int, default=256,
                        help='Latent dimension')
    parser.add_argument('--dynamics_type', type=str, default='gru',
                        choices=['gru', 'lstm', 'transformer'],
                        help='Type of dynamics model')

    # Training
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--n_epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')

    # Loss weights
    parser.add_argument('--recon_weight', type=float, default=1.0,
                        help='Reconstruction loss weight')
    parser.add_argument('--pred_weight', type=float, default=1.0,
                        help='Prediction loss weight')
    parser.add_argument('--existence_weight', type=float, default=0.1,
                        help='Existence loss weight')

    # Other
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Checkpoint directory')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='Log directory')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    args = parser.parse_args()
    main(args)
