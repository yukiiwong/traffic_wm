"""
PyTorch Dataset for Multi-Agent Trajectories

Provides DataLoader-ready dataset for training world models.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, Optional, Tuple


class TrajectoryDataset(Dataset):
    """
    Dataset for multi-agent trajectory episodes.

    Each episode contains:
        - states: [T, K, F] vehicle states over time
        - masks: [T, K] binary mask (1 = valid vehicle, 0 = padding)
        - scene_id: integer scene identifier
    """

    def __init__(
        self,
        data_path: str,
        normalize: bool = True,
        stats_path: Optional[str] = None
    ):
        """
        Initialize dataset from preprocessed NPZ file.

        Args:
            data_path: Path to the episodes.npz file
            normalize: Whether to normalize states
            stats_path: Path to normalization statistics (mean, std)
        """
        self.data_path = Path(data_path)
        self.normalize = normalize

        # Load data
        data = np.load(self.data_path)
        self.states = torch.from_numpy(data['states']).float()  # [N, T, K, F]
        self.masks = torch.from_numpy(data['masks']).float()    # [N, T, K]
        self.scene_ids = torch.from_numpy(data['scene_ids']).long()  # [N]

        self.n_episodes = len(self.states)
        self.T = self.states.shape[1]
        self.K = self.states.shape[2]
        self.F = self.states.shape[3]

        # Compute or load normalization statistics
        if self.normalize:
            if stats_path and Path(stats_path).exists():
                self._load_stats(stats_path)
            else:
                self._compute_stats()
            self._normalize_data()

        print(f"Loaded {self.n_episodes} episodes")
        print(f"Shape: T={self.T}, K={self.K}, F={self.F}")

    def _compute_stats(self) -> None:
        """Compute mean and std for normalization (only on valid vehicles)."""
        # Expand masks to match feature dimension
        valid_mask = self.masks.unsqueeze(-1)  # [N, T, K, 1]

        # Compute statistics only on valid entries
        valid_states = self.states * valid_mask
        n_valid = valid_mask.sum()

        # Mean and std per feature
        self.mean = valid_states.sum(dim=(0, 1, 2)) / n_valid  # [F]

        # Variance
        diff = (valid_states - self.mean) * valid_mask
        self.std = torch.sqrt((diff ** 2).sum(dim=(0, 1, 2)) / n_valid)  # [F]

        # Avoid division by zero
        self.std = torch.clamp(self.std, min=1e-6)

        print(f"Normalization stats computed:")
        print(f"  Mean: {self.mean}")
        print(f"  Std: {self.std}")

    def _load_stats(self, stats_path: str) -> None:
        """Load pre-computed normalization statistics."""
        stats = np.load(stats_path)
        self.mean = torch.from_numpy(stats['mean']).float()
        self.std = torch.from_numpy(stats['std']).float()
        print(f"Loaded normalization stats from {stats_path}")

    def _normalize_data(self) -> None:
        """Apply z-score normalization to states."""
        self.states = (self.states - self.mean) / self.std
        # Multiply by mask to ensure padding remains zero
        self.states = self.states * self.masks.unsqueeze(-1)

    def save_stats(self, save_path: str) -> None:
        """Save normalization statistics for later use."""
        np.savez(
            save_path,
            mean=self.mean.numpy(),
            std=self.std.numpy()
        )
        print(f"Saved normalization stats to {save_path}")

    def __len__(self) -> int:
        """Return number of episodes."""
        return self.n_episodes

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single episode.

        Args:
            idx: Episode index

        Returns:
            Dictionary with:
                - states: [T, K, F]
                - masks: [T, K]
                - scene_id: scalar
        """
        return {
            'states': self.states[idx],      # [T, K, F]
            'masks': self.masks[idx],        # [T, K]
            'scene_id': self.scene_ids[idx]  # scalar
        }


def get_dataloader(
    data_path: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    normalize: bool = True,
    stats_path: Optional[str] = None
) -> DataLoader:
    """
    Create a DataLoader for training.

    Args:
        data_path: Path to episodes.npz
        batch_size: Batch size
        shuffle: Whether to shuffle episodes
        num_workers: Number of worker processes
        normalize: Whether to normalize states
        stats_path: Path to normalization stats

    Returns:
        PyTorch DataLoader
    """
    dataset = TrajectoryDataset(
        data_path=data_path,
        normalize=normalize,
        stats_path=stats_path
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader


if __name__ == '__main__':
    # Example usage
    dataloader = get_dataloader(
        data_path='../../data/processed/train_episodes.npz',
        batch_size=16,
        shuffle=True
    )

    # Test iteration
    batch = next(iter(dataloader))
    print(f"Batch states shape: {batch['states'].shape}")  # [B, T, K, F]
    print(f"Batch masks shape: {batch['masks'].shape}")    # [B, T, K]
    print(f"Scene IDs: {batch['scene_id']}")               # [B]
