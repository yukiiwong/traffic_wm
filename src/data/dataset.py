"""
PyTorch Dataset for Multi-Agent Trajectories

Provides DataLoader-ready dataset for training world models.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, Optional, Tuple
import json


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

        # ✅ FIX: Enforce stats_path for val/test datasets
        # Detect if this is val/test by filename
        filename = self.data_path.stem  # e.g., 'train_episodes', 'val_episodes'
        is_val_or_test = 'val' in filename or 'test' in filename

        if is_val_or_test and normalize and not stats_path:
            raise ValueError(
                f"Val/test dataset '{filename}' MUST provide stats_path to use train statistics. "
                f"Prevent normalization inconsistency by loading train stats."
            )

        # Load data
        data = np.load(self.data_path)
        self.states = torch.from_numpy(data['states']).float()  # [N, T, K, F]
        self.masks = torch.from_numpy(data['masks']).float()    # [N, T, K]
        self.scene_ids = torch.from_numpy(data['scene_ids']).long()  # [N]

        self.n_episodes = len(self.states)
        self.T = self.states.shape[1]
        self.K = self.states.shape[2]
        self.F = self.states.shape[3]

        # Load metadata to identify discrete features
        self._load_discrete_feature_indices()

        # Compute or load normalization statistics
        if self.normalize:
            if stats_path and Path(stats_path).exists():
                self._load_stats(stats_path)
            else:
                self._compute_stats()
            self._normalize_data()

        # ✅ FIX: Clamp and validate discrete features after loading
        self._validate_and_clamp_discrete_features()

        print(f"Loaded {self.n_episodes} episodes")
        print(f"Shape: T={self.T}, K={self.K}, F={self.F}")
        if self.normalize:
            print(f"Normalized continuous features: {self.continuous_indices}")
            print(f"Preserved discrete features: {self.discrete_indices}")

    def _load_discrete_feature_indices(self) -> None:
        """Load discrete feature indices from metadata."""
        metadata_path = self.data_path.parent / 'metadata.json'

        # Default: assume no discrete features if metadata not found
        self.discrete_indices = []
        self.continuous_indices = list(range(self.F))
        self.num_lanes = 0
        self.num_classes = 0
        self.num_sites = 0

        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)

                # Get discrete feature indices from metadata
                validation_info = metadata.get('validation_info', {})
                discrete_features = validation_info.get('discrete_features', {})

                if discrete_features:
                    # ✅ FIX: Filter out None values before sorting
                    discrete_values = [v for v in discrete_features.values() if v is not None]
                    self.discrete_indices = sorted(set(discrete_values))
                    self.continuous_indices = [i for i in range(self.F)
                                                if i not in self.discrete_indices]
                    print(f"Loaded discrete feature indices from metadata: {self.discrete_indices}")

                    # Load bounds for validation
                    self.num_lanes = validation_info.get('num_lanes', len(metadata.get('lane_mapping', {})))
                    self.num_sites = len(metadata.get('sites_processed', [])) or 9  # A-I = 9 sites
                    # Assuming class_id is small, default to 10 if not specified
                    self.num_classes = 10

                    print(f"Discrete feature bounds: lanes={self.num_lanes}, sites={self.num_sites}, classes={self.num_classes}")

                    # ✅ FIX: Validate discrete feature indices
                    self._validate_discrete_indices()
                else:
                    print("Warning: No discrete features found in metadata, normalizing all features")
            except Exception as e:
                print(f"Warning: Could not load metadata from {metadata_path}: {e}")
                print("Normalizing all features as fallback")
        else:
            print(f"Warning: Metadata not found at {metadata_path}, normalizing all features")

    def _validate_discrete_indices(self) -> None:
        """Validate discrete feature indices are within valid range."""
        for idx in self.discrete_indices:
            if idx < 0 or idx >= self.F:
                raise ValueError(
                    f"Invalid discrete feature index {idx}: must be in range [0, {self.F-1}]"
                )

        # Check for duplicates (should already be handled by set(), but double-check)
        if len(self.discrete_indices) != len(set(self.discrete_indices)):
            raise ValueError(f"Duplicate discrete indices found: {self.discrete_indices}")

        # Check continuous and discrete are disjoint and complete
        all_indices = set(self.continuous_indices) | set(self.discrete_indices)
        expected_indices = set(range(self.F))
        if all_indices != expected_indices:
            missing = expected_indices - all_indices
            extra = all_indices - expected_indices
            raise ValueError(
                f"Feature index mismatch! Missing: {missing}, Extra: {extra}"
            )

        print(f"✅ Discrete index validation passed: {self.discrete_indices}")

    def _compute_stats(self) -> None:
        """Compute mean and std for normalization (only on continuous features)."""
        # Expand masks to match feature dimension
        valid_mask = self.masks.unsqueeze(-1)  # [N, T, K, 1]

        if len(self.continuous_indices) == 0:
            print("Warning: No continuous features to normalize!")
            return

        # Extract continuous features only
        continuous_states = self.states[..., self.continuous_indices]  # [N, T, K, n_continuous]

        # Compute statistics only on valid entries
        valid_continuous = continuous_states * valid_mask[..., :len(self.continuous_indices)]
        n_valid = valid_mask.sum()

        # Mean and std per continuous feature
        self.mean = valid_continuous.sum(dim=(0, 1, 2)) / n_valid  # [n_continuous]

        # Variance
        diff = (valid_continuous - self.mean) * valid_mask[..., :len(self.continuous_indices)]
        self.std = torch.sqrt((diff ** 2).sum(dim=(0, 1, 2)) / n_valid)  # [n_continuous]

        # Avoid division by zero
        self.std = torch.clamp(self.std, min=1e-6)

        print(f"Normalization stats computed for {len(self.continuous_indices)} continuous features:")
        print(f"  Mean shape: {self.mean.shape}")
        print(f"  Std shape: {self.std.shape}")
        print(f"  Mean (first 5): {self.mean[:5]}")
        print(f"  Std (first 5): {self.std[:5]}")

    def _load_stats(self, stats_path: str) -> None:
        """Load pre-computed normalization statistics."""
        stats = np.load(stats_path)
        self.mean = torch.from_numpy(stats['mean']).float()
        self.std = torch.from_numpy(stats['std']).float()

        # Load continuous/discrete indices if available
        if 'continuous_indices' in stats:
            loaded_continuous = stats['continuous_indices'].tolist()
            if loaded_continuous != self.continuous_indices:
                print(f"Warning: Loaded continuous_indices {loaded_continuous} differ from metadata {self.continuous_indices}")
                print(f"Using indices from stats file")
                self.continuous_indices = loaded_continuous
        if 'discrete_indices' in stats:
            loaded_discrete = stats['discrete_indices'].tolist()
            if loaded_discrete != self.discrete_indices:
                print(f"Warning: Loaded discrete_indices {loaded_discrete} differ from metadata {self.discrete_indices}")
                print(f"Using indices from stats file")
                self.discrete_indices = loaded_discrete

        print(f"Loaded normalization stats from {stats_path}")

    def _normalize_data(self) -> None:
        """Apply z-score normalization to continuous features only."""
        if len(self.continuous_indices) == 0:
            print("No continuous features to normalize, skipping normalization")
            return

        # Extract continuous features
        continuous_feats = self.states[..., self.continuous_indices]  # [N, T, K, n_continuous]

        # Normalize
        continuous_feats = (continuous_feats - self.mean) / self.std

        # Apply mask to ensure padding remains zero
        continuous_feats = continuous_feats * self.masks.unsqueeze(-1)

        # Put normalized features back into states
        self.states[..., self.continuous_indices] = continuous_feats

        # Discrete features (if any) remain unchanged at their original integer values
        print(f"Normalized {len(self.continuous_indices)} continuous features")
        print(f"Preserved {len(self.discrete_indices)} discrete features without normalization")

    def _validate_and_clamp_discrete_features(self) -> None:
        """
        Validate and clamp discrete features to ensure valid embedding indices.

        Actions:
        - Clamp negative values to 0 (invalid → special token)
        - Set padding positions (mask=0) to 0
        - Optionally warn about out-of-range values
        """
        if len(self.discrete_indices) == 0:
            return  # No discrete features to validate

        for idx in self.discrete_indices:
            discrete_feat = self.states[..., idx]  # [N, T, K]

            # Count issues before clamping
            n_negative = (discrete_feat < 0).sum().item()

            # Clamp negative values to 0
            discrete_feat = torch.clamp(discrete_feat, min=0)

            # Set padding positions to 0 (where mask=0)
            discrete_feat = discrete_feat * self.masks  # [N, T, K] * [N, T, K]

            # Store back (still as float, will convert to long in __getitem__)
            self.states[..., idx] = discrete_feat

            # Report issues
            if n_negative > 0:
                print(f"⚠️  Clamped {n_negative} negative values in discrete feature {idx} to 0")

            # Optional: Check range for specific features
            max_val = discrete_feat.max().item()
            if idx == self.discrete_indices[0] and hasattr(self, 'num_lanes') and max_val >= self.num_lanes:
                print(f"⚠️  Warning: lane_id max={max_val} >= num_lanes={self.num_lanes}")
            elif idx == self.discrete_indices[-1] and hasattr(self, 'num_sites') and max_val >= self.num_sites:
                print(f"⚠️  Warning: site_id max={max_val} >= num_sites={self.num_sites}")

        print(f"✅ Discrete feature validation complete: clamped negative values, zeroed padding")

    def save_stats(self, save_path: str) -> None:
        """Save normalization statistics for later use."""
        np.savez(
            save_path,
            mean=self.mean.numpy(),
            std=self.std.numpy(),
            continuous_indices=np.array(self.continuous_indices),
            discrete_indices=np.array(self.discrete_indices)
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
                - states: [T, K, F] (float32, normalized continuous + raw discrete)
                - masks: [T, K] (float32)
                - scene_id: scalar (int64)
                - discrete_features: [T, K, n_discrete] (int64, ready for embeddings)
        """
        states = self.states[idx]  # [T, K, F]
        masks = self.masks[idx]    # [T, K]

        # ✅ FIX: Extract discrete features as separate LongTensor for embeddings
        if len(self.discrete_indices) > 0:
            # Extract discrete features
            discrete_feats = states[..., self.discrete_indices]  # [T, K, n_discrete]

            # Already clamped and zeroed in _validate_and_clamp_discrete_features
            # Convert to long for embedding layers
            discrete_feats = discrete_feats.long()
        else:
            # No discrete features, return empty tensor
            discrete_feats = torch.zeros(self.T, self.K, 0, dtype=torch.long)

        return {
            'states': states,                  # [T, K, F] - full state vector
            'masks': masks,                    # [T, K]
            'scene_id': self.scene_ids[idx],   # scalar
            'discrete_features': discrete_feats  # [T, K, n_discrete] - LongTensor for embeddings
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
