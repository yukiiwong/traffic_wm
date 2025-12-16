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
        self.metadata = {}  # Initialize empty metadata dict

        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)

                # Store metadata as instance attribute
                self.metadata = metadata

                # Get discrete feature indices from metadata
                validation_info = metadata.get('validation_info', {})
                discrete_features = validation_info.get('discrete_features', {})

                # Get angle index (should not be normalized with z-score)
                angle_idx = validation_info.get('angle_idx', None)
                non_normalized_indices = [v for v in discrete_features.values() if v is not None]
                if angle_idx is not None:
                    non_normalized_indices.append(angle_idx)
                
                # Get explicitly excluded features (e.g., sin_angle, cos_angle)
                exclude_features = validation_info.get('exclude_features', [])
                if exclude_features:
                    print(f"Excluding features from continuous_indices: {exclude_features}")
                    exclude_names = validation_info.get('exclude_feature_names', [])
                    if exclude_names:
                        print(f"  Feature names: {exclude_names}")
                    non_normalized_indices.extend(exclude_features)

                if discrete_features:
                    # ✅ FIX: Filter out None values before sorting
                    discrete_values = [v for v in discrete_features.values() if v is not None]
                    self.discrete_indices = sorted(set(discrete_values))
                    # Continuous features: exclude discrete, angle, AND explicitly excluded features
                    self.continuous_indices = [i for i in range(self.F)
                                                if i not in non_normalized_indices]
                    
                    # Exclude following vehicle features: has_following (10) and rel_*_following (16-19)
                    following_features = [10, 16, 17, 18, 19]
                    self.continuous_indices = [i for i in self.continuous_indices 
                                                if i not in following_features]
                    
                    # Add derived features: velocity_direction (20), headway (21), ttc (22), preceding_distance (23)
                    self.continuous_indices.extend([20, 21, 22, 23])
                    
                    print(f"Loaded discrete feature indices from metadata: {self.discrete_indices}")
                    print(f"Excluded following vehicle features: {following_features}")
                    print(f"Added derived interaction features: [20:velocity_direction, 21:headway, 22:ttc, 23:preceding_distance]")
                    if angle_idx is not None:
                        print(f"Angle feature at index {angle_idx} will NOT be normalized (periodic)")
                        self.angle_idx = angle_idx
                    else:
                        self.angle_idx = None

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
        
        # Compute derived features and append temporarily
        # Feature 20: velocity_direction
        vx = self.states[:, :, :, 2]  # [N, T, K]
        vy = self.states[:, :, :, 3]  # [N, T, K]
        vel_dir = torch.atan2(vy, vx)  # [N, T, K]
        
        # Feature 21: headway (from rel_x_preceding)
        headway = self.states[:, :, :, 12]  # [N, T, K]
        
        # Feature 22: ttc
        rel_x = self.states[:, :, :, 12]  # [N, T, K]
        rel_y = self.states[:, :, :, 13]  # [N, T, K]
        rel_vx = self.states[:, :, :, 14]  # [N, T, K]
        
        distance = torch.sqrt(rel_x**2 + rel_y**2)
        ttc = torch.where(
            rel_vx < -0.1,
            -distance / rel_vx,
            torch.tensor(100.0)  # Use finite value for stats
        )
        ttc = torch.clamp(ttc / 30.0, 0, 100)  # Convert to seconds
        
        # Feature 23: preceding_distance
        preceding_distance = distance
        
        # Append all derived features [20, 21, 22, 23]
        states_with_derived = torch.cat([
            self.states,
            vel_dir.unsqueeze(-1),
            headway.unsqueeze(-1),
            ttc.unsqueeze(-1),
            preceding_distance.unsqueeze(-1)
        ], dim=-1)  # [N, T, K, 24]

        # Extract continuous features only
        continuous_states = states_with_derived[..., self.continuous_indices]  # [N, T, K, n_continuous]

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
        
        # Filter continuous_indices to only include features that exist in self.states
        # Feature 20 (velocity_direction) is added dynamically in __getitem__
        existing_continuous = [i for i in self.continuous_indices if i < self.F]
        
        if len(existing_continuous) == 0:
            print("No existing continuous features to normalize")
            return

        # Extract continuous features (exclude velocity_direction for now)
        continuous_feats = self.states[..., existing_continuous]  # [N, T, K, n_continuous-1]

        # Get corresponding mean and std (exclude velocity_direction stats)
        mean_existing = self.mean[:len(existing_continuous)]
        std_existing = self.std[:len(existing_continuous)]

        # Normalize
        continuous_feats = (continuous_feats - mean_existing) / std_existing

        # Apply mask to ensure padding remains zero
        continuous_feats = continuous_feats * self.masks.unsqueeze(-1)

        # Put normalized features back into states
        self.states[..., existing_continuous] = continuous_feats

        # Discrete features (if any) remain unchanged at their original integer values
        # Note: velocity_direction (feature 20) will be computed and normalized in __getitem__
        print(f"Normalized {len(existing_continuous)} continuous features (excluding velocity_direction)")
        print(f"Preserved {len(self.discrete_indices)} discrete features without normalization")
        print(f"velocity_direction (feature 20) will be computed dynamically in __getitem__")

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
                - states: [T, K, F+1] (float32, normalized continuous + raw discrete + velocity_direction)
                - masks: [T, K] (float32)
                - scene_id: scalar (int64) - site identifier (0=A, 1=B, ..., 8=I)
                - site_id: scalar (int64) - alias for scene_id, used as episode-level conditioning
                - discrete_features: [T, K, n_discrete] (int64, ready for embeddings)
        """
        states = self.states[idx]  # [T, K, F=20]
        masks = self.masks[idx]    # [T, K]
        site_id = self.scene_ids[idx]  # scalar site identifier
        
        # === Dynamically add derived features ===
        
        # Feature 20: velocity_direction (from vx=2, vy=3)
        vx = states[:, :, 2]  # [T, K]
        vy = states[:, :, 3]  # [T, K]
        vel_dir = torch.atan2(vy, vx)  # [T, K] in radians [-π, π]
        
        # Feature 21: headway (longitudinal distance to preceding vehicle)
        # From rel_x_preceding (feature 12)
        headway = states[:, :, 12]  # [T, K] in pixels
        
        # Feature 22: ttc (Time-To-Collision with preceding vehicle)
        # Computed from relative position and velocity
        rel_x = states[:, :, 12]  # [T, K]
        rel_y = states[:, :, 13]  # [T, K]
        rel_vx = states[:, :, 14]  # [T, K]
        rel_vy = states[:, :, 15]  # [T, K]
        
        distance = torch.sqrt(rel_x**2 + rel_y**2)  # [T, K]
        rel_speed = torch.sqrt(rel_vx**2 + rel_vy**2)  # [T, K]
        
        # TTC = -distance / relative_velocity (only if approaching)
        # Negative rel_vx means closing in
        ttc = torch.where(
            rel_vx < -0.1,  # Approaching (threshold 0.1 px/s)
            -distance / rel_vx,
            torch.tensor(float('inf'))  # Not approaching
        )
        # Clamp to reasonable range [0, 100] seconds
        # Convert from pixels/frame to seconds: multiply by frame_time
        # Assuming 30 FPS: 1 frame = 1/30 sec
        ttc = torch.clamp(ttc / 30.0, 0, 100)  # [T, K] in seconds
        
        # Feature 23: preceding_distance (total distance)
        preceding_distance = distance  # [T, K] in pixels
        
        # Normalize derived features
        if hasattr(self, 'mean') and hasattr(self, 'std'):
            # velocity_direction: use stats at index -4
            vel_dir_normalized = (vel_dir - self.mean[-4]) / self.std[-4]
            # headway: use stats at index -3
            headway_normalized = (headway - self.mean[-3]) / self.std[-3]
            # ttc: use stats at index -2
            ttc_normalized = (ttc - self.mean[-2]) / self.std[-2]
            # preceding_distance: use stats at index -1
            distance_normalized = (preceding_distance - self.mean[-1]) / self.std[-1]
        else:
            # Fallback normalization
            vel_dir_normalized = vel_dir / np.pi
            headway_normalized = headway / 100.0  # rough normalization
            ttc_normalized = ttc / 10.0
            distance_normalized = preceding_distance / 100.0
        
        # Concatenate all derived features [20, 21, 22, 23]
        states = torch.cat([
            states,
            vel_dir_normalized.unsqueeze(-1),
            headway_normalized.unsqueeze(-1),
            ttc_normalized.unsqueeze(-1),
            distance_normalized.unsqueeze(-1)
        ], dim=-1)  # [T, K, 24]

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
            'scene_id': site_id,               # scalar (backward compatibility)
            'site_id': site_id,                # scalar (explicit site identifier for conditioning)
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
