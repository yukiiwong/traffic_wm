"""
CSV to Episode Preprocessing

Converts raw drone trajectory CSV files into fixed-shape tensor episodes.
Supports both basic (6 features) and extended (10 features) modes.

Output format: [T, K, F] where
  - T: time steps (e.g., 30 seconds at 1 Hz)
  - K: max number of vehicles per frame (with padding)
  - F: per-vehicle features (6 basic, or 10 extended)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import logging
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_csv_trajectory(csv_path: str) -> pd.DataFrame:
    """
    Load a single CSV file containing vehicle trajectories.

    Args:
        csv_path: Path to the CSV file

    Returns:
        DataFrame with trajectory data
    """
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {csv_path}: {len(df)} rows, {df['track_id'].nunique()} unique vehicles")
    return df


def compute_velocities(df: pd.DataFrame, dt: float = 1.0 / 30.0) -> pd.DataFrame:
    """
    Compute velocity components (vx, vy) from position differences.

    Args:
        df: DataFrame with 'track_id', 'frame', 'center_x', 'center_y'
        dt: Time step in seconds (default: 1/30 for 30 FPS)

    Returns:
        DataFrame with added 'vx' and 'vy' columns
    """
    df = df.sort_values(['track_id', 'frame']).copy()

    # Compute velocities within each track
    df['vx'] = df.groupby('track_id')['center_x'].diff() / dt
    df['vy'] = df.groupby('track_id')['center_y'].diff() / dt

    # Fill first frame velocities with 0
    df['vx'] = df['vx'].fillna(0.0)
    df['vy'] = df['vy'].fillna(0.0)

    return df


def compute_acceleration(df: pd.DataFrame, dt: float = 1.0 / 30.0) -> pd.DataFrame:
    """
    Compute acceleration from velocity.

    Args:
        df: DataFrame with vx, vy
        dt: Time step

    Returns:
        DataFrame with ax, ay columns
    """
    df = df.sort_values(['track_id', 'frame']).copy()

    df['ax'] = df.groupby('track_id')['vx'].diff() / dt
    df['ay'] = df.groupby('track_id')['vy'].diff() / dt

    df['ax'] = df['ax'].fillna(0.0)
    df['ay'] = df['ay'].fillna(0.0)

    return df


def encode_lane(lane_str: str, lane_mapping: Dict[str, int]) -> int:
    """
    Encode lane string to integer.

    Args:
        lane_str: Lane identifier (e.g., 'A1', 'B2', 'crossroads1')
        lane_mapping: Dictionary mapping lane names to integers

    Returns:
        Encoded lane ID
    """
    if pd.isna(lane_str):
        return 0  # Unknown lane

    lane_str = str(lane_str).strip()
    if lane_str not in lane_mapping:
        lane_mapping[lane_str] = len(lane_mapping) + 1

    return lane_mapping[lane_str]

def get_site_id_from_frames(df: pd.DataFrame, frames: List[int]) -> int:
    """
    Derive a numeric site_id (0,1,2,...) from the 'site' column.

    Convention:
      - If site is like 'A', 'B', ..., 'I'  -> A=0, B=1, ...
      - If site is like 'Site A'           -> A=0, B=1, ...
      - Otherwise                          -> 0 (default)

    Args:
        df: Full trajectory DataFrame (must contain 'site' if used)
        frames: Frames belonging to this episode (for robustness)

    Returns:
        Integer site_id in [0, ...]
    """
    if 'site' not in df.columns:
        return 0

    # Prefer frames within this episode; fallback to whole df if empty
    df_sub = df[df['frame'].isin(frames)]
    if len(df_sub) == 0:
        df_sub = df

    if len(df_sub) == 0:
        return 0

    raw = str(df_sub['site'].iloc[0]).strip()

    # Case 1: single letter 'A'...'Z'
    if len(raw) == 1 and raw.isalpha():
        return max(0, ord(raw.upper()) - ord('A'))

    # Case 2: 'Site A', 'site B', etc.
    parts = raw.split()
    last = parts[-1]
    if len(last) == 1 and last.isalpha():
        return max(0, ord(last.upper()) - ord('A'))

    # Fallback
    return 0

def extract_episodes(
    df: pd.DataFrame,
    episode_length: int = 30,
    max_vehicles: int = 50,
    overlap: int = 0,
    use_extended_features: bool = False,
    use_acceleration: bool = False,
    use_site_id: bool = False,
) -> List[Dict[str, np.ndarray]]:

    """
    Extract fixed-length episodes from trajectory data.

    Args:
        df: DataFrame with trajectory data
        episode_length: Number of time steps per episode (T)
        max_vehicles: Maximum number of vehicles to track (K)
        overlap: Number of overlapping frames between episodes
        use_extended_features: Include lane and relationship features
        use_acceleration: Include acceleration features

    Returns:
        List of episode dictionaries
    """
    episodes = []
    frames = sorted(df['frame'].unique())

    # Sliding window over frames
    step = max(1, episode_length - overlap)

    for start_idx in tqdm(range(0, len(frames) - episode_length + 1, step),
                          desc="Extracting episodes"):
        episode_frames = frames[start_idx:start_idx + episode_length]
        episode_data = extract_single_episode(
            df,
            episode_frames,
            max_vehicles,
            use_extended_features=use_extended_features,
            use_acceleration=use_acceleration,
            use_site_id=use_site_id,
        )
        episodes.append(episode_data)

    return episodes


def extract_single_episode(
    df: pd.DataFrame,
    frames: List[int],
    max_vehicles: int,
    use_extended_features: bool = False,
    use_acceleration: bool = False,
    use_site_id: bool = False,
) -> Dict[str, np.ndarray]:

    """
    Extract a single episode for the given frames.

    Args:
        df: DataFrame with trajectory data
        frames: List of frame indices for this episode
        max_vehicles: Maximum number of vehicles (K)
        use_extended_features: Include extended features
        use_acceleration: Include acceleration
        use_site_id: Include site_id as an extra feature dimension (last feature)

    Returns:
        Dictionary with 'states', 'masks', 'scene_id', etc.
    """
    T = len(frames)
    # Derive a scene-level site_id (used as both scene_id and per-vehicle feature if enabled)
    site_id = get_site_id_from_frames(df, frames)

    # Determine number of features
    F = 6  # Basic: x, y, vx, vy, angle, type
    if use_acceleration:
        F += 2  # ax, ay
    if use_extended_features:
        F += 3  # lane, has_preceding, has_following
        if F < 10:  # Ensure at least 10 features for extended mode
            F = 10
    # Add one extra dim for site_id if enabled
    if use_site_id:
        F += 1
    states = np.zeros((T, max_vehicles, F), dtype=np.float32)
    masks = np.zeros((T, max_vehicles), dtype=np.float32)
    track_ids = np.zeros((T, max_vehicles), dtype=np.int32)

    # Process each time step
    for t, frame_id in enumerate(frames):
        frame_data = df[df['frame'] == frame_id].copy()
        frame_data = frame_data.sort_values('track_id')

        n_vehicles = min(len(frame_data), max_vehicles)

        if n_vehicles > 0:
            feature_idx = 0

            # Basic features: position
            states[t, :n_vehicles, 0:2] = frame_data[['center_x', 'center_y']].values[:n_vehicles]
            feature_idx = 2

            # Velocity
            states[t, :n_vehicles, 2:4] = frame_data[['vx', 'vy']].values[:n_vehicles]
            feature_idx = 4

            # Acceleration (if enabled)
            if use_acceleration and 'ax' in frame_data.columns:
                states[t, :n_vehicles, 4:6] = frame_data[['ax', 'ay']].values[:n_vehicles]
                feature_idx = 6

            # Heading
            states[t, :n_vehicles, feature_idx] = frame_data['angle'].values[:n_vehicles]
            feature_idx += 1

            # Vehicle type
            states[t, :n_vehicles, feature_idx] = frame_data['class_id'].values[:n_vehicles]
            feature_idx += 1

            # Extended features (if enabled)
            if use_extended_features:
                # Lane
                if 'lane_encoded' in frame_data.columns:
                    states[t, :n_vehicles, feature_idx] = frame_data['lane_encoded'].values[:n_vehicles]
                feature_idx += 1

                # Has preceding vehicle
                if 'preceding_id' in frame_data.columns:
                    states[t, :n_vehicles, feature_idx] = \
                        (~frame_data['preceding_id'].isna()).astype(float).values[:n_vehicles]
                feature_idx += 1

                # Has following vehicle
                if 'following_id' in frame_data.columns:
                    states[t, :n_vehicles, feature_idx] = \
                        (~frame_data['following_id'].isna()).astype(float).values[:n_vehicles]
                feature_idx += 1

            # Store track IDs
            track_ids[t, :n_vehicles] = frame_data['track_id'].values[:n_vehicles]
            masks[t, :n_vehicles] = 1.0
            # Site feature (if enabled) — use the same site_id for all vehicles in this episode
            if use_site_id:
                # feature_idx currently points to the next free feature slot
                states[t, :n_vehicles, feature_idx] = float(site_id)
                feature_idx += 1

    # Extract scene_id from site column
    # Use the derived site_id as scene_id for this episode
    scene_id = site_id

    return {
        'states': states,
        'masks': masks,
        'scene_id': scene_id,
        'track_ids': track_ids,
        'start_frame': frames[0],
        'end_frame': frames[-1]
    }


def compute_dataset_statistics(episodes: List[Dict]) -> Dict:
    """
    Compute statistics about the dataset.

    Args:
        episodes: List of episode dictionaries

    Returns:
        Dictionary with statistics
    """
    all_states = np.concatenate([ep['states'] for ep in episodes], axis=0)
    all_masks = np.concatenate([ep['masks'] for ep in episodes], axis=0)

    valid_states = all_states[all_masks.astype(bool)]

    stats = {
        'n_episodes': len(episodes),
        'total_timesteps': sum(ep['states'].shape[0] for ep in episodes),
        'mean_vehicles_per_frame': float(all_masks.sum(axis=1).mean()),
        'max_vehicles_observed': int(all_masks.sum(axis=1).max()),
        'min_vehicles_observed': int(all_masks.sum(axis=1).min()),
        'feature_means': valid_states.mean(axis=0).tolist(),
        'feature_stds': valid_states.std(axis=0).tolist(),
        'feature_mins': valid_states.min(axis=0).tolist(),
        'feature_maxs': valid_states.max(axis=0).tolist(),
    }

    return stats


def split_episodes(
    episodes: List[Dict],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split episodes into train/val/test sets.

    Args:
        episodes: List of episodes
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        seed: Random seed

    Returns:
        (train_episodes, val_episodes, test_episodes)
    """
    np.random.seed(seed)

    n_total = len(episodes)
    indices = np.random.permutation(n_total)

    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]

    train_episodes = [episodes[i] for i in train_indices]
    val_episodes = [episodes[i] for i in val_indices]
    test_episodes = [episodes[i] for i in test_indices]

    logger.info(f"Split: Train={len(train_episodes)}, Val={len(val_episodes)}, Test={len(test_episodes)}")

    return train_episodes, val_episodes, test_episodes


def preprocess_trajectories(
    input_dir: str,
    output_dir: str,
    episode_length: int = 30,
    max_vehicles: int = 50,
    overlap: int = 5,
    fps: float = 30.0,
    use_extended_features: bool = False,
    use_acceleration: bool = False,
    use_site_id: bool = False,
    split_data: bool = False,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    save_metadata: bool = True
) -> None:

    """
    Main preprocessing pipeline: CSV -> Episodes -> NPZ files.

    Args:
        input_dir: Directory containing CSV files
        output_dir: Directory to save processed episodes
        episode_length: Time steps per episode
        max_vehicles: Max vehicles per frame
        overlap: Overlapping frames between episodes
        fps: Frames per second in original data
        use_extended_features: Include lane and relationship features (10 features total)
        use_acceleration: Include acceleration features
        split_data: Whether to split into train/val/test
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        save_metadata: Whether to save metadata JSON
        use_site_id: Include site_id as an extra feature dimension

    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dt = 1.0 / fps

    # Find CSV files
    csv_files = sorted(list(input_path.glob('drone_*.csv')))
    if len(csv_files) == 0:
        csv_files = sorted(list(input_path.glob('*.csv')))

    logger.info(f"Found {len(csv_files)} CSV files")

    if len(csv_files) == 0:
        logger.error(f"No CSV files found in {input_dir}")
        return

    all_episodes = []
    lane_mapping = {}  # Global lane mapping

    # Process each CSV file
    for csv_file in csv_files:
        logger.info(f"Processing {csv_file.name}...")

        # Load data
        df = load_csv_trajectory(str(csv_file))

        # Compute velocities
        df = compute_velocities(df, dt=dt)

        # Optionally compute acceleration
        if use_acceleration:
            df = compute_acceleration(df, dt=dt)

        # Encode lanes if using extended features
        if use_extended_features and 'lane' in df.columns:
            df['lane_encoded'] = df['lane'].apply(lambda x: encode_lane(x, lane_mapping))

        # Extract episodes
        episodes = extract_episodes(
            df,
            episode_length=episode_length,
            max_vehicles=max_vehicles,
            overlap=overlap,
            use_extended_features=use_extended_features,
            use_acceleration=use_acceleration,
            use_site_id=use_site_id,
        )


        all_episodes.extend(episodes)
        logger.info(f"  Extracted {len(episodes)} episodes")

    logger.info(f"\nTotal episodes extracted: {len(all_episodes)}")

    # Compute statistics
    logger.info("Computing dataset statistics...")
    stats = compute_dataset_statistics(all_episodes)

    logger.info(f"Dataset statistics:")
    logger.info(f"  Total episodes: {stats['n_episodes']}")
    logger.info(f"  Mean vehicles per frame: {stats['mean_vehicles_per_frame']:.2f}")
    logger.info(f"  Max vehicles observed: {stats['max_vehicles_observed']}")
    logger.info(f"  Min vehicles observed: {stats['min_vehicles_observed']}")

    # Split data if requested
    if split_data:
        train_eps, val_eps, test_eps = split_episodes(
            all_episodes, train_ratio, val_ratio, test_ratio
        )

        # Save train set
        np.savez(
            output_path / 'train_episodes.npz',
            states=np.array([ep['states'] for ep in train_eps]),
            masks=np.array([ep['masks'] for ep in train_eps]),
            scene_ids=np.array([ep['scene_id'] for ep in train_eps])
        )

        # Save val set
        if len(val_eps) > 0:
            np.savez(
                output_path / 'val_episodes.npz',
                states=np.array([ep['states'] for ep in val_eps]),
                masks=np.array([ep['masks'] for ep in val_eps]),
                scene_ids=np.array([ep['scene_id'] for ep in val_eps])
            )

        # Save test set
        if len(test_eps) > 0:
            np.savez(
                output_path / 'test_episodes.npz',
                states=np.array([ep['states'] for ep in test_eps]),
                masks=np.array([ep['masks'] for ep in test_eps]),
                scene_ids=np.array([ep['scene_id'] for ep in test_eps])
            )

        logger.info(f"\nSaved train/val/test splits to {output_path}")
    else:
        # Save all episodes
        output_file = output_path / 'episodes.npz'
        np.savez(
            output_file,
            states=np.array([ep['states'] for ep in all_episodes]),
            masks=np.array([ep['masks'] for ep in all_episodes]),
            scene_ids=np.array([ep['scene_id'] for ep in all_episodes])
        )
        logger.info(f"\nSaved {len(all_episodes)} episodes to {output_file}")

    # Save metadata
    if save_metadata:
        # Determine feature count
        F = 6
        if use_acceleration:
            F += 2
        if use_extended_features:
            F = max(F + 3, 10)  # base + lane/preceding/following, at least 10
        if use_site_id:
            F += 1  # Extra dim for site_id

        metadata = {
            'n_episodes': len(all_episodes),
            'episode_length': episode_length,
            'max_vehicles': max_vehicles,
            'n_features': F,
            'fps': fps,
            'dt': dt,
            'use_extended_features': use_extended_features,
            'use_acceleration': use_acceleration,
            'use_site_id': use_site_id,
            'lane_mapping': lane_mapping if use_extended_features else {},
            'statistics': stats
        }


        with open(output_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved metadata to {output_path / 'metadata.json'}")

    logger.info("\nPreprocessing complete!")
    logger.info(f"Feature count: {all_episodes[0]['states'].shape[-1]}")


if __name__ == '__main__':
    # Example usage - Extended features mode (recommended)
    preprocess_trajectories(
        input_dir='../../data/raw',
        output_dir='../../data/processed',
        episode_length=30,
        max_vehicles=50,
        overlap=5,
        fps=30.0,
        use_extended_features=True,  # Enable lane + relationship features
        use_acceleration=False,       # Optional: add acceleration
        split_data=True,              # Split into train/val/test
        save_metadata=True            # Save metadata JSON
    )

    # For basic mode (6 features), use:
    # preprocess_trajectories(
    #     input_dir='../../data/raw',
    #     output_dir='../../data/processed',
    #     use_extended_features=False,
    #     split_data=False
    # )
