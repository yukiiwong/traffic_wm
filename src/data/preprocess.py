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


def encode_lane(lane_str: str, lane_mapping: Dict[str, int], site: str = None) -> int:
    """
    Encode lane string to integer with site-specific token to prevent collisions.

    Args:
        lane_str: Lane identifier (e.g., 'A1', 'B2', 'crossroads1')
        lane_mapping: Dictionary mapping lane names to integers
        site: Site identifier (e.g., 'A', 'B', 'C') for creating unique lane tokens

    Returns:
        Encoded lane ID
    """
    if pd.isna(lane_str):
        return 0  # Unknown lane

    lane_str = str(lane_str).strip()

    # Create site-specific lane token to prevent cross-site collisions
    if site is not None:
        lane_token = f"{site}:{lane_str}"
    else:
        lane_token = lane_str

    if lane_token not in lane_mapping:
        lane_mapping[lane_token] = len(lane_mapping) + 1

    return lane_mapping[lane_token]


def build_global_timeline(site_dfs: List[Tuple[str, pd.DataFrame]], fps: float = 30.0) -> pd.DataFrame:
    """
    Build a per-site global timeline by concatenating multiple CSV files.

    Handles:
    - Frame resets across files
    - Track ID collisions across files
    - Creates global_frame and global_track_id

    Args:
        site_dfs: List of tuples (filename, DataFrame) for a single site
        fps: Frames per second (default: 30.0)

    Returns:
        Merged DataFrame with global_frame and global_track_id columns
    """
    merged_dfs = []

    # Sort files by filename to maintain consistent ordering
    site_dfs = sorted(site_dfs, key=lambda x: x[0])

    for file_id, (filename, df) in enumerate(site_dfs):
        df = df.copy()

        # Create globally unique track ID: file_id * 1_000_000 + original track_id
        df['global_track_id'] = file_id * 1_000_000 + df['track_id']

        # Offset frames to create continuous timeline
        if file_id == 0:
            df['global_frame'] = df['frame']
            prev_max_frame = df['frame'].max()
        else:
            # Add offset based on previous file's max frame + 1
            df['global_frame'] = df['frame'] + prev_max_frame + 1
            prev_max_frame = df['global_frame'].max()

        # Store original file_id for reference
        df['file_id'] = file_id
        df['source_file'] = filename

        merged_dfs.append(df)

    # Concatenate all files
    merged_df = pd.concat(merged_dfs, ignore_index=True)

    # Sort by global_frame, then global_track_id
    merged_df = merged_df.sort_values(['global_frame', 'global_track_id']).reset_index(drop=True)

    logger.info(f"  Built global timeline: {len(site_dfs)} files, "
                f"{merged_df['global_frame'].nunique()} unique frames, "
                f"{merged_df['global_track_id'].nunique()} unique vehicles")

    return merged_df


def detect_gaps_and_split_segments(df: pd.DataFrame, max_gap: int = 1) -> List[Tuple[int, int]]:
    """
    Detect gaps in the global_frame timeline and split into continuous segments.

    Args:
        df: DataFrame with global_frame column
        max_gap: Maximum allowed gap between consecutive frames (default: 1)

    Returns:
        List of (start_frame, end_frame) tuples for continuous segments
    """
    unique_frames = sorted(df['global_frame'].unique())

    if len(unique_frames) == 0:
        return []

    segments = []
    segment_start = unique_frames[0]

    for i in range(1, len(unique_frames)):
        gap = unique_frames[i] - unique_frames[i-1]

        if gap > max_gap:
            # Found a gap, close current segment
            segments.append((segment_start, unique_frames[i-1]))
            segment_start = unique_frames[i]

    # Add final segment
    segments.append((segment_start, unique_frames[-1]))

    logger.info(f"  Detected {len(segments)} continuous segments")
    for i, (start, end) in enumerate(segments):
        logger.info(f"    Segment {i+1}: frames {start}-{end} (length: {end-start+1})")

    return segments


def extract_fixed_stride_episodes(
    df: pd.DataFrame,
    segments: List[Tuple[int, int]],
    episode_length: int = 80,
    stride: int = 15,
    max_vehicles: int = 50,
    use_extended_features: bool = False,
    use_acceleration: bool = False,
    use_site_id: bool = False,
    use_relative_features: bool = True,  # Add relative position/velocity features
) -> List[Dict[str, np.ndarray]]:
    """
    Extract fixed-stride episodes from continuous segments.

    Args:
        df: DataFrame with global_frame and global_track_id
        segments: List of (start_frame, end_frame) continuous segments
        episode_length: Length of each episode in frames (T)
        stride: Step size between episode starts (S)
        max_vehicles: Maximum vehicles per frame
        use_extended_features: Include lane and relationship features
        use_acceleration: Include acceleration features
        use_site_id: Include site_id feature

    Returns:
        List of episode dictionaries
    """
    episodes = []

    for seg_idx, (seg_start, seg_end) in enumerate(segments):
        # Get frames in this segment
        segment_frames = sorted(df[
            (df['global_frame'] >= seg_start) &
            (df['global_frame'] <= seg_end)
        ]['global_frame'].unique())

        # Slide window with fixed stride
        for start_idx in range(0, len(segment_frames) - episode_length + 1, stride):
            episode_frames = segment_frames[start_idx:start_idx + episode_length]

            # Extract episode using global_frame instead of original frame
            episode_data = extract_single_episode_from_global(
                df,
                episode_frames,
                max_vehicles,
                use_extended_features=use_extended_features,
                use_acceleration=use_acceleration,
                use_site_id=use_site_id,
                use_relative_features=use_relative_features,
            )

            # Add metadata about segment and global timing
            episode_data['segment_id'] = seg_idx
            episode_data['episode_start_global_frame'] = episode_frames[0]
            episode_data['episode_end_global_frame'] = episode_frames[-1]

            episodes.append(episode_data)

    logger.info(f"  Extracted {len(episodes)} episodes with stride={stride}")

    return episodes


def extract_single_episode_from_global(
    df: pd.DataFrame,
    global_frames: List[int],
    max_vehicles: int,
    use_extended_features: bool = False,
    use_acceleration: bool = False,
    use_site_id: bool = False,    use_relative_features: bool = True,) -> Dict[str, np.ndarray]:
    """
    Extract a single episode using global_frame instead of original frame.

    This version uses global_frame for indexing and maintains stable slot assignment.

    Args:
        df: DataFrame with global_frame and global_track_id
        global_frames: List of global frame indices for this episode
        max_vehicles: Maximum number of vehicles (K)
        use_extended_features: Include extended features
        use_acceleration: Include acceleration
        use_site_id: Include site_id feature

    Returns:
        Dictionary with 'states', 'masks', 'scene_id', etc.
    """
    T = len(global_frames)
    site_id = get_site_id_from_frames(df, global_frames)

    # Determine number of features
    F = 6  # Basic: x, y, vx, vy, angle, type
    if use_acceleration:
        F += 2  # ax, ay
    if use_extended_features:
        F += 3  # lane, has_preceding, has_following
        if F < 10:
            F = 10
    if use_site_id:
        F += 1
    if use_relative_features:
        F += 8  # preceding (dx, dy, dvx, dvy) + following (dx, dy, dvx, dvy)

    states = np.zeros((T, max_vehicles, F), dtype=np.float32)
    masks = np.zeros((T, max_vehicles), dtype=np.float32)
    track_ids = np.zeros((T, max_vehicles), dtype=np.int64)

    # Build stable slot assignment for this episode
    # Collect all vehicles that appear in this episode
    episode_data = df[df['global_frame'].isin(global_frames)]
    unique_vehicles = episode_data['global_track_id'].unique()

    # Assign slots to vehicles (prioritize by frequency or first appearance)
    vehicle_counts = episode_data['global_track_id'].value_counts()
    top_vehicles = vehicle_counts.head(max_vehicles).index.tolist()
    vehicle_to_slot = {vid: slot for slot, vid in enumerate(top_vehicles)}

    # Process each time step
    for t, global_frame in enumerate(global_frames):
        frame_data = df[df['global_frame'] == global_frame].copy()

        for _, row in frame_data.iterrows():
            vehicle_id = row['global_track_id']

            # Check if this vehicle has an assigned slot
            if vehicle_id not in vehicle_to_slot:
                continue

            k = vehicle_to_slot[vehicle_id]

            # Fill features
            feature_idx = 0

            # Position
            states[t, k, 0:2] = [row['center_x'], row['center_y']]
            feature_idx = 2

            # Velocity
            states[t, k, 2:4] = [row['vx'], row['vy']]
            feature_idx = 4

            # Acceleration (if enabled)
            if use_acceleration and 'ax' in row:
                states[t, k, 4:6] = [row['ax'], row['ay']]
                feature_idx = 6

            # Heading
            states[t, k, feature_idx] = row['angle']
            feature_idx += 1

            # Vehicle type
            states[t, k, feature_idx] = row['class_id']
            feature_idx += 1

            # Extended features
            if use_extended_features:
                if 'lane_encoded' in row:
                    states[t, k, feature_idx] = row['lane_encoded']
                feature_idx += 1

                if 'preceding_id' in row:
                    states[t, k, feature_idx] = 0.0 if pd.isna(row['preceding_id']) else 1.0
                feature_idx += 1

                if 'following_id' in row:
                    states[t, k, feature_idx] = 0.0 if pd.isna(row['following_id']) else 1.0
                feature_idx += 1

            # Site ID
            if use_site_id:
                states[t, k, feature_idx] = float(site_id)
                feature_idx += 1

            # Relative features (preceding and following vehicle)
            if use_relative_features:
                # Preceding vehicle relative features
                if 'preceding_id' in row and not pd.isna(row['preceding_id']):
                    prec_id = row['preceding_id']
                    prec_data = frame_data[frame_data['global_track_id'] == prec_id]
                    if len(prec_data) > 0:
                        prec = prec_data.iloc[0]
                        states[t, k, feature_idx] = prec['center_x'] - row['center_x']  # dx
                        states[t, k, feature_idx+1] = prec['center_y'] - row['center_y']  # dy
                        states[t, k, feature_idx+2] = prec['vx'] - row['vx']  # dvx
                        states[t, k, feature_idx+3] = prec['vy'] - row['vy']  # dvy
                    # else: keep zeros
                # else: keep zeros (no preceding vehicle)
                feature_idx += 4

                # Following vehicle relative features
                if 'following_id' in row and not pd.isna(row['following_id']):
                    foll_id = row['following_id']
                    foll_data = frame_data[frame_data['global_track_id'] == foll_id]
                    if len(foll_data) > 0:
                        foll = foll_data.iloc[0]
                        states[t, k, feature_idx] = foll['center_x'] - row['center_x']  # dx
                        states[t, k, feature_idx+1] = foll['center_y'] - row['center_y']  # dy
                        states[t, k, feature_idx+2] = foll['vx'] - row['vx']  # dvx
                        states[t, k, feature_idx+3] = foll['vy'] - row['vy']  # dvy
                    # else: keep zeros
                # else: keep zeros (no following vehicle)
                feature_idx += 4

            # Mark as valid
            masks[t, k] = 1.0
            track_ids[t, k] = vehicle_id

    return {
        'states': states,
        'masks': masks,
        'scene_id': site_id,
        'track_ids': track_ids,
        'start_frame': global_frames[0],
        'end_frame': global_frames[-1]
    }

def get_site_id_from_frames(df: pd.DataFrame, frames: List[int]) -> int:
    """
    Derive a numeric site_id (0,1,2,...) from the 'site' column.

    Convention:
      - If site is like 'A', 'B', ..., 'I'  -> A=0, B=1, ...
      - If site is like 'Site A'           -> A=0, B=1, ...
      - Otherwise                          -> 0 (default)

    Args:
        df: Full trajectory DataFrame (must contain 'site' if used)
        frames: Frames belonging to this episode (global_frame or original frame)

    Returns:
        Integer site_id in [0, ...]
    """
    if 'site' not in df.columns:
        return 0

    # Try global_frame first, then fall back to frame
    frame_col = 'global_frame' if 'global_frame' in df.columns else 'frame'

    # Prefer frames within this episode; fallback to whole df if empty
    df_sub = df[df[frame_col].isin(frames)]
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
    use_relative_features: bool = True,
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

            # Site feature (if enabled) — use the same site_id for all vehicles in this episode
            if use_site_id:
                # feature_idx currently points to the next free feature slot
                states[t, :n_vehicles, feature_idx] = float(site_id)
                feature_idx += 1

            # Relative features (preceding and following vehicle)
            if use_relative_features:
                for k in range(n_vehicles):
                    vehicle_row = frame_data.iloc[k]
                    
                    # Preceding vehicle relative features
                    if 'preceding_id' in frame_data.columns and not pd.isna(vehicle_row['preceding_id']):
                        prec_id = vehicle_row['preceding_id']
                        prec_data = frame_data[frame_data['track_id'] == prec_id]
                        if len(prec_data) > 0:
                            prec = prec_data.iloc[0]
                            states[t, k, feature_idx] = prec['center_x'] - vehicle_row['center_x']
                            states[t, k, feature_idx+1] = prec['center_y'] - vehicle_row['center_y']
                            states[t, k, feature_idx+2] = prec['vx'] - vehicle_row['vx']
                            states[t, k, feature_idx+3] = prec['vy'] - vehicle_row['vy']
                    # Preceding features at feature_idx:feature_idx+4
                    
                    # Following vehicle relative features
                    foll_feature_idx = feature_idx + 4
                    if 'following_id' in frame_data.columns and not pd.isna(vehicle_row['following_id']):
                        foll_id = vehicle_row['following_id']
                        foll_data = frame_data[frame_data['track_id'] == foll_id]
                        if len(foll_data) > 0:
                            foll = foll_data.iloc[0]
                            states[t, k, foll_feature_idx] = foll['center_x'] - vehicle_row['center_x']
                            states[t, k, foll_feature_idx+1] = foll['center_y'] - vehicle_row['center_y']
                            states[t, k, foll_feature_idx+2] = foll['vx'] - vehicle_row['vx']
                            states[t, k, foll_feature_idx+3] = foll['vy'] - vehicle_row['vy']
                    # Following features at foll_feature_idx:foll_feature_idx+4

            # Store track IDs and masks
            track_ids[t, :n_vehicles] = frame_data['track_id'].values[:n_vehicles]
            masks[t, :n_vehicles] = 1.0

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


def preprocess_single_site_with_global_timeline(
    csv_files: List[Path],
    site_name: str,
    episode_length: int = 80,
    stride: int = 15,
    max_vehicles: int = 50,
    fps: float = 30.0,
    use_extended_features: bool = False,
    use_acceleration: bool = False,
    use_site_id: bool = False,
    lane_mapping: Optional[Dict[str, int]] = None,
    frame_range: Optional[Tuple[int, int]] = None,
    split_name: Optional[str] = None,
) -> Tuple[List[Dict[str, np.ndarray]], Dict[str, int]]:
    """
    Preprocess a single site using global timeline approach.

    This implements the improved pipeline:
    1. Build global timeline across all CSV files for this site
    2. Compute velocities and accelerations
    3. Encode lanes with site-specific tokens
    4. Detect gaps and split into continuous segments
    5. Extract fixed-stride episodes (T=80, stride=15 by default)

    Args:
        csv_files: List of CSV file paths for this site
        site_name: Site identifier (e.g., 'A', 'B', 'C')
        episode_length: Episode length in frames (default: 80 for C=65 + H=15)
        stride: Stride between episode starts (default: 15)
        max_vehicles: Maximum vehicles per frame
        fps: Frames per second (default: 30.0)
        use_extended_features: Include lane and relationship features
        use_acceleration: Include acceleration features
        use_site_id: Include site_id feature
        lane_mapping: Shared lane mapping dictionary (modified in-place)
        frame_range: Optional (min_frame, max_frame) to restrict episode extraction
                     to prevent overlap across splits when stride < T
        split_name: Optional name of split ('train', 'val', 'test') for logging

    Returns:
        (episodes, updated_lane_mapping)
    """
    if lane_mapping is None:
        lane_mapping = {}

    dt = 1.0 / fps

    logger.info(f"Processing site {site_name} with {len(csv_files)} files...")

    # Step 1: Load all CSV files for this site
    site_dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        # Ensure site column exists
        if 'site' not in df.columns:
            df['site'] = site_name
        site_dfs.append((csv_file.name, df))

    # Step 2: Build global timeline
    merged_df = build_global_timeline(site_dfs, fps=fps)

    # Step 3: Compute velocities (using global_track_id)
    merged_df = merged_df.sort_values(['global_track_id', 'global_frame']).copy()
    merged_df['vx'] = merged_df.groupby('global_track_id')['center_x'].diff() / dt
    merged_df['vy'] = merged_df.groupby('global_track_id')['center_y'].diff() / dt
    merged_df['vx'] = merged_df['vx'].fillna(0.0)
    merged_df['vy'] = merged_df['vy'].fillna(0.0)

    # Step 4: Compute acceleration if needed
    if use_acceleration:
        merged_df['ax'] = merged_df.groupby('global_track_id')['vx'].diff() / dt
        merged_df['ay'] = merged_df.groupby('global_track_id')['vy'].diff() / dt
        merged_df['ax'] = merged_df['ax'].fillna(0.0)
        merged_df['ay'] = merged_df['ay'].fillna(0.0)

    # Step 5: Encode lanes with site-specific tokens
    if use_extended_features and 'lane' in merged_df.columns:
        merged_df['lane_encoded'] = merged_df['lane'].apply(
            lambda x: encode_lane(x, lane_mapping, site=site_name)
        )

    # Step 6: Detect gaps and split into continuous segments
    segments = detect_gaps_and_split_segments(merged_df, max_gap=1)

    # Step 6.5: Apply frame_range restriction if provided
    if frame_range is not None:
        min_frame, max_frame = frame_range
        # Filter segments to only include those within the frame range
        filtered_segments = []
        for seg_start, seg_end in segments:
            # Clip segment to frame_range
            clipped_start = max(seg_start, min_frame)
            clipped_end = min(seg_end, max_frame)
            # Only include if there's still a valid range
            if clipped_start <= clipped_end and (clipped_end - clipped_start + 1) >= episode_length:
                filtered_segments.append((clipped_start, clipped_end))
        segments = filtered_segments
        split_info = f" ({split_name}, frames {min_frame}-{max_frame})" if split_name else ""
        logger.info(f"  Applied frame range restriction{split_info}: {len(segments)} segments remaining")

    # Step 7: Extract fixed-stride episodes
    episodes = extract_fixed_stride_episodes(
        merged_df,
        segments,
        episode_length=episode_length,
        stride=stride,
        max_vehicles=max_vehicles,
        use_extended_features=use_extended_features,
        use_acceleration=use_acceleration,
        use_site_id=use_site_id,
        use_relative_features=use_relative_features,
    )

    split_info = f" ({split_name})" if split_name else ""
    logger.info(f"Site {site_name}{split_info}: extracted {len(episodes)} episodes")

    return episodes, lane_mapping


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
    use_relative_features: bool = True,
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

        # Encode lanes if using extended features (with site-specific tokens)
        if use_extended_features and 'lane' in df.columns:
            # Get site from dataframe
            site = df['site'].iloc[0] if 'site' in df.columns else None
            df['lane_encoded'] = df['lane'].apply(lambda x: encode_lane(x, lane_mapping, site))

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
        if use_relative_features:
            F += 8  # preceding (dx, dy, dvx, dvy) + following (dx, dy, dvx, dvy)

        # Build feature layout mapping
        feature_layout = {}
        idx = 0
        feature_layout[str(idx)] = "center_x"; idx += 1
        feature_layout[str(idx)] = "center_y"; idx += 1
        feature_layout[str(idx)] = "vx"; idx += 1
        feature_layout[str(idx)] = "vy"; idx += 1
        
        if use_acceleration:
            feature_layout[str(idx)] = "ax"; idx += 1
            feature_layout[str(idx)] = "ay"; idx += 1
        
        feature_layout[str(idx)] = "angle"; idx += 1
        angle_idx = idx - 1  # Remember angle index
        feature_layout[str(idx)] = "class_id"; idx += 1
        class_id_idx = idx - 1
        
        if use_extended_features:
            feature_layout[str(idx)] = "lane_id"; idx += 1
            lane_id_idx = idx - 1
            feature_layout[str(idx)] = "has_preceding"; idx += 1
            feature_layout[str(idx)] = "has_following"; idx += 1
        
        if use_site_id:
            feature_layout[str(idx)] = "site_id"; idx += 1
            site_id_idx = idx - 1
        
        if use_relative_features:
            feature_layout[str(idx)] = "preceding_rel_x"; idx += 1
            feature_layout[str(idx)] = "preceding_rel_y"; idx += 1
            feature_layout[str(idx)] = "preceding_rel_vx"; idx += 1
            feature_layout[str(idx)] = "preceding_rel_vy"; idx += 1
            feature_layout[str(idx)] = "following_rel_x"; idx += 1
            feature_layout[str(idx)] = "following_rel_y"; idx += 1
            feature_layout[str(idx)] = "following_rel_vx"; idx += 1
            feature_layout[str(idx)] = "following_rel_vy"; idx += 1

        # Discrete features
        discrete_features = {"class_id": class_id_idx}
        do_not_normalize = ["class_id", "angle"]
        
        if use_extended_features:
            discrete_features["lane_id"] = lane_id_idx
            do_not_normalize.append("lane_id")
        
        if use_site_id:
            discrete_features["site_id"] = site_id_idx
            do_not_normalize.append("site_id")

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
            'use_relative_features': use_relative_features,
            'feature_layout': feature_layout,
            'validation_info': {
                'discrete_features': discrete_features,
                'do_not_normalize': do_not_normalize,
                'angle_idx': angle_idx,
            },
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
        use_relative_features=True,   # Enable relative position/velocity features
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
