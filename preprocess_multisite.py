"""
Multi-Site Data Preprocessing Script

Preprocesses trajectory data from multiple sites (A-I) with 80/10/10 split.

Usage:
    python preprocess_multisite.py
    python preprocess_multisite.py --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1

Author: Traffic World Model Team
"""

import argparse
import sys
from pathlib import Path
import shutil
import pandas as pd

# Get project root and add to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.split_strategy import MultiSiteDataSplitter, chronological_split_episodes
from src.data.preprocess import preprocess_single_site_with_global_timeline
import logging
import numpy as np
import json

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def preprocess_all_sites(
    raw_data_dir: str = None,
    output_dir: str = None,
    use_extended_features: bool = True,
    use_acceleration: bool = True,
    episode_length: int = 80,
    stride: int = 15,
    max_vehicles: int = 50,
    fps: float = 30.0,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    save_split_config: bool = True,
    use_site_id: bool = True,
    use_chronological_split: bool = True,
    sites: list = None,
):
    """
    Preprocess all sites using improved pipeline with global timeline.

    Implements the improved preprocessing approach from improved.md:
    - Per-site global timeline construction (handles frame resets and track_id collisions)
    - Gap detection and continuous segment splitting
    - Fixed-stride episode extraction (T=80, stride=15 by default)
    - Site-specific lane vectorization
    - Chronological split (time-based) to prevent temporal leakage

    Args:
        raw_data_dir: Directory containing site folders (A-I). If None, uses project_root/data/raw
        output_dir: Output directory for processed data. If None, uses project_root/data/processed
        use_extended_features: Use extended features (lane, preceding, following)
        use_acceleration: Include acceleration features
        episode_length: Frames per episode (default: 80 for C=65 + H=15)
        stride: Stride between episode starts (default: 15)
        max_vehicles: Max vehicles per frame
        fps: Frames per second (default: 30.0 for 30 FPS data)
        train_ratio: Training ratio (default: 0.7)
        val_ratio: Validation ratio (default: 0.15)
        test_ratio: Test ratio (default: 0.15)
        save_split_config: Save split config to JSON
        use_site_id: Whether to add site_id as an extra feature dimension (default: True)
        use_chronological_split: Use time-based split (default: True)
        sites: List of site names to process (e.g., ['A', 'B']). If None, processes all available sites
    """
    # Set default paths relative to project root
    if raw_data_dir is None:
        raw_data_dir = PROJECT_ROOT / 'data' / 'raw'
    else:
        raw_data_dir = Path(raw_data_dir)

    if output_dir is None:
        output_dir = PROJECT_ROOT / 'data' / 'processed'
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("IMPROVED MULTI-SITE PREPROCESSING")
    logger.info("=" * 60)
    logger.info(f"Raw data dir: {raw_data_dir.absolute()}")
    logger.info(f"Output dir: {output_dir.absolute()}")
    logger.info(f"Episode config: T={episode_length} (C={episode_length-15}, H=15), stride={stride}")
    logger.info(f"FPS: {fps} (dt={1.0/fps:.4f}s)")
    logger.info(f"Extended features: {use_extended_features}")
    logger.info(f"Acceleration: {use_acceleration}")
    logger.info(f"Site ID: {use_site_id}")
    logger.info(f"Split type: {'Chronological' if use_chronological_split else 'Random'}")
    logger.info(f"Split ratios: {train_ratio:.0%} train / {val_ratio:.0%} val / {test_ratio:.0%} test")
    logger.info("=" * 60)

    # Get available sites
    all_sites = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']

    # Filter by requested sites if specified
    if sites is not None:
        # Convert to uppercase and filter
        requested_sites = [s.upper() for s in sites]
        available_sites = [s for s in requested_sites if (raw_data_dir / s).exists()]
        missing_sites = [s for s in requested_sites if s not in available_sites]

        if missing_sites:
            logger.warning(f"Requested sites not found: {missing_sites}")
    else:
        # Process all available sites
        available_sites = [s for s in all_sites if (raw_data_dir / s).exists()]

    if not available_sites:
        logger.error(f"No site folders found in {raw_data_dir}")
        return

    logger.info(f"Processing sites: {available_sites}")

    # Process each site with split-aware episode extraction to avoid overlap
    # This implements "Scheme A" from IMPROVEMENTS_todo.md:
    # 1. For each site, determine frame cutoffs based on split ratios
    # 2. Extract episodes independently within each split's frame range
    # 3. This prevents episodes from spanning split boundaries when stride < T

    splits = {'train': [], 'val': [], 'test': []}
    lane_mapping = {}  # Shared lane mapping across all sites

    for site_name in available_sites:
        site_dir = raw_data_dir / site_name
        csv_files = sorted(list(site_dir.glob("*.csv")))

        if len(csv_files) == 0:
            logger.warning(f"No CSV files found for site {site_name}")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Processing Site {site_name}")
        logger.info(f"{'='*60}")

        if use_chronological_split:
            # SCHEME A: Determine frame cutoffs first, then extract episodes per split
            # This prevents overlap when stride < T

            # First pass: build global timeline to determine frame range
            from src.data.preprocess import build_global_timeline
            site_dfs = []
            for csv_file in csv_files:
                df = pd.read_csv(csv_file)
                if 'site' not in df.columns:
                    df['site'] = site_name
                site_dfs.append((csv_file.name, df))

            merged_df = build_global_timeline(site_dfs, fps=fps)
            min_frame = merged_df['global_frame'].min()
            max_frame = merged_df['global_frame'].max()
            total_frames = max_frame - min_frame + 1

            # Minimum frames needed for a split to generate at least one episode
            min_frames_needed = episode_length + stride

            # Calculate frame cutoffs with boundary checks
            train_end = min_frame + int(total_frames * train_ratio)
            val_end = train_end + int(total_frames * val_ratio)

            # Check if val/test splits are too short
            val_frames = val_end - train_end
            test_frames = max_frame - val_end

            if val_frames < min_frames_needed:
                logger.warning(f"  Val split too short ({val_frames} frames < {min_frames_needed} needed)")
                logger.warning(f"  Adjusting to ensure at least {min_frames_needed} frames for val")
                val_end = min(train_end + min_frames_needed, max_frame - min_frames_needed)
                train_end = val_end - min_frames_needed

            if test_frames < min_frames_needed:
                logger.warning(f"  Test split too short ({test_frames} frames < {min_frames_needed} needed)")
                logger.warning(f"  Adjusting to ensure at least {min_frames_needed} frames for test")
                val_end = max(min_frame + min_frames_needed, max_frame - min_frames_needed)

            logger.info(f"  Total frames: {total_frames} (frames {min_frame}-{max_frame})")
            logger.info(f"  Train: frames {min_frame}-{train_end} ({train_end - min_frame + 1} frames)")
            logger.info(f"  Val:   frames {train_end+1}-{val_end} ({val_end - train_end} frames)")
            logger.info(f"  Test:  frames {val_end+1}-{max_frame} ({max_frame - val_end} frames)")

            # Extract episodes for each split independently
            for split_name, frame_range in [
                ('train', (min_frame, train_end)),
                ('val', (train_end + 1, val_end)),
                ('test', (val_end + 1, max_frame))
            ]:
                episodes, lane_mapping = preprocess_single_site_with_global_timeline(
                    csv_files=csv_files,
                    site_name=site_name,
                    episode_length=episode_length,
                    stride=stride,
                    max_vehicles=max_vehicles,
                    fps=fps,
                    use_extended_features=use_extended_features,
                    use_acceleration=use_acceleration,
                    use_site_id=use_site_id,
                    lane_mapping=lane_mapping,
                    frame_range=frame_range,
                    split_name=split_name,
                )
                splits[split_name].extend(episodes)
        else:
            # Legacy random split: extract all episodes then shuffle
            episodes, lane_mapping = preprocess_single_site_with_global_timeline(
                csv_files=csv_files,
                site_name=site_name,
                episode_length=episode_length,
                stride=stride,
                max_vehicles=max_vehicles,
                fps=fps,
                use_extended_features=use_extended_features,
                use_acceleration=use_acceleration,
                use_site_id=use_site_id,
                lane_mapping=lane_mapping,
            )

            # Random split
            np.random.shuffle(episodes)
            n_total = len(episodes)
            n_train = int(n_total * train_ratio)
            n_val = int(n_total * val_ratio)
            splits['train'].extend(episodes[:n_train])
            splits['val'].extend(episodes[n_train:n_train + n_val])
            splits['test'].extend(episodes[n_train + n_val:])

    total_episodes = sum(len(splits[s]) for s in ['train', 'val', 'test'])
    logger.info(f"\n{'='*60}")
    logger.info(f"Total episodes extracted: {total_episodes}")
    logger.info(f"  Train: {len(splits['train'])}")
    logger.info(f"  Val:   {len(splits['val'])}")
    logger.info(f"  Test:  {len(splits['test'])}")
    logger.info(f"{'='*60}")

    # Save each split
    logger.info("\n" + "=" * 60)
    logger.info("SAVING SPLITS")
    logger.info("=" * 60)

    for split_name in ['train', 'val', 'test']:
        eps = splits[split_name]
        if len(eps) == 0:
            logger.warning(f"No episodes in {split_name} split!")
            continue

        states = np.array([ep['states'] for ep in eps])
        masks = np.array([ep['masks'] for ep in eps])
        scene_ids = np.array([ep['scene_id'] for ep in eps])

        # Extract temporal metadata for validation
        start_frames = np.array([ep.get('episode_start_global_frame', ep.get('start_frame', 0)) for ep in eps])
        end_frames = np.array([ep.get('episode_end_global_frame', ep.get('end_frame', 0)) for ep in eps])

        output_file = output_dir / f"{split_name}_episodes.npz"
        np.savez(
            output_file,
            states=states,
            masks=masks,
            scene_ids=scene_ids,
            start_frames=start_frames,
            end_frames=end_frames
        )
        logger.info(f"✅ Saved {len(eps)} episodes to {output_file}")
        logger.info(f"   Frame range: {start_frames.min()}-{end_frames.max()}")

    # Save metadata with validation info
    F = 6  # Basic features
    if use_acceleration:
        F += 2
    if use_extended_features:
        F = max(F + 3, 10)
    if use_site_id:
        F += 1

    # Calculate num_lanes dynamically from lane_mapping
    num_lanes = len(lane_mapping) + 1  # +1 for unknown lane (id=0)

    # Feature layout documentation with index tracking
    feature_layout = {
        0: "center_x",
        1: "center_y",
        2: "vx",
        3: "vy"
    }
    idx = 4
    if use_acceleration:
        feature_layout[idx] = "ax"
        feature_layout[idx + 1] = "ay"
        idx += 2
    feature_layout[idx] = "angle"
    idx += 1

    # Track class_id index
    class_id_idx = idx
    feature_layout[idx] = "class_id"
    idx += 1

    # Track lane_id index
    lane_id_idx = None
    if use_extended_features:
        lane_id_idx = idx
        feature_layout[idx] = "lane_id"
        idx += 1
        feature_layout[idx] = "has_preceding"
        idx += 1
        feature_layout[idx] = "has_following"
        idx += 1

    # Track site_id index
    site_id_idx = None
    if use_site_id:
        site_id_idx = idx
        feature_layout[idx] = "site_id"
        idx += 1

    # Validation info with correct indices
    # angle is at index 6 (before class_id at 7, lane_id at 8)
    angle_idx = 6
    validation_info = {
        'num_lanes': num_lanes,
        'lane_token_format': 'site:lane (e.g., A:A1, B:crossroads1)',
        'discrete_features': {
            'lane_id': lane_id_idx,
            'class_id': class_id_idx,
            'site_id': site_id_idx
        },
        'angle_idx': angle_idx,  # angle feature (radian, periodic)
        'do_not_normalize': ['lane_id', 'class_id', 'site_id', 'angle'],  # angle should not use z-score normalization
        'context_length': episode_length - 15,  # C = T - H
        'rollout_horizon': 15  # H
    }

    metadata = {
        'n_episodes': total_episodes,
        'episode_length': episode_length,
        'context_length': episode_length - 15,  # C = T - H
        'rollout_horizon': 15,  # H
        'stride': stride,
        'max_vehicles': max_vehicles,
        'n_features': F,
        'fps': fps,
        'dt': 1.0 / fps,
        'use_extended_features': use_extended_features,
        'use_acceleration': use_acceleration,
        'use_site_id': use_site_id,
        'use_chronological_split': use_chronological_split,
        'train_ratio': train_ratio,
        'val_ratio': val_ratio,
        'test_ratio': test_ratio,
        'lane_mapping': lane_mapping,
        'sites_processed': available_sites,
        'split_sizes': {
            'train': len(splits['train']),
            'val': len(splits['val']),
            'test': len(splits['test'])
        },
        'feature_layout': feature_layout,
        'validation_info': validation_info
    }

    metadata_path = output_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"✅ Saved metadata to {metadata_path}")

    # Save split config if requested
    if save_split_config:
        split_config = {
            'type': 'chronological' if use_chronological_split else 'random',
            'train_ratio': train_ratio,
            'val_ratio': val_ratio,
            'test_ratio': test_ratio,
            'episode_length': episode_length,
            'stride': stride,
            'total_episodes': total_episodes,
            'splits': {
                'train': len(splits['train']),
                'val': len(splits['val']),
                'test': len(splits['test'])
            }
        }
        config_path = output_dir / "split_config.json"
        with open(config_path, 'w') as f:
            json.dump(split_config, f, indent=2)
        logger.info(f"✅ Saved split config to {config_path}")

    logger.info("\n" + "=" * 60)
    logger.info("✅ PREPROCESSING COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"Output files:")
    logger.info(f"  - {output_dir}/train_episodes.npz ({len(splits['train'])} episodes)")
    logger.info(f"  - {output_dir}/val_episodes.npz ({len(splits['val'])} episodes)")
    logger.info(f"  - {output_dir}/test_episodes.npz ({len(splits['test'])} episodes)")
    logger.info(f"  - {output_dir}/metadata.json")
    if save_split_config:
        logger.info(f"  - {output_dir}/split_config.json")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess multi-site drone trajectory data (80/10/10 split)"
    )

    # Paths
    parser.add_argument('--raw_data_dir', type=str, default=None,
                       help='Input directory with site folders (default: project_root/data/raw)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: project_root/data/processed)')

    # Feature configuration
    parser.add_argument('--use_extended_features', action='store_true', default=True,
                       help='Use extended features (10 instead of 6)')
    parser.add_argument('--use_acceleration', action='store_true', default=True,
                       help='Include acceleration features')

    # Episode configuration
    parser.add_argument('--episode_length', type=int, default=80,
                       help='Frames per episode (default: 80 for C=65 + H=15)')
    parser.add_argument('--stride', type=int, default=15,
                       help='Stride between episode starts (default: 15)')
    parser.add_argument('--max_vehicles', type=int, default=50,
                       help='Max vehicles per frame (default: 50)')
    parser.add_argument('--fps', type=float, default=30.0,
                       help='Frames per second (default: 30.0)')

    # Split ratios
    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='Training ratio (default: 0.7)')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                       help='Validation ratio (default: 0.15)')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                       help='Test ratio (default: 0.15)')
    parser.add_argument('--use_chronological_split', action='store_true', default=True,
                       help='Use chronological time-based split (default: True)')
    parser.add_argument('--use_random_split', dest='use_chronological_split',
                       action='store_false',
                       help='Use random split instead of chronological')

    # Site selection
    parser.add_argument('--sites', type=str, nargs='+', default=None,
                       help='Specific sites to process (e.g., A B C). If not specified, processes all available sites')

    # Other
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--no_save_split_config', action='store_true',
                       help='Do not save split config JSON')

    args = parser.parse_args()

    # Run preprocessing
    preprocess_all_sites(
        raw_data_dir=args.raw_data_dir,
        output_dir=args.output_dir,
        use_extended_features=args.use_extended_features,
        use_acceleration=args.use_acceleration,
        episode_length=args.episode_length,
        stride=args.stride,
        max_vehicles=args.max_vehicles,
        fps=args.fps,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        save_split_config=not args.no_save_split_config,
        use_chronological_split=args.use_chronological_split,
        sites=args.sites
    )


if __name__ == '__main__':
    main()
