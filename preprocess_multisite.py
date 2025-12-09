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

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.data.split_strategy import MultiSiteDataSplitter
from src.data.preprocess import preprocess_trajectories
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def preprocess_all_sites(
    raw_data_dir: str = "data/raw",
    output_dir: str = "data/processed",
    use_extended_features: bool = True,
    use_acceleration: bool = True,
    episode_length: int = 30,
    max_vehicles: int = 50,
    overlap: int = 5,
    fps: float = 30.0,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    save_split_config: bool = True
):
    """
    Preprocess all sites and split into train/val/test.

    Args:
        raw_data_dir: Directory containing site folders (A-I)
        output_dir: Output directory for processed data
        use_extended_features: Use 10 features instead of 6
        use_acceleration: Include acceleration features
        episode_length: Frames per episode
        max_vehicles: Max vehicles per frame
        overlap: Frame overlap between episodes
        fps: Frames per second
        train_ratio: Training ratio (default: 0.8)
        val_ratio: Validation ratio (default: 0.1)
        test_ratio: Test ratio (default: 0.1)
        seed: Random seed for reproducibility
        save_split_config: Save split config to JSON
    """
    logger.info("=" * 60)
    logger.info("MULTI-SITE PREPROCESSING (80/10/10 SPLIT)")
    logger.info("=" * 60)
    logger.info(f"Raw data dir: {raw_data_dir}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Extended features: {use_extended_features}")
    logger.info(f"Split: {train_ratio:.0%} train / {val_ratio:.0%} val / {test_ratio:.0%} test")
    logger.info("=" * 60)

    # Initialize splitter and get splits
    splitter = MultiSiteDataSplitter(raw_data_dir=raw_data_dir)

    splits = splitter.split_data(
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed
    )

    # Save split config
    if save_split_config:
        config_path = Path(output_dir) / "split_config.json"
        splitter.save_split_config(splits, str(config_path))

    # Create temporary directories for each split
    temp_dirs = {}
    for split_name in ['train', 'val', 'test']:
        temp_dir = Path(output_dir) / f"temp_{split_name}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_dirs[split_name] = temp_dir

        # Copy files to temp directory
        for file_path in splits[split_name]:
            dest_path = temp_dir / f"{file_path.parent.name}_{file_path.name}"

            # Read and add site column if not present
            df = pd.read_csv(file_path)
            if 'site' not in df.columns:
                site_name = file_path.parent.name
                df['site'] = f"Site {site_name}"

            df.to_csv(dest_path, index=False)

        logger.info(f"Prepared {len(splits[split_name])} files for {split_name}")

    # Preprocess each split
    logger.info("\n" + "=" * 60)
    logger.info("PREPROCESSING SPLITS")
    logger.info("=" * 60)

    for split_name in ['train', 'val', 'test']:
        logger.info(f"\nProcessing {split_name.upper()} set...")

        preprocess_trajectories(
            input_dir=str(temp_dirs[split_name]),
            output_dir=output_dir,
            episode_length=episode_length,
            max_vehicles=max_vehicles,
            overlap=overlap,
            fps=fps,
            use_extended_features=use_extended_features,
            use_acceleration=use_acceleration,
            split_data=False,  # We're already splitting
            save_metadata=(split_name == 'train')  # Only save metadata once
        )

        # Rename output file
        output_file = Path(output_dir) / "episodes.npz"
        final_file = Path(output_dir) / f"{split_name}_episodes.npz"

        if output_file.exists():
            output_file.rename(final_file)
            logger.info(f"✅ Saved to {final_file}")

    # Clean up temp directories
    for temp_dir in temp_dirs.values():
        shutil.rmtree(temp_dir)

    logger.info("\n" + "=" * 60)
    logger.info("✅ PREPROCESSING COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"Output files:")
    logger.info(f"  - {output_dir}/train_episodes.npz")
    logger.info(f"  - {output_dir}/val_episodes.npz")
    logger.info(f"  - {output_dir}/test_episodes.npz")
    logger.info(f"  - {output_dir}/metadata.json")
    if save_split_config:
        logger.info(f"  - {output_dir}/split_config.json")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess multi-site drone trajectory data (80/10/10 split)"
    )

    # Paths
    parser.add_argument('--raw_data_dir', type=str, default='data/raw',
                       help='Input directory with site folders (default: data/raw)')
    parser.add_argument('--output_dir', type=str, default='data/processed',
                       help='Output directory (default: data/processed)')

    # Feature configuration
    parser.add_argument('--use_extended_features', action='store_true', default=True,
                       help='Use extended features (10 instead of 6)')
    parser.add_argument('--use_acceleration', action='store_true', default=True,
                       help='Include acceleration features')

    # Episode configuration
    parser.add_argument('--episode_length', type=int, default=30,
                       help='Frames per episode (default: 30)')
    parser.add_argument('--max_vehicles', type=int, default=50,
                       help='Max vehicles per frame (default: 50)')
    parser.add_argument('--overlap', type=int, default=5,
                       help='Frame overlap between episodes (default: 5)')
    parser.add_argument('--fps', type=float, default=30.0,
                       help='Frames per second (default: 30.0)')

    # Split ratios
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='Training ratio (default: 0.8)')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                       help='Validation ratio (default: 0.1)')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                       help='Test ratio (default: 0.1)')

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
        max_vehicles=args.max_vehicles,
        overlap=args.overlap,
        fps=args.fps,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        save_split_config=not args.no_save_split_config
    )


if __name__ == '__main__':
    main()
