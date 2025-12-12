"""
Multi-Site Data Splitter

Mixes all sites (A-I) and randomly splits into train/val/test sets.

Author: Traffic World Model Team
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiSiteDataSplitter:
    """
    Collects data from all sites and splits randomly.
    """

    def __init__(
        self,
        raw_data_dir: Optional[str] = None,
        sites: Optional[List[str]] = None
    ):
        """
        Initialize the splitter.

        Args:
            raw_data_dir: Directory containing site folders (A, B, C, ...)
                         If None, uses project_root/data/raw
            sites: List of site names (default: A-I)
        """
        # Get project root directory
        if raw_data_dir is None:
            # This file is in: project_root/src/data/split_strategy.py
            project_root = Path(__file__).parent.parent.parent
            self.raw_data_dir = project_root / 'data' / 'raw'
        else:
            self.raw_data_dir = Path(raw_data_dir)

        self.sites = sites or ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']

        # Validate sites exist
        self.available_sites = []
        for site in self.sites:
            site_path = self.raw_data_dir / site
            if site_path.exists():
                self.available_sites.append(site)
            else:
                logger.warning(f"Site {site} not found at {site_path}")

        if not self.available_sites:
            logger.error(f"No sites found in {self.raw_data_dir}")
        else:
            logger.info(f"Available sites: {self.available_sites}")

    def get_site_files(self, site: str) -> List[Path]:
        """Get all CSV files for a given site."""
        site_path = self.raw_data_dir / site
        if not site_path.exists():
            return []
        return sorted(site_path.glob("*.csv"))

    def split_data(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42
    ) -> Dict[str, List[Path]]:
        """
        Mix all sites together, then randomly split into train/val/test.

        Args:
            train_ratio: Proportion for training (default: 0.8)
            val_ratio: Proportion for validation (default: 0.1)
            test_ratio: Proportion for testing (default: 0.1)
            seed: Random seed for reproducibility

        Returns:
            Dictionary with 'train', 'val', 'test' keys mapping to file lists
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"

        # Collect all files from all sites
        all_files = []
        for site in self.available_sites:
            site_files = self.get_site_files(site)
            all_files.extend(site_files)
            logger.info(f"Site {site}: {len(site_files)} files")

        if not all_files:
            raise ValueError(f"No CSV files found in {self.raw_data_dir}")

        logger.info(f"Total files: {len(all_files)}")

        # Shuffle
        np.random.seed(seed)
        indices = np.random.permutation(len(all_files))

        # Split
        n_train = int(len(all_files) * train_ratio)
        n_val = int(len(all_files) * val_ratio)

        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]

        splits = {
            'train': [all_files[i] for i in train_indices],
            'val': [all_files[i] for i in val_indices],
            'test': [all_files[i] for i in test_indices]
        }

        self._log_split_info(splits)
        return splits

    def _log_split_info(self, splits: Dict[str, List[Path]]):
        """Log information about the split."""
        logger.info("=" * 60)
        logger.info("Data Split Summary:")
        logger.info("=" * 60)

        for split_name in ['train', 'val', 'test']:
            files = splits[split_name]
            logger.info(f"{split_name.upper()}: {len(files)} files")

            # Count by site
            site_counts = {}
            for file_path in files:
                site = file_path.parent.name
                site_counts[site] = site_counts.get(site, 0) + 1

            logger.info(f"  Sites: {dict(sorted(site_counts.items()))}")

        logger.info("=" * 60)

    def save_split_config(
        self,
        splits: Dict[str, List[Path]],
        output_path: Optional[str] = None
    ):
        """
        Save split configuration to JSON for reproducibility.

        Args:
            splits: Split dictionary
            output_path: Path to save JSON config. If None, uses project_root/data/processed/split_config.json
        """
        if output_path is None:
            project_root = Path(__file__).parent.parent.parent
            output_path = project_root / 'data' / 'processed' / 'split_config.json'
        else:
            output_path = Path(output_path)

        config = {}
        for split_name, files in splits.items():
            config[split_name] = {}
            for file_path in files:
                site = file_path.parent.name
                filename = file_path.name

                if site not in config[split_name]:
                    config[split_name][site] = []
                config[split_name][site].append(filename)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)

        logger.info(f"Split config saved to {output_path}")


def chronological_split_episodes(
    episodes: List[Dict],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Dict[str, List[Dict]]:
    """
    Split episodes chronologically (time-based) to prevent temporal leakage.

    Episodes are sorted by their start time, then split sequentially:
    - First train_ratio portion -> train
    - Next val_ratio portion -> val
    - Last test_ratio portion -> test

    Args:
        episodes: List of episode dictionaries (must contain 'episode_start_global_frame')
        train_ratio: Training ratio (default: 0.7)
        val_ratio: Validation ratio (default: 0.15)
        test_ratio: Test ratio (default: 0.15)

    Returns:
        Dictionary with 'train', 'val', 'test' keys mapping to episode lists
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    # Sort episodes by start time
    sorted_episodes = sorted(
        episodes,
        key=lambda ep: ep.get('episode_start_global_frame', ep.get('start_frame', 0))
    )

    n_total = len(sorted_episodes)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    splits = {
        'train': sorted_episodes[:n_train],
        'val': sorted_episodes[n_train:n_train + n_val],
        'test': sorted_episodes[n_train + n_val:]
    }

    logger.info("=" * 60)
    logger.info("Chronological Split Summary:")
    logger.info("=" * 60)
    for split_name in ['train', 'val', 'test']:
        eps = splits[split_name]
        if len(eps) > 0:
            start_frame = min(ep.get('episode_start_global_frame', ep.get('start_frame', 0))
                            for ep in eps)
            end_frame = max(ep.get('episode_end_global_frame', ep.get('end_frame', 0))
                          for ep in eps)
            logger.info(f"{split_name.upper()}: {len(eps)} episodes, "
                       f"frames {start_frame}-{end_frame}")
        else:
            logger.info(f"{split_name.upper()}: {len(eps)} episodes")
    logger.info("=" * 60)

    return splits


if __name__ == '__main__':
    # Example usage
    splitter = MultiSiteDataSplitter()

    print("\n" + "=" * 60)
    print("MIXED SPLIT (80/10/10)")
    print("=" * 60)

    splits = splitter.split_data(
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=42
    )

    # Save config
    splitter.save_split_config(splits)
    print("\n✅ Split config saved!")
