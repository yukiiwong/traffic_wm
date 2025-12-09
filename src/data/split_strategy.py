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
        raw_data_dir: str = "data/raw",
        sites: Optional[List[str]] = None
    ):
        """
        Initialize the splitter.

        Args:
            raw_data_dir: Directory containing site folders (A, B, C, ...)
            sites: List of site names (default: A-I)
        """
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
        output_path: str
    ):
        """
        Save split configuration to JSON for reproducibility.

        Args:
            splits: Split dictionary
            output_path: Path to save JSON config
        """
        config = {}
        for split_name, files in splits.items():
            config[split_name] = {}
            for file_path in files:
                site = file_path.parent.name
                filename = file_path.name

                if site not in config[split_name]:
                    config[split_name][site] = []
                config[split_name][site].append(filename)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)

        logger.info(f"Split config saved to {output_path}")


if __name__ == '__main__':
    # Example usage
    splitter = MultiSiteDataSplitter(raw_data_dir="data/raw")

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
    splitter.save_split_config(splits, "data/split_config.json")
    print("\n✅ Split config saved to data/split_config.json")
