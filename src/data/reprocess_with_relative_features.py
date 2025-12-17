#!/usr/bin/env python3
"""Reprocess training data (recommended entry point).

This script is a thin wrapper around the current preprocessing pipeline
(`src/data/preprocess_multisite.py`). It exists so you can regenerate processed
episodes after changing preprocessing logic (e.g., frame-gap-aware velocity
estimation) without relying on old, interactive one-off scripts.

Examples:

  # Reprocess a single site (A)
  python -m src.data.reprocess_with_relative_features \
    --raw_data_dir data/raw \
    --output_dir data/processed_siteA_20 \
    --sites A

  # Reprocess multiple sites
  python -m src.data.reprocess_with_relative_features \
    --raw_data_dir data/raw \
    --output_dir data/processed_all \
    --sites A B C

Notes:
- Extended features include preceding/following relative features in this repo.
- The pipeline will also write normalization_stats.npz in the output dir.
"""

from __future__ import annotations

import argparse

from src.data.preprocess_multisite import preprocess_all_sites


def main() -> None:
    parser = argparse.ArgumentParser(description="Reprocess training data (multi-site)")
    parser.add_argument("--raw_data_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--sites", type=str, nargs="+", default=None, help="Sites to process, e.g. A B C")
    parser.add_argument("--episode_length", type=int, default=80)
    parser.add_argument("--stride", type=int, default=15)
    parser.add_argument("--max_vehicles", type=int, default=50)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--no_save_split_config", action="store_true")
    parser.add_argument("--use_random_split", dest="use_chronological_split", action="store_false")
    parser.add_argument("--no-save_normalization_stats", dest="save_normalization_stats", action="store_false")
    parser.set_defaults(use_chronological_split=True, save_normalization_stats=True)

    args = parser.parse_args()

    preprocess_all_sites(
        raw_data_dir=args.raw_data_dir,
        output_dir=args.output_dir,
        use_extended_features=True,
        use_acceleration=True,
        use_site_id=True,
        use_chronological_split=bool(args.use_chronological_split),
        episode_length=args.episode_length,
        stride=args.stride,
        max_vehicles=args.max_vehicles,
        fps=args.fps,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        save_split_config=not args.no_save_split_config,
        save_normalization_stats=bool(args.save_normalization_stats),
        sites=args.sites,
    )


if __name__ == "__main__":
    main()
