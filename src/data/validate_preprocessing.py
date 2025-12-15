"""
Validation Script for Preprocessed Data

Performs sanity checks based on IMPROVEMENTS_todo.md:
- Verify metadata.json consistency
- Check lane_mapping format (site:lane tokens)
- Verify train/val/test splits are non-overlapping
- Check feature dimensions and layout
- Validate episode temporal ranges

Usage:
    python validate_preprocessing.py [--processed_dir data/processed]
"""

import argparse
import json
import numpy as np
from pathlib import Path
import sys


def load_metadata(processed_dir):
    """Load and return metadata.json"""
    metadata_path = processed_dir / 'metadata.json'
    if not metadata_path.exists():
        print(f"❌ ERROR: metadata.json not found at {metadata_path}")
        return None

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    print(f"✅ Loaded metadata from {metadata_path}")
    return metadata


def check_metadata_consistency(metadata):
    """Check metadata.json for required fields and correct values"""
    print("\n" + "="*60)
    print("CHECK 1: Metadata Consistency")
    print("="*60)

    issues = []

    # Check fps
    expected_fps = 30.0
    if metadata.get('fps') != expected_fps:
        issues.append(f"fps should be {expected_fps}, got {metadata.get('fps')}")
    else:
        print(f"✅ fps = {metadata['fps']}")

    # Check dt
    expected_dt = 1.0 / 30.0
    if abs(metadata.get('dt', 0) - expected_dt) > 1e-6:
        issues.append(f"dt should be {expected_dt:.6f}, got {metadata.get('dt')}")
    else:
        print(f"✅ dt = {metadata['dt']:.6f}")

    # Check episode_length
    expected_episode_length = 80
    if metadata.get('episode_length') != expected_episode_length:
        issues.append(f"episode_length should be {expected_episode_length}, got {metadata.get('episode_length')}")
    else:
        print(f"✅ episode_length = {metadata['episode_length']}")

    # Check stride
    if 'stride' not in metadata:
        issues.append("stride not found in metadata")
    else:
        print(f"✅ stride = {metadata['stride']}")

    # Check context_length and rollout_horizon (top-level and in validation_info)
    C_top = metadata.get('context_length', 0)
    H_top = metadata.get('rollout_horizon', 0)

    if C_top == 65 and H_top == 15:
        print(f"✅ Context length (C) = {C_top}, Rollout horizon (H) = {H_top} (top-level)")
    elif C_top != 0 or H_top != 0:
        issues.append(f"Expected C=65, H=15 (top-level), got C={C_top}, H={H_top}")

    # Also check validation_info for consistency
    if 'validation_info' in metadata:
        val_info = metadata['validation_info']
        C_val = val_info.get('context_length', 0)
        H_val = val_info.get('rollout_horizon', 0)

        if C_val != C_top or H_val != H_top:
            issues.append(f"C/H mismatch: top-level ({C_top}/{H_top}) vs validation_info ({C_val}/{H_val})")
        else:
            print(f"✅ C/H consistent in both top-level and validation_info")
    else:
        print("⚠️  validation_info not found in metadata")

    if issues:
        print("\n❌ Issues found:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("\n✅ All metadata checks passed!")
        return True


def check_lane_mapping(metadata):
    """Check lane_mapping uses site:lane format"""
    print("\n" + "="*60)
    print("CHECK 2: Lane Mapping Format")
    print("="*60)

    lane_mapping = metadata.get('lane_mapping', {})
    if not lane_mapping:
        print("⚠️  No lane_mapping found (using basic features?)")
        return True

    issues = []
    correct_format_count = 0

    for lane_token, lane_id in lane_mapping.items():
        # Check if token follows "site:lane" format
        if ':' in lane_token:
            parts = lane_token.split(':')
            if len(parts) == 2 and len(parts[0]) == 1 and parts[0].isalpha():
                correct_format_count += 1
            else:
                issues.append(f"Lane token '{lane_token}' has unexpected format")
        else:
            issues.append(f"Lane token '{lane_token}' missing site prefix (should be 'site:lane')")

    print(f"Total lane tokens: {len(lane_mapping)}")
    print(f"Correctly formatted: {correct_format_count}")

    if issues:
        print("\n❌ Issues found:")
        for issue in issues[:10]:  # Show first 10
            print(f"  - {issue}")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more")
        return False
    else:
        print(f"✅ All {len(lane_mapping)} lane tokens correctly formatted (site:lane)")
        # Show a few examples
        examples = list(lane_mapping.items())[:5]
        print("\nExample lane tokens:")
        for token, id in examples:
            print(f"  '{token}' -> {id}")
        return True


def check_split_ranges(processed_dir, metadata):
    """Check that train/val/test splits don't overlap in time"""
    print("\n" + "="*60)
    print("CHECK 3: Split Temporal Ranges (Non-Overlapping)")
    print("="*60)

    if not metadata.get('use_chronological_split', False):
        print("⚠️  Chronological split not used, skipping temporal check")
        return True

    split_ranges = {}
    split_temporal_info = {}

    for split_name in ['train', 'val', 'test']:
        npz_path = processed_dir / f'{split_name}_episodes.npz'
        if not npz_path.exists():
            print(f"❌ {split_name}_episodes.npz not found")
            return False

        data = np.load(npz_path, allow_pickle=True)
        print(f"✅ {split_name}: {len(data['states'])} episodes")

        split_ranges[split_name] = len(data['states'])

        # Check if temporal metadata exists
        if 'start_frames' in data and 'end_frames' in data:
            start_frames = data['start_frames']
            end_frames = data['end_frames']
            split_temporal_info[split_name] = {
                'min_start': int(start_frames.min()),
                'max_start': int(start_frames.max()),
                'min_end': int(end_frames.min()),
                'max_end': int(end_frames.max())
            }
            print(f"   Frame range: {split_temporal_info[split_name]['min_start']}-{split_temporal_info[split_name]['max_end']}")

    # Check split sizes match metadata
    if 'split_sizes' in metadata:
        expected = metadata['split_sizes']
        for split_name in ['train', 'val', 'test']:
            if split_ranges[split_name] != expected[split_name]:
                print(f"❌ {split_name} size mismatch: expected {expected[split_name]}, got {split_ranges[split_name]}")
                return False
        print("\n✅ Split sizes match metadata")

    # Check temporal non-overlap if metadata available
    if len(split_temporal_info) == 3:
        print("\n" + "-"*60)
        print("Checking temporal non-overlap...")
        print("-"*60)

        train_max_end = split_temporal_info['train']['max_end']
        val_min_start = split_temporal_info['val']['min_start']
        val_max_end = split_temporal_info['val']['max_end']
        test_min_start = split_temporal_info['test']['min_start']

        issues = []

        # Check train -> val boundary
        if train_max_end >= val_min_start:
            issues.append(f"Train overlaps with val: train max_end={train_max_end} >= val min_start={val_min_start}")
        else:
            gap_train_val = val_min_start - train_max_end
            print(f"✅ Train -> Val gap: {gap_train_val} frames")

        # Check val -> test boundary
        if val_max_end >= test_min_start:
            issues.append(f"Val overlaps with test: val max_end={val_max_end} >= test min_start={test_min_start}")
        else:
            gap_val_test = test_min_start - val_max_end
            print(f"✅ Val -> Test gap: {gap_val_test} frames")

        if issues:
            print("\n❌ Temporal overlap detected:")
            for issue in issues:
                print(f"  - {issue}")
            return False
        else:
            print("\n✅ All splits are temporally non-overlapping!")
            return True
    else:
        print("\n⚠️  Temporal metadata (start_frames/end_frames) not found in NPZ files")
        print("    Cannot verify temporal non-overlap")
        print("    (This is expected for data preprocessed before this update)")
        return True


def check_feature_dimensions(processed_dir, metadata):
    """Check feature dimensions match metadata"""
    print("\n" + "="*60)
    print("CHECK 4: Feature Dimensions")
    print("="*60)

    expected_features = metadata.get('n_features')
    if not expected_features:
        print("❌ n_features not found in metadata")
        return False

    print(f"Expected n_features: {expected_features}")

    # Check a sample from train set
    train_path = processed_dir / 'train_episodes.npz'
    if not train_path.exists():
        print("❌ train_episodes.npz not found")
        return False

    data = np.load(train_path)
    states = data['states']

    B, T, K, F = states.shape
    print(f"Actual data shape: [B={B}, T={T}, K={K}, F={F}]")

    if F != expected_features:
        print(f"❌ Feature dimension mismatch: expected {expected_features}, got {F}")
        return False

    if T != metadata.get('episode_length', 80):
        print(f"❌ Episode length mismatch: expected {metadata.get('episode_length')}, got {T}")
        return False

    if K != metadata.get('max_vehicles', 50):
        print(f"❌ Max vehicles mismatch: expected {metadata.get('max_vehicles')}, got {K}")
        return False

    print(f"✅ All dimensions match!")

    # Show feature layout
    if 'feature_layout' in metadata:
        print("\nFeature layout:")
        for idx, name in sorted(metadata['feature_layout'].items(), key=lambda x: int(x[0])):
            print(f"  [{idx}] {name}")

    return True


def check_discrete_features(metadata):
    """Check discrete features are properly documented"""
    print("\n" + "="*60)
    print("CHECK 5: Discrete Features")
    print("="*60)

    if 'validation_info' not in metadata:
        print("⚠️  validation_info not found")
        return True

    val_info = metadata['validation_info']

    if 'discrete_features' in val_info:
        discrete = val_info['discrete_features']
        print("Discrete features (do not normalize):")
        for name, idx in discrete.items():
            if idx is not None:
                print(f"  {name}: index {idx}")
        print(f"\n✅ Discrete features documented: {list(discrete.keys())}")

    if 'do_not_normalize' in val_info:
        print(f"✅ Normalization exclusions: {val_info['do_not_normalize']}")

    if 'num_lanes' in val_info:
        print(f"✅ num_lanes (dynamic): {val_info['num_lanes']}")

    return True


def main():
    parser = argparse.ArgumentParser(description='Validate preprocessed data')
    parser.add_argument('--processed_dir', type=str, default='data/processed',
                       help='Path to processed data directory')
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)

    print("="*60)
    print("PREPROCESSING VALIDATION")
    print("="*60)
    print(f"Checking: {processed_dir.absolute()}\n")

    # Load metadata
    metadata = load_metadata(processed_dir)
    if metadata is None:
        sys.exit(1)

    # Run all checks
    results = []
    results.append(("Metadata Consistency", check_metadata_consistency(metadata)))
    results.append(("Lane Mapping Format", check_lane_mapping(metadata)))
    results.append(("Split Ranges", check_split_ranges(processed_dir, metadata)))
    results.append(("Feature Dimensions", check_feature_dimensions(processed_dir, metadata)))
    results.append(("Discrete Features", check_discrete_features(metadata)))

    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)

    for check_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {check_name}")

    all_passed = all(passed for _, passed in results)

    if all_passed:
        print("\n✅ All validation checks passed!")
        print("\n✓ Data is ready for training")
        print("\nNext steps:")
        print("  1. python src/training/train_world_model.py \\")
        print(f"       --train_data {processed_dir}/train_episodes.npz \\")
        print(f"       --val_data {processed_dir}/val_episodes.npz \\")
        print(f"       --input_dim {metadata['n_features']}")
        return 0
    else:
        print("\n❌ Some validation checks failed!")
        print("\n✗ Please fix the issues before training")
        return 1


if __name__ == '__main__':
    sys.exit(main())
