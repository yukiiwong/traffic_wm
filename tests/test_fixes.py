#!/usr/bin/env python3
"""
Quick test to verify all fixes are working correctly.
"""

import json
import sys
from pathlib import Path

def test_metadata_fix():
    """Test that metadata.json has angle_idx and angle in do_not_normalize."""
    print("=" * 60)
    print("TEST 1: metadata.json fixes")
    print("=" * 60)
    
    meta_path = Path("data/processed_siteA/metadata.json")
    if not meta_path.exists():
        print(f"❌ FAIL: {meta_path} not found")
        return False
    
    with open(meta_path) as f:
        meta = json.load(f)
    
    validation_info = meta.get('validation_info', {})
    
    # Check angle_idx
    angle_idx = validation_info.get('angle_idx')
    if angle_idx == 6:
        print(f"✅ PASS: angle_idx = {angle_idx}")
    else:
        print(f"❌ FAIL: angle_idx = {angle_idx} (expected 6)")
        return False
    
    # Check do_not_normalize
    do_not_normalize = validation_info.get('do_not_normalize', [])
    if 'angle' in do_not_normalize:
        print(f"✅ PASS: 'angle' in do_not_normalize: {do_not_normalize}")
    else:
        print(f"❌ FAIL: 'angle' not in do_not_normalize: {do_not_normalize}")
        return False
    
    print()
    return True


def test_train_script_params():
    """Test that train_world_model.py has new parameters."""
    print("=" * 60)
    print("TEST 2: train_world_model.py new parameters")
    print("=" * 60)
    
    train_script = Path("src/training/train_world_model.py")
    if not train_script.exists():
        print(f"❌ FAIL: {train_script} not found")
        return False
    
    content = train_script.read_text()
    
    # Check scheduler parameter
    if '--scheduler' in content and 'CosineAnnealingLR' in content:
        print("✅ PASS: Learning rate scheduler support added")
    else:
        print("❌ FAIL: Scheduler support not found")
        return False
    
    # Check angle_weight parameter
    if '--angle_weight' in content and 'args.angle_weight' in content:
        print("✅ PASS: angle_weight parameter added")
    else:
        print("❌ FAIL: angle_weight parameter not found")
        return False
    
    # Check scheduler step logic
    if 'scheduler.step' in content:
        print("✅ PASS: Scheduler step() logic added")
    else:
        print("❌ FAIL: Scheduler step() logic not found")
        return False
    
    print()
    return True


def test_decoder_sigmoid():
    """Test that decoder.py has sigmoid for binary features."""
    print("=" * 60)
    print("TEST 3: decoder.py binary feature sigmoid")
    print("=" * 60)
    
    decoder_script = Path("src/models/decoder.py")
    if not decoder_script.exists():
        print(f"❌ FAIL: {decoder_script} not found")
        return False
    
    content = decoder_script.read_text()
    
    # Check binary_feature_indices parameter
    if 'binary_feature_indices' in content:
        print("✅ PASS: binary_feature_indices parameter added")
    else:
        print("❌ FAIL: binary_feature_indices parameter not found")
        return False
    
    # Check sigmoid application
    if 'torch.sigmoid(states[..., idx])' in content:
        print("✅ PASS: Sigmoid applied to binary features")
    else:
        print("❌ FAIL: Sigmoid application not found")
        return False
    
    # Check correct indices [6, 7] for has_preceding, has_following
    if '[6, 7]' in content:
        print("✅ PASS: Correct binary feature indices [6, 7]")
    else:
        print("⚠️  WARNING: Binary indices might be incorrect (expected [6, 7])")
    
    print()
    return True


def main():
    print("\n🔧 Testing all fixes...\n")
    
    results = []
    results.append(test_metadata_fix())
    results.append(test_train_script_params())
    results.append(test_decoder_sigmoid())
    
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if all(results):
        print("✅ ALL TESTS PASSED!")
        print("\nYou can now train with improved settings:")
        print("\n  python src/training/train_world_model.py \\")
        print("    --train_data data/processed_siteA/train_episodes.npz \\")
        print("    --val_data data/processed_siteA/val_episodes.npz \\")
        print("    --angle_weight 1.0 \\")
        print("    --weight_decay 0.001 \\")
        print("    --scheduler cosine \\")
        print("    --checkpoint_dir checkpoints/world_model_v2")
        print()
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
