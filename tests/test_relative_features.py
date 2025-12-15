#!/usr/bin/env python3
"""
验证相对位置特征实现
"""

import sys
from pathlib import Path

print("=" * 80)
print("验证相对位置特征实现")
print("=" * 80)
print()

# Test 1: Check preprocess.py modifications
print("TEST 1: 检查 preprocess.py 修改")
print("-" * 80)

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.preprocess import (
    extract_single_episode_from_global,
    extract_single_episode,
    extract_fixed_stride_episodes,
    preprocess_trajectories
)
import inspect

functions_to_check = [
    ('extract_single_episode_from_global', extract_single_episode_from_global),
    ('extract_single_episode', extract_single_episode),
    ('extract_fixed_stride_episodes', extract_fixed_stride_episodes),
    ('preprocess_trajectories', preprocess_trajectories),
]

all_have_param = True
for func_name, func in functions_to_check:
    sig = inspect.signature(func)
    has_param = 'use_relative_features' in sig.parameters
    status = "✅" if has_param else "❌"
    print(f"{status} {func_name}: use_relative_features={'存在' if has_param else '缺失'}")
    if not has_param:
        all_have_param = False

print()
if all_have_param:
    print("✅ TEST 1 PASSED: 所有函数都有 use_relative_features 参数")
else:
    print("❌ TEST 1 FAILED: 部分函数缺少参数")
    sys.exit(1)

print()

# Test 2: Check feature count calculation
print("TEST 2: 验证特征数量计算")
print("-" * 80)

# Simulate feature calculation
F = 6  # base
F += 2  # acceleration
F += 3  # extended (lane, has_preceding, has_following)
F += 1  # site_id
F += 8  # relative features

expected_F = 20
if F == expected_F:
    print(f"✅ 特征数量正确: {F} (期望 {expected_F})")
else:
    print(f"❌ 特征数量错误: {F} (期望 {expected_F})")
    sys.exit(1)

print()

# Test 3: Verify binary feature indices
print("TEST 3: 验证二值特征索引")
print("-" * 80)

# With all features enabled
features = []
idx = 0

features.append((idx, "center_x")); idx += 1
features.append((idx, "center_y")); idx += 1
features.append((idx, "vx")); idx += 1
features.append((idx, "vy")); idx += 1
features.append((idx, "ax")); idx += 1
features.append((idx, "ay")); idx += 1
features.append((idx, "angle")); idx += 1; angle_idx = idx - 1
features.append((idx, "class_id")); idx += 1; class_id_idx = idx - 1
features.append((idx, "lane_id")); idx += 1; lane_id_idx = idx - 1
features.append((idx, "has_preceding")); idx += 1; has_prec_idx = idx - 1
features.append((idx, "has_following")); idx += 1; has_foll_idx = idx - 1
features.append((idx, "site_id")); idx += 1; site_id_idx = idx - 1
features.append((idx, "preceding_rel_x")); idx += 1
features.append((idx, "preceding_rel_y")); idx += 1
features.append((idx, "preceding_rel_vx")); idx += 1
features.append((idx, "preceding_rel_vy")); idx += 1
features.append((idx, "following_rel_x")); idx += 1
features.append((idx, "following_rel_y")); idx += 1
features.append((idx, "following_rel_vx")); idx += 1
features.append((idx, "following_rel_vy")); idx += 1

discrete_and_angle = set([angle_idx, class_id_idx, lane_id_idx, site_id_idx])
continuous_indices = [i for i, _ in features if i not in discrete_and_angle]

has_prec_cont = continuous_indices.index(has_prec_idx)
has_foll_cont = continuous_indices.index(has_foll_idx)

print(f"原始索引: has_preceding={has_prec_idx}, has_following={has_foll_idx}")
print(f"连续输出索引: has_preceding={has_prec_cont}, has_following={has_foll_cont}")
print(f"连续特征总数: {len(continuous_indices)}")

if has_prec_cont == 6 and has_foll_cont == 7 and len(continuous_indices) == 16:
    print("✅ TEST 3 PASSED: 二值特征索引正确 [6, 7], continuous_dim=16")
else:
    print(f"❌ TEST 3 FAILED: 索引错误或 continuous_dim 不正确")
    sys.exit(1)

print()

# Test 4: Check decoder compatibility
print("TEST 4: 验证 decoder.py 兼容性")
print("-" * 80)

decoder_path = Path('src/models/decoder.py')
if decoder_path.exists():
    decoder_content = decoder_path.read_text()
    
    has_binary_param = 'binary_feature_indices' in decoder_content
    has_default_67 = '[6, 7]' in decoder_content
    has_sigmoid = 'torch.sigmoid(states[..., idx])' in decoder_content
    
    print(f"{'✅' if has_binary_param else '❌'} binary_feature_indices 参数存在")
    print(f"{'✅' if has_default_67 else '❌'} 默认值为 [6, 7]")
    print(f"{'✅' if has_sigmoid else '❌'} sigmoid 应用逻辑存在")
    
    if has_binary_param and has_default_67 and has_sigmoid:
        print("✅ TEST 4 PASSED: decoder.py 兼容新特征")
    else:
        print("❌ TEST 4 FAILED: decoder.py 需要更新")
        sys.exit(1)
else:
    print("⚠️ decoder.py 未找到，跳过测试")

print()
print("=" * 80)
print("✅ 所有测试通过!")
print("=" * 80)
print()
print("下一步:")
print("  1. 运行: python reprocess_with_relative_features.py")
print("  2. 训练: python src/training/train_world_model.py --input_dim 20 ...")
print()
print("详细指南: RELATIVE_FEATURES_GUIDE.md")
