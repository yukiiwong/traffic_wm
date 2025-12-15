#!/usr/bin/env python3
"""
重新预处理数据以包含相对位置特征

运行此脚本以生成包含相对位置/速度特征的新数据集。
"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.preprocess import preprocess_trajectories

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    print("=" * 80)
    print("重新预处理数据 - 添加相对位置特征")
    print("=" * 80)
    print()
    print("配置:")
    print("  - use_extended_features: True (lane, has_preceding, has_following)")
    print("  - use_acceleration: True (ax, ay)")
    print("  - use_relative_features: True (相对位置/速度)")
    print("  - use_site_id: True")
    print()
    print("预期特征数: 20")
    print("  0-1:   center_x, center_y")
    print("  2-3:   vx, vy")
    print("  4-5:   ax, ay")
    print("  6:     angle")
    print("  7:     class_id (离散)")
    print("  8:     lane_id (离散)")
    print("  9-10:  has_preceding, has_following (二值)")
    print("  11:    site_id (离散)")
    print("  12-15: preceding 相对特征 (dx, dy, dvx, dvy)")
    print("  16-19: following 相对特征 (dx, dy, dvx, dvy)")
    print()
    print("输出目录: data/processed_siteA_with_relative")
    print("=" * 80)
    print()
    
    confirm = input("继续处理? (yes/no): ")
    if confirm.lower() not in ['yes', 'y']:
        print("取消处理")
        sys.exit(0)
    
    logger.info("开始预处理...")
    
    try:
        preprocess_trajectories(
            input_dir='data/raw/siteA',
            output_dir='data/processed_siteA_with_relative',
            episode_length=80,
            max_vehicles=50,
            overlap=65,  # context_length
            fps=30.0,
            use_extended_features=True,    # lane, has_preceding, has_following
            use_acceleration=True,         # ax, ay
            use_site_id=True,              # site_id
            use_relative_features=True,    # ⭐ NEW: preceding/following relative features
            split_data=True,               # train/val/test split
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            save_metadata=True
        )
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("✅ 处理完成!")
        logger.info("=" * 80)
        logger.info("")
        logger.info("验证特征数...")
        
        import numpy as np
        import json
        
        train_data = np.load('data/processed_siteA_with_relative/train_episodes.npz')
        states = train_data['states']
        logger.info(f"  states shape: {states.shape}")
        logger.info(f"  特征数 (F): {states.shape[-1]}")
        
        if states.shape[-1] == 20:
            logger.info("  ✅ 特征数正确 (20)")
        else:
            logger.warning(f"  ⚠️ 特征数不符合预期 (期望 20, 实际 {states.shape[-1]})")
        
        with open('data/processed_siteA_with_relative/metadata.json') as f:
            meta = json.load(f)
        
        logger.info("")
        logger.info("Metadata 检查:")
        logger.info(f"  n_features: {meta['n_features']}")
        logger.info(f"  use_relative_features: {meta.get('use_relative_features', False)}")
        logger.info(f"  angle_idx: {meta.get('validation_info', {}).get('angle_idx', 'N/A')}")
        
        logger.info("")
        logger.info("下一步 - 训练模型:")
        logger.info("  python src/training/train_world_model.py \\")
        logger.info("    --train_data data/processed_siteA_with_relative/train_episodes.npz \\")
        logger.info("    --val_data data/processed_siteA_with_relative/val_episodes.npz \\")
        logger.info("    --input_dim 20 \\")
        logger.info("    --latent_dim 384 \\")
        logger.info("    --angle_weight 1.0 \\")
        logger.info("    --weight_decay 0.001 \\")
        logger.info("    --scheduler cosine \\")
        logger.info("    --epochs 50 \\")
        logger.info("    --checkpoint_dir checkpoints/world_model_with_relative \\")
        logger.info("    --log_dir logs/world_model_with_relative")
        
    except Exception as e:
        logger.error(f"❌ 预处理失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    print(f"  angle_idx: {meta['validation_info']['angle_idx']}")
    print(f"  continuous_dim: {len([i for i in range(meta['n_features']) if i not in meta['validation_info']['discrete_features'].values() and i != meta['validation_info']['angle_idx']])}")
    
    print()
    print("Feature layout:")
    for idx in sorted([int(k) for k in meta['feature_layout'].keys()]):
        print(f"  {idx}: {meta['feature_layout'][str(idx)]}")
