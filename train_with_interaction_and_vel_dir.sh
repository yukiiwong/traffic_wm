#!/bin/bash

# 训练脚本：简化特征 + velocity_direction + velocity_direction_loss
# 
# 改进点：
# 1. 只保留关键特征 (15个)
#    - 基本运动: x, y, vx, vy, ax, ay (6个)
#    - 前车交互: has_preceding, rel_x/y/vx/vy_preceding (5个)
#    - 派生特征: velocity_direction, headway, ttc, preceding_distance (4个)
# 2. 排除后车信息 (has_following, rel_*_following)
# 3. 使用velocity_direction_loss监督训练
# 4. 期望：velocity_direction_error从60°降到20-30°

DATA_DIR="data/processed_siteA_20"
EXPERIMENT_NAME="simplified_interaction_vel_dir"
LOG_DIR="experiments/${EXPERIMENT_NAME}"

echo "========================================="
echo "Training with Simplified Features"
echo "========================================="
echo ""
echo "特征总数: 24 (20 原始 + 4 派生)"
echo "Continuous特征: 15个"
echo ""
echo "基本运动 (6):"
echo "  [0-5]: x, y, vx, vy, ax, ay"
echo ""
echo "前车交互 (5):"
echo "  [9]: has_preceding"
echo "  [12-15]: rel_x/y/vx/vy_preceding"
echo ""
echo "派生特征 (4):"
echo "  [20]: velocity_direction (atan2(vy, vx))"
echo "  [21]: headway (纵向距离)"
echo "  [22]: ttc (Time-To-Collision)"
echo "  [23]: preceding_distance (总距离)"
echo ""
echo "排除特征:"
echo "  [6]: angle"
echo "  [10]: has_following"
echo "  [16-19]: rel_*_following (后车信息)"
echo ""
echo "Loss Function:"
echo "  - reconstruction_loss (MSE)"
echo "  - prediction_loss (MSE)"
echo "  - velocity_direction_loss (angular) weight=0.3"
echo ""

python src/training/train_world_model.py \
  --train_data "${DATA_DIR}/train_episodes.npz" \
  --val_data "${DATA_DIR}/val_episodes.npz" \
  --log_dir "${LOG_DIR}" \
  --epochs 200 \
  --batch_size 32 \
  --learning_rate 1e-4 \
  --latent_dim 512 \
  --dynamics_layers 6 \
  --num_heads 16 \
  --recon_weight 1.0 \
  --pred_weight 1.0 \
  --velocity_direction_weight 0.3 \
  --velocity_threshold 0.5 \
  --eval_interval 10 \
  --save_interval 20 \
  --num_workers 4

echo ""
echo "========================================="
echo "Training completed!"
echo "========================================="
echo "Log directory: ${LOG_DIR}"
echo ""
echo "To evaluate:"
echo "  python src/evaluation/rollout_eval.py \\"
echo "    --checkpoint ${LOG_DIR}/best_model.pth \\"
echo "    --test_data ${DATA_DIR}/test_episodes.npz"
