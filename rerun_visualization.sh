#!/bin/bash
# 重新运行可视化的完整流程

echo "="
echo "步骤 1: 检查checkpoint和数据"
echo "========================================"

if [ ! -f "checkpoints/best_model.pt" ]; then
    echo "❌ 未找到checkpoint: checkpoints/best_model.pt"
    echo "   请先训练模型或指定正确的checkpoint路径"
    exit 1
fi

if [ ! -f "data/processed/test_episodes.npz" ]; then
    echo "❌ 未找到测试数据: data/processed/test_episodes.npz"
    echo "   请先运行预处理: python preprocess_multisite.py"
    exit 1
fi

echo "✅ 数据检查完成"

echo ""
echo "========================================"
echo "步骤 2: 运行可视化"
echo "========================================"

python src/evaluation/visualize_predictions.py \
    --checkpoint checkpoints/best_model.pt \
    --test_data data/processed/test_episodes.npz \
    --metadata data/processed/metadata.json \
    --site_images_dir src/evaluation/sites \
    --context_length 65 \
    --rollout_horizon 15 \
    --batch_size 8 \
    --output_dir results/visualizations \
    --num_samples 5 \
    --max_agents 10

echo ""
echo "✅ 可视化完成!"
echo "查看结果: results/visualizations/"

