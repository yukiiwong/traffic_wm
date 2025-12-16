# Traffic World Model - 多智能体轨迹预测

基于Transformer的潜在世界模型，用于多站点无人机轨迹数据的预测和仿真。

## 特征配置 (v2.5)

- **总特征数**: 24 (20原始 + 4派生)
- **Continuous**: 15个 - 基本运动(6) + 前车交互(5) + 派生特征(4)
- **派生特征**: velocity_direction, headway, ttc, preceding_distance
- **Loss**: reconstruction + prediction + velocity_direction_loss

---

## 快速开始

### 1. 数据预处理

**代码**: `src/data/preprocess_multisite.py`

```bash
# 预处理Site A数据（20维特征）
python src/data/preprocess_multisite.py \
    --raw_data_dir data/raw \
    --output_dir data/processed_siteA_20 \
    --sites A \
    --episode_length 80 \
    --stride 15 \
    --max_agents 50

# 查看生成的文件
ls data/processed_siteA_20/
# 输出: train_episodes.npz  val_episodes.npz  test_episodes.npz  metadata.json  normalization_stats.npz
```

### 2. 模型训练

**代码**: `src/training/train_world_model.py`, `src/training/losses.py`

```bash
# 使用训练脚本（推荐）
./train_with_interaction_and_vel_dir.sh

# 或手动训练
python src/training/train_world_model.py \
    --train_data data/processed_siteA_20/train_episodes.npz \
    --val_data data/processed_siteA_20/val_episodes.npz \
    --log_dir experiments/simplified_vel_dir \
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
```

### 3. 模型评估

**代码**: `src/evaluation/rollout_eval.py`, `src/evaluation/prediction_metrics.py`

```bash
# Rollout评估（生成完整轨迹预测）
python src/evaluation/rollout_eval.py \
    --checkpoint experiments/simplified_vel_dir/best_model.pth \
    --test_data data/processed_siteA_20/test_episodes.npz \
    --output_dir results/rollout_eval \
    --num_episodes 100 \
    --save_predictions

# 查看评估指标
cat results/rollout_eval/metrics.json
```

### 4. 可视化

**代码**: `src/evaluation/visualize_predictions_detailed.py`

```bash
# 生成详细可视化（GT vs Pred轨迹对比）
python src/evaluation/visualize_predictions_detailed.py \
    --checkpoint experiments/simplified_vel_dir/best_model.pth \
    --test_data data/processed_siteA_20/test_episodes.npz \
    --output_dir results/visualization \
    --num_episodes 10 \
    --selection_mode presence

# 输出: results/visualization/episode_*.png
```

---

## 特征详细说明

### 输入特征 (20维原始)

```
基本运动 (6):
  [0-5]: center_x, center_y, vx, vy, ax, ay

离散特征 (3):
  [6]: angle       - 车辆朝向角
  [7]: class_id    - 车辆类别  
  [8]: lane_id     - 车道ID
  [11]: site_id    - 站点ID

前车交互 (5):
  [9]:  has_preceding      - 是否有前车
  [12]: rel_x_preceding    - 前车相对x
  [13]: rel_y_preceding    - 前车相对y
  [14]: rel_vx_preceding   - 前车相对vx
  [15]: rel_vy_preceding   - 前车相对vy

后车交互 (5) - 已排除:
  [10]: has_following      
  [16-19]: rel_*_following
```

### 派生特征 (4维，动态计算)

```
[20]: velocity_direction  = atan2(vy, vx)
[21]: headway            = rel_x_preceding
[22]: ttc                = -distance / rel_vx (if approaching)
[23]: preceding_distance = sqrt(rel_x² + rel_y²)
```

### Continuous特征 (15维用于训练)

```
[0,1,2,3,4,5,9,12,13,14,15,20,21,22,23]
```

---

## 代码文件说明

### 数据处理
```
src/data/preprocess_multisite.py  - 主预处理脚本
src/data/dataset.py                - PyTorch Dataset (动态添加派生特征)
src/data/split_strategy.py         - Train/Val/Test划分策略
```

### 模型训练
```
src/training/train_world_model.py  - 训练主脚本
src/training/losses.py             - Loss函数 (含velocity_direction_loss)
src/models/world_model.py          - 完整模型
src/models/encoder.py              - Transformer Encoder
src/models/dynamics.py             - Transformer Dynamics
src/models/decoder.py              - MLP Decoder
```

### 评估与可视化
```
src/evaluation/rollout_eval.py                - Rollout评估（主要）
src/evaluation/prediction_metrics.py          - 指标计算
src/evaluation/visualize_predictions_detailed.py - 可视化
```

---

## 评估指标

**基础指标**:
- `ADE` (Average Displacement Error): 平均位置误差
- `FDE` (Final Displacement Error): 终点位置误差

**扩展指标** (v2.5):
- `moving_ade`: 只计算运动车辆的ADE
- `velocity_direction_error`: 速度方向误差 (度)
- `acceleration_error`: 加速度误差
- `position_variance`: 位置方差（平滑度）

**输出**:
```json
{
  "ade": 0.025,
  "fde": 0.033,
  "moving_ade": 0.40,
  "velocity_direction_error": 25.3,
  "acceleration_error": 1.82,
  "position_variance": 0.15
}
```

---

## 高级用法

### 自定义训练参数

```bash
python src/training/train_world_model.py \
    --train_data data/processed_siteA_20/train_episodes.npz \
    --val_data data/processed_siteA_20/val_episodes.npz \
    --log_dir experiments/custom_run \
    --epochs 200 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --latent_dim 512 \
    --dynamics_layers 6 \
    --num_heads 16 \
    --dropout 0.1 \
    --recon_weight 1.0 \
    --pred_weight 1.0 \
    --velocity_direction_weight 0.3 \
    --velocity_threshold 0.5 \
    --grad_clip 1.0 \
    --eval_interval 10 \
    --save_interval 20 \
    --num_workers 4
```

### 分布式训练

```bash
# 多GPU训练
CUDA_VISIBLE_DEVICES=0,1 python src/training/train_world_model.py \
    --train_data data/processed_siteA_20/train_episodes.npz \
    --val_data data/processed_siteA_20/val_episodes.npz \
    --log_dir experiments/multi_gpu \
    --batch_size 64 \
    --num_workers 8
```

### 评估特定checkpoint

```bash
# 评估最佳模型
python src/evaluation/rollout_eval.py \
    --checkpoint experiments/simplified_vel_dir/best_model.pth \
    --test_data data/processed_siteA_20/test_episodes.npz \
    --output_dir results/best_eval \
    --num_episodes 500

# 评估特定epoch
python src/evaluation/rollout_eval.py \
    --checkpoint experiments/simplified_vel_dir/model_epoch_100.pth \
    --test_data data/processed_siteA_20/test_episodes.npz \
    --output_dir results/epoch_100_eval
```

### 可视化选项

```bash
# 选择存在时间最长的车辆
python src/evaluation/visualize_predictions_detailed.py \
    --checkpoint experiments/simplified_vel_dir/best_model.pth \
    --test_data data/processed_siteA_20/test_episodes.npz \
    --output_dir results/viz_presence \
    --selection_mode presence \
    --num_episodes 20

# 选择随机车辆
python src/evaluation/visualize_predictions_detailed.py \
    --checkpoint experiments/simplified_vel_dir/best_model.pth \
    --test_data data/processed_siteA_20/test_episodes.npz \
    --output_dir results/viz_random \
    --selection_mode random \
    --num_episodes 20
```

---

## 项目结构

```
traffic_wm/
├── data/
│   ├── raw/                        # 原始CSV数据
│   └── processed_siteA_20/         # 预处理后的数据
│       ├── train_episodes.npz
│       ├── val_episodes.npz
│       ├── test_episodes.npz
│       ├── metadata.json
│       └── normalization_stats.npz
│
├── src/
│   ├── data/
│   │   ├── preprocess_multisite.py  # 数据预处理
│   │   ├── dataset.py               # Dataset类
│   │   └── split_strategy.py        # 划分策略
│   ├── models/
│   │   ├── world_model.py           # 完整模型
│   │   ├── encoder.py               # Encoder
│   │   ├── dynamics.py              # Dynamics
│   │   └── decoder.py               # Decoder
│   ├── training/
│   │   ├── train_world_model.py     # 训练脚本
│   │   └── losses.py                # Loss函数
│   └── evaluation/
│       ├── rollout_eval.py          # 评估
│       ├── prediction_metrics.py    # 指标
