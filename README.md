"""Traffic World Model (traffic_wm)

基于 Transformer 的多智能体轨迹世界模型，用于多站点无人机车辆轨迹数据的
teacher-forced one-step 训练与 open-loop rollout 评估/可视化。

本仓库的默认坐标单位是像素（pixel），时间采样为 30 FPS。
"""

# Traffic World Model - 多智能体轨迹预测

## 快速开始

### 0) 新建 Conda 环境（推荐）

```bash
# 1) 创建并激活环境
conda create -n traffic_wm python=3.10 -y
conda activate traffic_wm

# 2) 安装依赖（包含 torch / pandas / matplotlib 等）
pip install -U pip
pip install -r requirements.txt
```

说明:
- 如果你需要 GPU/CUDA 版本的 PyTorch，请根据你机器的 CUDA 版本安装匹配的 `torch` 构建；本仓库其余依赖不依赖特定 CUDA。

### 1) 数据预处理

唯一入口脚本: `src/data/preprocess_multisite.py`（推荐用 `./preprocess.sh` 包一层快捷调用）

```bash
python src/data/preprocess_multisite.py \
  --raw_data_dir data/raw \
  --output_dir data/processed_siteA \
  --sites A \
  --episode_length 80 \
  --stride 15 \
  --max_agents 50

# 等价写法（推荐，参数原样透传）
# ./preprocess.sh --raw_data_dir data/raw --output_dir data/processed_siteA --sites A --episode_length 80 --stride 15 --max_agents 50

ls data/processed_siteA/
# train_episodes.npz  val_episodes.npz  test_episodes.npz  metadata.json  normalization_stats.npz
```

关键点:
- `.npz` 存储的是预处理的基础特征（默认 20 维）。
- Dataset 会在加载时动态计算并追加 4 个派生特征（见下文）。
- 速度/加速度的差分会按真实帧间隔（frame gap）进行时间尺度修正，避免缺帧造成速度爆炸。
- `src/data/preprocess.py` 是底层/legacy helper（被 `preprocess_multisite.py` 调用），不作为对外入口。

### 2) 训练

入口: `src/training/train_world_model.py`

```bash
python src/training/train_world_model.py \
  --train_data data/processed_siteA/train_episodes.npz \
  --val_data data/processed_siteA/val_episodes.npz \
  --stats_path data/processed_siteA/normalization_stats.npz \
  --log_dir experiments/wm_siteA \
  --epochs 200 \
  --batch_size 32 \
  --learning_rate 1e-4 \
  --eval_interval 10 \
  --save_interval 20
```

训练相关的常用稳定器开关（可按需启用）:
- `--disable_vxy_supervision`: vx/vy 仍作为输入，但不再作为监督目标；日志中 vx/vy-based 的方向指标会标注为 diag-only。
- `--short_rollout_horizon`, `--short_rollout_weight`: 短 open-loop rollout 的 xy loss（更贴近 rollout 行为）。
- `--scheduled_sampling_start`, `--scheduled_sampling_end`: scheduled sampling（逐步从 teacher forcing 过渡到自回归）。
- `--boundary_weight` 等: soft boundary penalty（约束轨迹不要跑出画面范围）。

### 3) 评估（rollout）

入口: `src/evaluation/rollout_eval.py`

```bash
python src/evaluation/rollout_eval.py \
  --checkpoint experiments/wm_siteA/best_model.pth \
  --test_data data/processed_siteA/test_episodes.npz \
  --stats_path data/processed_siteA/normalization_stats.npz \
  --output_dir results/rollout_eval \
  --num_episodes 100 \
  --save_predictions
```

### 4) 可视化

静态图: `src/evaluation/visualize_predictions_detailed.py`

```bash
python src/evaluation/visualize_predictions_detailed.py \
  --checkpoint experiments/wm_siteA/best_model.pth \
  --test_data data/processed_siteA/test_episodes.npz \
  --stats_path data/processed_siteA/normalization_stats.npz \
  --output_dir results/visualization \
  --num_episodes 10 \
  --selection_mode presence
```

说明:
- 绘图是 mask-aware 的：在 mask 断点处会断线，避免 padding → real 的“超长线”伪像。

## 特征说明（高层）

### 预处理输出（默认 20 维，存储于 .npz）

基础运动 (6):
- `[0-5]`: center_x, center_y, vx, vy, ax, ay

离散/ID 类特征（来自 metadata 配置，通常包含）:
- angle / class_id / lane_id / site_id 等

交互（通常包含前车与后车相关字段）:
- has_preceding + 相对位置/速度
- has_following + 相对位置/速度

### Dataset 动态追加派生特征（4 维）

在 `src/data/dataset.py` 中动态计算并拼接到 state 末尾:
- `[20]`: velocity_direction = atan2(vy, vx)
- `[21]`: headway = rel_x_preceding
- `[22]`: ttc（接近时的 time-to-collision）
- `[23]`: preceding_distance = sqrt(rel_x^2 + rel_y^2)

最终 `__getitem__` 返回的 `states` 维度是 `[T, K, 24]`。

更完整的模块说明与关键实现细节见 `CODE_DOCUMENTATION.md`。
特征转换 (dataset.py.__getitem__)
  ├─ 排除后车特征 [10,16-19]
  ├─ 计算派生特征 [20-23]
  └─ 输出15维continuous特征
           ↓
训练 (train_world_model.py)
  ├─ 从dataset读取continuous_indices
  ├─ 自动适配特征维度
  └─ Loss计算基于15维特征
           ↓
评估 (rollout_eval.py)
  ├─ 从dataset读取continuous_indices
  ├─ 自动适配特征维度
  └─ Metrics计算基于15维特征
```

**代码自动适配机制**:
- ✅ 训练代码从`dataset.continuous_indices`自动读取特征配置
- ✅ 评估代码优先使用`dataset.continuous_indices`，否则从metadata读取
- ✅ 切换预处理参数后，只需重新生成数据，训练/评估代码无需修改
- ⚠️ 如果使用基础配置（12维），派生特征和部分Loss无法使用

### 模型训练
```
src/training/train_world_model.py  - 训练主脚本（自动读取continuous_indices）
src/training/losses.py             - Loss函数（含velocity_direction_loss）
src/models/world_model.py          - 完整模型（特征维度自适应）
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
│   │   ├── preprocess.py             # 底层/legacy helper（不作为入口）
│   │   ├── dataset.py               # Dataset类
│   │   ├── split_strategy.py        # 划分策略
│   │   └── validate_preprocessing.py # 预处理结果自检
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
