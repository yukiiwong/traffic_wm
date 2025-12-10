# Traffic World Model

多智能体潜在世界模型，用于无人机轨迹预测和交通仿真。

---

## 📋 目录

- [项目简介](#项目简介)
- [快速开始](#快速开始)
- [数据处理](#数据处理)
- [模型训练](#模型训练)
- [参数调节](#参数调节)
- [GitHub上传](#github上传)
- [项目结构](#项目结构)
- [常见问题](#常见问题)

---

## 项目简介

本项目实现了基于Transformer的多智能体轨迹预测系统：

- 🚗 支持多站点（A-I）无人机数据处理
- 🎯 自动80/10/10数据分割
- 🧠 多种编码器架构（基础/增强/相对位置）
- 📊 完整的训练和评估流程
- 🎨 注意力可视化工具

### 核心功能

- ✅ 多站点数据自动混合分割（80/10/10）
- ✅ 灵活特征配置（6/8/10/11维，推荐10维扩展模式）
- ✅ 空间位置编码
- ✅ 社交池化（局部交互建模）
- ✅ 多种动态模型（GRU/LSTM/Transformer）
- ✅ 完整评估指标（ADE/FDE/速度/角度/碰撞率）

---

## 快速开始

### 1. 环境安装

```bash
# 克隆项目
git clone https://github.com/YOUR_USERNAME/traffic-world-model.git
cd traffic-world-model

# 安装依赖
pip install -r requirements.txt
```

**依赖包：**
- Python 3.10+
- PyTorch 2.0+
- numpy, pandas
- tqdm, pyyaml

---

### 2. 准备数据

将你的数据按以下结构放置：

```
traffic-world-model/
└── data/
    └── raw/
        ├── A/
        │   ├── drone_1.csv
        │   ├── drone_2.csv
        │   └── ...
        ├── B/
        │   └── ...
        ...
        └── I/
```

**CSV格式要求：**
- `track_id`: 车辆ID
- `frame`: 帧号
- `center_x`, `center_y`: 中心坐标
- `angle`: 角度
- `class_id`: 车辆类型
- `lane`: 车道ID（可选）
- `preceding_id`, `following_id`: 前后车ID（可选）

---

### 3. 数据预处理

```bash
# 最简单的方式（自动80/10/10分割，默认11维完整特征）
python preprocess_multisite.py
```

**默认配置：**
- `use_extended_features=True` → 包含车道和前后车信息
- `use_acceleration=True` → 自动计算并包含加速度
- **实际输出：11维完整特征**（位置+速度+加速度+角度+类型+车道+前后车）
- `episode_length=30` → 每个episode 30帧
- `max_vehicles=50` → 最多跟踪50辆车
- `overlap=5` → 相邻episode重叠5帧

**输出：**
```
data/processed/
├── train_episodes.npz      # 训练集 (80%)
├── val_episodes.npz        # 验证集 (10%)
├── test_episodes.npz       # 测试集 (10%)
├── metadata.json           # 元数据（含特征维度信息）
└── split_config.json       # 分割配置（记录哪些文件在哪个集合）
```

**自定义参数示例：**
```bash
python preprocess_multisite.py \
    --train_ratio 0.7 \
    --val_ratio 0.15 \
    --test_ratio 0.15 \
    --episode_length 30 \
    --max_vehicles 50 \
    --overlap 10
```

**注意：**
- `use_extended_features`和`use_acceleration`默认已启用（11维特征）
- 训练时`--input_dim`必须与预处理的特征维度匹配（默认为11）

---

### 4. 训练模型

**基础训练（默认11维特征）：**
```bash
python src/training/train_world_model.py \
    --train_data data/processed/train_episodes.npz \
    --val_data data/processed/val_episodes.npz \
    --input_dim 11 \
    --batch_size 32 \
    --n_epochs 100 \
    --latent_dim 256
```

**重要：** `--input_dim`必须与预处理时的特征维度匹配！检查`data/processed/metadata.json`中的`n_features`字段

**查看训练日志：**
```bash
tail -f logs/trainer.log
```

---

## 数据处理

### 工作流程

```
原始CSV → 混合分割 → 特征提取 → Episode生成 → NPZ保存
  (A-I)     (80/10/10)   (6/8/10/11维)  (T×K×F)      (训练数据)
```

**说明：** 特征维度F取决于配置（默认11维完整模式，推荐10维扩展模式）

### 特征说明

本系统支持4种特征配置，通过`use_extended_features`和`use_acceleration`控制：

**基础模式 (6维):** `use_extended_features=False, use_acceleration=False`
- `[0:2]` 位置 (center_x, center_y)
- `[2:4]` 速度 (vx, vy)
- `[4]` 角度 (angle)
- `[5]` 车辆类型 (class_id)

**基础+加速度 (8维):** `use_extended_features=False, use_acceleration=True`
- `[0:2]` 位置 (center_x, center_y)
- `[2:4]` 速度 (vx, vy)
- `[4:6]` 加速度 (ax, ay)
- `[6]` 角度 (angle)
- `[7]` 车辆类型 (class_id)

**扩展模式 (10维):** ⭐ 推荐 `use_extended_features=True, use_acceleration=False`
- `[0:2]` 位置 (center_x, center_y)
- `[2:4]` 速度 (vx, vy)
- `[4]` 角度 (angle)
- `[5]` 车辆类型 (class_id)
- `[6]` 车道ID（编码后的整数）
- `[7]` 是否有前车 (0/1)
- `[8]` 是否有后车 (0/1)
- `[9]` 填充（保证维度为10）

**完整模式 (11维):** `use_extended_features=True, use_acceleration=True`
- `[0:2]` 位置 (center_x, center_y)
- `[2:4]` 速度 (vx, vy)
- `[4:6]` 加速度 (ax, ay)
- `[6]` 角度 (angle)
- `[7]` 车辆类型 (class_id)
- `[8]` 车道ID（编码后的整数）
- `[9]` 是否有前车 (0/1)
- `[10]` 是否有后车 (0/1)

### 数据分割策略

**混合策略（当前实现）：**
1. 收集所有站点的所有CSV文件
2. 随机打乱
3. 按比例分配到train/val/test
4. 分别处理每个集合

**优点：**
- ✅ 最大化性能
- ✅ 每个集合都包含所有站点的数据
- ✅ 数据分布均衡

---

## 模型训练

### 模型架构

```
输入 [B, T, K, F]
  ↓
编码器 (Encoder)
  ├─ 基础: Transformer
  ├─ 增强: Transformer + 空间编码 + 社交池化
  └─ 相对: 图神经网络 + 相对位置
  ↓
潜在表示 [B, T, D]
  ↓
动态模型 (Dynamics)
  ├─ GRU (快速)
  ├─ LSTM (平衡)
  └─ Transformer (最佳)
  ↓
潜在预测 [B, T', D]
  ↓
解码器 (Decoder)
  ↓
预测轨迹 [B, T', K, F]
```

### 核心参数

| 参数 | 代码默认 | 说明 | 推荐 |
|-----|-------|------|------|
| `--input_dim` | 10 | 特征维度 | **必须匹配预处理！** 默认预处理输出11维 |
| `--latent_dim` | 256 | 潜在空间维度 | 128/256/512 |
| `--dynamics_type` | gru | 动态模型 | gru/lstm/transformer |
| `--batch_size` | 32 | 批次大小 | 根据显存调整 |
| `--learning_rate` | 1e-3 | 学习率 | 3e-4最保险 |
| `--n_epochs` | 100 | 训练轮数 | 100-300 |
| `--recon_weight` | 1.0 | 重建损失权重 | 保持1.0 |
| `--pred_weight` | 1.0 | 预测损失权重 | 1.0-2.0 |
| `--existence_weight` | 0.1 | 存在性损失权重 | 0.1-0.5 |

**完整参数列表：**
```bash
python src/training/train_world_model.py --help
```

---

## 参数调节

### 三阶段训练策略

#### 阶段1: 快速验证（1-2小时）
```bash
python src/training/train_world_model.py \
    --train_data data/processed/train_episodes.npz \
    --val_data data/processed/val_episodes.npz \
    --input_dim 11 \
    --latent_dim 128 \
    --n_epochs 10 \
    --batch_size 32
```
**目标：** 验证代码和数据正常

#### 阶段2: 基准测试（4-8小时）⭐ 推荐
```bash
python src/training/train_world_model.py \
    --train_data data/processed/train_episodes.npz \
    --val_data data/processed/val_episodes.npz \
    --input_dim 11 \
    --latent_dim 256 \
    --dynamics_type gru \
    --batch_size 32 \
    --n_epochs 100 \
    --learning_rate 3e-4 \
    --recon_weight 1.0 \
    --pred_weight 1.0 \
    --existence_weight 0.1
```
**目标：** 获得基准性能

**说明：** 使用默认11维完整特征（如用其他维度，需修改`--input_dim`）

#### 阶段3: 性能优化（1-3天）
```bash
python src/training/train_world_model.py \
    --train_data data/processed/train_episodes.npz \
    --val_data data/processed/val_episodes.npz \
    --input_dim 11 \
    --latent_dim 512 \
    --dynamics_type transformer \
    --batch_size 64 \
    --n_epochs 300 \
    --learning_rate 3e-4 \
    --pred_weight 1.5 \
    --existence_weight 0.2
```
**目标：** 获得最佳性能

**说明：** 使用Transformer动态模型和更大的latent_dim以获得最佳预测精度

---

### 关键参数调节指南

#### 1. latent_dim (模型大小)
- **128**: 快速实验，资源受限
- **256**: ⭐ 推荐，平衡性能和速度
- **512**: 追求最佳性能
- **1024**: 数据充足时使用

#### 2. learning_rate (学习率)
- **3e-4 (0.0003)**: ⭐ 最保险
- **1e-3 (0.001)**: 默认，训练快但可能不稳定
- **1e-4 (0.0001)**: 慢但稳定

**诊断：**
- Loss震荡 → 降低学习率
- 收敛太慢 → 提高学习率

#### 3. batch_size (批次大小)
- **8-16**: 小显存GPU (4GB)
- **32**: ⭐ 推荐，中等GPU (8GB)
- **64**: 大显存GPU (16GB+)

**诊断：**
- 显存不足 → 减小batch_size
- 训练太慢 → 增大batch_size

#### 4. dynamics_type (动态模型)
| 类型 | 速度 | 性能 | 显存 |
|-----|------|------|------|
| `gru` | ⭐⭐⭐ | ⭐⭐ | 低 |
| `lstm` | ⭐⭐ | ⭐⭐⭐ | 中 |
| `transformer` | ⭐ | ⭐⭐⭐⭐ | 高 |

#### 5. 损失权重
```bash
--recon_weight 1.0 \      # 重建损失（默认1.0）
--pred_weight 1.0 \       # 预测损失（默认1.0）
--existence_weight 0.1    # 存在性损失（默认0.1）
```

**调整策略：**
- 预测精度差 → 增加 `pred_weight` 到 1.5-2.0
- 车辆出现/消失不准 → 增加 `existence_weight` 到 0.2-0.5
- 重建质量差 → 保持 `recon_weight=1.0`，调整其他权重

---

### 常见问题诊断

| 问题 | 原因 | 解决方案 |
|-----|------|---------|
| Loss是NaN | 学习率太大 | `--learning_rate 1e-4` |
| Loss不下降 | 学习率太小或模型太小 | 提高学习率或增大模型 |
| 显存不足 | 批次或模型太大 | 减小 `batch_size` 或 `latent_dim` |
| 过拟合 | 模型太大或正则化不足 | `--weight_decay 1e-4` |
| 训练太慢 | 批次小或模型大 | 增大 `batch_size` 或用 `gru` |

---

## GitHub上传

### .gitignore已配置

以下文件**不会**上传到GitHub：
- ✅ `data/raw/` - 原始数据
- ✅ `data/processed/*.npz` - 处理后数据
- ✅ `checkpoints/` - 模型检查点
- ✅ `logs/` - 日志文件

### 上传步骤

```bash
# 1. 初始化Git
cd traffic-world-model
git init

# 2. 添加文件（大文件自动排除）
git add .

# 3. 提交
git commit -m "Initial commit: Traffic World Model"

# 4. 连接GitHub
git remote add origin https://github.com/YOUR_USERNAME/traffic-world-model.git
git branch -M main

# 5. 推送
git push -u origin main
```

### 验证排除

```bash
# 检查哪些文件会被上传
git status

# 测试特定文件是否被忽略
git check-ignore -v data/raw/A/drone_1.csv

# 应该输出：
# .gitignore:44:data/raw/    data/raw/A/drone_1.csv
```

### 后续更新

```bash
git add .
git commit -m "Update training script"
git push
```

---

## 项目结构

```
traffic-world-model/
├── data/
│   ├── raw/                    # 原始数据（不上传）
│   │   ├── A/
│   │   ├── B/
│   │   └── ...
│   └── processed/              # 处理后数据（不上传）
│       ├── train_episodes.npz
│       ├── val_episodes.npz
│       ├── test_episodes.npz
│       └── metadata.json
│
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── preprocess.py       # 数据预处理
│   │   ├── split_strategy.py   # 数据分割
│   │   └── dataset.py          # PyTorch Dataset
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── encoder.py          # 编码器（含增强版本）
│   │   ├── dynamics.py         # 动态模型
│   │   ├── decoder.py          # 解码器
│   │   └── world_model.py      # 完整模型
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train_world_model.py  # 训练脚本
│   │   └── losses.py           # 损失函数
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── prediction_metrics.py     # 评估指标
│   │   ├── rollout_eval.py           # Rollout评估
│   │   ├── visualization.py          # 可视化
│   │   └── attention_visualization.py # 注意力可视化
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logger.py           # 日志工具
│       ├── common.py           # 通用函数
│       └── config.py           # 配置管理
│
├── checkpoints/                # 模型检查点（不上传）
├── logs/                       # 日志（不上传）
├── experiments/                # 实验配置
│
├── preprocess_multisite.py     # 多站点预处理脚本
├── requirements.txt            # 依赖列表
├── .gitignore                  # Git忽略配置
└── README.md                   # 本文件
```

---

## 常见问题

### 数据处理相关

**Q: 权限错误：`Permission denied: '../../data'`**

A: 使用项目根目录运行，或创建必要目录：
```bash
mkdir -p data/raw data/processed checkpoints logs
python preprocess_multisite.py
```

**Q: 如何修改数据分割比例？**

A: 使用命令行参数：
```bash
python preprocess_multisite.py \
    --train_ratio 0.7 \
    --val_ratio 0.15 \
    --test_ratio 0.15
```

**Q: 预处理很慢怎么办？**

A: 减少 `max_vehicles` 或增加 `overlap`：
```bash
python preprocess_multisite.py \
    --max_vehicles 30 \
    --overlap 10
```

---

### 训练相关

**Q: 显存不足**

A:
```bash
# 方案1: 减小batch_size
--batch_size 16

# 方案2: 减小模型
--latent_dim 128

# 方案3: 使用GRU
--dynamics_type gru
```

**Q: Loss不下降**

A: 检查以下几点：
1. 学习率是否太小？尝试 `--learning_rate 1e-3`
2. 模型是否太小？尝试 `--latent_dim 512`
3. 数据是否正确加载？检查日志

**Q: 如何恢复训练？**

A:
```bash
python src/training/train_world_model.py \
    --train_data data/processed/train_episodes.npz \
    --val_data data/processed/val_episodes.npz \
    --resume checkpoints/checkpoint_epoch_50.pt
```

---

### GitHub相关

**Q: 在新电脑上克隆后，数据会被push吗？**

A: **不会**。`.gitignore` 会被一起克隆，自动排除数据文件。

**Q: 如何验证数据不会被上传？**

A:
```bash
git status  # 不应该看到 data/raw/
git check-ignore -v data/raw/A/drone_1.csv  # 应该显示被忽略
```

**Q: 意外提交了大文件怎么办？**

A:
```bash
git rm --cached -r data/raw/
git commit -m "Remove large files"
git push
```

---

## 性能指标

### 预期性能

| 指标 | 1秒预测 | 3秒预测 | 5秒预测 |
|-----|---------|---------|---------|
| ADE (平均位移误差) | < 1.5m | < 3.0m | < 5.0m |
| FDE (最终位移误差) | < 2.5m | < 6.0m | < 10.0m |

### 评估命令

```python
from src.evaluation.rollout_eval import evaluate_rollout
from src.data.dataset import TrajectoryDataset
from torch.utils.data import DataLoader

# 加载测试集
test_dataset = TrajectoryDataset('data/processed/test_episodes.npz')
test_loader = DataLoader(test_dataset, batch_size=32)

# 评估
metrics = evaluate_rollout(
    model=model,
    data_loader=test_loader,
    context_length=10,
    rollout_length=20
)

print(f"ADE: {metrics['ade']:.3f}m")
print(f"FDE: {metrics['fde']:.3f}m")
```

---

## 依赖要求

### requirements.txt

```
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
tqdm>=4.65.0
pyyaml>=6.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

### 安装

```bash
pip install -r requirements.txt
```

---

## 引用

如果使用本项目，请引用：

```bibtex
@software{traffic_world_model,
  title={Traffic World Model: Multi-Agent Trajectory Prediction},
  author={Your Name},
  year={2025},
  url={https://github.com/YOUR_USERNAME/traffic-world-model}
}
```

---

## 许可证

MIT License

---

## 更新日志

### v2.0 (2025-12-09)
- ✅ 统一数据预处理（混合分割策略）
- ✅ 修复路径处理问题
- ✅ 简化文档结构
- ✅ 整合所有编码器到单一文件

### v1.0 (2024)
- ✅ 初始版本发布
- ✅ 基础模型实现
- ✅ 多站点数据支持

---

**最后更新：** 2025-12-09
**状态：** ✅ 生产就绪
