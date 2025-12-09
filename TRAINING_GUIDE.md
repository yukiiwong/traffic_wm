# World Model 训练参数调节指南

## 📋 目录

- [快速开始](#快速开始)
- [参数完整列表](#参数完整列表)
- [参数调节策略](#参数调节策略)
- [常见场景配置](#常见场景配置)
- [性能优化](#性能优化)
- [故障排除](#故障排除)

---

## 快速开始

### 基础训练命令

```bash
python src/training/train_world_model.py \
    --train_data data/processed/train_episodes.npz \
    --val_data data/processed/val_episodes.npz \
    --batch_size 32 \
    --n_epochs 100 \
    --latent_dim 256
```

---

## 参数完整列表

### 1. 数据参数

| 参数 | 默认值 | 说明 | 调节建议 |
|-----|-------|------|---------|
| `--train_data` | **必需** | 训练数据路径 | `data/processed/train_episodes.npz` |
| `--val_data` | None | 验证数据路径 | `data/processed/val_episodes.npz` |
| `--num_workers` | 4 | 数据加载线程数 | CPU核心数的一半 |

---

### 2. 模型架构参数 ⭐ 重要

| 参数 | 默认值 | 说明 | 调节建议 |
|-----|-------|------|---------|
| `--input_dim` | 10 | 每个agent的特征数 | 6 (基础) 或 10 (扩展) |
| `--max_agents` | 50 | 每帧最大车辆数 | 与预处理时一致 |
| `--latent_dim` | 256 | 潜在空间维度 | 🔥 **关键参数** 见下方 |
| `--dynamics_type` | 'gru' | 动态模型类型 | gru/lstm/transformer |

#### latent_dim 调节指南

- **128**: 小模型，训练快，适合快速实验
- **256**: ⭐ **推荐**，平衡性能和速度
- **512**: 大模型，性能更好，但需要更多内存和时间
- **1024**: 超大模型，仅在数据量充足时使用

#### dynamics_type 选择

| 类型 | 速度 | 性能 | 显存占用 | 适用场景 |
|-----|------|------|---------|---------|
| `gru` | ⭐⭐⭐ | ⭐⭐ | 低 | 快速实验，资源受限 |
| `lstm` | ⭐⭐ | ⭐⭐⭐ | 中 | 需要长期依赖 |
| `transformer` | ⭐ | ⭐⭐⭐⭐ | 高 | 追求最佳性能 |

---

### 3. 训练参数 🎯 核心

| 参数 | 默认值 | 说明 | 调节建议 |
|-----|-------|------|---------|
| `--batch_size` | 32 | 批次大小 | 🔥 **关键参数** 见下方 |
| `--n_epochs` | 100 | 训练轮数 | 100-300 epochs |
| `--learning_rate` | 1e-3 | 学习率 | 🔥 **关键参数** 见下方 |
| `--weight_decay` | 1e-5 | 权重衰减（L2正则化） | 1e-5 ~ 1e-4 |

#### batch_size 调节指南

**影响因素：**
- GPU显存大小
- 训练稳定性
- 训练速度

**推荐值：**
- **8-16**: 小显存GPU（4GB）
- **32**: ⭐ 推荐，中等GPU（8GB）
- **64**: 大显存GPU（16GB+）
- **128**: 超大显存（24GB+），数据充足

**注意：** batch_size 越大，训练越稳定，但需要更多显存

#### learning_rate 调节指南

**默认配置：**
```
初始学习率: 1e-3 (0.001)
调度器: CosineAnnealingLR
最小学习率: 1e-6
```

**推荐值：**
- **3e-4**: ⭐ **最保险**，适合大多数情况
- **1e-3**: 默认值，训练快但可能不稳定
- **5e-4**: 平衡速度和稳定性
- **1e-4**: 慢但稳定，适合微调

**调节策略：**
1. 如果loss震荡 → 降低学习率
2. 如果收敛太慢 → 提高学习率
3. 如果在最优点附近震荡 → 降低最小学习率

---

### 4. 损失函数权重 ⚖️

| 参数 | 默认值 | 说明 | 调节建议 |
|-----|-------|------|---------|
| `--recon_weight` | 1.0 | 重建损失权重 | 保持1.0作为基准 |
| `--pred_weight` | 1.0 | 预测损失权重 | 1.0 ~ 2.0 |
| `--existence_weight` | 0.1 | 存在性损失权重 | 0.1 ~ 0.5 |

#### 损失权重调节策略

**场景1: 模型重建好但预测差**
```bash
--recon_weight 1.0 \
--pred_weight 2.0 \      # 增加预测权重
--existence_weight 0.1
```

**场景2: 车辆出现/消失预测不准**
```bash
--recon_weight 1.0 \
--pred_weight 1.0 \
--existence_weight 0.5   # 增加存在性权重
```

**场景3: 平衡配置（推荐）**
```bash
--recon_weight 1.0 \
--pred_weight 1.5 \
--existence_weight 0.2
```

---

### 5. 其他参数

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| `--seed` | 42 | 随机种子 |
| `--checkpoint_dir` | './checkpoints' | 检查点保存目录 |
| `--log_dir` | './logs' | 日志保存目录 |
| `--resume` | None | 从检查点恢复训练 |

---

## 参数调节策略

### 阶段1: 快速原型（1-2小时）

**目标：** 验证数据和代码是否正常

```bash
python src/training/train_world_model.py \
    --train_data data/processed/train_episodes.npz \
    --val_data data/processed/val_episodes.npz \
    --input_dim 10 \
    --max_agents 50 \
    --latent_dim 128 \              # 小模型
    --dynamics_type gru \           # 快速
    --batch_size 32 \
    --n_epochs 10 \                 # 少量epochs
    --learning_rate 1e-3
```

**期望结果：**
- 训练loss下降
- 验证loss下降
- 没有NaN或爆炸

---

### 阶段2: 基准测试（4-8小时）

**目标：** 获得基准性能

```bash
python src/training/train_world_model.py \
    --train_data data/processed/train_episodes.npz \
    --val_data data/processed/val_episodes.npz \
    --input_dim 10 \
    --max_agents 50 \
    --latent_dim 256 \              # 标准大小
    --dynamics_type gru \
    --batch_size 32 \
    --n_epochs 100 \                # 完整训练
    --learning_rate 3e-4 \          # 保守学习率
    --recon_weight 1.0 \
    --pred_weight 1.5 \
    --existence_weight 0.2
```

**期望结果：**
- ADE < 5.0m (1s预测)
- FDE < 10.0m (3s预测)
- 训练稳定

---

### 阶段3: 性能优化（1-3天）

**目标：** 获得最佳性能

#### 方案A: 增大模型容量

```bash
python src/training/train_world_model.py \
    --train_data data/processed/train_episodes.npz \
    --val_data data/processed/val_episodes.npz \
    --input_dim 10 \
    --latent_dim 512 \              # 大模型
    --dynamics_type transformer \   # 更强的模型
    --batch_size 64 \               # 更大batch
    --n_epochs 200 \
    --learning_rate 3e-4
```

#### 方案B: 调整损失权重

```bash
# 多次实验，尝试不同权重组合
python src/training/train_world_model.py \
    ... \
    --recon_weight 1.0 \
    --pred_weight 2.0 \             # 更重视预测
    --existence_weight 0.3
```

#### 方案C: 学习率调优

```bash
# 实验1: 较大学习率
python src/training/train_world_model.py \
    ... \
    --learning_rate 1e-3

# 实验2: 较小学习率
python src/training/train_world_model.py \
    ... \
    --learning_rate 1e-4

# 实验3: 中等学习率（通常最好）
python src/training/train_world_model.py \
    ... \
    --learning_rate 5e-4
```

---

## 常见场景配置

### 场景1: 资源受限（小GPU）

```bash
python src/training/train_world_model.py \
    --train_data data/processed/train_episodes.npz \
    --val_data data/processed/val_episodes.npz \
    --input_dim 10 \
    --latent_dim 128 \
    --dynamics_type gru \
    --batch_size 8 \                # 小batch
    --n_epochs 150 \
    --learning_rate 3e-4
```

---

### 场景2: 追求速度

```bash
python src/training/train_world_model.py \
    --train_data data/processed/train_episodes.npz \
    --input_dim 6 \                 # 使用基础特征
    --latent_dim 128 \
    --dynamics_type gru \
    --batch_size 64 \               # 大batch加速
    --n_epochs 50 \                 # 少量epochs
    --num_workers 8
```

---

### 场景3: 追求性能

```bash
python src/training/train_world_model.py \
    --train_data data/processed/train_episodes.npz \
    --val_data data/processed/val_episodes.npz \
    --input_dim 10 \                # 扩展特征
    --latent_dim 512 \              # 大模型
    --dynamics_type transformer \   # 强模型
    --batch_size 32 \
    --n_epochs 300 \                # 充分训练
    --learning_rate 3e-4 \
    --recon_weight 1.0 \
    --pred_weight 2.0 \
    --existence_weight 0.3
```

---

### 场景4: 长期预测

```bash
python src/training/train_world_model.py \
    --train_data data/processed/train_episodes.npz \
    --val_data data/processed/val_episodes.npz \
    --latent_dim 512 \
    --dynamics_type lstm \          # LSTM适合长期依赖
    --pred_weight 3.0 \             # 非常重视预测
    --n_epochs 200
```

---

## 性能优化

### GPU优化

1. **使用混合精度训练**（需要修改代码）
   - 可加速2-3倍
   - 减少50%显存占用

2. **增大batch_size**
   - 充分利用GPU并行能力
   - 提高训练稳定性

3. **多GPU训练**（需要修改代码）
   - 使用 DataParallel 或 DistributedDataParallel

### 数据加载优化

```bash
--num_workers 8  # 增加数据加载线程
```

### 超参数搜索

创建一个脚本尝试不同组合：

```bash
# search_hyperparams.sh
for lr in 1e-3 5e-4 3e-4 1e-4; do
  for latent_dim in 128 256 512; do
    python src/training/train_world_model.py \
        --train_data data/processed/train_episodes.npz \
        --val_data data/processed/val_episodes.npz \
        --latent_dim $latent_dim \
        --learning_rate $lr \
        --n_epochs 100 \
        --checkpoint_dir checkpoints/lr_${lr}_dim_${latent_dim}
  done
done
```

---

## 故障排除

### 问题1: Loss是NaN

**原因：** 学习率过大，梯度爆炸

**解决：**
```bash
--learning_rate 1e-4  # 降低学习率
```

---

### 问题2: Loss不下降

**可能原因和解决方案：**

1. **学习率太小**
   ```bash
   --learning_rate 1e-3  # 提高学习率
   ```

2. **模型容量不足**
   ```bash
   --latent_dim 512  # 增大模型
   ```

3. **数据问题**
   - 检查数据是否正确加载
   - 检查数据是否归一化

---

### 问题3: 训练/验证Loss差距大（过拟合）

**解决方案：**

1. **增加正则化**
   ```bash
   --weight_decay 1e-4  # 增加权重衰减
   ```

2. **减小模型**
   ```bash
   --latent_dim 128
   ```

3. **早停**
   - 监控验证loss，及时停止

---

### 问题4: 显存不足

**解决方案：**

1. **减小batch_size**
   ```bash
   --batch_size 16  # 或更小
   ```

2. **减小模型**
   ```bash
   --latent_dim 128
   ```

3. **使用GRU而不是Transformer**
   ```bash
   --dynamics_type gru
   ```

---

### 问题5: 训练太慢

**解决方案：**

1. **增大batch_size**
   ```bash
   --batch_size 64
   ```

2. **增加数据加载线程**
   ```bash
   --num_workers 8
   ```

3. **使用更简单的模型**
   ```bash
   --dynamics_type gru
   --latent_dim 128
   ```

---

## 监控训练进度

### 查看日志

```bash
# 实时查看训练日志
tail -f logs/trainer.log

# 搜索最佳验证loss
grep "Val Loss" logs/trainer.log | sort -k6 -n | head -5
```

### 使用TensorBoard（需要添加）

如果后续添加TensorBoard支持：

```bash
tensorboard --logdir logs/
```

---

## 推荐实验流程

```
1. 快速原型（latent_dim=128, 10 epochs）
   └─> 验证代码和数据正常

2. 基准实验（latent_dim=256, 100 epochs, lr=3e-4）
   └─> 获得基准ADE/FDE

3. 学习率扫描（lr in [1e-4, 3e-4, 5e-4, 1e-3]）
   └─> 找到最佳学习率

4. 模型大小扫描（latent_dim in [128, 256, 512]）
   └─> 平衡性能和速度

5. 损失权重调优（pred_weight in [1.0, 1.5, 2.0, 3.0]）
   └─> 优化预测性能

6. 最终训练（最佳配置, 200-300 epochs）
   └─> 获得最佳模型
```

---

## 参数速查表

| 目标 | 推荐配置 |
|-----|---------|
| 快速实验 | `latent_dim=128, batch_size=32, n_epochs=10` |
| 基准测试 | `latent_dim=256, lr=3e-4, n_epochs=100` |
| 最佳性能 | `latent_dim=512, dynamics=transformer, n_epochs=300` |
| 资源受限 | `latent_dim=128, batch_size=8, dynamics=gru` |
| 长期预测 | `dynamics=lstm, pred_weight=3.0` |

---

**最后更新:** 2025-12-09
**版本:** 1.0
