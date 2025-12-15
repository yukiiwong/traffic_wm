# 训练问题诊断与解决方案

## 问题分析

### 1. Angle (朝向角) 学习失败 ⚠️ 最严重

**现象**:
- Angle MAE 始终高达 0.84-0.89 弧度（48-51°）
- 16 个 epoch 几乎没有改善
- 理论上应该降到 0.09-0.17 弧度（5-10°）

**可能原因**:
```python
# 检查 1: Angular distance loss 是否正确应用？
# src/training/losses.py 中需要确认：
# - angle_weight 参数是否传入
# - _angular_loss() 是否被调用
# - angle_idx 是否正确识别为 6

# 检查 2: Angle head 是否启用？
# src/models/decoder.py 中需要确认：
# - enable_angle_head=True
# - forward() 返回 angle

# 检查 3: Angle prior 是否应用？
# src/models/world_model.py 中需要确认：
# - _kinematic_prior_angle() 是否被调用
# - 混合权重是否合理 (0.7 * prior + 0.3 * pred)
```

**解决方案**:
1. **立即检查代码**:
   ```bash
   # 查看 losses.py 是否使用了 angular loss
   grep -n "angle" src/training/losses.py
   
   # 查看 train_world_model.py 是否传入 angle_idx
   grep -n "angle_idx\|angle_weight" src/training/train_world_model.py
   ```

2. **如果代码正确，调整参数**:
   ```python
   # 在 train_world_model.py 中增大 angle_weight
   loss_fn = WorldModelLoss(
       ...,
       angle_weight=1.5  # 从 0.5 增加到 1.5
   )
   
   # 在 world_model.py 中增大 prior 权重
   pred_angle = 0.85 * angle_prior + 0.15 * pred_angle_base
   ```

3. **检查 angle 是否被归一化了**:
   ```python
   # 在 dataset.py 中确认 angle 不在 continuous_indices 中
   print(f"Continuous indices: {continuous_indices}")
   print(f"Angle idx: {angle_idx}")
   assert angle_idx not in continuous_indices
   ```

---

### 2. 二值特征 (has_preceding/following) 预测差 ⚠️

**现象**:
- has_preceding MAE: 0.67-0.70 (理论应该 < 0.3)
- has_following MAE: 0.63-0.68 (理论应该 < 0.3)

**可能原因**:
- 二值特征应该用 BCELoss 而非 Huber Loss
- 或者这些特征本身就很难预测（与周围车辆有关）

**解决方案**:
1. **暂时接受** - 这些特征不影响核心预测（x, y, v）
2. **如果要改进**:
   ```python
   # 在 decoder.py 中为二值特征添加 sigmoid
   has_preceding = torch.sigmoid(states[..., 7])
   has_following = torch.sigmoid(states[..., 8])
   
   # 在 losses.py 中对这两个特征单独用 BCE loss
   ```

---

### 3. Val Loss 过早停滞（过拟合）

**现象**:
- Train loss: 1.78 → 1.35 (持续下降)
- Val loss: 1.60 → 1.38 → 停滞波动

**解决方案**:

#### 方案 A: 增加正则化 (推荐)
```bash
python src/training/train_world_model.py \
    --weight_decay 0.001 \  # 从 0.0001 增加 10 倍
    --dropout 0.2 \          # 如果支持，增加 dropout
    ...
```

#### 方案 B: 减小模型容量
```bash
python src/training/train_world_model.py \
    --latent_dim 256 \       # 从 512 减少到 256
    --dynamics_layers 3 \    # 从 4 减少到 3
    ...
```

#### 方案 C: 数据增强
```python
# 在 dataset.py 中添加随机噪声
if self.training:
    continuous_feats += torch.randn_like(continuous_feats) * 0.01
```

#### 方案 D: Learning Rate 调整
```bash
# 使用学习率调度器（如果还没有）
python src/training/train_world_model.py \
    --lr 1e-4 \              # 降低学习率
    --scheduler cosine \     # 添加 cosine annealing
    ...
```

---

## 🚀 立即行动计划

### Step 1: 诊断 Angle 问题（最优先）
```bash
# 1. 检查训练脚本
cat src/training/train_world_model.py | grep -A 5 "angle"

# 2. 检查 loss 函数
cat src/training/losses.py | grep -A 10 "angular"

# 3. 检查 world_model
cat src/models/world_model.py | grep -A 5 "angle"
```

### Step 2: 查看 metadata
```bash
# 确认 angle 确实没有被归一化
cat data/processed_siteA/metadata.json | grep -A 10 "validation_info"
```

### Step 3: 修复并重新训练
1. 如果发现 angle 相关代码缺失 → 参考 `ANGLE_IMPROVEMENT_GUIDE.md` 实现
2. 如果代码正确但效果差 → 调整 `angle_weight` 和 prior 混合权重
3. 重新训练并监控 angle MAE

---

## 📈 期望改进

修复后的训练应该看到：

**Epoch 10** (修复后):
```
[RECON MAE PER FEATURE]
  angle: 0.65  # 从 0.84 降到 0.65 (37°)
  
[PRED MAE PER FEATURE]  
  angle: 0.68  # 从 0.85 降到 0.68 (39°)
```

**Epoch 30** (修复后):
```
[RECON MAE PER FEATURE]
  angle: 0.20  # 降到 0.20 (11°)
  
[PRED MAE PER FEATURE]
  angle: 0.25  # 降到 0.25 (14°)
```

**最终目标**:
- Angle MAE < 0.17 弧度 (< 10°)
- Val loss 继续下降到 1.2 以下
- ADE/FDE 保持稳定
