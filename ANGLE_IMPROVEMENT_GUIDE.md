# Angle (Heading) 改进实现指南

## 问题根源

之前 `heading_error` 一直很高的原因：

1. **角度归一化破坏周期性**：z-score 归一化将 `-π` 和 `π`（同一方向）变成两个很远的数值
2. **损失函数不合适**：Huber loss 不理解角度的周期性（预测 `3.0` vs 真值 `-3.0` 被认为差 `6.0`，实际只差 `0.28`）
3. **缺少物理先验**：x/y 有运动学先验，但 angle 没有（速度方向本身就是 heading 的强先验）

## 解决方案（方案 C）

### 1. 数据预处理层面

**修改文件**：`preprocess_multisite.py`
- 将 `angle` (index 6) 添加到 `do_not_normalize` 列表
- 在 metadata 中记录 `angle_idx: 6`

**修改文件**：`src/data/dataset.py`
- 解析 `angle_idx` 并从 `continuous_indices` 中排除
- angle 保持原始 radian 值，不做 z-score 归一化

### 2. 模型架构层面

**修改文件**：`src/models/decoder.py`
- 增加 `enable_angle_head` 参数（默认 True）
- 新增 `angle_head: Linear(hidden, max_agents)`，输出未归一化的 angle
- `forward()` 返回新增 `angle: [B, T, K]`（raw radians）

**修改文件**：`src/models/world_model.py`
- 新增 `idx_angle: int = 6` 参数
- 新增 `_kinematic_prior_angle()` 方法：基于速度计算 angle 先验
  ```python
  angle_prior = torch.atan2(vy, vx)  # 速度方向
  ```
- `forward()` 中：
  - Decoder 输出 `recon_angle` 和 `pred_angle_base`
  - 将 angle prior 与 decoder 预测混合：`pred_angle = 0.7 * prior + 0.3 * pred_base`
  - 返回新增字段：`reconstructed_angle`, `predicted_angle`

### 3. 损失函数层面

**修改文件**：`src/training/losses.py`
- 新增 `angle_weight` 参数（默认 0.5）
- 新增 `_angular_distance()` 静态方法：使用 `atan2(sin, cos)` 计算周期性角度差
- 新增 `_angular_loss()` 方法：对 angle 使用 angular distance
- `forward()` 中计算：
  - `recon_angle_loss`：重建时的 angle 误差
  - `pred_angle_loss`：预测时的 angle 误差（one-step ahead）
  - 总损失加入：`angle_weight * (recon_angle_loss + pred_angle_loss)`

**修改文件**：`src/training/train_world_model.py`
- 从 metadata 读取 `angle_idx`
- 实例化 `WorldModelLoss` 时传入 `angle_idx` 和 `angle_weight=0.5`
- 日志中显示 angle loss

## 使用步骤

### 1. 重新预处理数据

```bash
python preprocess_multisite.py
```

这会生成新的 `metadata.json`，其中：
- `validation_info.angle_idx: 6`
- `validation_info.do_not_normalize: ['lane_id', 'class_id', 'site_id', 'angle']`

### 2. 训练模型

```bash
python src/training/train_world_model.py \
  --train_data data/processed/train_episodes.npz \
  --val_data data/processed/val_episodes.npz \
  --checkpoint_dir checkpoints/world_model/v3_angle_fix \
  --epochs 50 \
  --batch_size 16
```

训练日志会显示：
```
[Epoch 1/50] train_loss=... val_loss=... recon=... pred=...
  [ANGLE] recon_angle=0.15  pred_angle=0.12
  [DIAG] ADE=... FDE=... heading_error=...  <- 应该明显降低
```

### 3. 关键参数调优

**angle prior 混合权重**（在 `world_model.py`）：
```python
pred_angle = 0.7 * angle_prior + 0.3 * pred_angle_base
```
- 当前：70% 物理先验 + 30% 网络学习
- 如果 heading_error 仍高：提高先验权重到 0.8 或 0.9
- 如果过拟合先验：降低到 0.5

**angle loss 权重**（在 `train_world_model.py`）：
```python
angle_weight=0.5
```
- 默认：0.5（与 pred_weight=1.0 相比，angle 占总损失的 1/3）
- 如果 angle 学习太慢：提高到 1.0
- 如果 angle 过拟合牺牲 x/y：降低到 0.3

## 预期效果

1. **heading_error 显著降低**：从 30-50° 降至 5-10°
2. **angle 收敛更快**：物理先验提供强引导
3. **训练稳定性提升**：angular distance 避免周期性带来的梯度爆炸

## 向后兼容性

**旧数据**：必须重新预处理（angle 需要保持原始值）
**旧模型**：不兼容（decoder 结构变化，增加了 angle_head）
**评估脚本**：需要更新（处理新增的 `reconstructed_angle` 和 `predicted_angle` 字段）

## 检查清单

- [ ] 重新运行 `preprocess_multisite.py`
- [ ] 确认 `metadata.json` 中 `angle_idx: 6` 和 `do_not_normalize` 包含 `'angle'`
- [ ] 训练时日志显示 `recon_angle_loss` 和 `pred_angle_loss`
- [ ] 验证集上 `heading_error` < 10°
- [ ] 可视化预测轨迹，确认车辆朝向与速度方向一致

## 故障排查

**问题：angle loss 始终是 0**
- 检查 decoder 是否启用了 `enable_angle_head=True`
- 检查 forward 是否调用了 `return_angle=True`

**问题：angle loss 很大（> 1.0 rad）**
- 检查 metadata 中 angle 是否真的没被归一化
- 打印 `states[..., 6]` 确认范围在 `[-π, π]` 而非归一化后的值

**问题：训练崩溃 NaN**
- angle prior 可能在速度为 0 时不稳定
- 在 `_kinematic_prior_angle` 中加入 eps 保护：`angle_prior = torch.atan2(vy + 1e-6, vx + 1e-6)`
