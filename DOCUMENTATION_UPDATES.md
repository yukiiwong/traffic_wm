# 文档更新补充 - Angle 处理与工具函数

本文档包含对 README.md 和 CODE_DOCUMENTATION.md 的重要补充更新，主要涵盖：
1. Angle (朝向角) 特殊处理架构
2. 新增工具函数文档
3. 训练/评估流程更新

---

## 1. Angle 处理架构补充

### 1.1 为什么需要特殊处理 Angle

**问题根源**:
- ❌ **Z-score 归一化破坏周期性**: `-π` 和 `π` 是同一方向，但归一化后相差很远
- ❌ **Huber loss 不理解周期性**: 预测 `3.0` vs 真值 `-3.0` 被认为误差 `6.0`，实际只差 `0.28`
- ❌ **缺少物理先验**: x/y 有运动学先验，但 angle 没有

**解决方案 (方案 C - 最彻底)**:
1. **预处理**: Angle 不做 z-score 归一化，保持原始弧度 `[-π, π]`
2. **架构**: 专用 angle_head 输出层
3. **物理先验**: `angle ≈ atan2(vy, vx)` (速度方向)
4. **损失函数**: Angular distance loss (处理周期性)

### 1.2 代码修改清单

#### `preprocess_multisite.py`
```python
# 添加到 do_not_normalize 列表
do_not_normalize = ['lane_id', 'class_id', 'site_id', 'angle']

# 在 validation_info 中记录
validation_info = {
    "discrete_features": {...},
    "do_not_normalize": do_not_normalize,
    "angle_idx": 6  # 新增
}
```

#### `src/data/dataset.py`
```python
# _load_discrete_feature_indices()
angle_idx = validation_info.get('angle_idx', 6)
# 从 continuous_indices 中排除 angle
continuous_indices = [i for i in range(n_features) 
                      if i not in discrete_indices and i != angle_idx]
```

#### `src/models/decoder.py`
```python
class StateDecoder(nn.Module):
    def __init__(self, ..., enable_angle_head=True):
        ...
        if enable_angle_head:
            self.angle_head = nn.Linear(hidden_dim, max_agents)
            nn.init.normal_(self.angle_head.weight, std=0.01)
            nn.init.zeros_(self.angle_head.bias)
    
    def forward(self, latent, return_residual_xy=False, return_angle=False):
        ...
        angle = None
        if return_angle and self.enable_angle_head:
            angle = self.angle_head(h).view(B, T, max_agents)
        return states, existence_logits, residual_xy, angle
```

#### `src/models/world_model.py`
```python
class WorldModel(nn.Module):
    def __init__(self, ..., idx_angle=6):
        self.idx_angle = idx_angle
        ...
    
    def _kinematic_prior_angle(self, prev_states):
        """基于速度方向计算 angle 先验"""
        vx = self._denorm(prev_states[..., self.idx_vx], self.idx_vx)
        vy = self._denorm(prev_states[..., self.idx_vy], self.idx_vy)
        angle_prior = torch.atan2(vy, vx)  # [-π, π]
        return angle_prior
    
    def forward(self, states, masks):
        ...
        # Decoder 输出 angle
        recon_states, exist_logits, _, recon_angle = \
            decoder(latent, return_angle=True)
        pred_states_base, pred_exist_logits, residual_xy, pred_angle_base = \
            decoder(predicted_latent, return_residual_xy=True, return_angle=True)
        
        # 混合先验与预测
        angle_prior = self._kinematic_prior_angle(states)
        pred_angle = 0.7 * angle_prior + 0.3 * pred_angle_base
        
        return {
            ...,
            "reconstructed_angle": recon_angle,
            "predicted_angle": pred_angle
        }
```

#### `src/training/losses.py`
```python
class WorldModelLoss(nn.Module):
    def __init__(self, ..., angle_idx=6, angle_weight=0.5):
        self.angle_idx = angle_idx
        self.angle_weight = angle_weight
    
    @staticmethod
    def _angular_distance(pred_angle, target_angle):
        """计算周期性角度距离"""
        diff = pred_angle - target_angle
        distance = torch.atan2(torch.sin(diff), torch.cos(diff))
        return distance.abs()
    
    def _angular_loss(self, pred_angle, target_angle, mask):
        """Masked angular loss"""
        distance = self._angular_distance(pred_angle, target_angle)
        masked_distance = distance * mask
        return masked_distance.sum() / mask.sum().clamp(min=1.0)
    
    def forward(self, predictions, targets):
        ...
        # Angle losses
        recon_angle_loss = self._angular_loss(
            predictions['reconstructed_angle'],
            targets['states'][..., self.angle_idx],
            targets['masks']
        )
        pred_angle_loss = self._angular_loss(
            predictions['predicted_angle'][:, :-1],
            targets['states'][:, 1:, :, self.angle_idx],
            targets['masks'][:, :-1]
        )
        
        total_loss += self.angle_weight * (recon_angle_loss + pred_angle_loss)
        
        return {
            ...,
            "recon_angle_loss": recon_angle_loss.detach(),
            "pred_angle_loss": pred_angle_loss.detach()
        }
```

#### `src/training/train_world_model.py`
```python
# 从 metadata 读取 angle_idx
meta = train_loader.dataset.metadata
angle_idx = meta.get('validation_info', {}).get('angle_idx', 6)

# 创建 loss 函数
loss_fn = WorldModelLoss(
    ...,
    angle_idx=angle_idx,
    angle_weight=0.5
)

# 训练循环中打印 angle loss
print(f"[Epoch {epoch}] ... recon_angle={metrics['recon_angle_loss']:.4f} "
      f"pred_angle={metrics['pred_angle_loss']:.4f}")
```

### 1.3 预期效果

**训练指标**:
- Heading error: 30-50° → 5-10°
- Angle收敛更快 (物理先验提供强引导)
- 训练稳定性提升 (angular distance 避免梯度爆炸)

**参数调优**:
```python
# world_model.py - angle prior 混合权重
pred_angle = 0.7 * angle_prior + 0.3 * pred_angle_base
# 增大先验权重 (0.8-0.9) 如果 heading_error 仍高
# 减小先验权重 (0.5) 如果过拟合先验

# train_world_model.py - angle loss 权重
angle_weight = 0.5
# 增大 (1.0) 如果 angle 学习太慢
# 减小 (0.3) 如果 angle 过拟合牺牲 x/y
```

---

## 2. 工具函数文档

### 2.1 `src/utils/common.py` - parse_discrete_feature_indices_from_metadata

**函数签名**:
```python
def parse_discrete_feature_indices_from_metadata(
    metadata: dict
) -> Tuple[List[int], Optional[int], Optional[int], Optional[int]]:
```

**作用**: 从 metadata 中解析离散特征索引 (集中化解析逻辑，避免重复代码)

**参数**:
- `metadata`: dict - 包含 `validation_info` 字段的 metadata

**返回**:
```python
(
    discrete_indices: [7, 8, 11],  # sorted list
    idx_lane: 8,                    # lane_id index (or None)
    idx_class: 7,                   # class_id index (or None)
    idx_site: 11                    # site_id index (or None)
)
```

**Fallback 逻辑**:
- 如果 metadata 缺失 `validation_info` 或 `discrete_features`
- 返回默认值: `([7, 8, 11], 8, 7, 11)`

**使用示例**:
```python
# 在训练脚本中
meta = train_loader.dataset.metadata
discrete_indices, idx_lane, idx_class, idx_site = \
    parse_discrete_feature_indices_from_metadata(meta)

# 传递给 WorldModel
model = WorldModel(
    ...,
    lane_feature_idx=idx_lane,
    class_feature_idx=idx_class,
    site_feature_idx=idx_site
)

# 在评估脚本中
from src.utils.common import parse_discrete_feature_indices_from_metadata
discrete_indices, idx_lane, _, _ = \
    parse_discrete_feature_indices_from_metadata(metadata)
```

**好处**:
1. ✅ 集中化解析逻辑，避免 hardcode
2. ✅ 所有脚本使用统一接口
3. ✅ Fallback 保证兼容性
4. ✅ 减少配置漂移问题

---

## 3. 训练流程更新

### 3.1 完整训练命令 (含 Angle)

```bash
# 1. 重新预处理数据 (确保 angle 不被归一化)
python preprocess_multisite.py \
    --raw_data_dir data/raw \
    --output_dir data/processed \
    --sites A B C D E F G H I \
    --episode_length 80 \
    --stride 15

# 2. 验证预处理
python validate_preprocessing.py
# 检查: angle_idx: 6, do_not_normalize 包含 'angle'

# 3. 训练模型 (自动使用 angle 改进)
python src/training/train_world_model.py \
    --train_data data/processed/train_episodes.npz \
    --val_data data/processed/val_episodes.npz \
    --checkpoint_dir checkpoints/world_model/v3_angle_fix \
    --input_dim 12 \
    --latent_dim 256 \
    --batch_size 16 \
    --epochs 50 \
    --lr 3e-4
```

### 3.2 训练日志示例

```
[Epoch 1/50] train_loss=12.3456  val_loss=13.4567
  [STATE] recon=10.234 pred=2.345 exist=0.123 pred_exist=0.098
  [ANGLE] recon_angle=0.15  pred_angle=0.12  <- 新增
  
[Epoch 25/50] train_loss=3.4567  val_loss=4.1234
  [STATE] recon=2.345 pred=0.987 exist=0.098 pred_exist=0.087
  [ANGLE] recon_angle=0.05  pred_angle=0.04  <- 显著降低
```

### 3.3 评估更新

**Rollout 评估**:
```bash
python src/evaluation/rollout_eval.py \
    --checkpoint checkpoints/world_model/v3_angle_fix/checkpoint_best.pt \
    --test_data data/processed/test_episodes.npz \
    --context_length 65 \
    --rollout_horizon 15
```

**期望指标**:
```json
{
  "ade": 0.10,              // 10cm (unchanged)
  "fde": 0.12,              // 12cm (unchanged)
  "velocity_error": 0.08,   // 8cm/s (unchanged)
  "heading_error": 5.2,     // 5.2° (improved from 30-50°) ✅
  "collision_rate": 5.2%
}
```

---

## 4. 故障排查

### 4.1 Angle Loss 始终是 0

**检查**:
```python
# 1. Decoder 是否启用 angle_head
assert model.decoder.enable_angle_head == True

# 2. Forward 是否调用 return_angle=True
preds = model(states, masks)
assert 'reconstructed_angle' in preds
assert 'predicted_angle' in preds
```

### 4.2 Angle Loss 很大 (> 1.0 rad)

**检查**:
```python
# 1. Metadata 中 angle 是否真的没被归一化
with open('data/processed/metadata.json') as f:
    meta = json.load(f)
    assert 'angle' in meta['validation_info']['do_not_normalize']

# 2. 数据范围检查
states = train_loader.dataset.states
angle_values = states[..., 6]
print(f"Angle range: [{angle_values.min():.2f}, {angle_values.max():.2f}]")
# 应该是 [-3.14, 3.14] 而非归一化后的值
```

### 4.3 训练崩溃 NaN

**原因**: angle prior 在速度为 0 时可能不稳定

**修复**:
```python
# src/models/world_model.py: _kinematic_prior_angle
def _kinematic_prior_angle(self, prev_states):
    vx = self._denorm(prev_states[..., self.idx_vx], self.idx_vx)
    vy = self._denorm(prev_states[..., self.idx_vy], self.idx_vy)
    
    # 添加 eps 保护
    eps = 1e-6
    angle_prior = torch.atan2(vy + eps, vx + eps)
    return angle_prior
```

---

## 5. 向后兼容性

**旧数据**: 
- ❌ **必须重新预处理** (angle 需要保持原始值，不能使用旧的归一化数据)

**旧模型**:
- ❌ **不兼容** (decoder 结构变化，增加了 angle_head)
- 需要重新训练

**评估脚本**:
- ⚠️ **需要更新** (处理新增的 `reconstructed_angle` 和 `predicted_angle` 字段)

---

## 6. 检查清单

训练前:
- [ ] 重新运行 `preprocess_multisite.py`
- [ ] 确认 `metadata.json` 中 `angle_idx: 6`
- [ ] 确认 `do_not_normalize` 包含 `'angle'`
- [ ] 运行 `validate_preprocessing.py` 通过

训练中:
- [ ] 日志显示 `recon_angle_loss` 和 `pred_angle_loss`
- [ ] Angle loss 逐步下降 (初期 ~0.3，后期 ~0.05)

训练后:
- [ ] 验证集上 `heading_error < 10°`
- [ ] 可视化预测轨迹，确认车辆朝向与速度方向一致
- [ ] 对比旧模型，heading_error 显著降低

---

**文档版本**: 1.0
**生成时间**: 2025-12-14
**相关文档**: 
- `ANGLE_IMPROVEMENT_GUIDE.md` - 详细实现指南
- `README.md` - 用户使用指南
- `CODE_DOCUMENTATION.md` - 代码级文档
