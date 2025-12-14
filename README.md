# Traffic World Model - 多智能体轨迹预测

基于Transformer的潜在世界模型，用于多站点无人机轨迹数据的预测和仿真。

**核心特性**:
- 🚁 多站点无人机数据支持 (Sites A-I)
- 🧠 Transformer编码器 + Transformer时序动力学 + 物理先验
- 🎯 12维特征 + Site/Lane/Class embeddings
- ⏱️ 时序无重叠的train/val/test划分
- 🔧 完整的预处理、训练、评估流程

---

## 📋 目录

1. [快速开始](#-快速开始)
2. [数据预处理](#-数据预处理)
3. [模型训练](#-模型训练)
4. [模型评估](#-模型评估)
5. [可视化](#-可视化)
6. [项目结构](#-项目结构)
7. [代码文件详解](#-代码文件详解)
8. [重要说明](#-重要说明)
9. [故障排查](#-故障排查)

---

## 🚀 快速开始

### 环境安装

```bash
# Python 3.10+
pip install -r requirements.txt
```

**核心依赖**:
- PyTorch >= 2.0.0
- NumPy >= 1.24.0
- Pandas >= 2.0.0
- tqdm, matplotlib, seaborn, opencv-python

### 完整流程（4步）

```bash
# 1. 数据预处理
python preprocess_multisite.py

# 2. 验证数据
python validate_preprocessing.py

# 3. 训练模型
python src/training/train_world_model.py \
    --train_data data/processed/train_episodes.npz \
    --val_data data/processed/val_episodes.npz \
    --input_dim 12 \
    --latent_dim 256 \
    --batch_size 16 \
    --epochs 50 \
    --lr 3e-4

# 4. 评估模型
python src/evaluation/rollout_eval.py \
    --checkpoint checkpoints/best_model.pt \
    --test_data data/processed/test_episodes.npz \
    --context_length 65 \
    --rollout_horizon 15
```

---

## 📊 数据预处理

### 步骤1: 数据准备

**输入数据结构**:
```
data/raw/
├── A/
│   ├── drone_1.csv
│   ├── drone_2.csv
│   └── ...
├── B/
│   ├── drone_1.csv
│   └── ...
...
└── I/
    └── ...
```

**CSV必需列**:
- `track_id`: 车辆ID
- `frame`: 帧号
- `center_x`, `center_y`: 中心坐标
- `angle`: 朝向角度
- `class_id`: 车辆类别

**可选列**:
- `lane`: 车道ID
- `preceding_id`, `following_id`: 前后车ID
- `timestamp`: 时间戳

### 步骤2: 运行预处理

**使用的代码文件**:
- 📄 **主脚本**: `preprocess_multisite.py`
- 📄 **核心逻辑**: `src/data/preprocess.py`
- 📄 **数据划分**: `src/data/split_strategy.py`

**命令**:
```bash
# 默认配置（推荐）
python preprocess_multisite.py

# 自定义参数
python preprocess_multisite.py \
    --raw_data_dir data/raw \
    --output_dir data/processed \
    --episode_length 80 \
    --stride 15 \
    --fps 30.0 \
    --train_ratio 0.7 \
    --val_ratio 0.15 \
    --test_ratio 0.15
```

**处理流程**:

```
preprocess_multisite.py
  ↓ 调用
src/data/preprocess.py
  ├─ build_global_timeline()          # 1. 构建每个site的全局时间线
  │   └─ 合并所有CSV, 处理frame重置
  │
  ├─ detect_gaps_and_split_segments() # 2. 检测时序gap并分段
  │   └─ 确保episodes不跨越gap
  │
  ├─ extract_fixed_stride_episodes()  # 3. 固定步长提取episodes
  │   └─ T=80帧, stride=15帧
  │
  ├─ encode_lane()                     # 4. 编码lane为site-specific token
  │   └─ "A:A1", "B:crossroads1"等
  │
  └─ extract_extended_features()       # 5. 提取12维特征
      ├─ center_x, center_y (标准化)
      ├─ vx, vy (速度)
      ├─ ax, ay (加速度)
      ├─ angle (朝向)
      ├─ class_id (离散, 不标准化)
      ├─ lane_id (离散, 不标准化)
      ├─ has_preceding, has_following
      └─ site_id (离散, 不标准化)
  ↓
src/data/split_strategy.py
  └─ chronological_split_episodes()    # 6. 时序划分
      └─ 按时间顺序分train/val/test
```

**输出文件**:
```
data/processed/
├── train_episodes.npz               # [N_train, 80, 50, 12]
│   ├── 'states'        → [N, T, K, F] 状态矩阵
│   ├── 'masks'         → [N, T, K] 有效性mask
│   ├── 'scene_ids'     → [N] site ID
│   ├── 'start_frames'  → [N] episode起始帧
│   └── 'end_frames'    → [N] episode结束帧
│
├── val_episodes.npz                 # 同上
├── test_episodes.npz                # 同上
│
├── metadata.json                    # 元数据
│   ├── n_features: 12
│   ├── episode_length: 80
│   ├── context_length: 65
│   ├── rollout_horizon: 15
│   ├── fps: 30.0
│   ├── feature_layout: {...}
│   ├── lane_mapping: {...}
│   └── validation_info: {
│       ├── discrete_features: {7, 8, 11}
│       └── do_not_normalize: [...]
│   }
│
└── split_config.json                # 划分配置
    ├── train_files: [...]
    ├── val_files: [...]
    └── test_files: [...]
```

**重要参数**:
- `T=80`: Episode长度 (80帧 ≈ 2.67秒 @ 30 FPS)
- `C=65`: Context长度 (warm-up, 前65帧)
- `H=15`: Rollout horizon (预测后15帧)
- `K=50`: 最大车辆数（padding）
- `F=12`: 特征维度

### 步骤3: 特征说明

预处理生成**12维特征向量** (`src/data/preprocess.py:extract_extended_features()`):

| 索引 | 特征名 | 类型 | 说明 | 代码位置 |
|------|--------|------|------|---------|
| 0 | center_x | 连续 | X坐标（z-score标准化） | `extract_extended_features()` L385 |
| 1 | center_y | 连续 | Y坐标（z-score标准化） | L386 |
| 2 | vx | 连续 | X方向速度 | L388 |
| 3 | vy | 连续 | Y方向速度 | L389 |
| 4 | ax | 连续 | X方向加速度 | L390 |
| 5 | ay | 连续 | Y方向加速度 | L391 |
| 6 | angle | 连续 | 朝向角度 | L392 |
| 7 | class_id | **离散** | 车辆类别（不标准化） | L393 |
| 8 | lane_id | **离散** | 车道ID（不标准化） | L394 |
| 9 | has_preceding | 二值 | 是否有前车 | L395 |
| 10 | has_following | 二值 | 是否有后车 | L396 |
| 11 | site_id | **离散** | 站点ID 0-8（不标准化） | L397 |

**关键**:
- ✅ **连续特征** (0-6, 9-10): z-score标准化 (mean~0, std~1)
- ❌ **离散特征** (7, 8, 11): **不进行标准化**，保持原始整数值
- 离散特征用于embedding，必须保持整数形式

### 步骤4: 验证预处理结果

**使用的代码文件**:
- 📄 **验证脚本**: `validate_preprocessing.py`

**命令**:
```bash
python validate_preprocessing.py
```

**检查项**:
```python
# validate_preprocessing.py 检查:
✅ 1. 元数据一致性 (fps=30, T=80, C=65, H=15)
✅ 2. Lane token格式 ("site:lane")
✅ 3. Train/Val/Test时序无重叠
✅ 4. 离散特征未被标准化
✅ 5. 特征维度正确 (F=12)
✅ 6. Episode数量合理
```

**期望输出**:
```
✅ All preprocessing checks passed!
- Metadata: fps=30.0, T=80, C=65, H=15
- Features: 12 (9 continuous, 3 discrete)
- Lane tokens: site:lane format OK
- Splits: No temporal overlap
- Train: 44100 episodes
- Val: 6300 episodes
- Test: 6300 episodes
```

---

## 🎓 模型训练

### 步骤1: 训练前准备

**使用的代码文件**:
- 📄 **主训练脚本**: `src/training/train_world_model.py`
- 📄 **Loss函数**: `src/training/losses.py`
- 📄 **Dataset加载**: `src/data/dataset.py`
- 📄 **模型定义**: `src/models/world_model.py`
  - 📄 Encoder: `src/models/encoder.py`
  - 📄 Dynamics: `src/models/dynamics.py`
  - 📄 Decoder: `src/models/decoder.py`

**检查元数据**:
```bash
cat data/processed/metadata.json | grep n_features
# 输出: "n_features": 12
```

### 步骤2: 训练命令

```bash
python src/training/train_world_model.py \
    --train_data data/processed/train_episodes.npz \
    --val_data data/processed/val_episodes.npz \
    --checkpoint_dir checkpoints/world_model \
    --input_dim 12 \
    --latent_dim 256 \
    --batch_size 16 \
    --epochs 50 \
    --lr 3e-4 \
    --weight_decay 1e-4 \
    --grad_clip 1.0
```

**关键参数**:
- `--input_dim 12`: **必须与metadata.json中的n_features一致**
- `--latent_dim 256`: 潜在空间维度（推荐128-512）
- `--batch_size 16`: 根据GPU内存调整（默认16）
- `--epochs 50`: 训练轮数
- `--lr 3e-4`: 学习率（AdamW优化器）
- `--dynamics_layers 4`: Transformer动力学层数
- `--dynamics_heads 8`: 注意力头数
- `--max_dynamics_len 512`: 最大序列长度
- `--max_dynamics_context 128`: Rollout时的最大上下文长度

### 步骤3: 模型架构详解

**整体架构**: Encoder → Transformer Dynamics → Decoder (with Kinematic Prior)

**完整的前向传播流程**:

```
输入: states [B, T=80, K=50, F=12], masks [B, T, K]
  ↓
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【src/models/encoder.py: MultiAgentEncoder】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  1. 特征分离 (forward L133-164)
     # 将12维特征分为连续和离散特征

     连续特征提取 (L139):
     ├─ continuous_indices = [0,1,2,3,4,5,6,9,10]  # 排除7,8,11
     └─ cont = states[..., continuous_indices]  # [B, T, K, 9]

     离散特征embedding (L145-161):
     ├─ lane_id [8] → lane_embedding(nn.Embedding(num_lanes, 16))
     ├─ class_id [7] → class_embedding(nn.Embedding(num_classes, 8))  # ← 新增
     └─ site_id [11] → site_embedding(nn.Embedding(num_sites, 8))

  2. 连续特征投影 (L69-74)
     cont_emb = continuous_projector(cont)
     # Sequential(Linear(9→256), LayerNorm, ReLU, Dropout)
     → [B, T, K, hidden_dim=256]

  3. 特征融合 (L92-96, L163)
     fused_dim = 256 + 16 + 8 + 8 = 288
     agent_feats = concat([cont_emb, lane_emb, class_emb, site_emb])  # [B,T,K,288]
     agent_feats = fusion(agent_feats)  # Linear(288→256) + ReLU
     → [B, T, K, hidden_dim=256]

  4. Transformer Attention over Agents (L98-106, L169)
     # 对每个时间步独立处理，在agent维度K上做attention
     states_flat = states.reshape(B*T, K, F)  # [B*T, K, 256]

     for layer in transformer_layers:  # n_layers=2
         agent_feats = TransformerEncoderLayer(
             agent_feats,
             src_key_padding_mask=pad  # [B*T, K] mask无效agent
         )
     → [B*T, K, hidden_dim=256]

  5. Masked Mean Pooling (L172-173)
     # 聚合K个agent到单一场景表示
     weights = masks_flat.unsqueeze(-1)  # [B*T, K, 1]
     pooled = (agent_feats * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1e-6)
     → [B*T, hidden_dim=256]

  6. 投影到Latent空间 (L108-111, L175)
     latent = to_latent(pooled)  # Linear(256→256) + LayerNorm
     latent = latent.view(B, T, latent_dim)
     → [B, T, latent_dim=256]
  ↓
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【src/models/dynamics.py: LatentDynamics (Transformer-only)】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ⚠️ 重要变化: 现在只支持Transformer，移除了GRU/LSTM选项

  1. 位置编码 (forward L45-49, L114)
     pos_emb: 可学习参数 [1, max_len=512, latent_dim=256]
     (或使用sinusoidal positional encoding)

     x = latent + pos_emb[:, :T, :]  # 添加位置信息
     → [B, T, D]

  2. Causal Transformer (L51-59, L78-86, L116-122)
     # 因果mask确保时间步t只能attend到<=t的历史
     causal_mask = _causal_mask(T, device, dtype)  # [T, T]
     # Upper triangular (excluding diagonal) = -inf

     out = transformer(
         x,
         mask=causal_mask,  # 因果mask
         src_key_padding_mask=time_padding_mask  # [B,T] 可选padding mask
     )
     # TransformerEncoder:
     #   - n_layers=4
     #   - n_heads=8
     #   - dim_feedforward=1024 (4*latent_dim)
     #   - norm_first=True (Pre-LN)
     → [B, T, D]

  3. Output投影 (L62-65, L124)
     predicted_latent = output_proj(out)
     # Sequential(LayerNorm, Linear(D→D))
     → [B, T, latent_dim=256]

  4. 单步预测方法 (step L127-154)
     # Rollout时使用，支持truncated context
     next_latent = step(latent_history, max_context=128)
     # 只用最近128步历史来预测下一步（效率优化）
  ↓
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【src/models/decoder.py: StateDecoder】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ⚠️ 重要新增: (x,y)残差头，用于物理先验修正

  1. MLP Backbone (forward L34-42, L81)
     h = backbone(latent)
     # Sequential(
     #   Linear(256→256), LayerNorm, ReLU, Dropout,
     #   Linear(256→256), ReLU, Dropout
     # )
     → [B, T, hidden_dim=256]

  2. 绝对状态预测 (L45, L83)
     states = state_head(h).view(B, T, K=50, F=12)
     # Linear(256 → 50*12=600)
     → [B, T, K, F]

  3. Existence Logits (L48, L84)
     existence_logits = existence_head(h)
     # Linear(256 → 50)
     → [B, T, K]

  4. (x,y)残差头 (L51-57, L86-90) ← 新增
     IF enable_xy_residual:
         residual_xy = residual_xy_head(h).view(B, T, K, 2)
         # Linear(256 → 50*2=100)
         # ✅ 初始化为0 (从纯物理先验开始学习)
         → [B, T, K, 2]
  ↓
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【src/models/world_model.py: WorldModel】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ⚠️ 核心创新: Kinematic Prior + Residual

  完整流程 (forward L173-215):

  1. 编码 (L190)
     latent = encoder(states, masks)  # [B, T, D]

  2. 时间padding mask (L193)
     time_padding = (masks.sum(dim=-1) == 0)  # [B,T] bool
     # True表示该时间步所有agent都不存在

  3. 动力学预测 (L195)
     predicted_latent, _ = dynamics(latent, time_padding_mask=time_padding)
     → [B, T, D]

  4. 解码重建分支 (L197)
     recon_states, exist_logits, _ = decoder(latent, return_residual_xy=False)
     # 不使用residual_xy，直接输出绝对状态

  5. 解码预测分支 (L198)
     pred_states_base, pred_exist_logits, residual_xy = decoder(
         predicted_latent,
         return_residual_xy=True  # ← 获取residual
     )

  6. 🔥 物理先验 + 残差 (L200-207)
     # 计算运动学先验 (在原始空间计算，然后重新标准化)
     prior_xy = _kinematic_prior_xy(states)  # [B,T,K,2]

     # _kinematic_prior_xy内部流程 (L150-171):
     #   1. Denormalize到原始空间
     #      x = denorm(states[..., idx_x=0])
     #      y = denorm(states[..., idx_y=1])
     #      vx = denorm(states[..., idx_vx=2])
     #      vy = denorm(states[..., idx_vy=3])
     #      ax = denorm(states[..., idx_ax=4])
     #      ay = denorm(states[..., idx_ay=5])
     #
     #   2. 应用运动学方程 (使用加速度)
     #      x_next = x + vx*dt + 0.5*ax*dt^2
     #      y_next = y + vy*dt + 0.5*ay*dt^2
     #      # dt = 1/30 ≈ 0.0333 秒
     #
     #   3. Renormalize回标准化空间
     #      x_next_norm = renorm(x_next)
     #      y_next_norm = renorm(y_next)
     #      return [x_next_norm, y_next_norm]

     # 应用残差修正 (mask掉padding agents)
     residual_xy = residual_xy * masks.unsqueeze(-1)
     pred_states[..., idx_x] = prior_xy[..., 0] + residual_xy[..., 0]
     pred_states[..., idx_y] = prior_xy[..., 1] + residual_xy[..., 1]

     # 其他特征 (vx,vy,ax,ay,angle等) 直接使用decoder输出

  7. 返回 (L209-215)
     return {
         "latent": latent,                          # [B,T,D]
         "reconstructed_states": recon_states,      # [B,T,K,F]
         "predicted_states": pred_states,           # [B,T,K,F] with prior+residual
         "existence_logits": exist_logits,          # [B,T,K]
         "predicted_existence_logits": pred_exist_logits,  # [B,T,K]
     }
```

**架构亮点**:
1. ✅ **Transformer-only动力学**: 移除RNN，全面使用Transformer建模时序
2. ✅ **Causal Masking**: 确保预测时只能看到过去信息
3. ✅ **物理先验 + 学习残差**: 结合运动学方程和神经网络修正
4. ✅ **三种embeddings**: Lane, Class, Site三种离散特征embedding
5. ✅ **Normalization-aware**: 物理先验在原始空间计算，保证正确性

### 步骤4: Loss计算详解

**使用的代码文件**: `src/training/losses.py`

**Loss组成** (WorldModelLoss L67-109):
```python
total_loss = recon_weight * recon_loss +        # 重建loss
             pred_weight * pred_loss +           # 预测loss
             exist_weight * (exist_loss +        # 存在性loss (重建)
                            pred_exist_loss)     # 存在性loss (预测)
```

**关键实现**:

```python
class WorldModelLoss(nn.Module):
    def __init__(
        self,
        recon_weight: float = 1.0,
        pred_weight: float = 1.0,
        exist_weight: float = 0.1,
        huber_beta: float = 1.0,  # Huber loss平滑参数
        continuous_indices: Optional[List[int]] = None,  # ← 关键
        use_pred_existence_loss: bool = True,
    ):
        ...

    def _masked_huber_loss(self, pred, target, mask):
        """
        L39-57: 仅对continuous_indices计算Huber loss
        """
        if self.continuous_indices is not None:
            pred = pred[..., self.continuous_indices]    # ← 过滤到连续特征
            target = target[..., self.continuous_indices]

        # Huber loss (beta=1.0)
        diff = pred - target
        abs_diff = diff.abs()
        loss = torch.where(
            abs_diff < beta,
            0.5 * (diff ** 2) / beta,  # 小误差: quadratic
            abs_diff - 0.5 * beta       # 大误差: linear (robust)
        )

        # 应用mask
        loss = loss * mask.unsqueeze(-1)
        return loss.sum() / (mask.sum() * loss.shape[-1]).clamp(min=1.0)
```

**Loss计算流程** (forward L67-109):

```python
# 1. 重建loss (L86): 对齐当前帧
recon_loss = _masked_huber_loss(
    recon_states,  # decoder(latent)
    states,        # ground truth
    masks
)

# 2. 预测loss (L89): t预测t+1
#    ⚠️ 时间对齐很关键
pred_loss = _masked_huber_loss(
    pred_states[:, :-1],  # 预测: t=0到t=T-2
    states[:, 1:],        # 目标: t=1到t=T-1
    masks[:, :-1]         # mask对齐
)

# 3. 存在性loss (L91): 重建分支
exist_loss = _existence_loss(exist_logits, masks)
# BCEWithLogitsLoss: sigmoid(logits) vs ground truth masks

# 4. 预测存在性loss (L94-95): 预测分支
if use_pred_existence_loss:
    pred_exist_loss = _existence_loss(
        predicted_existence_logits[:, :-1],  # 时间对齐
        masks[:, 1:]
    )
```

**为什么只对连续特征计算loss**:
```
离散特征 (7, 8, 11):
- class_id, lane_id, site_id是类别变量
- 不应该用Huber/MSE回归
- 模型通过embedding学习这些特征
- 回归loss会误导学习方向 (把整数当连续值优化)

连续特征 (0-6, 9-10):
- center_x, center_y, vx, vy, ax, ay, angle, has_preceding, has_following
- 适合回归任务
- Huber loss robust to outliers (相比MSE)
```

**continuous_indices从哪里来**:
```python
# train_world_model.py L154
continuous_indices = train_loader.dataset.continuous_indices
# 由dataset.py从metadata.json读取

# dataset.py自动计算:
discrete_features = {lane_idx=8, class_idx=7, site_idx=11}
continuous_indices = [i for i in range(12) if i not in discrete_features]
# → [0, 1, 2, 3, 4, 5, 6, 9, 10]
```

### 步骤5: 训练流程详解

**代码文件**: `src/training/train_world_model.py`

**主要流程**:

```python
def main():
    # 1. 解析参数 (L30-56)
    args = parse_args()

    # 2. 创建TRAIN loader并计算normalization stats (L96-106)
    train_loader = get_dataloader(
        args.train_data,
        batch_size=args.batch_size,
        shuffle=True,
        normalize=True,
        stats_path=None  # 首次运行，自动计算stats
    )

    # 保存normalization stats (用于VAL/TEST)
    stats_path = ckpt_dir / "normalization_stats.npz"
    if not stats_path.exists():
        train_loader.dataset.save_stats(str(stats_path))

    # 3. 创建VAL loader (复用TRAIN的stats) (L108-116)
    val_loader = get_dataloader(
        args.val_data,
        batch_size=args.batch_size,
        shuffle=False,
        normalize=True,
        stats_path=str(stats_path)  # ← 使用TRAIN的stats
    )

    # 4. 从metadata读取配置 (L118-123)
    meta = train_loader.dataset.metadata
    dt = float(meta.get("dt", 1.0/30.0))  # 0.0333秒
    num_lanes = int(meta.get("num_lanes", 100))
    num_sites = int(meta.get("num_sites", 10))
    num_classes = int(meta.get("num_classes", 10))

    # 5. 创建WorldModel (L127-140)
    model = WorldModel(
        input_dim=args.input_dim,
        max_agents=args.max_agents,
        latent_dim=args.latent_dim,
        dynamics_layers=args.dynamics_layers,      # Transformer层数
        dynamics_heads=args.dynamics_heads,        # 注意力头数
        dt=dt,
        max_dynamics_len=args.max_dynamics_len,    # 512
        max_dynamics_context=args.max_dynamics_context,  # 128
        num_lanes=num_lanes,
        num_sites=num_sites,
        num_classes=num_classes,
        use_acceleration=bool(meta.get("use_acceleration", True)),
    ).to(device)

    # 6. ✅ 设置normalization stats到model (L142-147)
    #    ⚠️ 关键: kinematic prior需要这些stats来denorm/renorm
    model.set_normalization_stats(
        train_loader.dataset.mean,  # [n_continuous]
        train_loader.dataset.std,   # [n_continuous]
        train_loader.dataset.continuous_indices,  # [0,1,2,3,4,5,6,9,10]
    )

    # 7. 创建Loss函数 (L149-156)
    loss_fn = WorldModelLoss(
        recon_weight=1.0,
        pred_weight=1.0,
        exist_weight=0.1,
        huber_beta=1.0,
        continuous_indices=train_loader.dataset.continuous_indices,
        use_pred_existence_loss=True,
    )

    # 8. 创建Optimizer (L158)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # 9. 训练循环 (L162-199)
    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            states = batch["states"].to(device)  # [B,T,K,F]
            masks = batch["masks"].to(device)    # [B,T,K]

            optimizer.zero_grad(set_to_none=True)

            # Forward
            preds = model(states, masks)

            # Compute loss
            losses = loss_fn(preds, {"states": states, "masks": masks})
            loss = losses["total_loss"]

            # Backward
            loss.backward()
            if args.grad_clip > 0:
                clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        # Validation
        val_metrics = evaluate(model, val_loader, loss_fn, device)

        # 打印 (L190-192)
        print(f"[Epoch {epoch+1}] "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"recon={val_metrics['recon_loss']:.4f} "
              f"pred={val_metrics['pred_loss']:.4f} "
              f"exist={val_metrics['exist_loss']:.4f}")

        # 保存checkpoint (L194, L196-198)
        save_checkpoint(ckpt_dir / "checkpoint_last.pt", ...)
        if val_loss < best_val:
            save_checkpoint(ckpt_dir / "checkpoint_best.pt", ...)
```

**Checkpoint保存内容** (save_checkpoint L59-68):
```python
{
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
}
```

**Normalization stats保存** (dataset.py):
```python
# checkpoints/world_model/normalization_stats.npz:
{
    "mean": [n_continuous],  # 仅连续特征的mean
    "std": [n_continuous],   # 仅连续特征的std
    "continuous_indices": [0,1,2,3,4,5,6,9,10],
}
```

### 步骤6: 监控训练

**实时查看训练输出**:
```bash
# 脚本直接打印到stdout
python src/training/train_world_model.py ... | tee train.log
```

**期望输出**:
```
[Epoch 1] train_loss=12.3456  val_loss=13.4567  recon=10.234 pred=2.345 exist=0.123 pred_exist=0.098
[Epoch 2] train_loss=10.1234  val_loss=11.2345  recon=8.456 pred=1.987 exist=0.112 pred_exist=0.089
...
[Epoch 25] train_loss=3.4567  val_loss=4.1234  recon=2.345 pred=0.987 exist=0.098 pred_exist=0.087
```

**健康指标**:
- ✅ train_loss和val_loss逐epoch下降
- ✅ recon_loss通常略大于pred_loss (重建更难)
- ✅ exist_loss和pred_exist_loss收敛到0.05-0.15
- ✅ val_loss略高于train_loss,但gap不应过大 (避免过拟合)
- ✅ 无NaN或Inf (若出现,降低lr或检查数据)

**Checkpoint文件**:
```
checkpoints/world_model/
├── checkpoint_last.pt          # 最新epoch
├── checkpoint_best.pt          # 最佳val_loss
└── normalization_stats.npz     # 标准化统计量
```

---

## 📈 模型评估

### 步骤1: Rollout评估

**使用的代码文件**:
- 📄 **评估脚本**: `src/evaluation/rollout_eval.py`
- 📄 **指标计算**: `src/evaluation/prediction_metrics.py`
- 📄 **工具函数**: `src/utils/common.py`

**命令**:
```bash
python src/evaluation/rollout_eval.py \
    --checkpoint checkpoints/best_model.pt \
    --test_data data/processed/test_episodes.npz \
    --context_length 65 \
    --rollout_horizon 15 \
    --output_dir results/
```

**评估流程**:

```
src/evaluation/rollout_eval.py
  ↓
1. 加载模型 (L320-395)
   ├─ 读取checkpoint
   ├─ 推断模型配置 (latent_dim, dynamics_type, hidden_dim)
   │   └─ 通过权重矩阵形状推断 (修复后的逻辑)
   └─ 创建WorldModel并加载权重
  ↓
2. 加载测试数据 (L369-375)
   └─ 使用src/data/dataset.py的TrajectoryDataset
  ↓
3. Rollout评估 (evaluate_rollout L23-97)
   for batch in test_loader:
       # 分割context和target
       context_states = states[:, :C=65]
       target_states = states[:, C:C+H=15]

       # Rollout预测 (调用world_model.py:rollout)
       rollout_output = model.rollout(
           initial_states=context_states,
           initial_masks=context_masks,
           n_steps=H=15,
           teacher_forcing=False  # Open-loop
       )

       # 计算指标 (调用prediction_metrics.py)
       metrics = compute_all_metrics(
           predicted=rollout_output['predicted_states'],
           ground_truth=target_states,
           masks=target_masks,
           convert_to_meters=True  # 转换为米
       )
  ↓
4. 保存结果 (L412-431)
   └─ results/rollout_metrics.json
```

**World Model Rollout详解**:

**新rollout实现** (src/models/world_model.py L217-296):

```python
@torch.no_grad()
def rollout(
    initial_states,      # [B, T0=65, K, F]
    initial_masks,       # [B, T0, K]
    n_steps=15,          # 预测步数
    threshold=0.5,       # 存在性阈值
    teacher_forcing=False,
    ground_truth_states=None,
):
    """
    🔥 新版rollout: 使用Transformer dynamics.step() + Kinematic Prior

    关键改进:
    1. 使用dynamics.step()进行单步预测 (支持truncated context)
    2. 应用kinematic prior + residual修正(x,y)
    3. 累积latent历史 (用于Transformer的attention)
    """
    B, T0, K, F = initial_states.shape

    # 1. 编码context (L243)
    latent_ctx = encoder(initial_states, initial_masks)  # [B,T0,D]
    time_padding = (initial_masks.sum(dim=-1) == 0)  # [B,T0]

    # 2. 通过dynamics获取context的预测latent (L246-247)
    pred_latent_ctx, _ = dynamics(latent_ctx, time_padding_mask=time_padding)
    current_latent = pred_latent_ctx[:, -1:, :]  # [B,1,D] 最后一步的预测

    # 3. 初始化历史和状态 (L249-250)
    latent_hist = latent_ctx          # 历史latent序列 [B,T0,D]
    prev_state = initial_states[:, -1:, :, :]  # [B,1,K,F] 最后一帧

    out_states = []
    out_masks = []

    # 4. Autoregressive rollout循环 (L255-292)
    for step in range(n_steps):  # 15步
        # a. 解码当前latent (L257-259)
        base_states, exist_logits, residual_xy = decoder(
            current_latent,
            return_residual_xy=True  # 获取(x,y)残差
        )
        pred_state = base_states.clone()  # [B,1,K,F]

        # b. 🔥 应用kinematic prior (L262-266)
        prior_xy = _kinematic_prior_xy(prev_state)  # [B,1,K,2]
        # prior_xy基于prev_state的(vx,vy,ax,ay)计算物理预测

        if residual_xy is not None:
            pred_state[..., idx_x] = prior_xy[..., 0] + residual_xy[..., 0]
            pred_state[..., idx_y] = prior_xy[..., 1] + residual_xy[..., 1]

        # c. 存在性mask (L269-270)
        exist_prob = torch.sigmoid(exist_logits)  # logits → prob
        pred_mask = (exist_prob > threshold).float()  # [B,1,K]

        out_states.append(pred_state)
        out_masks.append(pred_mask)

        # d. 决定下一步的"prev_state" (L275-284)
        if teacher_forcing and ground_truth_states is not None:
            # 使用ground truth (用于训练阶段的scheduled sampling)
            gt_state = ground_truth_states[:, T0+step:T0+step+1, :, :]
            prev_state = gt_state
            gt_mask = (gt_state.abs().sum(dim=-1) > 0).float()
            current_latent = encoder(gt_state, gt_mask)
        else:
            # 使用预测结果 (open-loop)
            prev_state = pred_state * pred_mask.unsqueeze(-1)

        # e. 累积latent历史，预测下一步 (L287-292)
        latent_hist = torch.cat([latent_hist, current_latent], dim=1)
        # latent_hist: [B, T0+step+1, D]

        next_latent = dynamics.step(
            latent_hist,
            max_context=self.max_dynamics_context  # 128 (truncate长历史)
        ).view(B, 1, -1)  # [B,1,D]

        current_latent = next_latent

    # 5. 拼接输出 (L294-296)
    predicted_states = torch.cat(out_states, dim=1)  # [B,n_steps=15,K,F]
    predicted_masks = torch.cat(out_masks, dim=1)    # [B,n_steps,K]

    return predicted_states, predicted_masks
```

**关键架构特点**:

1. **Transformer dynamics.step() (L288-291)**:
   ```python
   # dynamics.py: step method (L127-154)
   def step(latent_history, time_padding_mask=None, max_context=None):
       """
       单步预测，支持truncated context
       """
       if max_context and latent_history.size(1) > max_context:
           # 只保留最近max_context步 (效率优化)
           latent_history = latent_history[:, -max_context:, :]

       pred, _ = forward(latent_history, time_padding_mask)
       return pred[:, -1, :]  # 返回最后一个token的预测
   ```

2. **Kinematic Prior应用 (L262-266)**:
   - 物理先验在**原始空间**计算 (denormalize → physics → renormalize)
   - 残差从decoder输出,初始化为0
   - 只修正(x,y),其他特征直接使用decoder输出

3. **累积Latent历史 (L287)**:
   - Transformer需要完整历史来做attention
   - 使用truncated context (128步) 避免超长序列

4. **Open-loop vs Teacher Forcing (L275-284)**:
   - Open-loop: 使用自己的预测 `prev_state = pred_state`
   - Teacher forcing: 使用ground truth (训练时可用)

**与旧版本的区别**:
| 特性 | 旧版 (GRU/LSTM) | 新版 (Transformer) |
|------|----------------|-------------------|
| Dynamics | RNN hidden state | Latent历史序列 |
| 单步预测 | `dynamics(current_latent, hidden)` | `dynamics.step(latent_hist, max_context=128)` |
| 物理先验 | 无 | Kinematic prior + residual |
| Context | Hidden state | Truncated latent history |
| (x,y)预测 | 直接输出 | Prior + Residual |

### 步骤2: 指标计算

**代码文件**: `src/evaluation/prediction_metrics.py`

**指标详解**:

```python
# prediction_metrics.py: compute_all_metrics (L257-319)

def compute_all_metrics(predicted, ground_truth, masks, convert_to_meters=True):
    """
    计算所有评估指标

    指标列表:
    1. ADE (Average Displacement Error)  - L18-50
    2. FDE (Final Displacement Error)    - L53-86
    3. Velocity Error                     - L89-121
    4. Heading Error                      - L124-160
    5. Collision Rate                     - L163-217
    """

    # 坐标转换 (L282-301)
    if convert_to_meters:
        # 使用src/utils/common.py:convert_pixels_to_meters
        pixel_to_meter = get_pixel_to_meter_conversion()  # ≈ 0.077

        predicted = convert_pixels_to_meters(
            predicted,
            pixel_to_meter,
            position_indices=(0, 1),
            velocity_indices=(2, 3),
            acceleration_indices=(4, 5)
        )
        ground_truth = convert_pixels_to_meters(ground_truth, ...)

    # 计算各项指标 (L303-318)
    metrics = {
        'ade': compute_ade(predicted, ground_truth, masks),
        'fde': compute_fde(predicted, ground_truth, masks),
        'velocity_error': compute_velocity_error(...),
        'heading_error': compute_heading_error(...),
        'collision_rate': compute_collision_rate(...)
    }

    return metrics
```

**ADE/FDE计算**:
```python
# ADE: 平均位移误差 (L18-50)
def compute_ade(predicted, ground_truth, masks):
    # 提取位置 (x, y)
    pred_pos = predicted[..., :2]  # [B, T, K, 2]
    gt_pos = ground_truth[..., :2]

    # L2距离
    displacement = torch.norm(pred_pos - gt_pos, dim=-1)  # [B, T, K]

    # 应用mask并求平均
    masked_displacement = displacement * masks
    ade = masked_displacement.sum() / masks.sum().clamp(min=1)

    return ade.item()  # 单位: 米

# FDE: 最终位移误差 (L53-86)
def compute_fde(predicted, ground_truth, masks):
    # 仅最后一帧
    pred_final = predicted[:, -1, :, :2]  # [B, K, 2]
    gt_final = ground_truth[:, -1, :, :2]
    mask_final = masks[:, -1, :]

    # L2距离
    displacement = torch.norm(pred_final - gt_final, dim=-1)

    # 平均
    fde = (displacement * mask_final).sum() / mask_final.sum().clamp(min=1)

    return fde.item()  # 单位: 米
```

**期望结果** (良好模型):
```json
{
  "ade": 0.10,          // 10厘米平均误差
  "fde": 0.12,          // 12厘米最终误差
  "velocity_error": 0.08,   // 8cm/s速度误差
  "heading_error": 1.5,     // 1.5度朝向误差
  "collision_rate": 5.2     // 5.2% (取决于safety_margin)
}
```

---

## 🎨 可视化

### 步骤1: 轨迹可视化

**使用的代码文件**:
- 📄 **可视化脚本**: `src/evaluation/visualize_predictions.py`
- 📄 **航拍图**: `src/evaluation/sites/SiteA.jpg` ~ `SiteI.jpg`

**命令**:
```bash
python src/evaluation/visualize_predictions.py \
    --checkpoint checkpoints/best_model.pt \
    --test_data data/processed/test_episodes.npz \
    --site_images_dir src/evaluation/sites \
    --context_length 65 \
    --rollout_horizon 15 \
    --output_dir results/visualizations \
    --num_samples 5 \
    --max_agents 10
```

**可视化流程**:

```
src/evaluation/visualize_predictions.py
  ↓
1. 加载站点航拍图 (L150-157)
   for site in A-I:
       load SiteX.jpg
  ↓
2. 收集测试样本 (L161-222)
   for batch in test_loader:
       # 分割context和target
       context = states[:, :C=65]      # 蓝色轨迹
       target = states[:, C:C+H=15]    # 绿色轨迹(真实)

       # 归一化给模型
       context_norm = normalize_states(context, mean, std, continuous_indices)

       # Rollout预测
       predictions_norm = model.rollout(context_norm, n_steps=15)

       # 反归一化回像素坐标
       predictions = denormalize_states(
           predictions_norm, mean, std, continuous_indices
       )  # 红色轨迹(预测)

       # 收集样本 (每个site 5个样本)
       samples_by_site[site_id].append({
           'context': context,
           'ground_truth': target,
           'predicted': predictions
       })
  ↓
3. 绘制轨迹 (L224-306)
   for site in A-I:
       for sample in samples:
           # 在航拍图上绘制
           img = load_site_image(site)

           # 绘制每个agent的轨迹 (最多10个)
           for agent_idx in valid_agents[:10]:
               # 蓝色: context轨迹
               draw_trajectory(img, context[:, agent_idx, :2], color=(0,0,255))

               # 绿色: ground truth轨迹
               draw_trajectory(img, gt[:, agent_idx, :2], color=(0,255,0))

               # 红色: 预测轨迹
               draw_trajectory(img, pred[:, agent_idx, :2], color=(255,0,0))

           # 添加图例
           add_legend(img)

           # 保存
           save(f'site_{site}_sample_{idx}.jpg')
  ↓
输出: results/visualizations/
    ├── site_A_sample_1.jpg
    ├── site_A_sample_2.jpg
    ...
    └── site_I_sample_5.jpg
```

**绘制函数详解**:
```python
# visualize_predictions.py: draw_trajectory_on_image (L58-106)

def draw_trajectory_on_image(img, trajectory, color, thickness=2):
    """
    在图像上绘制单条轨迹

    参数:
        img: 航拍图 [H, W, 3]
        trajectory: [T, 2] 轨迹坐标 (像素)
        color: (R, G, B) 颜色
    """
    import cv2

    # 过滤无效点
    valid_mask = (trajectory[:, 0] > 0) & (trajectory[:, 1] > 0)
    trajectory = trajectory[valid_mask]

    # 绘制连线
    for i in range(len(trajectory) - 1):
        pt1 = (int(trajectory[i, 0]), int(trajectory[i, 1]))
        pt2 = (int(trajectory[i+1, 0]), int(trajectory[i+1, 1]))
        cv2.line(img, pt1, pt2, color, thickness)

    # 绘制点
    for pt in trajectory:
        cv2.circle(img, (int(pt[0]), int(pt[1])), 3, color, -1)

    # 起点: 大圆圈
    cv2.circle(img, (int(trajectory[0, 0]), int(trajectory[0, 1])), 6, color, 2)

    # 终点: 方块
    end_pt = (int(trajectory[-1, 0]), int(trajectory[-1, 1]))
    cv2.rectangle(img, (end_pt[0]-4, end_pt[1]-4), (end_pt[0]+4, end_pt[1]+4), color, -1)

    return img
```

**可视化结果示例**:
```
┌─────────────────────────────────────┐
│ 航拍图: Site A                      │
│                                     │
│  蓝色线条 ━━━━━━━━━━┐               │
│                     ↓ Context (65帧)│
│                     ●               │
│  绿色线条 ━━━━━━━━━┐ Ground Truth   │
│                     ↓ (15帧)        │
│                     ■               │
│  红色线条 ━━━━━━━━━┐ Prediction     │
│                     ↓ (15帧)        │
│                     ■               │
│                                     │
│ 图例: Blue=Context, Green=GT,       │
│       Red=Prediction                │
└─────────────────────────────────────┘
```

### 步骤2: Attention权重可视化

**代码文件**: `src/evaluation/attention_visualization.py`

```bash
python src/evaluation/attention_visualization.py \
    --checkpoint checkpoints/best_model.pt \
    --test_data data/processed/test_episodes.npz \
    --output_dir results/attention_maps
```

**可视化内容**:
- Encoder的Transformer attention权重
- 跨agent的attention pattern
- 分析哪些agent之间交互最强

---

## 📁 项目结构

```
traffic_wm/
├── 📄 README.md                        # 本文件 - 用户指南
├── 📄 CLAUDE.md                        # 开发者指南 (详细技术说明)
├── 📄 WORLD_MODEL_COMPARISON.md        # 与DreamerV3的架构对比
├── 📄 requirements.txt                 # Python依赖
│
├── 📄 preprocess_multisite.py          # ⭐ 预处理主脚本
├── 📄 validate_preprocessing.py        # ⭐ 验证脚本
│
├── 📂 data/
│   ├── raw/                            # 原始CSV数据 (用户提供)
│   │   ├── A/drone_1.csv, drone_2.csv, ...
│   │   ├── B/drone_1.csv, ...
│   │   └── I/...
│   └── processed/                      # 预处理输出
│       ├── train_episodes.npz          # [N, 80, 50, 12]
│       ├── val_episodes.npz
│       ├── test_episodes.npz
│       ├── metadata.json               # 元数据配置
│       └── split_config.json           # 数据划分记录
│
├── 📂 src/
│   ├── 📂 data/
│   │   ├── 📄 __init__.py
│   │   ├── 📄 preprocess.py            # 预处理核心逻辑
│   │   │   ├─ build_global_timeline()
│   │   │   ├─ detect_gaps_and_split_segments()
│   │   │   ├─ extract_fixed_stride_episodes()
│   │   │   └─ extract_extended_features()
│   │   ├── 📄 split_strategy.py        # 数据划分策略
│   │   │   └─ chronological_split_episodes()
│   │   └── 📄 dataset.py               # PyTorch Dataset/DataLoader
│   │       └─ TrajectoryDataset
│   │
│   ├── 📂 models/
│   │   ├── 📄 __init__.py
│   │   ├── 📄 encoder.py               # ⭐ MultiAgentEncoder
│   │   │   ├─ Feature Embedding
│   │   │   ├─ Site/Lane Embeddings
│   │   │   ├─ Transformer Attention
│   │   │   └─ Masked Mean Pooling
│   │   ├── 📄 decoder.py               # ⭐ StateDecoder
│   │   │   ├─ MLP Layers
│   │   │   ├─ State Head ([K, F])
│   │   │   └─ Existence Head ([K])
│   │   ├── 📄 dynamics.py              # ⭐ LatentDynamics
│   │   │   ├─ GRUDynamics
│   │   │   ├─ LSTMDynamics
│   │   │   └─ TransformerDynamics
│   │   └── 📄 world_model.py           # ⭐ WorldModel
│   │       ├─ forward()
│   │       └─ rollout()  ← 修复后的实现
│   │
│   ├── 📂 training/
│   │   ├── 📄 __init__.py
│   │   ├── 📄 train_world_model.py     # ⭐ 训练主脚本
│   │   │   ├─ Trainer class
│   │   │   ├─ train_epoch()
│   │   │   ├─ validate()
│   │   │   └─ save_checkpoint()
│   │   └── 📄 losses.py                # ⭐ Loss函数
│   │       ├─ WorldModelLoss
│   │       │   ├─ Reconstruction Loss (仅连续特征)
│   │       │   ├─ Prediction Loss (仅连续特征)
│   │       │   └─ Existence Loss
│   │       ├─ RolloutLoss
│   │       └─ ContrastiveLoss
│   │
│   ├── 📂 evaluation/
│   │   ├── 📄 __init__.py
│   │   ├── 📄 rollout_eval.py          # ⭐ Rollout评估
│   │   │   ├─ evaluate_rollout()
│   │   │   ├─ evaluate_multihorizon()
│   │   │   └─ evaluate_with_teacher_forcing()
│   │   ├── 📄 prediction_metrics.py    # ⭐ 评估指标
│   │   │   ├─ compute_ade()
│   │   │   ├─ compute_fde()
│   │   │   ├─ compute_velocity_error()
│   │   │   ├─ compute_heading_error()
│   │   │   ├─ compute_collision_rate()
│   │   │   └─ compute_all_metrics()
│   │   ├── 📄 visualize_predictions.py # ⭐ 轨迹可视化
│   │   │   ├─ visualize_batch_predictions()
│   │   │   ├─ draw_trajectory_on_image()
│   │   │   └─ denormalize_states()
│   │   ├── 📄 visualization.py         # 其他可视化
│   │   ├── 📄 attention_visualization.py # Attention权重可视化
│   │   └── 📂 sites/                   # 站点航拍图
│   │       ├── SiteA.jpg
│   │       ├── SiteB.jpg
│   │       └── ... SiteI.jpg
│   │
│   └── 📂 utils/
│       ├── 📄 __init__.py
│       ├── 📄 logger.py                # 日志工具
│       ├── 📄 common.py                # 通用工具函数
│       │   ├─ set_seed()
│       │   ├─ count_parameters()
│       │   ├─ get_pixel_to_meter_conversion()
│       │   └─ convert_pixels_to_meters()
│       └── 📄 config.py                # 配置管理
│
├── 📂 logs/                            # 训练日志
│   └── trainer.log
│
├── 📂 checkpoints/                     # 模型checkpoints
│   ├── best_model.pt                   # 最佳模型
│   └── checkpoint_epoch_N.pt           # 定期保存
│
├── 📂 results/                         # 评估结果
│   ├── rollout_metrics.json            # 评估指标
│   └── visualizations/                 # 可视化结果
│       ├── site_A_sample_1.jpg
│       └── ...
│
└── 📂 dreamerv3/                       # DreamerV3参考实现
    ├── nets.py                         # RSSM, MultiEncoder, MultiDecoder
    └── agent.py                        # WorldModel
```

---

## 🔍 代码文件详解

### 核心数据流

```
数据流向:

  CSV文件 (data/raw/)
    ↓ preprocess_multisite.py
    ↓ src/data/preprocess.py
  NPZ文件 (data/processed/)
    ↓ src/data/dataset.py: TrajectoryDataset
    ↓ torch.utils.data.DataLoader
  Batch [B, T, K, F]
    ↓ src/models/encoder.py: MultiAgentEncoder
  Latent [B, T, D]
    ↓ src/models/dynamics.py: LatentDynamics
  Predicted Latent [B, T, D]
    ↓ src/models/decoder.py: StateDecoder
  States [B, T, K, F] + Existence [B, T, K]
    ↓ src/training/losses.py: WorldModelLoss
  Loss (仅连续特征!)
    ↓ optimizer.step()
  Updated Model
```

### 关键函数调用链

**训练时**:
```python
src/training/train_world_model.py:main()
  └─ Trainer.__init__()
      ├─ src/data/dataset.py:TrajectoryDataset(normalize=True)
      ├─ src/models/world_model.py:WorldModel()
      │   ├─ src/models/encoder.py:MultiAgentEncoder()
      │   ├─ src/models/dynamics.py:LatentDynamics()
      │   └─ src/models/decoder.py:StateDecoder()
      └─ src/training/losses.py:WorldModelLoss()

  └─ Trainer.train()
      └─ for epoch in range(n_epochs):
          ├─ Trainer.train_epoch()
          │   └─ for batch in train_loader:
          │       ├─ model.forward(states, masks)
          │       ├─ loss_fn(predictions, targets)
          │       └─ optimizer.step()
          │
          ├─ Trainer.validate()
          └─ Trainer.save_checkpoint()
```

**评估时**:
```python
src/evaluation/rollout_eval.py:main()
  ├─ 加载checkpoint
  ├─ 创建WorldModel
  ├─ 创建TrajectoryDataset(test)
  │
  └─ evaluate_rollout()
      └─ for batch in test_loader:
          ├─ model.rollout(context, n_steps=15)
          │   ├─ encoder(context) → latent
          │   ├─ dynamics(latent) → predicted_latent_context
          │   └─ for step in range(15):
          │       ├─ dynamics(current_latent) → next_latent
          │       ├─ decoder(next_latent) → next_states
          │       └─ current_latent = next_latent
          │
          └─ src/evaluation/prediction_metrics.py:compute_all_metrics()
              ├─ compute_ade()
              ├─ compute_fde()
              ├─ compute_velocity_error()
              └─ ...
```

**可视化时**:
```python
src/evaluation/visualize_predictions.py:main()
  ├─ 加载checkpoint
  ├─ 加载站点图片
  ├─ 创建TrajectoryDataset(test, normalize=False)
  │
  └─ visualize_batch_predictions()
      └─ for batch in test_loader:
          ├─ 分割context/target
          ├─ normalize_states(context) → context_norm
          ├─ model.rollout(context_norm) → predictions_norm
          ├─ denormalize_states(predictions_norm) → predictions
          │
          └─ for agent in agents:
              ├─ draw_trajectory(context, color=blue)
              ├─ draw_trajectory(target, color=green)
              └─ draw_trajectory(predictions, color=red)
```

---

## ⚠️ 重要说明

### 1. 离散特征处理（关键！）

**为什么重要**: 这是最常见的错误来源！

**数据加载时** (`src/data/dataset.py:_normalize_data`):
```python
# ✅ 正确做法
continuous_feats = states[..., continuous_indices]  # [0,1,2,3,4,5,6,9,10]
continuous_feats = (continuous_feats - mean) / std

states[..., continuous_indices] = continuous_feats
# 离散特征 [7, 8, 11] 保持不变!
```

**模型中** (`src/models/encoder.py`):
```python
# ✅ 离散特征通过Embedding学习
site_id = states[..., 11].long()       # 提取site_id
lane_id = states[..., 8].long()        # 提取lane_id

site_embed = self.site_embedding(site_id)
lane_embed = self.lane_embedding(lane_id)
# 不参与连续特征的标准化!
```

**Loss计算时** (`src/training/losses.py`):
```python
# ✅ 仅对连续特征计算回归loss
continuous_indices = [0, 1, 2, 3, 4, 5, 6, 9, 10]

recon_loss = huber_loss(
    pred[..., continuous_indices],
    target[..., continuous_indices],
    mask
)
# 离散特征不参与loss计算!
```

**错误示例** ❌:
```python
# ❌ 错误: 对所有特征标准化
states = (states - mean) / std  # lane_id=150变成了150.3!

# ❌ 错误: 对离散特征计算回归loss
loss = mse(pred[:, :, :, :], target[:, :, :, :])  # 包括lane_id!

# ❌ 错误: 对lane_id做回归预测
predicted_lane = 150.73  # 应该是整数!
```

### 2. 输入维度匹配

**检查方法**:
```bash
# 1. 查看metadata中的特征数
cat data/processed/metadata.json | grep n_features
# 输出: "n_features": 12

# 2. 训练时必须匹配
python src/training/train_world_model.py --input_dim 12 ...
```

**常见错误**:
```bash
# ❌ 错误: input_dim不匹配
python src/training/train_world_model.py --input_dim 11 ...
# RuntimeError: Expected 11 features, got 12

# ✅ 正确
python src/training/train_world_model.py --input_dim 12 ...
```

### 3. Lane Token格式

**Lane mapping格式**: `"site:lane"`

**验证**:
```bash
cat data/processed/metadata.json | grep -A 10 lane_mapping
```

**期望输出**:
```json
"lane_mapping": {
    "A:A1": 1,
    "A:A2": 2,
    "A:B1": 3,
    "B:crossroads1": 4,
    ...
}
```

### 4. Rollout修复 (2025-12-14)

**修复前** ❌:
```python
# src/models/world_model.py:189 (旧版本)
current_latent = latent[:, -1:]  # 使用encoder的输出
```

**修复后** ✅:
```python
# src/models/world_model.py:191 (新版本)
predicted_latent_context, hidden = self.dynamics(latent)
current_latent = predicted_latent_context[:, -1:]  # 使用dynamics预测的输出
```

**为什么重要**:
- 训练时: dynamics预测latent[:, t] → latent[:, t+1]
- 推理时: 应该用dynamics预测的latent作为起点,保持一致性
- 修复后: context→prediction过渡更平滑,减少不连续性

---

## 🐛 故障排查

### 问题1: Loss不下降

**症状**: Train loss在高值plateau,不下降

**可能原因**:
1. Learning rate过高或过低
2. 离散特征被错误标准化
3. Input_dim不匹配
4. Batch size太小

**解决方案**:
```bash
# 1. 检查元数据
cat data/processed/metadata.json | grep -E "n_features|discrete_features"

# 2. 降低学习率
python src/training/train_world_model.py --learning_rate 1e-4

# 3. 增加batch size
python src/training/train_world_model.py --batch_size 64

# 4. 检查日志中的loss分量
tail -f logs/trainer.log
# 如果Recon loss很大 → encoder/decoder问题
# 如果Pred loss很大 → dynamics问题
# 如果Exist loss很大 → decoder existence head问题
```

### 问题2: Loss出现NaN

**症状**: 训练几个epoch后loss变成NaN

**可能原因**:
1. Learning rate过高
2. Gradient explosion
3. 数值不稳定

**解决方案**:
```bash
# 1. 降低学习率
python src/training/train_world_model.py --learning_rate 5e-4

# 2. 添加gradient clipping (需修改代码)
# 在train_world_model.py中添加:
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 3. 检查数据
python validate_preprocessing.py  # 确保数据正常
```

### 问题3: 预测不连续

**症状**: 可视化结果中,蓝色(context)和红色(prediction)之间有跳跃

**原因**: Rollout起点问题

**解决方案**: ✅ 已在2025-12-14修复!
```python
# src/models/world_model.py:191
# 确保使用dynamics预测的latent作为起点
current_latent = predicted_latent_context[:, -1:]
```

如果仍有问题:
1. 重新训练模型(使用修复后的代码)
2. 增加context_length
3. 调整loss权重(增大pred_weight)

### 问题4: Collision Rate异常

**症状**: Collision rate = 100%

**原因**: safety_margin设置过大

**解决方案**:
```python
# 修改src/evaluation/prediction_metrics.py:166
compute_collision_rate(predicted, masks, safety_margin=4.0)
# 调整为合理的车辆宽度(3-5米)
```

### 问题5: 模型加载失败

**症状**: RuntimeError: size mismatch for dynamics.rnn.weight_ih_l0

**原因**: 模型配置推断错误

**解决方案**: ✅ 已在2025-12-14修复!

现在`src/evaluation/rollout_eval.py`会自动:
1. 从checkpoint推断latent_dim
2. 通过权重矩阵形状推断dynamics_type (GRU vs LSTM)
3. 推断hidden_dim

如果仍有问题:
```bash
# 手动指定配置 (需修改代码添加参数)
python src/evaluation/rollout_eval.py \
    --checkpoint xxx.pt \
    --latent_dim 512 \
    --dynamics_type lstm \
    --dynamics_hidden 512
```

---

## 🤝 贡献

欢迎提交Issues和Pull Requests!

**贡献指南**:
1. Fork本仓库
2. 创建feature分支
3. 提交更改
4. 发起Pull Request

---

## 📄 许可

MIT License

---

## 📮 联系与文档

**详细文档**:
- 📘 `CLAUDE.md` - 详细开发指南 (中文)
- 📘 `WORLD_MODEL_COMPARISON.md` - 与DreamerV3架构对比
- 📘 `DEBUG_REPORT.md` - Debug报告
- 📘 `COMPLETE_DEBUG_SUMMARY_CN.md` - 完整debug总结

**快速查找**:
- 如何修改特征? → `src/data/preprocess.py:extract_extended_features`
- 如何修改模型架构? → `src/models/encoder.py`, `decoder.py`, `dynamics.py`
- 如何修改loss? → `src/training/losses.py`
- 如何添加新指标? → `src/evaluation/prediction_metrics.py`

---

**项目版本**: 1.0 (Production Ready ✅)
**最后更新**: 2025-12-14
**状态**: 所有已知bug已修复
