# Traffic World Model - 多智能体轨迹预测

基于Transformer的潜在世界模型，用于多站点无人机轨迹数据的预测和仿真。

**核心特性**:
- 🚁 多站点无人机数据支持 (Sites A-I)
- 🧠 Transformer编码器 + Transformer时序动力学 + 物理先验
- 🎯 支持 12 维基础特征 或 20 维增强特征（含相对位置/速度）
- ⏱️ 时序无重叠的train/val/test划分
- 🔧 完整的预处理、训练、评估流程
- 🔥 v2.4: 添加相对位置特征、Learning Rate Scheduler、Angle优化
- 🎓 v2.3: Decoder只输出连续特征，离散特征作为episode-level常量

---

## 🆕 最新更新 (v2.5)

### 简化特征配置 + 速度方向监督

1. **简化特征集** ⭐
   - 特征总数：24 (20原始 + 4派生)
   - Continuous特征：15个 (去除后车信息)
   - 保留：基本运动(6) + 前车交互(5) + 派生特征(4)

2. **派生交互特征** 🔥
   - velocity_direction: atan2(vy, vx) - 速度方向角
   - headway: 纵向车距
   - ttc: Time-To-Collision 碰撞时间
   - preceding_distance: 前车总距离

3. **速度方向损失** 🎯
   - velocity_direction_loss (weight=0.3)
   - 约束速度和方向一致性
   - 预期改进：velocity_direction_error 60° → 20-30°

4. **评估指标扩展**
   - moving_ade: 只计算运动车辆的ADE
   - velocity_direction_error: 速度方向误差
   - acceleration_error: 加速度预测误差
   - position_variance: 位置方差（轨迹平滑度）

---

## 📋 目录

1. [快速开始](#-快速开始)
2. [特征说明](#-特征说明)
3. [数据预处理](#-数据预处理)
4. [模型训练](#-模型训练)
5. [模型评估](#-模型评估)
6. [可视化](#-可视化)
7. [项目结构](#-项目结构)
8. [代码文件详解](#-代码文件详解)
9. [重要说明](#-重要说明)
10. [故障排查](#-故障排查)

---

## 🎯 特征说明

### 简化特征配置 (v2.5) ⭐ 当前使用

**特征总数**: 24 (20原始 + 4派生)  
**Continuous特征**: 15个

#### 基本运动 (6个)
```
[0-5]: center_x, center_y, vx, vy, ax, ay
```

#### 前车交互 - 原始特征 (5个)
```
[9]:  has_preceding      - 是否有前车 (0/1)
[12]: rel_x_preceding    - 前车相对x位置
[13]: rel_y_preceding    - 前车相对y位置
[14]: rel_vx_preceding   - 前车相对x速度
[15]: rel_vy_preceding   - 前车相对y速度
```

#### 派生交互特征 (4个) - 动态计算
```
[20]: velocity_direction  - 速度方向角 = atan2(vy, vx)
[21]: headway            - 纵向车距 = rel_x_preceding
[22]: ttc                - Time-To-Collision = -distance/rel_vx
[23]: preceding_distance - 总距离 = sqrt(rel_x² + rel_y²)
```

#### 排除的特征
```
[6]:     angle           - 车辆朝向角 (与速度方向可能不一致)
[7,8,11]: discrete       - class_id, lane_id, site_id
[10]:    has_following   - 后车标志
[16-19]: rel_*_following - 后车相对特征 (不需要)
```

**优势**:
- ✅ 更简洁：15个特征 vs 原来17个
- ✅ 更聚焦：只关注前车交互
- ✅ 更直观：headway/ttc直接对应驾驶行为
- ✅ 速度方向一致性：velocity_direction显式监督

---

## 🚀 快速开始

### 环境安装

```bash
# Python 3.9+ (推荐 3.10 或 3.11)
pip install -r requirements.txt
```

**核心依赖**:
- Python >= 3.9
- PyTorch >= 2.0.0
- NumPy >= 1.24.0
- Pandas >= 2.0.0
- tqdm, matplotlib, seaborn

### 完整流程（4步）

#### 方案 1: 多站点基础训练（12 维特征）

```bash
# 1. 数据预处理
python src/data/preprocess_multisite.py

# 2. 验证数据
python src/data/validate_preprocessing.py

# 3. 训练模型
python src/training/train_world_model.py \
    --train_data data/processed/train_episodes.npz \
    --val_data data/processed/val_episodes.npz \
    --input_dim 12 \
    --continuous_dim 9 \
    --latent_dim 256 \
    --batch_size 16 \
    --epochs 50 \
    --lr 3e-4 \
    --scheduler cosine \
    --angle_weight 2.0

# 4. 评估模型（新版checkpoint会自动读取配置，旧版可手动指定参数）
python src/evaluation/rollout_eval.py \
    --checkpoint checkpoints/best_model.pt \
    --test_data data/processed/test_episodes.npz \
    --metadata data/processed/metadata.json \
    --stats_path data/processed/train_episodes.npz \
    --context_length 65 \
    --rollout_horizon 15
```

#### 方案 2: 单站点增强训练（20 维特征）⭐ 推荐

```bash
# 1. 预处理单站点（启用相对特征）
python src/data/preprocess_multisite.py --sites A --output_dir data/processed_siteA --use_extended_features

# 2. 验证数据
python src/data/validate_preprocessing.py --data_dir data/processed_siteA

# 3. 训练模型
python src/training/train_world_model.py \
    --train_data data/processed_siteA/train_episodes.npz \
    --val_data data/processed_siteA/val_episodes.npz \
    --checkpoint_dir checkpoints/world_model_siteA_enhanced \
    --input_dim 20 \
    --continuous_dim 16 \
    --num_sites 1 \
    --num_lanes 19 \
    --batch_size 16 \
    --epochs 50 \
    --lr 3e-4 \
    --scheduler cosine \
    --lr_min 1e-6 \
    --weight_decay 1e-5 \
    --angle_weight 2.0

# 4. 评估模型（新版checkpoint会自动读取配置）
python src/evaluation/rollout_eval.py \
    --checkpoint checkpoints/world_model_siteA_enhanced/best_model.pt \
    --test_data data/processed_siteA/test_episodes.npz \
    --metadata data/processed_siteA/metadata.json \
    --stats_path data/processed_siteA/train_episodes.npz
```

### 训练参数说明

**新增参数** (v2.4):
- `--scheduler`: 学习率调度策略
  - `cosine`: 余弦退火（推荐）
  - `step`: 阶梯衰减
  - `plateau`: 自适应衰减
  - `none`: 无调度器
- `--lr_min`: 最低学习率（cosine 模式）
- `--weight_decay`: L2 正则化系数（推荐 1e-5）
- `--angle_weight`: Angle 损失权重（推荐 2.0，基础 1.0）

**关键超参数**:
- `--input_dim`: 输入特征维度（12 或 20）
- `--continuous_dim`: 连续特征维度（9 或 16）
- `--latent_dim`: 潜在空间维度（推荐 64-256）
- `--hidden_dim`: Transformer 隐藏维度（推荐 256-512）
- `--num_heads`: 注意力头数（推荐 4-8）
- `--num_layers`: Encoder 层数（推荐 3-4）
- `--num_dyn_layers`: Dynamics 层数（推荐 2-3）

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
- 📄 **主脚本**: `src/data/preprocess_multisite.py`
- 📄 **核心逻辑**: `src/data/preprocess.py`
- 📄 **数据划分**: `src/data/split_strategy.py`

**命令**:
```bash
# 基础配置（12 维特征）
python src/data/preprocess_multisite.py

# 增强配置（20 维特征）⭐ 推荐
python src/data/preprocess_multisite.py --use_extended_features

# 自定义参数
python src/data/preprocess_multisite.py \
    --raw_data_dir data/raw \
    --output_dir data/processed \
    --episode_length 80 \
    --stride 15 \
    --fps 30.0 \
    --train_ratio 0.7 \
    --val_ratio 0.15 \
    --test_ratio 0.15 \
    --use_extended_features
```

**处理流程**:

```
src/data/preprocess_multisite.py
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
  └─ extract_extended_features()       # 5. 提取特征
      ├─ **基础特征 (12 维)**:
      │   ├─ center_x, center_y (标准化)
      │   ├─ vx, vy (速度)
      │   ├─ ax, ay (加速度)
      │   ├─ angle (朝向)
      │   ├─ class_id (离散, 不标准化)
      │   ├─ lane_id (离散, 不标准化)
      │   ├─ has_preceding, has_following
      │   └─ site_id (离散, 不标准化)
      │
      └─ **增强特征 (新增 8 维)** 🆕:
          ├─ preceding_rel_x, preceding_rel_y
          ├─ preceding_rel_vx, preceding_rel_vy
          ├─ following_rel_x, following_rel_y
          └─ following_rel_vx, following_rel_vy
  ↓
src/data/split_strategy.py
  └─ chronological_split_episodes()    # 6. 时序划分
      └─ 按时间顺序分train/val/test
```

**输出文件**:
```
data/processed/
├── train_episodes.npz               # [N_train, 80, 50, F]  F=12或20
│   ├── 'states'        → [N, T, K, F] 状态矩阵
│   ├── 'masks'         → [N, T, K] 有效性mask
│   ├── 'scene_ids'     → [N] site ID
│   ├── 'start_frames'  → [N] episode起始帧
│   └── 'end_frames'    → [N] episode结束帧
│
├── val_episodes.npz                 # 同上
├── test_episodes.npz                # 同上
│
├── metadata.json                    # 元数据 🆕 v2.4更新
│   ├── n_features: 12 或 20
│   ├── episode_length: 80
│   ├── context_length: 65
│   ├── rollout_horizon: 15
│   ├── fps: 30.0
│   ├── feature_layout: {...}       # 特征索引映射
│   ├── lane_mapping: {...}
│   └── validation_info: {          # 🆕 关键配置
│       ├── discrete_features: {7, 8, 11}
│       ├── angle_idx: 6            # 🆕 Angle特征索引
│       └── do_not_normalize: [7, 8, 11, 6]  # 🆕 包含angle
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
- `F=12 or 20`: 特征维度（基础/增强）

### 步骤3: 特征说明

#### 基础特征（12 维）

预处理生成**12维输入特征向量** (`src/data/preprocess.py:extract_extended_features()`):

| 索引 | 特征名 | 类型 | 模型处理 | 说明 | 代码位置 |
|------|--------|------|---------|------|---------|
| 0 | center_x | 连续 | ✅ **预测** | X坐标（z-score标准化） | `extract_extended_features()` L385 |
| 1 | center_y | 连续 | ✅ **预测** | Y坐标（z-score标准化） | L386 |
| 2 | vx | 连续 | ✅ **预测** | X方向速度 | L388 |
| 3 | vy | 连续 | ✅ **预测** | Y方向速度 | L389 |
| 4 | ax | 连续 | ✅ **预测** | X方向加速度 | L390 |
| 5 | ay | 连续 | ✅ **预测** | Y方向加速度 | L391 |
| 6 | angle | **周期性** | ✅ **预测** | 🔥 朝向角度（弧度，不归一化） | L392 |
| 7 | class_id | **离散** | 🔒 **Embedding** | 车辆类别（不标准化，不预测） | L393 |
| 8 | lane_id | **离散** | 🔒 **Embedding** | 车道ID（不标准化，不预测） | L394 |
| 9 | has_preceding | **二值** | ✅ **预测** | 是否有前车（sigmoid输出）| L395 |
| 10 | has_following | **二值** | ✅ **预测** | 是否有后车（sigmoid输出）| L396 |
| 11 | site_id | **离散** | 🔒 **Episode-level** | 站点ID 0-8（不标准化，不预测） | L397 |

**连续预测特征**: 8 维 (去除 angle 和 3 个离散特征)

#### 增强特征（20 维）🆕

在基础特征基础上添加：

| 索引 | 特征名 | 类型 | 说明 |
|------|--------|------|------|
| 12 | preceding_rel_x | 连续 | 前车相对 x 距离（无前车时=0）|
| 13 | preceding_rel_y | 连续 | 前车相对 y 距离 |
| 14 | preceding_rel_vx | 连续 | 前车相对 x 速度 |
| 15 | preceding_rel_vy | 连续 | 前车相对 y 速度 |
| 16 | following_rel_x | 连续 | 后车相对 x 距离（无后车时=0）|
| 17 | following_rel_y | 连续 | 后车相对 y 距离 |
| 18 | following_rel_vx | 连续 | 后车相对 x 速度 |
| 19 | following_rel_vy | 连续 | 后车相对 y 速度 |

**连续预测特征**: 16 维

#### 特征处理架构

**v2.3 关键架构 (Continuous-Only Decoder)**:
- ✅ **连续特征 (8/16维)**: Decoder **直接预测**
- 🔒 **离散特征 (3维)**: [7,8,11] - 作为 **Embedding 输入**，episode内保持常量，**不参与预测**
- 📊 **Loss计算**: 仅在连续特征上计算回归loss (Huber)
- 🎯 **Rollout**: Decoder输出[B,T,K,8/16]，离散特征从初始状态复制

**🔥 Angle (朝向角) 特殊处理** 🆕 v2.4:

**问题**:
- ❌ Z-score归一化破坏周期性: `-π` 和 `π` 是同一方向
- ✅ **解决方案**: Angle 保持原始弧度值，不做归一化
- ✅ **元数据配置**: `angle_idx: 6` 添加到 `validation_info`

**🔥 二值特征处理** 🆕 v2.4:

- ✅ **has_preceding/has_following** 使用 Sigmoid 激活
- ✅ **Decoder**: `binary_feature_indices=[6,7]` (在连续特征输出中的索引)
- ✅ **更准确的二值预测**

详见: [src/models/decoder.py](src/models/decoder.py)

### 步骤4: 验证预处理结果

**使用的代码文件**:
- 📄 **验证脚本**: `src/data/validate_preprocessing.py`

**命令**:
```bash
python src/data/validate_preprocessing.py
```

**检查项**:
- ✅ 元数据一致性 (fps=30, T=80, C=65, H=15)
- ✅ Lane token格式 ("site:lane")
- ✅ Train/Val/Test时序无重叠
- ✅ 离散特征未被标准化
- ✅ Angle索引配置正确
- ✅ 特征维度正确 (F=12或20)
- ✅ Episode数量合理

**期望输出**:
```
✅ All preprocessing checks passed!
- Metadata: fps=30.0, T=80, C=65, H=15
- Features: 12/20 (8/16 continuous, 3 discrete, 1 angle)
- Lane tokens: site:lane format OK
- Splits: No temporal overlap
- Train: 44100 episodes
- Val: 6300 episodes
- Test: 6300 episodes
- Angle index: 6 ✅
- Binary features: [9, 10] ✅
```

---

## 🎓 模型训练

### 当前推荐训练方式 (v2.5) ⭐

使用简化特征配置 + velocity_direction_loss：

```bash
./train_with_interaction_and_vel_dir.sh
```

**训练配置**:
- 特征数: 24 (15个continuous)
- Batch size: 32
- Learning rate: 1e-4
- Epochs: 200
- Loss weights:
  - reconstruction: 1.0
  - prediction: 1.0
  - velocity_direction: 0.3

**Loss函数**:
- `reconstruction_loss`: MSE重建损失
- `prediction_loss`: MSE预测损失
- `velocity_direction_loss`: 速度方向角损失 (新增)

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
cat data/processed_siteA_20/metadata.json | grep num_features
# 输出: "num_features": 20 (原始特征)
# 动态添加: 4个派生特征 (velocity_direction, headway, ttc, preceding_distance)
```

### 步骤2: 手动训练命令（高级用法）

**推荐配置 (简化特征)**:
```bash
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
    --save_interval 20
```

**旧版配置 (完整特征) - 不推荐**:
```bash
python src/training/train_world_model.py \
    --train_data data/processed_siteA_20/train_episodes.npz \
    --val_data data/processed_siteA_20/val_episodes.npz \
    --checkpoint_dir checkpoints/world_model_siteA \
    --input_dim 12 \
    --continuous_dim 9 \
    --latent_dim 256 \
    --batch_size 16 \
    --epochs 50 \
    --lr 3e-4 \
    --num_sites 1 \
    --num_lanes 20 \
    --use_site_id False
```

**单站点 vs 多站点参数对比**:
| 参数 | 多站点 | 单站点 (Site A) | 说明 |
|------|--------|----------------|------|
| `--sites` | A B C D E F G H I | A | 预处理时指定站点 |
| `--output_dir` | data/processed | data/processed_siteA | 单独的输出目录 |
| `--num_sites` | 9 (默认) | 1 | 站点embedding数量 |
| `--num_lanes` | ~150 | ~20 | 单站点车道数较少 |
| `--use_site_id` | True (默认) | False | 单站点可禁用site_id特征 |
| `--checkpoint_dir` | checkpoints/world_model | checkpoints/world_model_siteA | 避免冲突 |

**单站点训练的适用场景**:
- ✅ **快速原型验证**: 数据量小，训练速度快
- ✅ **站点特异性研究**: 研究特定站点的交通模式
- ✅ **计算资源受限**: 单站点数据量约为多站点的1/9
- ✅ **迁移学习基线**: 可用于测试跨站点泛化能力

**多站点训练的优势**:
- ✅ **更强泛化性**: 学习跨站点的通用交通规律
- ✅ **更多训练数据**: 9个站点数据联合训练
- ✅ **站点条件化**: 模型能区分不同站点的特征
- ✅ **更鲁棒**: 对单站点特殊情况不易过拟合

**单站点训练示例（其他站点）**:
```bash
# Site B
python src/data/preprocess_multisite.py --sites B --output_dir data/processed_siteB
python src/training/train_world_model.py \
    --train_data data/processed_siteB/train_episodes.npz \
    --val_data data/processed_siteB/val_episodes.npz \
    --checkpoint_dir checkpoints/world_model_siteB \
    --num_sites 1 --num_lanes 25

# Site C
python src/data/preprocess_multisite.py --sites C --output_dir data/processed_siteC
python src/training/train_world_model.py \
    --train_data data/processed_siteC/train_episodes.npz \
    --val_data data/processed_siteC/val_episodes.npz \
    --checkpoint_dir checkpoints/world_model_siteC \
    --num_sites 1 --num_lanes 18
```

**多站点组合训练示例**:
```bash
# 训练Site A + B + C的组合
python src/data/preprocess_multisite.py --sites A B C --output_dir data/processed_ABC
python src/training/train_world_model.py \
    --train_data data/processed_ABC/train_episodes.npz \
    --val_data data/processed_ABC/val_episodes.npz \
    --checkpoint_dir checkpoints/world_model_ABC \
    --num_sites 3 --num_lanes 60
```

**关键参数**:
- `--input_dim 12`: 输入维度（必须与metadata.json中的n_features一致）
- `--continuous_dim 9`: Decoder输出的连续特征维度（12维模式）或 16（20维模式）
- `--latent_dim 256`: 潜在空间维度（推荐128-512）
- `--batch_size 16`: 根据GPU内存调整（默认16）
- `--epochs 50`: 训练轮数
- `--lr 3e-4`: 学习率（AdamW优化器）
- `--dynamics_layers 4`: Transformer动力学层数
- `--dynamics_heads 8`: 注意力头数
- `--max_dynamics_len 512`: 最大序列长度
- `--max_dynamics_context 128`: Rollout时的最大上下文长度

**参数说明**:
- `input_dim=12/20`: Encoder接收完整的12维或20维特征
- `continuous_dim=8/16`: Decoder输出8个（12维模式）或16个（20维模式）连续特征
  - 12维模式: 输出 [0,1,2,3,4,5,6,9,10] 共9个，但angle(6)单独处理，实际连续输出8个
  - 20维模式: 增加8个相对特征，连续输出16个
- 离散特征 [7,8,11] (class_id, lane_id, site_id) 通过embedding条件化模型，不参与decoder输出

### 步骤3: 模型架构详解

**整体架构**: Encoder → Transformer Dynamics → Decoder (with Kinematic Prior)
**v2.3关键变化**: Decoder只输出8维连续特征（12维模式）或16维（20维模式），离散特征作为episode-level常量

**完整的前向传播流程**:

```
输入: states [B, T=80, K=50, F=12], masks [B, T, K]
  ↓
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【src/models/encoder.py: MultiAgentEncoder】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  1. 特征分离 (forward L133-164)
     # 将12维特征分为连续(9维)和离散(3维)特征

     连续特征提取 (L139):
     ├─ continuous_indices = [0,1,2,3,4,5,6,9,10]  # 排除7,8,11
     └─ cont = states[..., continuous_indices]  # [B, T, K, 9]
     # 注意: angle(6) 保留在连续特征中，但不被归一化

     离散特征embedding (L145-161):
     ├─ lane_id [8] → lane_embedding(nn.Embedding(num_lanes, 16))
     ├─ class_id [7] → class_embedding(nn.Embedding(num_classes, 8))
     └─ site_id [11] → site_embedding(nn.Embedding(num_sites, 8))
     # ⚠️ 这些embedding仅用于条件化encoder，不参与decoder预测

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
【src/models/dynamics.py: LatentDynamics】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  基于Transformer的时序动力学模型

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

  状态解码器：输出连续特征(9维) + 存在性 + (x,y)残差

  1. MLP Backbone (forward L34-42, L81)
     h = backbone(latent)
     # Sequential(
     #   Linear(256→256), LayerNorm, ReLU, Dropout,
     #   Linear(256→256), ReLU, Dropout
     # )
     → [B, T, hidden_dim=256]

  2. 连续状态预测 (L45-47, L86)
     states = state_head(h).view(B, T, K=50, F_cont=9)
     # Linear(256 → 50*9=450)
     # 输出: [center_x, center_y, vx, vy, ax, ay, angle, has_preceding, has_following]
     → [B, T, K, F_cont=9]

  3. Existence Logits (L48, L87)
     existence_logits = existence_head(h)
     # Linear(256 → 50)
     → [B, T, K]

  4. (x,y)残差头 (L51-57, L89-93)
     IF enable_xy_residual:
         residual_xy = residual_xy_head(h).view(B, T, K, 2)
         # Linear(256 → 50*2=100)
         # ✅ 初始化为0 (从纯物理先验开始学习)
         → [B, T, K, 2]
  ↓
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【src/models/world_model.py: WorldModel】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  完整世界模型：Encoder → Dynamics → Decoder + 物理先验

  前向传播流程 (forward L173-215):

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

  6. 物理先验 + 残差 (L200-207)
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
         "reconstructed_states": recon_states,      # [B,T,K,9] 连续特征
         "predicted_states": pred_states,           # [B,T,K,9] 带物理先验
         "existence_logits": exist_logits,          # [B,T,K]
         "predicted_existence_logits": pred_exist_logits,  # [B,T,K]
     }
```

**架构亮点**:
1. ✅ **Transformer动力学**: 基于Transformer的时序建模，causal masking确保因果关系
2. ✅ **物理先验 + 学习残差**: 结合运动学方程和神经网络修正
3. ✅ **离散特征embedding**: Lane, Class, Site通过embedding条件化模型
4. ✅ **Continuous-Only Decoder**: Decoder只输出9个连续特征，离散特征作为episode常量
5. ✅ **Normalization-aware**: 物理先验在原始空间计算，保证正确性

### 步骤4: Loss计算详解

**使用的代码文件**: `src/training/losses.py`

**Loss组成**:
- **重建Loss**: 重建当前帧的连续特征
- **预测Loss**: 预测下一帧的连续特征
- **存在性Loss**: 预测车辆是否存在（mask）
- **总Loss** = recon_weight × recon_loss + pred_weight × pred_loss + exist_weight × exist_loss

**关键特性**:
- 仅对连续特征计算 Huber Loss
- 离散特征不参与loss计算
- Decoder输出已经是连续特征，target需要过滤

        # Huber loss (beta=1.0)
        diff = pred - target  # 现在两者都是 [B,T,K,9]
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

**Loss计算流程**:

1. **重建loss**: decoder(latent) vs ground truth (对齐当前帧)
2. **预测loss**: t预测t+1 (时间对齐很关键)
   - 预测: t=0到t=T-2
   - 目标: t=1到t=T-1
3. **存在性loss**: BCEWithLogitsLoss (sigmoid(logits) vs ground truth masks)
4. **预测存在性loss**: 预测分支
if use_pred_existence_loss:
    pred_exist_loss = _existence_loss(
        predicted_existence_logits[:, :-1],  # 时间对齐
        masks[:, 1:]
    )
```

**为什么只对连续特征计算loss**:

离散特征 (7=class_id, 8=lane_id, 11=site_id):
- 类别变量，不适合回归loss
- 作为episode-level常量，通过embedding条件化encoder
- 预测时从initial_states复制，保持整个episode不变

连续特征 (0-6, 9-10):
- center_x, center_y, vx, vy, ax, ay, angle, has_preceding, has_following
- 适合回归任务，Huber loss对outliers鲁棒
- Decoder只输出这9个特征 [B,T,K,9]

**continuous_indices配置**:
- 由 dataset.py 从 metadata.json 自动读取
- 自动计算：排除离散特征索引 {7, 8, 11}
- 结果：[0, 1, 2, 3, 4, 5, 6, 9, 10]

### 步骤5: 训练流程详解

**代码文件**: `src/training/train_world_model.py`

**主要流程**:

1. **解析参数** - 从命令行读取训练配置
2. **创建数据加载器**:
   - Train loader: 自动计算 normalization stats
   - Val loader: 复用 train 的 normalization stats
3. **创建模型** - WorldModel with Encoder, Dynamics, Decoder
4. **设置优化器** - AdamW + Learning Rate Scheduler
5. **训练循环** - Forward → Loss → Backward → Update
6. **验证和保存** - 每个 epoch 后验证并保存最佳模型
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

**Checkpoint保存内容**:
- epoch: 训练轮数
- model_state_dict: 模型参数
- optimizer_state_dict: 优化器状态

**Normalization stats保存**:
- mean: 连续特征的均值
- std: 连续特征的标准差
- continuous_indices: 连续特征索引列表

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
# 新版checkpoint（训练时保存了config）会自动读取所有配置
python src/evaluation/rollout_eval.py \
    --checkpoint checkpoints/best_model.pt \
    --test_data data/processed/test_episodes.npz \
    --metadata data/processed/metadata.json \
    --stats_path data/processed/train_episodes.npz \
    --context_length 65 \
    --rollout_horizon 15 \
    --output_dir results/

# 旧版checkpoint（无config）需手动指定模型架构参数
python src/evaluation/rollout_eval.py \
    --checkpoint checkpoints/old_model/checkpoint_best.pt \
    --test_data data/processed/test_episodes.npz \
    --metadata data/processed/metadata.json \
    --stats_path data/processed/train_episodes.npz \
    --input_dim 20 \
    --latent_dim 512 \
    --dynamics_layers 6 \
    --dynamics_heads 16
```

**参数优先级**: 命令行参数 > checkpoint config > 自动推断/默认值

**可选模型架构参数**（针对旧版checkpoint）:
- `--input_dim`: 输入特征维度（12或20），默认从metadata.json读取
- `--latent_dim`: 潜在空间维度，默认从checkpoint权重推断或使用256
- `--dynamics_layers`: Dynamics Transformer层数，默认4
- `--dynamics_heads`: 注意力头数，默认8

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

       # Rollout预测
       rollout_output = model.rollout(
           initial_states=context_states,
           initial_masks=context_masks,
           n_steps=H=15,
           teacher_forcing=False
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

**Rollout流程** (src/models/world_model.py):

**步骤**:
1. **编码Context**: 使用encoder编码初始65帧
2. **Dynamics预测**: 通过dynamics模型预测潜在状态
3. **循环生成**: 逐步预测未来15帧
4. **解码状态**: 使用decoder将潜在状态解码为车辆状态
5. **组合特征**: 连续特征来自decoder，离散特征从初始状态复制
    
    # 3. 提取离散特征模板（保持不变）
    discrete_template = initial_states[:, -1:, :, discrete_indices]
    
    # 4. Autoregressive rollout
    for step in range(n_steps):
        # a. 解码 → 连续特征预测
        pred_cont = decoder(current_latent)
        
        # b. 重建完整状态（连续 + 离散）
        pred_full[..., continuous_indices] = pred_cont
        pred_full[..., discrete_indices] = discrete_template
        
        # c. 应用物理先验
        prior_xy = _kinematic_prior_xy(prev_state_full)
        pred_cont[..., :2] = prior_xy + residual_xy
        
        # d. 预测下一步latent
        next_latent = dynamics.step(latent_hist, max_context=128)
        current_latent = next_latent
    
    return predicted_states  # [B, n_steps, K, 9]
```

**关键特性**:
- 使用`dynamics.step()`进行单步预测（支持truncated context）
- 离散特征从initial_states复制，保持episode常量
- 物理先验在原始空间计算
- Truncated context（max_context=128）避免内存爆炸

### 步骤2: 指标计算

**代码文件**: `src/evaluation/prediction_metrics.py`

**指标详解**:

**compute_all_metrics** (prediction_metrics.py):

计算所有评估指标:
1. ADE (Average Displacement Error) - 平均位移误差
2. FDE (Final Displacement Error) - 最终位移误差
3. Velocity Error - 速度误差
4. Heading Error - 航向误差
5. Collision Rate - 碰撞率

**坐标转换**:
- 使用 src/utils/common.py:convert_pixels_to_meters
- pixel_to_meter ≈ 0.077
- 转换位置、速度、加速度特征
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

**ADE (平均位移误差)**:
- 提取预测和真值位置 (x, y)
- 计算 L2 距离
- 应用mask并求平均
- 单位: 米

**FDE (最终位移误差)**:
- 仅最后一帧
- 计算预测和真值的 L2 距离
- 单位: 米

**期望结果** (良好模型):
- ADE: 0.10 (10厘米平均误差)
- FDE: 0.12 (12厘米最终误差)
- velocity_error: 0.08 (8cm/s速度误差)
- heading_error: 1.5 (1.5度朝向误差)
- collision_rate: 5.2 (5.2% 取决于safety_margin)

---

## 🎨 可视化

### 步骤1: 轨迹可视化

**使用的代码文件**:
- 📄 **可视化脚本**: `src/evaluation/visualize_predictions.py`
- 📄 **航拍图**: `src/evaluation/sites/SiteA.jpg` ~ `SiteI.jpg`

**命令**:
```bash
# 新版checkpoint（自动读取配置）
python src/evaluation/visualize_predictions_detailed.py \
    --checkpoint checkpoints/best_model.pt \
    --test_data data/processed/test_episodes.npz \
    --metadata data/processed/metadata.json \
    --site_images_dir src/evaluation/sites \
    --context_length 65 \
    --rollout_horizon 15 \
    --output_dir results/visualizations \
    --num_samples 50 \
    --max_agents 10

# 旧版checkpoint（手动指定架构参数）
python src/evaluation/visualize_predictions_detailed.py \
    --checkpoint checkpoints/old_model/checkpoint_best.pt \
    --test_data data/processed/test_episodes.npz \
    --metadata data/processed/metadata.json \
    --site_images_dir src/evaluation/sites \
    --input_dim 20 \
    --latent_dim 512 \
    --dynamics_layers 6 \
    --dynamics_heads 16 \
    --output_dir results/visualizations \
    --num_samples 50 \
    --max_agents 10
```

**注意**: `--max_agents` 是可视化参数（限制图中显示的agent数量），不是模型参数

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

**draw_trajectory_on_image** (visualize_predictions.py):

使用 OpenCV 在航拍图上绘制轨迹:
- 输入: 航拍图 + 轨迹坐标(像素) + 颜色
- 过滤无效点
- 绘制连线、起点圆圈、终点方块

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

**其他可用的可视化/调试脚本**:
- 📄 `src/evaluation/visualize_predictions.py` - 基础版可视化（简化版）
- 📄 `src/evaluation/visualize_predictions_wm.py` - 高级可视化（支持采样策略、agent选择）
- 📄 `src/evaluation/debug_world_model_checks.py` - 模型诊断（检查open-loop、teacher-forcing等模式）

**注**: 所有evaluation脚本都支持 `--input_dim`, `--latent_dim`, `--dynamics_layers`, `--dynamics_heads` 参数用于加载旧版checkpoint

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
├── 📄 README.md                        # 本文件 - 完整用户指南
├── 📄 DATA_GUIDE.md                    # 数据格式详解
├── 📄 CODE_DOCUMENTATION.md            # 代码结构和API文档
├── 📄 requirements.txt                 # Python依赖
│
├── � tests/                           # 测试文件 🆕
│   ├── 📄 test_fixes.py                # 修复验证测试
│   ├── 📄 test_relative_features.py    # 相对特征测试
│   └── 📄 test_continuity.py           # 连续性测试
│
├── 📂 data/
│   ├── raw/                            # 原始CSV数据 (用户提供)
│   │   ├── A/drone_1.csv, drone_2.csv, ...
│   │   ├── B/drone_1.csv, ...
│   │   └── I/...
│   └── processed/                      # 预处理输出
│       ├── train_episodes.npz          # [N, 80, 50, 12/20]
│       ├── val_episodes.npz
│       ├── test_episodes.npz
│       ├── metadata.json               # 元数据配置
│       └── split_config.json           # 数据划分记录
│
├── 📂 src/
│   ├── 📂 data/
│   │   ├── 📄 __init__.py
│   │   ├── 📄 preprocess_multisite.py  # ⭐ 预处理主脚本 🆕
│   │   ├── 📄 validate_preprocessing.py # ⭐ 验证脚本 🆕
│   │   ├── 📄 reprocess_with_relative_features.py # 重新预处理脚本 🆕
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
│   │   │   ├─ Binary Features Sigmoid 🆕
│   │   │   └─ Existence Head ([K])
│   │   ├── 📄 dynamics.py              # ⭐ LatentDynamics
│   │   │   ├─ GRUDynamics
│   │   │   ├─ LSTMDynamics
│   │   │   └─ TransformerDynamics
│   │   └── 📄 world_model.py           # ⭐ WorldModel
│   │       ├─ forward()
│   │       └─ rollout()
│   │
│   ├── 📂 training/
│   │   ├── 📄 __init__.py
│   │   ├── 📄 train_world_model.py     # ⭐ 训练主脚本
│   │   │   ├─ Trainer class
│   │   │   ├─ LR Scheduler 🆕
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

**可视化流程**:

1. **训练时**:
   - 初始化: TrajectoryDataset (normalize=True) + WorldModel + WorldModelLoss
   - 训练循环: forward → loss → backward → update
   - 验证与保存: validate() → save_checkpoint()

2. **评估时**:
   - 加载 checkpoint 和 WorldModel
   - 创建 test dataset
   - 对每个 batch:
     - model.rollout(context, n_steps=15)
     - compute_all_metrics(预测, 真值, masks)

3. **可视化时**:
   - 加载 checkpoint 和站点图片
   - 创建 test dataset (normalize=False)
   - 对每个 batch:
     - 分割 context/target
     - 标准化 context → model.rollout
     - 反标准化 → 绘制轨迹

3. **可视化时**:
   - 加载 checkpoint 和站点图片
   - 创建 test dataset (normalize=False)
   - 对每个 batch:
     - 分割 context/target
     - 标准化 context → model.rollout
     - 反标准化 → 绘制轨迹

---

## ⚠️ 重要说明

### 1. 离散特征处理（关键！）🔥 v2.3更新

**为什么重要**: 这是最常见的错误来源！v2.3彻底解决了离散特征问题。

**🔥 v2.3架构 - Decoder不输出离散特征**:

**数据归一化** (src/data/dataset.py):
- 仅对连续特征进行 z-score 归一化
- 离散特征 [7, 8, 11] 保持原始值不变
- Angle 特征保持原始弧度值（不归一化）

**模型Encoder** (src/models/encoder.py):
- 离散特征通过 Embedding 层学习
- Site/Lane/Class embeddings 用于条件化编码器
- 这些特征不参与decoder预测，仅影响潜在表示

**Decoder输出** (src/models/decoder.py):
- 只输出连续特征维度: 8 (12维模式) 或 16 (20维模式)
- 输出特征: center_x, center_y, vx, vy, ax, ay, angle, has_preceding, has_following
- 不输出离散特征: class_id, lane_id, site_id

**Loss计算** (src/training/losses.py):
- pred: decoder直接输出的连续特征
- target: 需要过滤到连续特征索引
- 只对连续特征计算 Huber Loss
- 离散特征不参与loss计算

**Rollout实现** (src/models/world_model.py):
- 离散特征从初始状态复制，在整个rollout过程中保持不变
- Decoder每步输出连续特征
- 重建完整状态时，连续部分来自decoder，离散部分复制模板

### 2. 输入维度匹配

**检查方法**:
```bash
# 1. 查看metadata中的特征数
cat data/processed/metadata.json | grep n_features
# 输出: "n_features": 12

# 2. 🔥 v2.3训练时需要两个维度参数
python src/training/train_world_model.py \
    --input_dim 12 \           # Encoder输入维度
    --continuous_dim 9 ...     # Decoder输出维度
```

**维度说明**:
```
输入: [B, T, K, 12]
  ├─ Encoder接收全部12维特征
  │   ├─ 连续特征 [0,1,2,3,4,5,6,9,10] → 9维
  │   └─ 离散特征 [7,8,11] → 3维 (转为embedding)
  ↓
Encoder输出: [B, T, latent_dim]
  ↓
Dynamics: [B, T, latent_dim]
  ↓
Decoder输出: [B, T, K, 9] 连续特征
```

**正确用法**:
```bash
python src/training/train_world_model.py \
    --input_dim 12 \
    --continuous_dim 9 \
    --latent_dim 256 ...
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

**单站点训练时如何确定num_lanes**:
```bash
# 预处理后查看metadata
cat data/processed_siteA/metadata.json | grep num_lanes
# 输出: "num_lanes": 19

# 或者查看lane_mapping的长度
cat data/processed_siteA/metadata.json | grep -A 100 lane_mapping | grep "A:" | wc -l
# 输出: 19 (Site A的车道数)

# 训练时使用该值
python src/training/train_world_model.py \
    --train_data data/processed_siteA/train_episodes.npz \
    --num_lanes 19  # 使用metadata中的实际值
```

**各站点车道数参考** (实际以metadata.json为准):
| 站点 | 大致车道数 | 备注 |
|------|----------|------|
| Site A | ~19 | A1-A6, B1-B7, C1-C5, crossroads1 |
| Site B | ~25 | 包含多个crossroads |
| Site C | ~18 | 较小站点 |
| ... | ... | 预处理后查看metadata确认 |
| **多站点** | ~150 | 所有站点车道union |

## 🐛 故障排查

### 问题1: Loss不下降 / 收敛缓慢 🆕

**症状**: Train loss在高值plateau,不下降或下降缓慢

**可能原因** 🆕 v2.4:
1. Learning rate过高或过低
2. 离散特征被错误标准化
3. **Angle特征未正确配置** - 检查 metadata.json 中是否有 `angle_idx: 6`
4. Input_dim/continuous_dim不匹配
5. Batch size太小
6. 没有使用学习率调度器

**解决方案**:
```bash
# 1. 检查元数据（关键！）
cat data/processed/metadata.json | grep -E "n_features|discrete_features|angle_idx"
# 应该看到: "angle_idx": 6

# 2. 使用推荐训练参数 🆕
python src/training/train_world_model.py \
    --learning_rate 1e-3 \
    --scheduler cosine \       # 🆕 使用余弦退火
    --lr_min 1e-6 \
    --weight_decay 1e-5 \      # 🆕 添加正则化
    --angle_weight 2.0         # 🆕 增加angle损失权重

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

# 2. 使用gradient clipping 🆕
python src/training/train_world_model.py --grad_clip 1.0

# 3. 检查数据
python src/data/validate_preprocessing.py  # 确保数据正常
```

### 问题3: 验证集Loss过早停滞（过拟合）

**症状**: 
- Train loss 持续下降 (1.78 → 1.35)
- Val loss 在某个值停滞或波动 (1.60 → 1.38 → 停滞)

**解决方案**:
```bash
# 方案 A: 增加正则化（推荐）
python src/training/train_world_model.py \
    --weight_decay 0.001 \
    --scheduler plateau     # 验证集loss不下降时降低学习率

# 方案 B: 使用更多数据
python src/data/preprocess_multisite.py --sites A B C  # 多站点训练

# 方案 C: Early stopping
# 在 train_world_model.py 中添加 patience 参数
```

### 问题4: Angle预测误差大 🆕

**症状**: Heading MAE > 0.5 rad (约30度)

**根本原因**:
1. **Angle被错误归一化** - angle是周期性特征，z-score会破坏周期性
2. **metadata.json缺少angle_idx配置**
3. **angle_weight过小** - 默认权重不足以让模型重视angle

**完整解决方案**:
```bash
# 1. 检查并修复 metadata.json
cat data/processed/metadata.json | grep angle_idx
# 如果没有，添加:
# "validation_info": {
#     "angle_idx": 6,
#     "do_not_normalize": [6, 7, 8, 11]
# }

# 2. 使用更大的 angle_weight
python src/training/train_world_model.py --angle_weight 2.0

# 3. 如果已经训练了错误的数据，需要重新预处理
python src/data/preprocess_multisite.py --sites A
```

**预期效果**:
- Angle MAE: 从 0.84 rad (48°) 降至 0.09-0.17 rad (5-10°)

### 问题5: 预测不连续

**症状**: 可视化结果中,context和prediction之间有跳跃

**解决方案**:
1. 增加context_length
2. 调整loss权重(增大pred_weight)
3. 检查normalization stats是否正确加载

### 问题6: 模型加载失败

**症状**: RuntimeError: size mismatch

**解决方案**:
1. 确保checkpoint与当前代码版本匹配
2. 检查metadata.json中的配置是否正确
3. 确认input_dim, continuous_dim等参数与训练时一致
4. 如果从12维升级到20维，需要重新训练

### 问题7: 相对特征数据不可用

**症状**: 加载数据时显示特征维度=12，但想使用20维

**解决方案**:
```bash
# 重新预处理数据，启用相对特征
python src/data/reprocess_with_relative_features.py --site A

# 或者从头开始
python src/data/preprocess_multisite.py --sites A --use_extended_features
```

### 问题8: 二值特征预测精度低

**症状**: has_preceding/has_following MAE > 0.6

**原因**: 二值特征难以预测（依赖周围车辆）

**解决方案**:
```bash
# v2.4 已添加 sigmoid 激活，如果还是不行：
# 1. 接受当前精度（不影响核心预测）
# 2. 或增加这些特征的loss权重（需修改代码）
```

---

## 📚 参考资料

### 核心概念

**World Model 是什么？**

World Model 是一个学习环境动态的神经网络，可以：
- 从观测序列中学习潜在表示
- 预测未来状态
- 用于轨迹预测、规划和仿真

**与 DreamerV3 的区别**:
- **DreamerV3**: 适用于强化学习，使用 RSSM (循环状态空间模型)，带有随机性
- **本项目**: 适用于轨迹预测，确定性模型，专注于多智能体交通场景

**关键特性**:
1. **Multi-agent encoder**: 使用 Transformer 处理多车辆交互
2. **Deterministic dynamics**: GRU/LSTM/Transformer 建模时序
3. **Physics-aware decoder**: 集成运动学先验
4. **Site/Lane conditioning**: 支持多场景泛化

### 架构设计原则

1. **Encoder**: 提取场景级潜在表示
   - 输入: [B, T, K, F] 多车辆状态
   - 输出: [B, T, latent_dim] 场景潜在向量

2. **Dynamics**: 建模时序演化
   - 1-step transition: z[t] → z[t+1]
   - 支持 GRU/LSTM/Transformer

3. **Decoder**: 重建车辆状态
   - 输入: [B, T, latent_dim]
   - 输出: [B, T, K, F_continuous] + masks

---

## 📚 文档索引

### 核心文档
- 📘 [README.md](README.md) - 本文档（完整使用指南）
---

## 📚 文档说明

本项目包含以下文档：

- **README.md** (本文档) - 完整的用户指南和参考手册
- **DATA_GUIDE.md** - 数据格式、预处理流程详解  
- **CODE_DOCUMENTATION.md** - 代码结构和API文档

所有功能说明、故障排查、最佳实践都已整合到本 README 中。

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

## 📮 版本信息

**项目版本**: v2.4 🆕 ✅

**更新日志**:
- **v2.4** (2025):
  - ✅ 添加相对位置特征（8维）
  - ✅ Learning Rate Scheduler (cosine/step/plateau)
  - ✅ Angle优化（修复归一化，添加angle_idx）
  - ✅ 二值特征Sigmoid激活
  - ✅ 完善metadata.json配置
  - ✅ 重组项目结构（脚本移至src/data，测试移至tests）

- **v2.3**:
  - ✅ Decoder只输出连续特征
  - ✅ 离散特征作为episode-level常量

- **v2.2**:
  - ✅ 多站点支持
  - ✅ 时序划分策略

**核心特性**:
- Transformer编码器 + 时序动力学 + 物理先验
- Decoder输出8/16个连续特征（离散特征作为常量）
- Angle专用处理（不归一化）
- 可选的相对位置特征（车辆交互建模）

---

**快速查找**:
- 如何修改特征? → [src/data/preprocess.py](src/data/preprocess.py):`extract_extended_features`
- 如何修改模型架构? → [src/models/](src/models/) (encoder.py, decoder.py, dynamics.py)
- 如何修改loss? → [src/training/losses.py](src/training/losses.py)
- 如何添加新指标? → [src/evaluation/prediction_metrics.py](src/evaluation/prediction_metrics.py)
- 数据格式详解? → [DATA_GUIDE.md](DATA_GUIDE.md)
- API文档? → [CODE_DOCUMENTATION.md](CODE_DOCUMENTATION.md)

