# Traffic World Model - 多智能体轨迹预测

基于Transformer的潜在世界模型，用于多站点无人机轨迹数据的预测和仿真。

---

## 🚀 快速开始

### 1. 环境安装

```bash
pip install -r requirements.txt
```

**依赖**: Python 3.10+, PyTorch 2.0+, NumPy, Pandas, PyYAML

### 2. 数据准备

将CSV文件按站点组织：

```
data/raw/
├── A/
│   ├── drone_1.csv
│   ├── drone_2.csv
│   └── ...
├── B/
│   └── ...
...
└── I/
```

**CSV必需列**: `track_id`, `frame`, `center_x`, `center_y`, `angle`, `class_id`
**可选列**: `lane`, `preceding_id`, `following_id`, `timestamp`

### 3. 数据预处理

```bash
python preprocess_multisite.py
```

**输出**:
- `data/processed/train_episodes.npz`
- `data/processed/val_episodes.npz`
- `data/processed/test_episodes.npz`
- `data/processed/metadata.json`

**默认配置**:
- Episode长度: T=80 (C=65 context + H=15 rollout)
- Stride: 15 frames (0.5秒 @ 30 FPS)
- Split: 70% train / 15% val / 15% test (chronological)
- Features: 12-dim (position, velocity, acceleration, angle, class, lane, preceding/following, site_id)

### 4. 验证数据

```bash
python validate_preprocessing.py
```

**检查项**:
- ✓ Metadata一致性 (fps=30, T=80, C=65, H=15)
- ✓ Lane tokens格式 ("site:lane")
- ✓ Split时间不重叠
- ✓ Feature dimensions

### 5. 训练模型

```bash
python src/training/train_world_model.py \
    --train_data data/processed/train_episodes.npz \
    --val_data data/processed/val_episodes.npz \
    --input_dim 12 \
    --latent_dim 256 \
    --batch_size 32 \
    --n_epochs 100
```

**训练监控**: 日志保存在 `logs/trainer.log`

---

## 📖 详细说明

### 数据预处理流程

预处理pipeline实现了以下关键改进：

**1. Per-Site Global Timeline**
- 每个站点的所有CSV文件合并为统一时间轴
- 处理frame重置和track_id冲突
- 创建`global_frame`和`global_track_id`

**2. Gap Detection & Segmentation**
- 自动检测时间间隙（gap > 1 frame）
- 将timeline分割为连续段
- Episodes不跨越时间断点

**3. Fixed-Stride Episode Extraction**
- T=80 frames per episode
- S=15 frames stride (equal interval sampling)
- Stable vehicle-to-slot assignment

**4. Site-Specific Lane Encoding**
- Lane tokens格式: "A:A1", "B:crossroads1"
- 避免跨站点lane冲突
- 动态计算num_lanes

**5. Chronological Split (Scheme A)**
- 先确定frame cutoffs (70%/15%/15%)
- 在各split内独立提取episodes
- 真正的时间不重叠，无temporal leakage

**Feature Layout (12-dim)**:
```
[0]  center_x        → continuous
[1]  center_y        → continuous
[2]  vx              → continuous
[3]  vy              → continuous
[4]  ax              → continuous
[5]  ay              → continuous
[6]  angle           → continuous
[7]  class_id        → discrete (do not normalize)
[8]  lane_id         → discrete (do not normalize, use embedding)
[9]  has_preceding   → binary
[10] has_following   → binary
[11] site_id         → discrete (do not normalize, use embedding)
```

### 预处理参数

```bash
python preprocess_multisite.py \
    --episode_length 80 \        # T = C + H
    --stride 15 \                # Step between episodes
    --fps 30.0 \                 # Frames per second
    --train_ratio 0.7 \
    --val_ratio 0.15 \
    --test_ratio 0.15 \
    --use_chronological_split    # Time-based split (default)
```

### 模型架构

**Encoder** (Multi-Agent Transformer):
- Feature embedding + site/lane embeddings
- Spatial positional encoding (optional)
- Social pooling (optional)
- Transformer attention over agents
- Masked mean pooling → latent

**Dynamics** (GRU/LSTM/Transformer):
- 1-step transition: latent[t] → latent[t+1]
- Teacher-forced during training
- Open-loop rollout during evaluation

**Decoder**:
- Latent → states reconstruction
- Existence prediction (which agents are present)

**Loss**:
- Reconstruction: L2(states_t, reconstructed_t)
- Prediction: L2(states_{t+1}, predicted_{t+1})
- Existence: BCE(masks, predicted_masks)

### 训练参数

```bash
python src/training/train_world_model.py \
    --train_data data/processed/train_episodes.npz \
    --val_data data/processed/val_episodes.npz \
    --input_dim 12 \              # Must match metadata.n_features
    --latent_dim 256 \
    --dynamics_type gru \         # gru/lstm/transformer
    --batch_size 32 \
    --n_epochs 100 \
    --learning_rate 1e-3 \
    --recon_weight 1.0 \
    --pred_weight 1.0 \
    --existence_weight 0.1
```

**重要**: `--input_dim` 必须与 `metadata.json` 中的 `n_features` 一致！

### 使用Lane和Site Embeddings

在encoder中启用embeddings（需修改代码）：

```python
encoder = MultiAgentEncoder(
    input_dim=12,
    use_site_id=True,
    num_sites=9,
    site_embed_dim=16,
    use_lane_embedding=True,
    num_lanes=len(lane_mapping) + 1,  # 从metadata读取
    lane_embed_dim=16,
    lane_feature_idx=8,
)
```

**注意**: 离散特征(lane_id, class_id, site_id)不要z-score normalize！

---

## 📊 数据格式

### NPZ文件结构

```python
{
    'states': [N, T, K, F],      # N episodes, T=80 timesteps, K=50 max agents, F=12 features
    'masks': [N, T, K],          # 1=real agent, 0=padding
    'scene_ids': [N],            # Site ID per episode
    'start_frames': [N],         # Episode start global_frame
    'end_frames': [N]            # Episode end global_frame
}
```

### Metadata.json关键字段

```json
{
  "n_features": 12,
  "episode_length": 80,
  "context_length": 65,
  "rollout_horizon": 15,
  "stride": 15,
  "fps": 30.0,
  "dt": 0.033333,
  "use_chronological_split": true,
  "lane_mapping": {
    "A:A1": 1,
    "A:B1": 2,
    "B:crossroads1": 3,
    ...
  },
  "validation_info": {
    "num_lanes": 150,
    "discrete_features": {
      "lane_id": 8,
      "class_id": 7,
      "site_id": 11
    },
    "do_not_normalize": ["lane_id", "class_id", "site_id"]
  }
}
```

---

## 🔍 验证检查清单

运行 `python validate_preprocessing.py` 会检查：

- [x] fps=30, dt=1/30
- [x] episode_length=80, stride=15
- [x] context_length=65, rollout_horizon=15
- [x] Lane tokens格式: "site:lane"
- [x] Feature dimensions匹配
- [x] Train/val/test时间不重叠
- [x] Discrete features正确标注

---

## 📁 项目结构

```
traffic_wm/
├── data/
│   ├── raw/              # 原始CSV (用户提供)
│   │   ├── A/
│   │   ├── B/
│   │   └── ...
│   └── processed/        # 预处理输出
│       ├── train_episodes.npz
│       ├── val_episodes.npz
│       ├── test_episodes.npz
│       └── metadata.json
│
├── src/
│   ├── data/
│   │   ├── preprocess.py        # 核心预处理逻辑
│   │   ├── split_strategy.py    # Chronological split
│   │   └── dataset.py           # PyTorch Dataset
│   ├── models/
│   │   ├── encoder.py           # Multi-Agent Encoder
│   │   ├── decoder.py           # Decoder
│   │   ├── dynamics.py          # GRU/LSTM/Transformer
│   │   └── world_model.py       # Complete model
│   ├── training/
│   │   ├── train_world_model.py # 训练脚本
│   │   └── losses.py            # Loss functions
│   ├── evaluation/
│   │   ├── rollout_eval.py      # Rollout evaluation
│   │   ├── prediction_metrics.py
│   │   └── visualization.py
│   └── utils/
│       ├── logger.py
│       └── common.py
│
├── preprocess_multisite.py      # 预处理主脚本
├── validate_preprocessing.py    # 验证脚本
├── requirements.txt
└── README.md
```

---

## ⚠️ 重要注意事项

### 1. Input Dimension匹配

**最常见错误**: `--input_dim` 与预处理不匹配

```bash
# 先检查
cat data/processed/metadata.json | grep n_features
# 输出: "n_features": 12

# 然后训练时使用相同值
python src/training/train_world_model.py --input_dim 12 ...
```

### 2. 时间参数一致性

- **Raw data**: 30 FPS
- **Preprocessing**: `--fps 30.0`
- **Episodes**: T=80 frames = 2.67秒
- **Context (C)**: 65 frames = 2.17秒
- **Rollout (H)**: 15 frames = 0.50秒

### 3. Chronological Split

**默认启用**，确保train/val/test在时间上不重叠。

如需随机split（不推荐）：
```bash
python preprocess_multisite.py --use_random_split
```

### 4. Lane Embedding

Lane tokens格式: `"site:lane"`
- ✓ Correct: `"A:A1"`, `"B:crossroads1"`
- ✗ Wrong: `"A1"`, `"crossroads1"`

验证: `cat data/processed/metadata.json | grep lane_mapping`

---

## 🐛 常见问题

**Q: 预处理后val/test没有episodes？**
A: 数据量太小。自动边界调整会确保至少 `episode_length + stride` 帧。

**Q: 训练时维度不匹配？**
A: 检查 `--input_dim` 是否与 `metadata.json` 的 `n_features` 一致。

**Q: Lane ID冲突？**
A: 确保预处理使用了site-specific tokens (`"A:lane"`)。运行 `validate_preprocessing.py` 检查。

**Q: 时间泄漏问题？**
A: 使用 `--use_chronological_split` (默认)。运行 `validate_preprocessing.py` 验证时间不重叠。

**Q: 如何可视化结果？**
A: 使用 `src/evaluation/visualization.py` (需自行实现rollout evaluation)

---

## 📝 完整工作流程

```bash
# 1. 准备数据
# 将CSV文件放入 data/raw/A/, data/raw/B/, ...

# 2. 预处理
python preprocess_multisite.py

# 3. 验证
python validate_preprocessing.py

# 4. 检查配置
cat data/processed/metadata.json

# 5. 训练
python src/training/train_world_model.py \
    --train_data data/processed/train_episodes.npz \
    --val_data data/processed/val_episodes.npz \
    --input_dim 12 \
    --latent_dim 256 \
    --batch_size 32 \
    --n_epochs 100

# 6. 监控训练
tail -f logs/trainer.log

# 7. (可选) Rollout evaluation
# 需实现 src/evaluation/rollout_eval.py
```

---

## 🎯 Pipeline特性总结

| 特性 | 实现方式 |
|------|---------|
| **Multi-site handling** | Per-site global timeline, site_id embedding |
| **Frame resets** | global_frame = frame + file_offset |
| **Track ID collisions** | global_track_id = file_id * 1M + track_id |
| **Gap detection** | Split when gap > 1 frame |
| **Episode extraction** | Fixed-stride (T=80, S=15) |
| **Lane encoding** | Site-specific tokens ("A:A1") |
| **Split strategy** | Chronological (time-based) |
| **Temporal leakage** | Scheme A: frame cutoffs → independent extraction |
| **Validation** | Automated checks via validate_preprocessing.py |

---

## 📚 参考

- **improved.md**: 原始改进规范
- **IMPROVEMENTS_todo.md**: 详细需求清单
- **BUGFIX_PATCH.md**: Bug修复说明

**所有改进已实施，代码已验证可用！** ✅

---

**License**: MIT
**Author**: Traffic World Model Team
**Last Updated**: 2025-12-12
