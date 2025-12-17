"""Traffic World Model - code documentation.

This repository evolved quickly and the previous function-by-function document
became stale. This file intentionally documents *entry points*, *data/layout
contracts*, and *behavioral gotchas* that affect training/eval correctness.
"""

# Traffic World Model - 代码文档（维护版）

## 入口与工作流

- 预处理（raw CSV → episodes.npz）: `src/data/preprocess_multisite.py`
- Dataset / DataLoader: `src/data/dataset.py`
- 模型（含 rollout & kinematic prior）: `src/models/world_model.py`
- 训练入口: `src/training/train_world_model.py`
- loss 组合与权重: `src/training/losses.py`
- rollout 评估: `src/evaluation/rollout_eval.py`
- 指标: `src/evaluation/prediction_metrics.py`
- 可视化（静态/动画）:
  - `src/evaluation/visualize_predictions_detailed.py`
  - `src/evaluation/visualize_predictions_wm.py`

## 数据契约（必须一致的部分）

### 时间与单位

- 默认帧率: 30 FPS
- 默认单位: 像素（pixel）

### masks 的语义

`masks[t, k] = 1` 表示 agent slot `k` 在时间 `t` 有效。
重要影响:
- 差分（位置→速度/加速度）若跨越 mask gap，会产生伪速度。
- 绘图若不在 mask gap 处断线，会出现“超长线”伪像。

## 预处理要点

文件: `src/data/preprocess.py`

关键点:
- 会构建全局时间线（避免跨 CSV 的 frame 重置）并做 chronological split。
- 速度/加速度的差分应按真实帧间隔（frame gap * dt）缩放，减弱缺帧导致的速度爆炸。

文件: `src/data/preprocess_multisite.py`

关键点:
- 负责多站点循环、split、保存 `train_episodes.npz/val_episodes.npz/test_episodes.npz` 与 `metadata.json`。

## Dataset 与特征布局

文件: `src/data/dataset.py`

核心行为:
- 从 `metadata.json` 读取离散特征索引（如 lane/class/site 等），并避免对其做 z-score。
- 会动态追加 4 个派生特征到 state 末尾（最终 `states` 为 `[T, K, 24]`）。
- val/test 必须显式提供 train 的 `stats_path`，避免归一化不一致。

派生特征（按 `__getitem__` 逻辑）:
- velocity_direction, headway, ttc, preceding_distance

## 训练与关键开关

文件: `src/training/train_world_model.py`

### `--disable_vxy_supervision`

含义:
- vx/vy 仍作为输入特征存在（模型可使用），但不作为回归监督目标。

原因:
- vx/vy 很容易被缺帧/重现（mask 0→1）引入的差分噪声污染。

配套行为:
- vx/vy-based 的 VEL-DIR 指标会在日志里标注为 diag-only。
- open-loop rollout 的 kinematic prior 在该模式下会优先使用由预测位置差分得到的速度（v = Δp / dt），避免依赖模型生成的 vx/vy。

### short open-loop rollout loss

- 用短 horizon 的 open-loop rollout 位置误差（xy-only）作为辅助 loss，更贴近真实 rollout 行为。

### scheduled sampling

- 在 teacher forcing 与自回归之间做平滑过渡，降低训练/推理暴露偏差。

### soft boundary penalty

- 对越界位置施加软约束，减少 open-loop 跑飞。

## 评估与可视化

### 指标计算空间

方向/角度相关指标应在反归一化后的物理/像素空间计算，避免在归一化空间因各向异性 std 扭曲角度。

### mask-aware 轨迹绘制

可视化脚本会在 mask gap 处插入 NaN 以断线，避免 padding → real 的连接导致误读。

## 调试脚本（保留在 src 下）

为避免仓库根目录堆积一次性脚本，方向一致性检查已迁移到:

- `src/evaluation/debug/gt_direction_consistency.py`
  4. 编码lanes (使用site-specific token: "A:A1")
  5. `detect_gaps_and_split_segments`
  6. **应用frame_range过滤**:
     ```python
     for seg_start, seg_end in segments:
         clipped_start = max(seg_start, min_frame)
         clipped_end = min(seg_end, max_frame)
         if clipped_end - clipped_start + 1 >= episode_length:
             filtered_segments.append((clipped_start, clipped_end))
     ```
  7. `extract_fixed_stride_episodes`
- **返回**: (episodes, updated_lane_mapping)

#### 辅助函数

**`extract_episodes(df, episode_length=30, overlap=0, ...) -> List[Dict]`**
- **作用**: 旧版episode提取(使用original frame,不使用global timeline)
- **注**: 已被`extract_fixed_stride_episodes`替代

**`extract_single_episode(df, frames, max_vehicles, ...) -> Dict`**
- **作用**: 旧版单episode提取(基于original frame)
- **注**: 已被`extract_single_episode_from_global`替代

**`compute_dataset_statistics(episodes: List[Dict]) -> Dict`**
- **作用**: 计算数据集统计量
- **输出**:
  - `n_episodes`
  - `mean_vehicles_per_frame`
  - `max/min_vehicles_observed`
  - `feature_means/stds/mins/maxs` (per feature)

**`split_episodes(episodes, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42) -> Tuple[...]`**
- **作用**: 随机划分episodes (旧版,已被chronological split替代)
- **算法**: Random permutation → 按比例切分
- **返回**: (train_episodes, val_episodes, test_episodes)

**`preprocess_trajectories(input_dir, output_dir, ...) -> None`**
- **作用**: 旧版完整预处理流程 (不使用global timeline)
- **注**: 主要用于向后兼容（内部函数，不作为对外CLI入口）

> ✅ 唯一推荐/支持的预处理入口是 `src/data/preprocess_multisite.py`（负责 multi-site + split + 写出 `normalization_stats.npz`）。

---

### 📄 `src/data/split_strategy.py`

数据划分策略,支持随机划分和时序划分。

#### MultiSiteDataSplitter类

**`class MultiSiteDataSplitter`**
- **作用**: 混合所有站点并随机划分文件
- **用途**: 旧版划分策略(已被chronological split替代)

**`__init__(raw_data_dir=None, sites=['A',...,'I'])`**
- **作用**: 初始化splitter
- **逻辑**: 检查站点目录是否存在,记录available_sites

**`get_site_files(site: str) -> List[Path]`**
- **作用**: 获取指定站点的所有CSV文件
- **返回**: sorted list of CSV paths

**`split_data(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42) -> Dict`**
- **作用**: 随机混合划分所有站点的文件
- **算法**:
  1. 收集所有站点的所有CSV文件
  2. 随机shuffle (使用seed)
  3. 按比例切分: train=80%, val=10%, test=10%
- **返回**: `{'train': [files], 'val': [files], 'test': [files]}`

**`save_split_config(splits: Dict, output_path=None)`**
- **作用**: 保存划分配置为JSON (用于复现)
- **格式**:
  ```json
  {
    "train": {
      "A": ["drone_1.csv", "drone_2.csv"],
      "B": [...]
    },
    "val": {...},
    "test": {...}
  }
  ```

#### 时序划分函数

**`chronological_split_episodes(episodes, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15) -> Dict`**
- **作用**: 按时间顺序划分episodes (防止temporal leakage)
- **算法**:
  1. 按`episode_start_global_frame`排序
  2. 顺序切分: first 70% → train, next 15% → val, last 15% → test
- **关键**: 确保train的最晚时刻 < val的最早时刻
- **返回**: `{'train': [...], 'val': [...], 'test': [...]}`
- **日志**: 打印每个split的episode数和frame范围

---

### 📄 `src/data/dataset.py`

PyTorch Dataset实现,用于训练和评估。

#### TrajectoryDataset类

**`class TrajectoryDataset(Dataset)`**
- **作用**: 多智能体轨迹episode的PyTorch Dataset
- **数据格式**:
  - `states`: [N, T, K, F] - N个episodes,每个T=80帧,K=50个agents,F=12维特征
  - `masks`: [N, T, K] - 有效agent标记 (1=有效, 0=padding)
  - `scene_ids`: [N] - 站点ID (0-8 for A-I)

**`__init__(data_path, normalize=True, stats_path=None)`**
- **作用**: 初始化dataset
- **关键逻辑**:
  1. **强制val/test使用train stats**:
     ```python
     is_val_or_test = 'val' in filename or 'test' in filename
     if is_val_or_test and normalize and not stats_path:
         raise ValueError("Val/test MUST provide stats_path")
     ```
  2. 加载NPZ数据 → 转为torch.Tensor
  3. 加载metadata → 识别discrete_features
  4. 计算或加载normalization stats (仅对continuous features)
  5. 归一化数据
  6. 验证并clamp离散特征
- **参数**:
  - `data_path`: NPZ文件路径
  - `normalize`: 是否归一化
  - `stats_path`: 归一化统计量路径 (val/test必须提供)

**`_load_discrete_feature_indices() -> None`**
- **作用**: 从metadata.json加载离散特征索引
- **逻辑**:
  ```python
  metadata = json.load('metadata.json')
  discrete_features = metadata['validation_info']['discrete_features']
  # {'lane_id': 8, 'class_id': 7, 'site_id': 11}
  self.discrete_indices = sorted([8, 7, 11])
  self.continuous_indices = [0,1,2,3,4,5,6,9,10]
  ```
- **边界**: 加载num_lanes, num_sites, num_classes用于后续验证

**`_validate_discrete_indices() -> None`**
- **作用**: 验证离散特征索引的有效性
- **检查**:
  - 索引在[0, F-1]范围内
  - 无重复
  - continuous + discrete = 完整特征集

**`_compute_stats() -> None`**
- **作用**: 计算归一化统计量 (仅对continuous features)
- **算法**:
  ```python
  continuous_states = states[..., continuous_indices]  # [N,T,K,9]
  valid_continuous = continuous_states * masks.unsqueeze(-1)

  mean = valid_continuous.sum(dim=(0,1,2)) / n_valid  # [9]
  diff = (valid_continuous - mean) * masks.unsqueeze(-1)
  std = sqrt((diff**2).sum(dim=(0,1,2)) / n_valid)  # [9]
  std = clamp(std, min=1e-6)  # 防止除零
  ```
- **重要**: 只对continuous_indices计算,discrete特征不参与

**`_load_stats(stats_path: str) -> None`**
- **作用**: 从NPZ文件加载预计算的归一化统计量
- **文件内容**:
  ```python
  {
      'mean': [9],  # 仅continuous features
      'std': [9],
      'continuous_indices': [0,1,2,3,4,5,6,9,10],
      'discrete_indices': [7,8,11]
  }
  ```

**`_normalize_data() -> None`**
- **作用**: 对continuous features应用z-score归一化
- **算法**:
  ```python
  continuous_feats = states[..., continuous_indices]  # [N,T,K,9]
  continuous_feats = (continuous_feats - mean) / std
  continuous_feats = continuous_feats * masks.unsqueeze(-1)  # 确保padding=0
  states[..., continuous_indices] = continuous_feats
  # discrete features保持原始整数值不变
  ```

**`_validate_and_clamp_discrete_features() -> None`**
- **作用**: 验证并修正离散特征值
- **操作**:
  1. Clamp负值到0: `discrete_feat = clamp(discrete_feat, min=0)`
  2. 设置padding位置为0: `discrete_feat = discrete_feat * masks`
  3. 检查是否超出范围 (警告但不报错)
- **目的**: 确保离散值是有效的embedding索引

**`save_stats(save_path: str) -> None`**
- **作用**: 保存归一化统计量供val/test使用
- **输出**: `normalization_stats.npz`

**`__len__() -> int`**
- **返回**: episode数量N

**`__getitem__(idx: int) -> Dict`**
- **作用**: 获取单个episode
- **返回**:
  ```python
  {
      'states': [T, K, F],  # float32, 归一化的连续+原始离散
      'masks': [T, K],      # float32
      'scene_id': int,      # int64
      'discrete_features': [T, K, n_discrete]  # int64, 用于embeddings
  }
  ```
- **关键**: 提取discrete_features为LongTensor,方便embedding层使用

#### DataLoader工厂函数

**`get_dataloader(data_path, batch_size=32, shuffle=True, num_workers=0, normalize=True, stats_path=None) -> DataLoader`**
- **作用**: 创建PyTorch DataLoader
- **参数**:
  - `shuffle`: 是否shuffle (train=True, val/test=False)
  - `num_workers`: 多进程加载 (Windows通常用0)
  - `stats_path`: val/test必须提供train的stats路径
- **返回**: 配置好的DataLoader
- **用法**:
  ```python
  train_loader = get_dataloader('train_episodes.npz', shuffle=True)
  val_loader = get_dataloader('val_episodes.npz', stats_path='train_stats.npz', shuffle=False)
  ```

---

## 工具模块

### 📄 `src/utils/common.py`

通用工具函数。

**`parse_discrete_feature_indices_from_metadata(metadata: dict) -> Tuple[List[int], Optional[int], Optional[int], Optional[int]]`**
- **作用**: 从 metadata 中解析离散特征索引 (集中化解析逻辑)
- **参数**: metadata dict (包含 validation_info 字段)
- **返回**: 
  ```python
  (
      discrete_indices: [7, 8, 11],  # sorted list
      idx_lane: 8,                    # lane_id index
      idx_class: 7,                   # class_id index  
      idx_site: 11                    # site_id index
  )
  ```
- **Fallback**: 如果 metadata 缺失字段，返回 `([7, 8, 11], 8, 7, 11)` (默认值)
- **用途**: 所有训练/评估脚本使用此函数统一解析，避免重复代码
- **代码位置**: `src/utils/common.py`
- **示例**:
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
  ```

**`set_seed(seed: int = 42)`**
- **作用**: 设置所有随机种子确保可复现
- **设置**: random, numpy, torch (CPU + CUDA), cudnn

**`count_parameters(model: nn.Module) -> int`**
- **作用**: 统计模型可训练参数数量
- **返回**: 参数总数

**`get_device(device_id: Optional[int] = None) -> torch.device`**
- **作用**: 获取PyTorch设备
- **逻辑**: 优先使用CUDA (如果可用),否则CPU

**`save_config(config: dict, save_path: str)`**
- **作用**: 保存配置字典为JSON

**`load_config(config_path: str) -> dict`**
- **作用**: 从JSON加载配置

**`format_time(seconds: float) -> str`**
- **作用**: 格式化时间为可读字符串 (e.g., "1h 23m 45s")

**`compute_gradient_norm(model: nn.Module) -> float`**
- **作用**: 计算梯度的L2范数
- **用途**: 监控训练稳定性,梯度爆炸检测

**`class EarlyStopping`**
- **作用**: Early stopping工具类
- **方法**:
  - `__init__(patience=10, min_delta=0, mode='min')`: 初始化
  - `__call__(metric_value) -> bool`: 检查是否应停止训练
- **逻辑**: 连续patience个epoch没有改善则early_stop=True

**`get_pixel_to_meter_conversion(lane_geometry_path=None, default_value=0.077) -> float`**
- **作用**: 获取像素到米的转换因子
- **来源**: 从lane_geometry_summary.json读取 (如果存在)
- **默认**: 0.07696103842104474

**`convert_pixels_to_meters(states, pixel_to_meter, position_indices=(0,1), ...) -> Tensor`**
- **作用**: 将像素坐标转换为米
- **转换**:
  - 位置: `pixels * pixel_to_meter`
  - 速度: `pixels/frame * pixel_to_meter` (已考虑dt)
  - 加速度: `pixels/frame^2 * pixel_to_meter` (已考虑dt)

---

### 📄 `src/utils/logger.py`

日志工具。

**`setup_logger(name='world_model', log_dir='./logs', log_file=None, level=logging.INFO) -> logging.Logger`**
- **作用**: 配置logger,同时输出到console和文件
- **配置**:
  - Console handler (stdout)
  - File handler (保存到log_dir)
  - Formatter: `'%(asctime)s - %(name)s - %(levelname)s - %(message)s'`
- **自动命名**: 如果log_file=None,使用`{name}_{timestamp}.log`

---

### 📄 `src/utils/config.py`

配置管理系统 (基于dataclass和YAML)。

#### 配置类

**`@dataclass DataConfig`**
- **字段**: train_path, val_path, episode_length, max_agents, input_dim, normalize, stats_path

**`@dataclass ModelConfig`**
- **字段**: latent_dim, encoder_hidden, encoder_n_heads, encoder_n_layers, dynamics_type, dynamics_hidden, decoder_hidden

**`@dataclass TrainingConfig`**
- **字段**: batch_size, n_epochs, learning_rate, weight_decay, max_grad_norm, scheduler_type, use_amp, use_ddp

**`@dataclass LossConfig`**
- **字段**: reconstruction_weight, prediction_weight, existence_weight, huber_delta

**`@dataclass EvaluationConfig`**
- **字段**: context_length, rollout_length, horizons, eval_frequency, save_visualizations

**`@dataclass LoggingConfig`**
- **字段**: checkpoint_dir, log_dir, save_frequency, use_tensorboard, use_wandb

**`@dataclass ExperimentConfig`**
- **作用**: 完整实验配置
- **包含**: data, model, training, loss, evaluation, logging子配置

#### 配置方法

**`ExperimentConfig.from_yaml(yaml_path: str) -> ExperimentConfig`**
- **作用**: 从YAML文件加载配置

**`ExperimentConfig.from_dict(config_dict: Dict) -> ExperimentConfig`**
- **作用**: 从字典创建配置

**`to_dict() -> Dict`**
- **作用**: 转换为字典

**`to_yaml(save_path: str)`**
- **作用**: 保存为YAML文件

**`to_json(save_path: str)`**
- **作用**: 保存为JSON文件

**`load_config(config_path: str) -> ExperimentConfig`**
- **作用**: 从YAML或JSON加载配置 (自动检测格式)

**`create_default_config(save_path='config.yaml')`**
- **作用**: 创建并保存默认配置模板

---

## 评估模块

### 📄 `src/evaluation/visualization.py`

轨迹可视化工具。

**`visualize_trajectories(predicted, ground_truth, masks, save_path=None, time_step=0, max_agents=20, figsize=(12,8))`**
- **作用**: 可视化单个时间步的预测vs真实轨迹
- **绘制**:
  - 绿色圆圈: Ground truth
  - 红色叉号: Prediction
  - 虚线连接: 误差向量
- **输入**: [T, K, F] arrays

**`visualize_rollout(predicted, ground_truth, masks, save_path='rollout_comparison.png', max_agents=10, figsize=(10,4))`**
- **作用**: 对比完整rollout轨迹
- **布局**: 左图GT,右图Prediction
- **绘制**: 为每个agent画连续轨迹线

**`visualize_error_heatmap(predicted, ground_truth, masks, save_path=None, figsize=(12,6))`**
- **作用**: 绘制误差热力图 (时间×agents)
- **计算**: L2 position error per (time, agent)
- **颜色**: 越红误差越大

**`plot_metrics_over_time(metrics_dict: dict, save_path=None, figsize=(12,8))`**
- **作用**: 绘制metrics随预测horizon变化
- **输入**: `{horizon: {metric_name: value}}`
- **绘制**: 2×2子图,每个metric一条曲线

**`create_animation(predicted, ground_truth, masks, save_path, fps=10, max_agents=20)`**
- **作用**: 创建rollout动画 (GIF或MP4)
- **实现**: 使用matplotlib FuncAnimation
- **显示**:
  - 当前帧位置
  - 历史轨迹trail
  - GT (绿线) vs Pred (红虚线)

---

### 📄 `src/evaluation/attention_visualization.py`

注意力机制可视化和分析。

**`visualize_attention_heatmap(attention_weights: [K,K], save_path=None, vehicle_ids=None, figsize=(12,10))`**
- **作用**: 绘制agent间注意力热力图
- **使用**: seaborn heatmap
- **轴标签**: "Attending From (Query)" vs "Attending To (Key)"

**`visualize_spatial_attention(attention_weights: [K,K], positions: [K,2], query_idx=0, save_path=None, figsize=(10,10))`**
- **作用**: 在空间坐标系中可视化注意力
- **绘制**:
  - Scatter: 所有agents,颜色表示被query agent注意的程度
  - 蓝色星号: Query vehicle
  - 箭头: Top-K注意力连接,宽度∝attention weight
  - 标注: 注意力权重数值

**`analyze_attention_patterns(attention_weights: [B,H,K,K], masks: [B,K], positions: [B,K,2]=None) -> Dict`**
- **作用**: 分析注意力模式,理解模型学习内容
- **分析指标**:
  - `avg_attention_per_head`: 每个head的平均注意力
  - `attention_entropy`: 注意力分布的熵 (高=分散,低=集中)
  - `self_attention_ratio`: 自注意力比例
  - `avg_attended_vehicles`: 平均关注的vehicle数 (阈值>0.1)
  - 如果提供positions:
    - `attention_distance_correlation`: 注意力与距离的相关性
    - `attention_by_distance`: 按距离bin统计的平均注意力

**`plot_attention_statistics(attention_analysis: Dict, save_path=None, figsize=(14,5))`**
- **作用**: 绘制注意力分析统计图
- **布局**: 3个子图
  1. 每个head的平均注意力 (柱状图)
  2. 关键指标: Entropy, Self-Attention, Avg Attended (柱状图)
  3. 注意力vs距离 (折线图)

**`extract_attention_from_model(model, states, masks, layer_idx=0) -> Tensor`**
- **作用**: 从训练好的模型提取注意力权重
- **实现**: 使用hook捕获Transformer层的attention
- **返回**: [B*T, H, K, K] attention weights

**`create_attention_report(model, dataloader, save_dir='./attention_analysis', n_samples=5, device='cpu')`**
- **作用**: 生成完整的注意力分析报告
- **输出**:
  - `attention_stats_sample_{i}.png`: 统计图
  - `spatial_attention_sample_{i}.png`: 空间注意力图
- **样本数**: n_samples个batch

---

## 模型架构模块

### 📄 `src/models/encoder.py`

多智能体编码器,使用Transformer进行per-frame的agent交互建模。

#### MultiAgentEncoder类

**`class MultiAgentEncoder(nn.Module)`**
- **作用**: 将多智能体状态编码为场景级latent表示
- **架构**: 连续特征投影 + 离散特征embedding → Fusion → Agent Transformer → Masked Pooling → Latent

**`__init__(...)`**
- **参数**:
  - `input_dim=12`: 输入特征维度
  - `hidden_dim=256`: 隐藏层维度
  - `latent_dim=256`: 输出latent维度
  - `max_agents=50`: 最大agent数
  - `n_layers=2`: Transformer层数
  - `n_heads=8`: 注意力头数
  - `dropout=0.1`: Dropout率
  - 离散特征配置:
    - `lane_feature_idx=8`: lane_id在features中的索引
    - `class_feature_idx=7`: class_id索引
    - `site_feature_idx=11`: site_id索引
    - `num_lanes=100`: lane vocabulary大小
    - `num_classes=10`: class vocabulary大小
    - `num_sites=10`: site vocabulary大小
    - `lane_embed_dim=16`, `class_embed_dim=8`, `site_embed_dim=8`: embedding维度
- **组件**:
  1. **连续特征投影器** (L69-74):
     ```python
     continuous_projector = Sequential(
         Linear(n_cont=9, hidden_dim=256),
         LayerNorm(256),
         ReLU(),
         Dropout(0.1)
     )
     ```
  2. **离散特征embeddings** (L76-89):
     - `lane_embedding`: nn.Embedding(num_lanes, 16)
     - `class_embedding`: nn.Embedding(num_classes, 8)
     - `site_embedding`: nn.Embedding(num_sites, 8)
  3. **特征融合层** (L91-96):
     ```python
     fusion = Sequential(
         Linear(fused_dim=256+16+8+8=288, hidden_dim=256),
         ReLU(),
         Dropout(0.1)
     )
     ```
  4. **Agent Transformer** (L98-106):
     - TransformerEncoder (d_model=256, nhead=8, dim_feedforward=1024)
     - batch_first=True, norm_first=True (Pre-LN)
     - n_layers=2
  5. **Latent投影** (L108-111):
     ```python
     to_latent = Sequential(
         Linear(256, latent_dim),
         LayerNorm(latent_dim)
     )
     ```

**`forward(states: [B,T,K,F], masks: [B,T,K]) -> [B,T,D]`**
- **流程**:
  1. **维度检查** (L122-131): 验证states为[B,T,K,F], masks为[B,T,K]
  2. **展平时间维度** (L133-136):
     ```python
     states_flat = states.reshape(B*T, K, F)  # [B*T, K, F]
     masks_flat = masks.reshape(B*T, K)       # [B*T, K]
     pad = (masks_flat == 0)                  # [B*T, K] bool
     ```
  3. **连续特征处理** (L138-140):
     ```python
     cont = states_flat[..., continuous_indices]  # [B*T, K, 9]
     cont_emb = continuous_projector(cont)         # [B*T, K, 256]
     ```
  4. **离散特征embedding** (L144-161):
     ```python
     # Lane embedding
     lane_ids = states_flat[..., 8].long()
     lane_ids = lane_ids.clamp(0, num_lanes-1)
     lane_ids = lane_ids.masked_fill(pad, 0)  # padding位置设为0
     lane_emb = lane_embedding(lane_ids)        # [B*T, K, 16]

     # 同样处理class_ids和site_ids
     ```
  5. **特征拼接与融合** (L163-164):
     ```python
     agent_feats = concat([cont_emb, lane_emb, class_emb, site_emb])  # [B*T,K,288]
     agent_feats = fusion(agent_feats)                                 # [B*T,K,256]
     ```
  6. **Agent Transformer** (L166-169):
     ```python
     agent_feats = agent_transformer(
         agent_feats,
         src_key_padding_mask=pad  # True=ignore this agent
     )  # [B*T, K, 256]
     ```
  7. **Masked Mean Pooling** (L171-173):
     ```python
     weights = masks_flat.unsqueeze(-1)  # [B*T, K, 1]
     pooled = (agent_feats * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1e-6)
     # pooled: [B*T, 256]
     ```
  8. **投影到latent空间** (L175):
     ```python
     latent = to_latent(pooled).view(B, T, latent_dim)  # [B, T, D]
     ```
- **返回**: [B, T, latent_dim] 场景级latent表示

---

### 📄 `src/models/dynamics.py`

基于Transformer的时序动力学模型 (Transformer-only)。

#### LatentDynamics类

**`class LatentDynamics(nn.Module)`**
- **作用**: 在latent空间建模时序演化
- **架构**: Positional Encoding → Causal Transformer → Output Projection
- **关键特性**:
  - ✅ Transformer-only (移除了GRU/LSTM)
  - ✅ Causal masking: output[t]只能attend到≤t的历史
  - ✅ 支持learned或sinusoidal位置编码
  - ✅ 支持time padding mask (忽略padding timesteps)

**`__init__(...)`**
- **参数**:
  - `latent_dim=256`: latent空间维度
  - `n_layers=4`: Transformer层数
  - `n_heads=8`: 注意力头数
  - `dropout=0.1`: Dropout率
  - `max_len=512`: 最大序列长度
  - `use_learned_pos_emb=True`: 使用可学习位置编码 (vs sinusoidal)
- **组件**:
  1. **位置编码** (L45-49):
     ```python
     if use_learned_pos_emb:
         pos_emb = Parameter(torch.zeros(1, max_len, latent_dim))
         nn.init.normal_(pos_emb, mean=0.0, std=0.02)
     else:
         pos_emb = _build_sinusoidal_pos_emb(max_len, latent_dim)
     ```
  2. **Transformer Encoder** (L51-59):
     - TransformerEncoderLayer (d_model=latent_dim, nhead=8, dim_feedforward=4*latent_dim)
     - batch_first=True, norm_first=True (Pre-LN)
     - num_layers=4
  3. **输出投影** (L62-65):
     ```python
     output_proj = Sequential(
         LayerNorm(latent_dim),
         Linear(latent_dim, latent_dim)
     )
     ```

**`_build_sinusoidal_pos_emb(max_len, d_model) -> [1, max_len, d_model]`** (静态方法, L67-75)
- **作用**: 构建sinusoidal位置编码
- **算法**:
  ```python
  position = arange(max_len).unsqueeze(1)  # [max_len, 1]
  div_term = exp(arange(0, d_model, 2) * (-log(10000.0) / d_model))
  pe[:, 0::2] = sin(position * div_term)
  pe[:, 1::2] = cos(position * div_term)
  ```

**`_causal_mask(T, device, dtype) -> [T, T]`** (静态方法, L77-86)
- **作用**: 生成causal attention mask
- **算法**:
  ```python
  mask = full((T, T), -inf)
  mask = triu(mask, diagonal=1)  # Upper triangular (excluding diagonal) = -inf
  ```
- **含义**: 位置t只能attend到位置≤t

**`forward(latent: [B,T,D], hidden=None, time_padding_mask: [B,T]=None) -> Tuple[[B,T,D], None]`**
- **流程**:
  1. **添加位置编码** (L114):
     ```python
     x = latent + pos_emb[:, :T, :]  # [B, T, D]
     ```
  2. **生成causal mask** (L116):
     ```python
     causal = _causal_mask(T, device, dtype)  # [T, T]
     ```
  3. **Transformer forward** (L118-122):
     ```python
     out = transformer(
         x,
         mask=causal,  # causal attention
         src_key_padding_mask=time_padding_mask  # [B,T] 忽略padding timesteps
     )  # [B, T, D]
     ```
  4. **输出投影** (L124):
     ```python
     out = output_proj(out)  # [B, T, D]
     ```
- **返回**: (predicted_latent [B,T,D], None)
- **注**: predicted_latent[:, t]预测时刻t+1的latent (one-step-ahead)

**`step(latent_history: [B,T,D], time_padding_mask=None, max_context=None) -> [B,D]`** (@torch.no_grad(), L127-154)
- **作用**: 使用完整(或truncated)历史进行单步预测
- **用途**: Rollout时使用
- **流程**:
  1. **Truncate历史** (L148-151):
     ```python
     if max_context and latent_history.size(1) > max_context:
         latent_history = latent_history[:, -max_context:, :]
         time_padding_mask = time_padding_mask[:, -max_context:]
     ```
  2. **Forward** (L153):
     ```python
     pred, _ = forward(latent_history, time_padding_mask=time_padding_mask)
     ```
  3. **返回最后一个token的预测** (L154):
     ```python
     return pred[:, -1, :]  # [B, D] 预测下一步latent
     ```

---

### 📄 `src/models/decoder.py`

状态解码器,从latent解码为状态和存在性。

#### StateDecoder类

**`class StateDecoder(nn.Module)`**
- **作用**: 将latent解码为agent states和existence logits
- **架构**: MLP Backbone → State Head + Existence Head + (可选)Residual XY Head
- **关键特性**:
  - ✅ 输出绝对状态 (在归一化空间)
  - ✅ 存在性logits (用sigmoid转为概率)
  - ✅ 可选(x,y)残差头 (用于物理先验修正)

**`__init__(...)`**
- **参数**:
  - `latent_dim=256`: 输入latent维度
  - `hidden_dim=256`: 隐藏层维度
  - `output_dim=12`: 输出状态维度 (F=12)
  - `max_agents=50`: 最大agent数
  - `dropout=0.1`: Dropout率
  - `enable_xy_residual=True`: 是否启用(x,y)残差头
- **组件**:
  1. **MLP Backbone** (L34-42):
     ```python
     backbone = Sequential(
         Linear(latent_dim, hidden_dim),
         LayerNorm(hidden_dim),
         ReLU(),
         Dropout(0.1),
         Linear(hidden_dim, hidden_dim),
         ReLU(),
         Dropout(0.1)
     )
     ```
  2. **State Head** (L45):
     ```python
     state_head = Linear(hidden_dim, max_agents * output_dim)
     ```
  3. **Existence Head** (L48):
     ```python
     existence_head = Linear(hidden_dim, max_agents)
     ```
  4. **Residual XY Head** (L51-57, 可选):
     ```python
     if enable_xy_residual:
         residual_xy_head = Linear(hidden_dim, max_agents * 2)
         # ✅ 初始化为0: 从纯物理先验开始学习
         nn.init.zeros_(residual_xy_head.weight)
         nn.init.zeros_(residual_xy_head.bias)
     ```

**`forward(latent: [B,T,D], return_residual_xy=False) -> Tuple[[B,T,K,F], [B,T,K], Optional[[B,T,K,2]]]`**
- **流程**:
  1. **Backbone** (L81):
     ```python
     h = backbone(latent)  # [B, T, hidden_dim]
     ```
  2. **State预测** (L83):
     ```python
     states = state_head(h).view(B, T, max_agents, output_dim)  # [B,T,K,F]
     ```
  3. **Existence logits** (L84):
     ```python
     existence_logits = existence_head(h)  # [B, T, K]
     ```
  4. **可选: Residual XY** (L86-90):
     ```python
     if return_residual_xy:
         residual_xy = residual_xy_head(h).view(B, T, max_agents, 2)  # [B,T,K,2]
     else:
         residual_xy = None
     ```
- **返回**: (states, existence_logits, residual_xy)

---

### 📄 `src/models/world_model.py`

完整的World Model: Encoder → Transformer Dynamics → Decoder (with Kinematic Prior)。

#### WorldModel类

**`class WorldModel(nn.Module)`**
- **作用**: 组装完整的world model
- **架构**: MultiAgentEncoder → LatentDynamics (Transformer) → StateDecoder
- **核心创新**: 🔥 **Kinematic Prior + Residual** 用于(x,y)预测

**`__init__(...)`**
- **参数**:
  - `input_dim=12`, `max_agents=50`, `latent_dim=256`
  - `dynamics_layers=4`, `dynamics_heads=8`
  - `dt=1.0/30`: 时间步长 (秒)
  - `max_dynamics_len=512`: Transformer最大序列长度
  - `max_dynamics_context=128`: Rollout时截断上下文长度
  - 特征索引: `idx_x=0`, `idx_y=1`, `idx_vx=2`, `idx_vy=3`, `idx_ax=4`, `idx_ay=5`
  - 🔥 `idx_angle=6`: **angle特征索引** (新增)
  - `use_acceleration=True`: 是否使用加速度
  - Embedding配置: `num_lanes`, `num_sites`, `num_classes`, 各自的`embed_dim`
- **组件初始化**:
  1. **Encoder** (L71-85): MultiAgentEncoder
  2. **Dynamics** (L88-93): LatentDynamics (Transformer-only)
  3. **Decoder** (L96-102): StateDecoder (enable_xy_residual=True, 🔥 enable_angle_head=True)
  4. **Normalization buffers** (L105-108):
     ```python
     register_buffer("norm_mean_cont", zeros(1))
     register_buffer("norm_std_cont", ones(1))
     register_buffer("cont_index_map", full((input_dim,), -1, dtype=long))
     ```

**`set_normalization_stats(mean_cont, std_cont, continuous_indices) -> None`** (L110-126)
- **作用**: 设置归一化统计量 (用于kinematic prior的denorm/renorm)
- **参数**:
  - `mean_cont`: [n_continuous] 连续特征的mean
  - `std_cont`: [n_continuous] 连续特征的std
  - `continuous_indices`: 连续特征的索引列表
- **逻辑**:
  ```python
  norm_mean_cont = tensor(mean_cont)  # [n_cont]
  norm_std_cont = tensor(std_cont).clamp(min=1e-6)

  # 创建feature_idx → continuous_idx的映射
  cont_index_map = full((input_dim,), -1)
  for j, feat_idx in enumerate(continuous_indices):
      cont_index_map[feat_idx] = j
  ```

**`_require_stats(feat_idx) -> (mean, std)`** (L128-140)
- **作用**: 获取指定特征的归一化统计量
- **返回**: (mean标量, std标量)

**`_denorm(x_norm, feat_idx) -> x_raw`** (L142-144)
- **作用**: 反归一化: `x_raw = x_norm * std + mean`

**`_renorm(x_raw, feat_idx) -> x_norm`** (L146-148)
- **作用**: 归一化: `x_norm = (x_raw - mean) / std`

**`_kinematic_prior_xy(prev_states: [B,T,K,F]) -> [B,T,K,2]`** (L150-171)
- **作用**: 🔥 计算运动学先验 (在原始空间)
- **算法**:
  ```python
  # 1. Denormalize到原始空间
  x = _denorm(prev_states[..., idx_x], idx_x)
  y = _denorm(prev_states[..., idx_y], idx_y)
  vx = _denorm(prev_states[..., idx_vx], idx_vx)
  vy = _denorm(prev_states[..., idx_vy], idx_vy)

  # 2. 应用运动学方程
  if use_acceleration:
      ax = _denorm(prev_states[..., idx_ax], idx_ax)
      ay = _denorm(prev_states[..., idx_ay], idx_ay)
      x_next = x + vx*dt + 0.5*ax*dt^2
      y_next = y + vy*dt + 0.5*ay*dt^2
  else:
      x_next = x + vx*dt
      y_next = y + vy*dt

  # 3. Renormalize回归一化空间
  x_next_norm = _renorm(x_next, idx_x)
  y_next_norm = _renorm(y_next, idx_y)
  return stack([x_next_norm, y_next_norm], dim=-1)  # [B,T,K,2]
  ```
- **关键**: 物理计算在原始空间进行,确保正确性

🔥 **`_kinematic_prior_angle(prev_states: [B,T,K,F]) -> [B,T,K]`** (新增)
- **作用**: 计算angle的物理先验 (基于速度方向)
- **算法**:
  ```python
  # 1. Denormalize速度
  vx = _denorm(prev_states[..., idx_vx], idx_vx)
  vy = _denorm(prev_states[..., idx_vy], idx_vy)
  
  # 2. 计算速度方向角 (即朝向角的先验)
  angle_prior = torch.atan2(vy, vx)  # [-π, π]
  
  # 3. 不需要 renormalize (因为 angle 不被归一化)
  return angle_prior  # [B,T,K]
  ```
- **物理意义**: 车辆朝向 ≈ 速度方向 (很强的物理约束)
- **处理边界**: 当 `vx ≈ 0, vy ≈ 0` 时 `atan2` 仍然有定义 (返回0)
- **使用**: 在 `forward()` 中与 decoder 预测混合

**`forward(states: [B,T,K,F], masks: [B,T,K]) -> Dict`** (L173-215)
- **作用**: 模型前向传播
- **流程**:
  1. **编码** (L190):
     ```python
     latent = encoder(states, masks)  # [B, T, D]
     ```
  2. **Time padding mask** (L193):
     ```python
     time_padding = (masks.sum(dim=-1) == 0)  # [B, T] bool
     # True=该时间步所有agent都不存在
     ```
  3. **Dynamics预测** (L195):
     ```python
     predicted_latent, _ = dynamics(latent, time_padding_mask=time_padding)
     ```
  4. **重建分支解码** (L197):
     ```python
     recon_states, exist_logits, _ = decoder(latent, return_residual_xy=False)
     ```
  5. **预测分支解码 (with residual)** (L198):
     ```python
     pred_states_base, pred_exist_logits, residual_xy = decoder(
         predicted_latent,
         return_residual_xy=True
     )
     ```
  6. **🔥 应用Kinematic Prior + Residual** (L200-207):
     ```python
     pred_states = pred_states_base.clone()
     if residual_xy is not None:
         prior_xy = _kinematic_prior_xy(states)  # [B,T,K,2] t预测t+1的prior
         residual_xy = residual_xy * masks.unsqueeze(-1)  # mask padding
         pred_states[..., idx_x] = prior_xy[..., 0] + residual_xy[..., 0]
         pred_states[..., idx_y] = prior_xy[..., 1] + residual_xy[..., 1]
     # 其他特征直接使用decoder输出
     ```
- **返回**:
  ```python
  {
      "latent": [B,T,D],
      "reconstructed_states": [B,T,K,F],
      "predicted_states": [B,T,K,F],  # with prior+residual
      "existence_logits": [B,T,K],
      "predicted_existence_logits": [B,T,K]
  }
  ```

**`rollout(initial_states: [B,T0,K,F], initial_masks: [B,T0,K], n_steps=20, threshold=0.5, teacher_forcing=False, ground_truth_states=None) -> Tuple[[B,n_steps,K,F], [B,n_steps,K]]`** (@torch.no_grad(), L217-296)
- **作用**: 🚀 Open-loop rollout预测
- **参数**:
  - `initial_states/masks`: Context (通常T0=65)
  - `n_steps`: 预测步数 (通常H=15)
  - `threshold=0.5`: 存在性阈值
  - `teacher_forcing`: 是否使用ground truth作为prev
  - `ground_truth_states`: [B,T0+n_steps,K,F] (teacher forcing时需要)
- **流程**:
  1. **编码context** (L243-244):
     ```python
     latent_ctx = encoder(initial_states, initial_masks)  # [B,T0,D]
     time_padding = (initial_masks.sum(dim=-1) == 0)
     ```
  2. **Dynamics预测context** (L246-247):
     ```python
     pred_latent_ctx, _ = dynamics(latent_ctx, time_padding_mask=time_padding)
     current_latent = pred_latent_ctx[:, -1:, :]  # [B,1,D] 最后一步的预测
     ```
  3. **初始化历史和状态** (L249-250):
     ```python
     latent_hist = latent_ctx  # [B, T0, D]
     prev_state = initial_states[:, -1:, :, :]  # [B, 1, K, F]
     ```
  4. **Autoregressive rollout循环** (L255-292):
     ```python
     for step in range(n_steps):
         # a. 解码当前latent
         base_states, exist_logits, residual_xy = decoder(
             current_latent, return_residual_xy=True
         )
         pred_state = base_states.clone()

         # b. 🔥 应用kinematic prior
         prior_xy = _kinematic_prior_xy(prev_state)  # 基于prev_state预测
         if residual_xy:
             pred_state[..., idx_x] = prior_xy[..., 0] + residual_xy[..., 0]
             pred_state[..., idx_y] = prior_xy[..., 1] + residual_xy[..., 1]

         # c. 存在性mask
         exist_prob = sigmoid(exist_logits)
         pred_mask = (exist_prob > threshold).float()

         out_states.append(pred_state)
         out_masks.append(pred_mask)

         # d. 决定下一步的"prev_state"
         if teacher_forcing and ground_truth_states:
             gt_state = ground_truth_states[:, T0+step:T0+step+1, :, :]
             prev_state = gt_state
             current_latent = encoder(gt_state, gt_mask)
         else:
             prev_state = pred_state * pred_mask.unsqueeze(-1)

         # e. 累积latent历史,预测下一步
         latent_hist = cat([latent_hist, current_latent], dim=1)
         next_latent = dynamics.step(
             latent_hist,
             max_context=max_dynamics_context  # 128 truncate
         ).view(B, 1, -1)
         current_latent = next_latent
     ```
  5. **拼接输出** (L294-296):
     ```python
     predicted_states = cat(out_states, dim=1)  # [B, n_steps, K, F]
     predicted_masks = cat(out_masks, dim=1)    # [B, n_steps, K]
     ```
- **返回**: (predicted_states, predicted_masks)
- **关键特性**:
  - 使用`dynamics.step()`进行单步预测
  - Truncated context (max_context=128) 避免内存爆炸
  - 每步应用kinematic prior + residual

---

---

## 训练模块

### 📄 `src/training/losses.py`

World Model的Loss函数实现。

#### WorldModelLoss类

**`class WorldModelLoss(nn.Module)`**
- **作用**: 计算world model的总loss
- **组成**: Reconstruction Loss + Prediction Loss + Existence Loss
- **关键**: ⚠️ **只对continuous features计算回归loss,discrete features不参与**

**`__init__(...)`**
- **参数**:
  - `recon_weight=1.0`: 重建loss权重
  - `pred_weight=1.0`: 预测loss权重
  - `exist_weight=0.1`: 存在性loss权重
  - `huber_beta=1.0`: Huber loss的beta参数
  - `continuous_indices`: **关键**! 连续特征索引列表 (e.g., [0,1,2,3,4,5,6,9,10])
  - `use_pred_existence_loss=True`: 是否计算预测分支的存在性loss

**`_masked_huber_loss(pred: [B,T,K,F], target: [B,T,K,F], mask: [B,T,K]) -> scalar`** (L39-57)
- **作用**: 计算masked Huber loss (仅对continuous features)
- **算法**:
  ```python
  # 1. 过滤到continuous features
  if continuous_indices is not None:
      pred = pred[..., continuous_indices]     # [B,T,K,9]
      target = target[..., continuous_indices] # [B,T,K,9]

  # 2. SmoothL1 Loss (Huber)
  diff = pred - target
  abs_diff = diff.abs()
  beta = huber_beta  # 1.0
  loss = where(
      abs_diff < beta,
      0.5 * (diff ** 2) / beta,  # 小误差: quadratic
      abs_diff - 0.5 * beta       # 大误差: linear (robust to outliers)
  )

  # 3. 应用mask
  loss = loss * mask.unsqueeze(-1)  # [B,T,K,9]

  # 4. 归一化
  denom = mask.sum() * loss.shape[-1]  # 有效agent数 × feature数
  return loss.sum() / denom.clamp(min=1.0)
  ```
- **为什么Huber**: 相比MSE,对outliers更robust

**`_existence_loss(logits: [B,T,K], mask: [B,T,K]) -> scalar`** (L59-65)
- **作用**: 计算存在性BCE loss
- **算法**:
  ```python
  loss = BCEWithLogitsLoss(logits, mask)
  # mask: 1=agent存在, 0=padding
  return loss.mean()
  ```

🔥 **`_angular_distance(pred_angle: [B,T,K], target_angle: [B,T,K]) -> [B,T,K]`** (static, 新增)
- **作用**: 计算周期性角度距离 (处理 `-π` 和 `π` 的等价性)
- **算法**:
  ```python
  diff = pred_angle - target_angle  # 可能超出 [-π, π]
  
  # 将差值映射到 [-π, π]
  distance = torch.atan2(torch.sin(diff), torch.cos(diff))
  # atan2(sin, cos) 自动处理周期性
  
  return distance.abs()  # [B,T,K] 非负距离
  ```
- **例子**:
  - `pred=3.0, target=-3.0`: 传统L1=6.0, angular distance=0.28 ✅
  - `pred=0.0, target=3.14`: 传统L1=3.14, angular distance=3.14 ✅
  - `pred=-3.1, target=3.1`: 传统L1=6.2, angular distance=0.08 ✅
- **优势**: 正确处理角度的周期性，避免梯度爆炸

🔥 **`_angular_loss(pred_angle: [B,T,K], target_angle: [B,T,K], mask: [B,T,K]) -> scalar`** (新增)
- **作用**: 计算masked angular distance loss
- **算法**:
  ```python
  distance = _angular_distance(pred_angle, target_angle)  # [B,T,K]
  
  # 应用mask
  masked_distance = distance * mask  # [B,T,K]
  
  # 平均
  loss = masked_distance.sum() / mask.sum().clamp(min=1.0)
  return loss
  ```
- **返回**: 平均角度误差 (弧度)

**`forward(predictions: Dict, targets: Dict) -> Dict`** (L67-109)
- **作用**: 计算总loss和各分项
- **输入**:
  - `targets`: `{'states': [B,T,K,F], 'masks': [B,T,K]}`
  - `predictions`: 从WorldModel.forward()的输出
- **流程**:
  1. **重建loss** (L86): 对齐t与t
     ```python
     recon_loss = _masked_huber_loss(
         reconstructed_states,  # [B, T, K, F]
         states,                # [B, T, K, F]
         masks                  # [B, T, K]
     )
     ```
  2. **预测loss** (L89): t预测t+1,忽略最后一步
     ```python
     pred_loss = _masked_huber_loss(
         pred_states[:, :-1],   # [B, T-1, K, F] 预测: t=0到T-2
         states[:, 1:],         # [B, T-1, K, F] 目标: t=1到T-1
         masks[:, :-1]          # [B, T-1, K]
     )
     ```
     **关键**: 时间对齐! pred_states[:, t]预测states[:, t+1]
  3. **存在性loss (重建分支)** (L91):
     ```python
     exist_loss = _existence_loss(existence_logits, masks)
     ```
  4. **存在性loss (预测分支)** (L93-95):
     ```python
     if use_pred_existence_loss:
         pred_exist_loss = _existence_loss(
             predicted_existence_logits[:, :-1],  # t=0到T-2
             masks[:, 1:]                         # t=1到T-1
         )
     ```
  5. **总loss** (L97-101):
     ```python
     total = (
         recon_weight * recon_loss +
         pred_weight * pred_loss +
         exist_weight * (exist_loss + pred_exist_loss)
     )
     ```
- **返回**:
  ```python
  {
      "total_loss": total,                      # 用于backward
      "recon_loss": recon_loss.detach(),        # 监控用
      "pred_loss": pred_loss.detach(),
      "exist_loss": exist_loss.detach(),
      "pred_exist_loss": pred_exist_loss.detach()
  }
  ```

**为什么只对continuous features计算loss**:
```
离散特征 (7=class_id, 8=lane_id, 11=site_id):
- 是类别变量,不应该用回归loss (Huber/MSE)
- 模型通过embedding层学习这些特征
- 回归loss会把整数当连续值优化,误导学习

连续特征 (0-6, 9-10):
- center_x, center_y, vx, vy, ax, ay, angle, has_preceding, has_following
- 适合回归任务
- Huber loss robust to outliers
```

---

### 📄 `src/training/train_world_model.py`

主训练脚本 (Transformer-only)。

#### 主要函数

**`parse_args() -> argparse.Namespace`** (L30-56)
- **作用**: 解析命令行参数
- **参数**:
  - 数据: `--train_data`, `--val_data`, `--checkpoint_dir`
  - 训练: `--epochs=50`, `--batch_size=16`, `--lr=3e-4`, `--weight_decay=1e-4`, `--grad_clip=1.0`
  - 模型: `--input_dim=12`, `--max_agents=50`, `--latent_dim=256`
  - Dynamics: `--dynamics_layers=4`, `--dynamics_heads=8`, `--max_dynamics_len=512`, `--max_dynamics_context=128`
  - 设备: `--device` (auto-detect CUDA)

**`save_checkpoint(path, model, optimizer, epoch) -> None`** (L59-68)
- **作用**: 保存训练checkpoint
- **保存内容**:
  ```python
  {
      "epoch": epoch,
      "model_state_dict": model.state_dict(),
      "optimizer_state_dict": optimizer.state_dict()
  }
  ```
- **注**: normalization stats单独保存为`normalization_stats.npz`

**`evaluate(model, loader, loss_fn, device) -> Dict[str, float]`** (@torch.no_grad(), L71-87)
- **作用**: 在validation set上评估
- **流程**:
  ```python
  model.eval()
  totals = {"total_loss": 0, "recon_loss": 0, "pred_loss": 0, ...}

  for batch in loader:
      states, masks = batch["states"].to(device), batch["masks"].to(device)
      preds = model(states, masks)
      losses = loss_fn(preds, {"states": states, "masks": masks})

      # 累积loss
      for k in totals:
          totals[k] += losses[k].item() * batch_size

  # 平均
  for k in totals:
      totals[k] /= total_samples

  return totals
  ```
- **返回**: `{total_loss, recon_loss, pred_loss, exist_loss, pred_exist_loss}`

**`main() -> None`** (L90-201)
- **作用**: 主训练循环
- **流程**:

  1. **解析参数** (L91-94):
     ```python
     args = parse_args()
     ckpt_dir = Path(args.checkpoint_dir)
     ckpt_dir.mkdir(parents=True, exist_ok=True)
     stats_path = ckpt_dir / "normalization_stats.npz"
     ```

  2. **创建TRAIN DataLoader并计算stats** (L96-106):
     ```python
     train_loader = get_dataloader(
         args.train_data,
         batch_size=args.batch_size,
         shuffle=True,
         normalize=True,
         stats_path=None  # ← 首次运行,自动计算
     )

     # 保存stats供VAL/TEST复用
     if not stats_path.exists():
         train_loader.dataset.save_stats(str(stats_path))
     ```
     **关键**: 只计算一次stats (从TRAIN),VAL/TEST复用!

  3. **创建VAL DataLoader (复用TRAIN stats)** (L108-116):
     ```python
     val_loader = get_dataloader(
         args.val_data,
         batch_size=args.batch_size,
         shuffle=False,
         normalize=True,
         stats_path=str(stats_path)  # ← 使用TRAIN的stats
     )
     ```

  4. **从metadata读取配置** (L118-123):
     ```python
     meta = train_loader.dataset.metadata
     dt = float(meta.get("dt", 1.0/30.0))
     num_lanes = int(meta.get("num_lanes", 100))
     num_sites = int(meta.get("num_sites", 10))
     num_classes = int(meta.get("num_classes", 10))
     ```

  5. **创建WorldModel** (L127-140):
     ```python
     model = WorldModel(
         input_dim=args.input_dim,
         max_agents=args.max_agents,
         latent_dim=args.latent_dim,
         dynamics_layers=args.dynamics_layers,
         dynamics_heads=args.dynamics_heads,
         dt=dt,
         max_dynamics_len=args.max_dynamics_len,
         max_dynamics_context=args.max_dynamics_context,
         num_lanes=num_lanes,
         num_sites=num_sites,
         num_classes=num_classes,
         use_acceleration=bool(meta.get("use_acceleration", True)),
     ).to(device)
     ```

  6. **🔥 设置normalization stats到model** (L142-147):
     ```python
     model.set_normalization_stats(
         train_loader.dataset.mean,
         train_loader.dataset.std,
         train_loader.dataset.continuous_indices
     )
     ```
     **关键**: kinematic prior需要这些stats来denorm/renorm!

  7. **创建Loss函数** (L149-156):
     ```python
     loss_fn = WorldModelLoss(
         recon_weight=1.0,
         pred_weight=1.0,
         exist_weight=0.1,
         huber_beta=1.0,
         continuous_indices=train_loader.dataset.continuous_indices,  # ← 关键!
         use_pred_existence_loss=True
     )
     ```

  8. **创建Optimizer** (L158):
     ```python
     optimizer = optim.AdamW(
         model.parameters(),
         lr=args.lr,
         weight_decay=args.weight_decay
     )
     ```

  9. **训练循环** (L162-199):
     ```python
     best_val = float("inf")

     for epoch in range(args.epochs):
         model.train()
         pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
         running_loss = 0.0
         n_samples = 0

         for batch in pbar:
             states = batch["states"].to(device)
             masks = batch["masks"].to(device)

             # Forward
             optimizer.zero_grad(set_to_none=True)
             preds = model(states, masks)
             losses = loss_fn(preds, {"states": states, "masks": masks})
             loss = losses["total_loss"]

             # Backward
             loss.backward()
             if args.grad_clip > 0:
                 clip_grad_norm_(model.parameters(), args.grad_clip)
             optimizer.step()

             # 更新进度条
             bs = states.size(0)
             running_loss += loss.item() * bs
             n_samples += bs
             pbar.set_postfix(loss=running_loss / n_samples)

         # Validation
         val_metrics = evaluate(model, val_loader, loss_fn, device)
         val_loss = val_metrics["total_loss"]

         # 打印
         print(f"[Epoch {epoch+1}] train_loss={running_loss/n_samples:.4f}  "
               f"val_loss={val_loss:.4f}  recon={val_metrics['recon_loss']:.4f} "
               f"pred={val_metrics['pred_loss']:.4f} exist={val_metrics['exist_loss']:.4f} "
               f"pred_exist={val_metrics['pred_exist_loss']:.4f}")

         # 保存checkpoints
         save_checkpoint(ckpt_dir / "checkpoint_last.pt", model, optimizer, epoch)

         if val_loss < best_val:
             best_val = val_loss
             save_checkpoint(ckpt_dir / "checkpoint_best.pt", model, optimizer, epoch)

     print("Training finished.")
     ```

**训练输出示例**:
```
[Epoch 1] train_loss=12.3456  val_loss=13.4567  recon=10.234 pred=2.345 exist=0.123 pred_exist=0.098
[Epoch 2] train_loss=10.1234  val_loss=11.2345  recon=8.456 pred=1.987 exist=0.112 pred_exist=0.089
...
[Epoch 50] train_loss=3.4567  val_loss=4.1234  recon=2.345 pred=0.987 exist=0.098 pred_exist=0.087
```

**完整训练命令示例**:
```bash
python src/training/train_world_model.py \
    --train_data data/processed/train_episodes.npz \
    --val_data data/processed/val_episodes.npz \
    --checkpoint_dir checkpoints/world_model \
    --input_dim 12 \
    --latent_dim 256 \
    --dynamics_layers 4 \
    --dynamics_heads 8 \
    --batch_size 16 \
    --epochs 50 \
    --lr 3e-4 \
    --weight_decay 1e-4 \
    --grad_clip 1.0
```

**关键设计**:
1. ✅ **ONE normalization stats**: 只从TRAIN计算,VAL/TEST复用
2. ✅ **Continuous indices**: 传递给loss,确保只回归连续特征
3. ✅ **Normalization stats设置到model**: kinematic prior需要
4. ✅ **Transformer-only**: 移除GRU/LSTM选项,简化架构
5. ✅ **Gradient clipping**: 防止梯度爆炸 (clip_norm=1.0)
6. ✅ **AdamW optimizer**: weight decay regularization

---

---

## 评估模块 (续)

### 📄 `src/evaluation/rollout_eval.py`

(待读取并补充)

### 📄 `src/evaluation/prediction_metrics.py`

(待读取并补充)

---

## 附录

### 特征布局 (12维)

```
[0]  center_x         → continuous (z-score)
[1]  center_y         → continuous (z-score)
[2]  vx               → continuous
[3]  vy               → continuous
[4]  ax               → continuous
[5]  ay               → continuous
[6]  angle            → continuous
[7]  class_id         → discrete (DO NOT normalize, use embedding)
[8]  lane_id          → discrete (DO NOT normalize, use embedding)
[9]  has_preceding    → continuous (binary 0/1)
[10] has_following    → continuous (binary 0/1)
[11] site_id          → discrete (DO NOT normalize, use embedding)
```

### 重要常量

- **FPS**: 30.0 (帧率)
- **dt**: 1/30 ≈ 0.0333 秒 (时间步长)
- **T**: 80 帧 (episode长度, ~2.67秒)
- **C**: 65 帧 (context长度, ~2.17秒)
- **H**: 15 帧 (rollout horizon, ~0.50秒)
- **S (stride)**: 15 帧 (episode间隔, ~0.50秒)
- **K (max_vehicles)**: 50
- **F (n_features)**: 12

### 数据流

```
原始CSV文件
  ↓ (preprocess_multisite.py)
全局时间线 → 连续段检测 → 固定stride episodes
  ↓
NPZ文件 [N, T=80, K=50, F=12]
  ↓ (dataset.py)
归一化 (continuous only) + Discrete validation
  ↓ (DataLoader)
Batch [B, T, K, F]
  ↓ (WorldModel)
Encoder → Transformer Dynamics → Decoder (+ Kinematic Prior)
  ↓
预测states [B, T, K, F]
```

---

**文档生成时间**: 2025-12-14
**项目版本**: v1.0
**状态**: 部分完成 (模型、训练、评估模块待补充)
