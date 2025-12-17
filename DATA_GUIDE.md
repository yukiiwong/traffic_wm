"""Raw data guide.

This document focuses on the *raw CSV* schema and the assumptions used by the
preprocessing pipeline.
"""

# 原始数据说明 (Raw Data Guide)

## 目录结构

```
data/raw/
    A/ ... I/   # Site A ~ Site I
```

## CSV 列定义（常见 22 列）

每个 CSV 文件通常包含如下字段（不同版本可能有增减，以实际文件为准）:

| 列名 | 类型 | 说明 |
|------|------|------|
| track_id | number | 车辆轨迹 ID（文件内唯一；跨文件可能重复） |
| frame | int | 帧编号 |
| center_x, center_y | float | 车辆中心点坐标（像素） |
| width, height | float | bbox 尺寸（像素） |
| angle | float | 车辆朝向角（弧度） |
| x1..y4 | float | 旋转 bbox 角点（像素） |
| confidence | float | 检测置信度 |
| class_id | number | 类别 ID |
| site | string | 站点（如 "Site A"） |
| lane | string | 车道标识（如 "B5" / "crossroads1"） |
| preceding_id, following_id | number | 前车/后车 track_id（可为空） |
| timestamp | string | 时间戳（可选） |

## 时间与单位

- 帧率: 30 FPS
- 时间步长: $dt = 1/30 \approx 0.0333$ 秒
- 坐标单位: 像素（pixel）

## 常见数据现象（会影响训练与指标）

### 1) 车辆的短暂消失与重现（mask gap）

检测失败/遮挡会导致某些轨迹在若干帧缺失。预处理会用 `masks` 标记有效性。

注意:
- 如果在缺失段前后直接做差分（padding → real），会产生极大的伪速度/伪加速度。
- 仓库的预处理对速度/加速度差分会按真实帧间隔进行时间尺度修正（frame gap * dt），用于缓解缺帧带来的速度爆炸。

### 2) track_id 的跨文件冲突

同一个站点的不同 drone CSV 之间，`track_id` 可能重复。预处理会构造全局 track id（例如 file_id 偏移）来避免冲突。

## 预处理产物（概念层）

预处理会生成 episode 张量:
- `states`: `[N, T, K, F]`
- `masks`: `[N, T, K]`
- `scene_ids`: `[N]`

具体的特征布局与离散特征索引以 `metadata.json` 为准。

**问题**: CSV文件之间可能存在时间间隙

**解决方案**:
- 自动检测 gap > 1 frame的位置
- 分割为连续段（segments）
- Episodes不跨越间隙提取

### 5. 车道编码

**问题**: 不同站点的车道名称可能冲突（如都有"A1"）

**解决方案**:
- 使用站点特定的车道token: `"site:lane"`
- 例如: `"A:A1"`, `"B:A1"` 不会冲突

---

## 📋 预处理流程

我们的预处理pipeline (`preprocess_multisite.py`) 处理这些原始数据的步骤：

### Step 1: 站点级全局时间轴构建

```python
# 对每个站点 (A-I):
for site in ['A', 'B', ..., 'I']:
    # 1. 收集该站点所有CSV文件
    csv_files = glob(f'data/raw/{site}/*.csv')

    # 2. 按顺序拼接，构建全局时间轴
    #    - global_frame = frame + file_offset
    #    - global_track_id = file_id * 1M + track_id

    # 3. 检测时间间隙，分割连续段
    #    - gap > 1 frame → 分割点

    # 4. 在连续段内提取episodes
```

### Step 2: 固定步长Episode提取

```python
# 参数:
T = 80          # Episode长度（帧）
stride = 15     # Episode间隔（帧）

# 提取逻辑:
for start_frame in range(segment_start, segment_end - T + 1, stride):
    episode = extract_episode(start_frame, T)
    # episode.shape = [T=80, K=50, F=12]
```

### Step 3: 时序划分（Chronological Split）

```python
# Scheme A: 先确定frame cutoffs，再独立提取
train_cutoff = total_frames * 0.70
val_cutoff = total_frames * 0.85

# 在各split的frame范围内独立提取episodes
train_episodes = extract_episodes(0, train_cutoff)
val_episodes = extract_episodes(train_cutoff, val_cutoff)
test_episodes = extract_episodes(val_cutoff, total_frames)

# ✅ 保证时间不重叠
```

### Step 4: 特征工程

从原始22列提取12维特征：

```python
features = [
    center_x,           # [0] 位置X（归一化）
    center_y,           # [1] 位置Y（归一化）
    vx,                 # [2] 速度X（计算得到）
    vy,                 # [3] 速度Y（计算得到）
    ax,                 # [4] 加速度X（计算得到）
    ay,                 # [5] 加速度Y（计算得到）
    angle,              # [6] 朝向角度
    class_id,           # [7] 车辆类别（离散）
    lane_id,            # [8] 车道ID（离散，编码为整数）
    has_preceding,      # [9] 是否有前车（0/1）
    has_following,      # [10] 是否有后车（0/1）
    site_id             # [11] 站点ID（0-8表示A-I）
]
```

### Step 5: 输出格式

```python
# NPZ文件 (train_episodes.npz, val_episodes.npz, test_episodes.npz)
{
    'states': [N, T=80, K=50, F=12],   # Episode数据
    'masks': [N, T=80, K=50],          # 1=真实车辆, 0=padding
    'scene_ids': [N],                  # 站点ID
    'start_frames': [N],               # Episode起始frame
    'end_frames': [N]                  # Episode结束frame
}

# Metadata (metadata.json)
{
    "n_features": 12,
    "episode_length": 80,
    "fps": 30.0,
    "lane_mapping": {"A:A1": 1, "A:B1": 2, ...},
    ...
}
```

---

## 🎯 数据使用示例

### 加载预处理后的数据

```python
import numpy as np
import json

# 加载训练数据
data = np.load('data/processed/train_episodes.npz')
states = data['states']      # [N, 80, 50, 12]
masks = data['masks']        # [N, 80, 50]

# 加载元数据
with open('data/processed/metadata.json', 'r') as f:
    metadata = json.load(f)

print(f"Episodes: {len(states)}")
print(f"Features: {metadata['n_features']}")
print(f"Lanes: {len(metadata['lane_mapping'])}")
```

### 特征索引

```python
# 从states中提取特征
positions = states[:, :, :, 0:2]      # [N, T, K, 2] - center_x, center_y
velocities = states[:, :, :, 2:4]     # [N, T, K, 2] - vx, vy
accelerations = states[:, :, :, 4:6]  # [N, T, K, 2] - ax, ay
angles = states[:, :, :, 6]           # [N, T, K] - angle
class_ids = states[:, :, :, 7]        # [N, T, K] - class_id (discrete)
lane_ids = states[:, :, :, 8]         # [N, T, K] - lane_id (discrete)
site_ids = states[:, :, :, 11]        # [N, T, K] - site_id (0-8)
```

---

## 📚 参考文档

- **README.md**: 完整的预处理和训练工作流程
- **CLAUDE.md**: 开发者技术指导
- **validate_preprocessing.py**: 数据验证脚本

---

## ⚠️ 重要提醒

1. **原始数据不要修改**: `data/raw/` 目录下的CSV文件应保持原样
2. **大文件处理**: 14GB数据需要足够的磁盘空间和内存
3. **处理时间**: 完整预处理可能需要10-30分钟（取决于硬件）
4. **站点差异**: 不同站点的车道结构、坐标范围都不同，需要站点级处理
5. **离散特征**: `class_id`, `lane_id`, `site_id` 不要归一化，使用embedding

---

**数据收集时间**: 2024年12月
**文档更新时间**: 2025-12-12
**数据版本**: v1.0
