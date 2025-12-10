````markdown
# 多站点 World Model 解决方案（仅采用策略 A：站点标识）

## 🚨 核心问题回顾

### 问题 1：坐标系不统一

- 每个站点有自己的 UAV 画面坐标系。
- 站点 A 的 `(100, 200)` 和站点 B 的 `(100, 200)` 在现实中可能相距几百米。
- 如果不加区分地把不同站点的坐标喂给同一个模型：
  - Transformer 会在**错误的空间关系**上做 self-attention；
  - 模型会误以为不在同一物理区域的车辆是“邻居”。

### 问题 2：当前混合策略的隐患

当前做法大致是：

1. 对每个站点单独生成 episodes；
2. 把不同站点的 episodes 全部混合、打乱；
3. 再划分 train / val / test。

问题在于：

- 每个 episode 内部只有单站点信息；
- 但模型**不知道**这是哪个站点的数据；
- 不同站点的坐标尺度、分布都可能不同，会增加学习难度、带来混淆。

---

## 🎯 目标（在只用策略 A 的前提下）

> 在**不做复杂标定、不统一全局坐标**的情况下，
> 让同一个世界模型能够：
> - 在多个站点上共享参数；
> - 正确认知“这是哪个站点”的数据；
> - 减少坐标混淆导致的训练不稳定。

我们**不做**：

- 多站点时间对齐的“走廊级 episode”；
- 复杂的全局坐标转换、标定；
- 站点之间的 GNN / 时空 Transformer 等高级结构。

只做一件事：  
**在现有 pipeline 基础上，给每辆车加一个可靠的 `site_id` 特征，并在模型中显式使用。**

---

## ✅ 策略 A：站点标识 + Site Embedding

### 核心思想

1. **数据层面：**
   - 保留各站点的本地坐标系；
   - 每条轨迹记录增加一个整数 `site_id`（例如 A=0, B=1, …, I=8）；
   - 可以继续按“单站点 episode”生成数据，再在 dataset 级别混合训练。

2. **模型层面：**
   - 将 11 维车辆特征和 1 维站点编号分开处理：
     - 车辆特征 → `feature_embed`
     - 站点编号 → `site_embedding`
   - 再拼接后送入 Transformer / Encoder。

这样模型在学习时知道：

- “这辆车来自站点 A/B/C…”，不会把不同站点的 `(x, y)` 混为一谈；
- 对不同站点可以学出不同的分布 / 风格，但仍然共享主干网络。

---

## 📋 数据预处理修改

### 1. 特征维度修改

**原始特征 (11 维)：**

```python
[center_x, center_y, vx, vy, ax, ay, angle,
 class_id, lane_id, has_pre, has_fol]
````

**修改后：增加 1 维 `site_id`，共 12 维：**

```python
[center_x, center_y, vx, vy, ax, ay, angle,
 class_id, lane_id, has_pre, has_fol,
 site_id]
#                        ^^^^^^  0~8 的整数，对应 A~I 站
```

> 站点编号约定示例：
> A → 0，B → 1，C → 2，…，I → 8
> 只要全流程一致即可。

---

### 2. 在 `preprocess_multisite.py` 中添加 site_id

伪代码示意（核心逻辑）：

```python
SITE_NAME_TO_ID = {
    "A": 0, "B": 1, "C": 2, "D": 3,
    "E": 4, "F": 5, "G": 6, "H": 7, "I": 8,
}

def get_site_id_from_path(csv_path: str) -> int:
    # 示例：data/raw/A/drone_1.csv → 站点 'A'
    site_name = csv_path.split('/')[-2]  # 例如 'A'
    return SITE_NAME_TO_ID[site_name]

def extract_episode(df: pd.DataFrame,
                    frames: List[int],
                    max_vehicles: int,
                    site_id: int,
                    use_site_id: bool = True):
    episode_length = len(frames)
    n_features = 11 + (1 if use_site_id else 0)
    states = np.zeros((episode_length, max_vehicles, n_features),
                      dtype=np.float32)
    masks = np.zeros((episode_length, max_vehicles),
                     dtype=np.float32)

    for t, frame in enumerate(frames):
        frame_df = df[df["frame"] == frame]
        # 选取前 max_vehicles 辆车
        frame_df = frame_df.head(max_vehicles)
        n_veh = len(frame_df)

        if n_veh == 0:
            continue

        # 填写前 11 维特征（归一化后的 x,y,vx,vy,ax,ay,...）
        states[t, :n_veh, :11] = extract_11d_features(frame_df)

        # 填写 mask
        masks[t, :n_veh] = 1.0

        if use_site_id:
            # 将 site_id 写入最后一维（对该帧真实车辆）
            states[t, :n_veh, 11] = float(site_id)

    return states, masks
```

上层调用示意：

```python
def process_single_csv(csv_path: str, ...):
    site_id = get_site_id_from_path(csv_path)
    df = load_and_preprocess_csv(csv_path, site_id=site_id)

    # 在这里按原有逻辑切 episode
    for frames in sliding_windows(...):
        states, masks = extract_episode(
            df=df,
            frames=frames,
            max_vehicles=max_vehicles,
            site_id=site_id,
            use_site_id=True,
        )
        # 保存到 episodes 列表中
```

---

### 3. 按站点做坐标归一化（强烈推荐）

即使有 `site_id`，也建议**按站点单独做归一化**：

```python
# 例：预先统计每个站点的均值与标准差
site_stats = {
    "A": {"x_mean": ..., "x_std": ..., "y_mean": ..., "y_std": ...},
    "B": {"x_mean": ..., "x_std": ..., "y_mean": ..., "y_std": ...},
    # ...
}

def normalize_xy(df: pd.DataFrame, site_name: str):
    stats = site_stats[site_name]
    df["center_x_norm"] = (df["center_x"] - stats["x_mean"]) / stats["x_std"]
    df["center_y_norm"] = (df["center_y"] - stats["y_mean"]) / stats["y_std"]
    return df
```

后续在 `extract_11d_features()` 中使用 `center_x_norm`, `center_y_norm` 作为位置特征即可。

---

### 4. 数据集组织方式保持不变

* 每个站点单独生成 episodes；
* 所有站点的 episodes 合并，并随机打乱；
* 按比例划分 train / val / test。

区别仅在于：

* 每条样本中都带有 `site_id`；
* 模型在训练时可以“感知”站点身份，从而减少混淆。

---

## 🧠 模型修改（仅在 Encoder 中支持 site_id）

### 1. 输入形状不变，只是 `F = 12`

* 仍使用形状：`states [B, T, K, F]`，`masks [B, T, K]`
* 原来 `F = 11`，现在 `F = 12`（多了 `site_id`）

---

### 2. Encoder 中添加 site embedding

```python
class MultiSiteEncoder(nn.Module):
    def __init__(self,
                 num_sites: int = 9,
                 d_feat: int = 128,
                 d_site: int = 16,
                 latent_dim: int = 256):
        super().__init__()

        # 11 维车辆特征嵌入
        self.feature_embed = nn.Linear(11, d_feat)

        # 站点 ID 嵌入：0~8 → d_site
        self.site_embedding = nn.Embedding(num_sites, d_site)

        # Transformer Encoder：在 K 维（agent 维度）上建模交互
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_feat + d_site,
                nhead=4,
                batch_first=True
            ),
            num_layers=2
        )

        # 将 pooled 表示投影到 latent 空间
        self.to_latent = nn.Sequential(
            nn.Linear(d_feat + d_site, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, states, masks):
        """
        states: [B, T, K, 12] (最后一维是 site_id)
        masks:  [B, T, K]
        """
        B, T, K, F = states.shape

        feats = states[..., :11]           # [B, T, K, 11]
        site_ids = states[..., 11].long()  # [B, T, K]

        feat_emb = self.feature_embed(feats)           # [B, T, K, d_feat]
        site_emb = self.site_embedding(site_ids)       # [B, T, K, d_site]

        x = torch.cat([feat_emb, site_emb], dim=-1)    # [B, T, K, d_feat + d_site]

        # 合并 B 和 T，方便在 K 维上做 self-attention
        x = x.view(B * T, K, -1)
        mask_bt = masks.view(B * T, K).bool()

        x = self.transformer(
            x,
            src_key_padding_mask=~mask_bt  # True 表示要被 mask 掉
        )  # [B*T, K, d_model]

        # Masked mean pooling over agents
        x = x.view(B, T, K, -1)
        mask_f = masks.unsqueeze(-1)  # [B, T, K, 1]
        masked_sum = (x * mask_f).sum(dim=2)      # [B, T, d_model]
        count = mask_f.sum(dim=2).clamp(min=1.0)  # [B, T, 1]
        pooled = masked_sum / count               # [B, T, d_model]

        latent = self.to_latent(pooled)           # [B, T, latent_dim]
        return latent
```

* Dynamics（GRU / LSTM / Transformer）和 Decoder 不需要大改，只要 `latent_dim` 一致即可。
* 训练损失（重建、预测、存在性）也可以保持原方案。

---

## ⚙️ 训练脚本参数修改示例

### 1. 预处理命令（示例）

```bash
python preprocess_multisite.py \
    --fps 1.0 \
    --episode_length 30 \
    --overlap 10 \
    --max_vehicles 50 \
    --use_site_id   # 在脚本中读取该 flag，启用 site_id 逻辑
```

### 2. 训练命令（示例）

```bash
python src/training/train_world_model.py \
    --train_data data/processed/train_episodes.npz \
    --val_data data/processed/val_episodes.npz \
    --input_dim 12 \          # 从 11 改为 12（多了 site_id）
    --latent_dim 256 \
    --dynamics_type gru \
    --batch_size 32 \
    --n_epochs 100 \
    --learning_rate 3e-4
```

---

## ⚠️ 注意事项与小建议

1. **site_id 约定要全流程一致**

   * 目录解析、预处理、可视化时都要保证 A~I 的编号一致；
   * 建议在一个单独的配置模块中定义 `SITE_NAME_TO_ID`。

2. **坐标归一化尽量在 CSV → features 这一步做完**

   * 不要在训练时再做归一化，避免重复计算；
   * 把归一化后的坐标直接写入保存的 npz 中。

3. **仍然是“单站点 episode + 多站点混训”**

   * 当前方案**不显式建模站点之间的交互和波动传递**；
   * 但可以显著减少“多站点坐标混淆”，让单站点预测更稳定、更可泛化。

4. **后续如果你想升级到走廊级建模**

   * 可以在此基础上引入“时间对齐多站点 episode”和更结构化的 latent 表示；
   * 当前这份文档完全兼容，将来升级时只需在此基础上扩展即可。

---

## 🔚 小结

在只采用策略 A 的前提下，你需要做的改动非常集中：

1. **预处理层：**

   * 为每辆车添加一个整型 `site_id`；
   * 按站点分别做坐标归一化；
   * 仍然按“单站点 episode + 混合”生成数据。

2. **模型层：**

   * `input_dim` 从 11 → 12；
   * 在 Encoder 中对 `site_id` 做 embedding，并与车辆特征 embedding 拼接；
   * 其它结构（Dynamics、Decoder、损失函数）可以基本保持不变。

这是一个工程量小、收益明显、非常适合作为多站点 world model 起步版本的方案。
后续你如果想再升级到“整条走廊 + 信号周期 + 拥堵波”级别，我们可以在这个基础上继续往上加层。👍

```
```
