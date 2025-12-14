# World Model 修改文档（离散特征处理 & site_id 设计）

> 目标：让模型**只回归连续状态**，把 `class_id / lane_id / site_id` 作为**条件/元数据**使用，而不是当作需要预测的连续变量；同时保证你仍然能明确知道车辆属于哪个站点（site）。

---

## 1. 为什么 Decoder 不输出 site_id 也不会“丢失站点信息”

- `site_id` 本质是 **episode-level 的条件变量（conditioning）/元数据（metadata）**  
  一个 episode 来自哪个站点是已知且固定的，不会在 rollout 过程中改变。
- 因此：
  - **站点信息应从数据集/episode 元数据中读取并携带**；
  - 模型预测的是未来的连续运动状态（x/y/v/a/角度等），不需要“生成” `site_id`。

---

## 2. 推荐方案（首选）：site_id 作为 episode-level metadata（最干净）

### 要做什么
- 在 `Dataset.__getitem__` 中显式返回 `site_id`（标量或 tensor）。
- 模型 forward 时把 `site_id` 传入 encoder/dynamics，查 embedding 后作为条件输入。
- rollout / imagined future 时 **沿用同一个 site_id**（同一 episode 不会跳站点）。

### 你最终怎么知道“车在哪个站点跑”
- 直接看该 episode 的 `site_id`（或来源文件/索引映射），无需 decoder 输出。

---
## 5. 与“只输出连续特征”的整体改动对齐（落地修改点）

### 5.1 Decoder 输出维度调整
- **Decoder 输出：仅连续特征（+可选 binary）**
- 不再输出：`class_id / lane_id / site_id`（离散特征）

> 说明：`has_preceding / has_following` 若你当前是按连续回归（0/1）也可暂时保留在连续集合中；后续可升级为 BCE 分类头。

### 5.2 Loss 计算调整
- 回归 loss（Huber/MAE）只对连续特征计算：
  - 连续：`center_x, center_y, vx, vy, ax, ay, angle, has_preceding, has_following`
  - 离散：`class_id, lane_id, site_id` 不参与回归 loss

### 5.3 站点信息如何进入模型（关键）
- `site_id` 不通过 decoder 预测，而通过：
  1) dataset 返回的 `site_id`
  2) embedding lookup 得到 `site_emb`
  3) 在 encoder/dynamics 中作为条件输入（concatenate / FiLM / additive bias 等）

---

## 6. rollout / imagined future 需要做的事（确保 site 不丢）

- rollout 函数输入增加 `site_id`（或 `site_emb`）。
- rollout 每一步：
  - 连续状态由模型预测
  - 离散条件（`site_id`）保持不变（常量传递）

如果你的管线仍然要输出 12 维：
- 最终拼回特征时，把 `site_id` 用常量填入即可。

---

## 7. 对比实验设置（与本修改一致）

### Experiment A：Single-site → single-site
- 只用单站点数据训练与测试（例如只用站点 A）。
- scaler：单站点 scaler（全局即可，因为只有一个站点）。
- 不需要 site embedding（可关掉 `use_site_id` 或固定为常量）。

### Experiment B：Multi-site（site embedding + per-site scaler）→ single-site
- 用多站点训练（A~I）：
  - 启用 `site embedding`
  - 启用 `per-site scaler`（每个站点单独统计均值方差/归一化器）
- 测试：只在某一个站点（例如站点 A）上评估
  - 使用对应站点的 scaler 进行 inverse/metrics
  - site_id 固定为该站点

---

## 8. 一句话结论

- **site_id 应该作为条件/元数据被“携带”和“注入”模型**，而不是作为 decoder 的预测输出。
- Decoder 只负责连续动态预测；站点归属始终由 episode 的 site_id 决定，rollout 时保持常量即可。
