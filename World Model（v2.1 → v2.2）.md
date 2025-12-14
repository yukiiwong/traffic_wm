# World Model 修改文档（v2.1 → v2.2）

本文件总结：当前训练/评估日志中暴露的问题应该如何改、改哪里；以及如何做两个对比实验：
- Single-site → single-site
- Multi-site（site embedding + per-site scaler）→ single-site

---

## 1. 数据与元数据（metadata.json）需要补充/调整的内容

### 1.1 为连续特征保存 scaler（mean/std），并明确哪些特征不归一化
**要做：**
- 在 `metadata.json` 里增加（或核对）连续特征的 `mean/std`（全局 or 分站点），并显式列出：
  - continuous_features：`center_x, center_y, vx, vy, ax, ay, angle(or sin/cos)`
  - binary_features：`has_preceding, has_following`
  - categorical_features：`class_id, lane_id, site_id`
  - do_not_normalize：至少包含 `class_id, lane_id, site_id`（你已有），建议也把 binary 放进 `do_not_normalize`（因为 BCE 不需要 z-score）

**改哪里：**
- `src/data/preprocess.py`（或你生成 episodes / metadata 的脚本）
  - 统计连续特征的 mean/std（全局 or per-site），写入 `metadata.json`
  - 确保写清楚 `feature_layout` 与上述分类字段

---

## 2. 特征表示需要修改：离散特征不要再用回归/MAE 处理

### 2.1 categorical 特征改为 embedding 输入（不参与 recon/pred 的回归 loss）
**要做：**
- `class_id / lane_id / site_id`：
  - 输入端：用 embedding（Embedding layer）转换为向量，再与连续特征拼接/融合
  - 输出端：默认不预测它们（不做 recon/pred loss），除非你明确要做 “多任务预测”（那也必须用 CE loss）

**改哪里：**
- `src/models/world_model.py`（或你的 world model 定义文件）
  - 加三个 embedding：
    - `class_emb = nn.Embedding(num_classes, d_class)`
    - `lane_emb  = nn.Embedding(num_lanes, d_lane)`
    - `site_emb  = nn.Embedding(num_sites, d_site)`（multi-site 实验必需；single-site 可关）
  - forward 时把 embedding 与连续特征融合（concat 或通过小 MLP 融合）
- `src/training/train_world_model.py`（或训练入口）
  - 增加/传入 `num_lanes / num_sites / num_classes`（从 metadata.json 读取）
  - 增加开关参数：`--use_lane_emb --use_site_emb --use_class_emb`

---

## 3. Loss 计算方式需要拆分：连续 / 二值 / 类别分头算

### 3.1 重写 recon/pred loss：只对连续特征算 L1/MSE
**要做：**
- 连续特征（建议默认）：`center_x, center_y, vx, vy, ax, ay, angle(or sin/cos)` → `L1` 或 `SmoothL1`
- 二值特征：`has_preceding, has_following` → `BCEWithLogitsLoss`
- 类别特征（如果你不做输出预测）：不算 loss
- 现有的存在性（existence）loss 保留：`existence_weight` OK

**改哪里：**
- `src/losses/world_model_loss.py`（若没有就新建）或现有 loss 文件
  - 把 `input_dim=12` 的“一锅端”loss 拆成：
    - `loss_recon_cont`
    - `loss_pred_cont`
    - `loss_bin`（可选）
    - `loss_exist`
- `src/training/train_world_model.py`
  - 日志里分别打印各 loss（连续 / 二值 / existence），不要再输出“lane_id MAE”

---

## 4. 指标（metrics）输出需要修复：不要再对离散特征算 MAE

### 4.1 修改 MAE-PER-STEP / MAE-PER-FEATURE 只统计连续特征
**要做：**
- `MAE-PER-STEP`：只针对连续特征维度聚合
- `RECON/PRED MAE PER FEATURE`：只列连续特征
- 如果你想看离散特征质量：
  - 类别预测 → 输出 accuracy/top-k（只有当你真的预测它们时）

**改哪里：**
- `src/evaluation/metrics.py`（或日志里 DIAG 计算位置）
  - 增加 `continuous_idx`、`binary_idx`、`categorical_idx`（从 metadata.json 读取更稳）
  - 统计时只对 `continuous_idx` 算 MAE/MSE
- `src/training/train_world_model.py`
  - 日志打印改为：
    - `MAE(step)`：continuous-only
    - `MAE(feature)`：continuous-only
    - `EXIST-*` 保留
    - `ADE/FDE` 保留

---

## 5. 速度误差 VelErr / ADE/FDE 口径统一（归一化/反归一化）

### 5.1 统一评估空间：要么全在 normalized 空间，要么全在 physical 空间
**要做：**
- 明确 `ADE/FDE/VelErr` 是在：
  - (A) normalized 空间算；或
  - (B) inverse-transform 回物理单位后算
- 推荐：对外汇报用 (B)，训练 early debug 用 (A) 也可以，但必须统一

**改哪里：**
- `src/evaluation/metrics.py`（或计算 ADE/FDE/VelErr 的函数）
  - 如果走 (B)：在算误差前对连续特征 inverse-transform（使用 metadata.json 的 scaler）
  - 确保 vx/vy 的 scaler 与 x/y 一致使用（同一套全局或 per-site）

---

## 6. angle 改为 sin/cos（强烈建议）

### 6.1 angle → (sin, cos) 两维连续输出
**要做：**
- 在 preprocess 阶段把 `angle` 替换为 `sin(angle), cos(angle)`
- `n_features` 将从 12 变为 13（如果只替换 angle 一维→两维，则 +1）

**改哪里：**
- `src/data/preprocess.py`
  - 生成 episode feature 时替换 angle
  - 更新 `feature_layout`
- `src/models/world_model.py` / `train_world_model.py`
  - 更新 `input_dim`

---

# 对比实验设计与落地步骤

## 实验 A：Single-site → single-site（训练=单站点，测试=同站点）
目标：建立一个“最干净”的基线，排除跨站点分布差异。

### A1. 数据准备
**要做：**
- 从原始数据中只保留一个 site（例如 `site = A`）
- 重新生成：
  - `data/processed/A_train_episodes.npz`
  - `data/processed/A_val_episodes.npz`
  - `data/processed/A_test_episodes.npz`
  - `data/processed/A_metadata.json`
- scaler：只用站点 A 的数据统计（per-site 就等于 single-site）

**改哪里：**
- `src/data/preprocess.py`
  - 增加参数 `--site_id A`（或 `--sites A`）
  - 输出按 site 命名的 npz 和 metadata

### A2. 训练与评估
**要做：**
- 训练：
  - `--train_data A_train_episodes.npz`
  - `--val_data   A_val_episodes.npz`
- 测试（single-site test）：
  - 用保存的 best checkpoint，在 `A_test_episodes.npz` 上跑一次 eval

**记录：**
- val_loss（continuous-only）
- ADE/FDE（统一口径）
- VelErr（统一口径）
- existence metrics

---

## 实验 B：Multi-site（site embedding + per-site scaler）→ single-site（只测试某一站点）
目标：验证“多站点学到更泛化的动态先验”，但在测试时聚焦某站点性能。

### B1. 数据准备（Multi-site + per-site scaler）
**要做：**
- 使用所有站点 A..I 生成统一的：
  - `data/processed/MS_train_episodes.npz`
  - `data/processed/MS_val_episodes.npz`
  - `data/processed/MS_test_episodes.npz`（可选）
  - `data/processed/MS_metadata.json`
- `MS_metadata.json` 里保存 **每个 site 的 scaler**（per-site scaler）：
  - `scaler_by_site: { "A": {mean,std}, "B": {...}, ... }`
- episode 内必须保留 `site_id`（你已有）

**改哪里：**
- `src/data/preprocess.py`
  - 统计每个站点的 mean/std（仅连续特征）
  - 写入 `scaler_by_site`
  - 同时也可以写一个 `global_scaler`（可选，用于 debug 对照）

### B2. 训练（Multi-site）
**要做：**
- 开启 site embedding：
  - `--use_site_emb true`
- 数据归一化方式（二选一，建议优先 per-site）：
  - (推荐) 训练时：对每条样本按其 `site_id` 使用对应 scaler 做 normalize
  - 或者：先 global normalize，再额外输入 site embedding（但通常不如 per-site 稳）

**改哪里：**
- `src/data/dataset.py`（或 DataLoader 取样处）
  - 在 `__getitem__` 里根据样本 `site_id` 选择 scaler
- `src/models/world_model.py`
  - `site_emb` 参与输入融合

### B3. 测试（只在单站点上评估）
**要做：**
- 从 Multi-site 的 test 集合中筛选出某站点（例如 A）样本：
  - 生成 `data/processed/MS_test_siteA.npz`（或运行时过滤）
- eval 时同样使用 per-site scaler（site=A 的 scaler）做 inverse-transform/metrics

**改哪里：**
- `src/evaluation/eval_world_model.py`（若无则在 train 脚本加 eval 模式）
  - 增加参数 `--eval_site A`：只评估该站点样本
- 或在 `dataset.py` 增加过滤选项 `site_filter`

---

## 对比实验输出建议（确保可复现）
每个实验都保存一份结果 JSON（或表格）：
- experiment_name
- train_sites
- test_site
- normalization: global / per-site
- use_site_emb: true/false
- best_val_epoch / best_val_loss
- ADE, FDE, VelErr（统一口径）
- existence Acc/Prec/Rec

输出路径示例：
- `results/exp_single_A_to_A.json`
- `results/exp_multisite_perSiteScaler_siteEmb_to_A.json`

---

# 快速自查清单（改完后你应该看到什么）
- log 中不再出现 `lane_id MAE ~ 89` 或 `MAE-PER-STEP ~ 99`
- `RECON/PRED MAE PER FEATURE` 只包含连续特征（x/y/v/acc/(sin,cos)）
- `val_loss` 更能真实反映轨迹质量
- `ADE/FDE`、`VelErr` 口径一致，不再出现“位置很好但速度很差却解释不清”的情况
