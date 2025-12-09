# 快速开始指南

## 多站点数据处理（80/10/10分割）

### 1. 数据准备

将你的9个站点(A-I)的数据按以下结构放置：

```
D:\DRIFT\A\traffic-world-model\data\raw\
├── A/
│   ├── drone_1.csv
│   ├── drone_2.csv
│   └── ...
├── B/
│   ├── drone_1.csv
│   └── ...
├── C/
├── D/
├── E/
├── F/
├── G/
├── H/
└── I/
```

### 2. 运行预处理

```bash
# 进入项目目录
cd D:\DRIFT\A\traffic-world-model

# 运行预处理（默认80/10/10分割）
python preprocess_multisite.py
```

这将：
- 自动收集所有站点的所有CSV文件
- 随机混合并分割成80% train / 10% val / 10% test
- 提取轨迹特征并保存为.npz文件

### 3. 输出结果

预处理完成后，会在 `data/processed/` 生成：

```
data/processed/
├── train_episodes.npz      # 训练集 (80%)
├── val_episodes.npz        # 验证集 (10%)
├── test_episodes.npz       # 测试集 (10%)
├── metadata.json           # 元数据（特征数、lane映射等）
└── split_config.json       # 分割配置（记录哪些文件在哪个集合）
```

### 4. 训练模型

```bash
python src/training/train_world_model.py \
    --train_data data/processed/train_episodes.npz \
    --val_data data/processed/val_episodes.npz \
    --batch_size 32 \
    --n_epochs 100 \
    --latent_dim 256
```

### 5. 评估模型

```python
from src.data.dataset import TrajectoryDataset
from src.models.world_model import WorldModel
from src.evaluation.rollout_eval import evaluate_rollout
from torch.utils.data import DataLoader

# 加载测试集
test_dataset = TrajectoryDataset('data/processed/test_episodes.npz')
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 加载模型
model = WorldModel()
model.load_state_dict(torch.load('checkpoints/best_model.pt'))

# 评估
metrics = evaluate_rollout(model, test_loader, context_length=10, rollout_length=20)

print(f"ADE: {metrics['ade']:.3f}m")
print(f"FDE: {metrics['fde']:.3f}m")
```

---

## 自定义选项

### 修改分割比例

```bash
# 例如：70% train, 15% val, 15% test
python preprocess_multisite.py \
    --train_ratio 0.7 \
    --val_ratio 0.15 \
    --test_ratio 0.15
```

### 修改episode参数

```bash
python preprocess_multisite.py \
    --episode_length 30 \
    --max_vehicles 50 \
    --overlap 5 \
    --fps 30.0
```

### 完整参数列表

```bash
python preprocess_multisite.py --help
```

---

## 常见问题

### Q: 数据在哪里？
A: 原始数据放在 `data/raw/A/`, `data/raw/B/`, ... `data/raw/I/`

### Q: 处理后的数据在哪里？
A: 处理后在 `data/processed/train_episodes.npz`, `val_episodes.npz`, `test_episodes.npz`

### Q: 如何保证每次分割一致？
A: 使用相同的 `--seed 42` 参数，或者保存 `split_config.json` 后复用

### Q: 处理很慢怎么办？
A: 减少 `--max_vehicles` 或增加 `--overlap` 来减少episode数量

### Q: 如何查看数据统计？
A: 查看生成的 `metadata.json` 文件

---

## 下一步

完成数据预处理后，可以：

1. ✅ 查看 `README.md` 了解完整功能
2. ✅ 修改 `experiments/config.yaml` 配置实验参数
3. ✅ 运行训练脚本开始训练
4. ✅ 使用可视化工具查看结果

---

**最后更新:** 2025-12-09
