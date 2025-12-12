import torch
from src.models.world_model import WorldModel
from src.data.dataset import get_dataloader
from src.evaluation.rollout_eval import evaluate_rollout

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 建立和训练时一致的模型
model = WorldModel(input_dim=12, latent_dim=256).to(device)

# 2. 读取 checkpoint（注意 map_location）
ckpt_path = r'/home/yukai/traffic_wm/checkpoints/world_model/dyn_gru_z256_seed1/checkpoint_epoch_200.pt'
checkpoint = torch.load(ckpt_path, map_location=device)

# 3. 只把其中的 model_state_dict 加载进模型
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print("✅ 模型参数加载成功")

# 4. 准备测试数据
test_loader = get_dataloader(
    data_path='data/processed/test_episodes.npz',
    batch_size=16,
    shuffle=False,
    normalize=True,
    stats_path=None  # 如果你有训练集 stats，可以在这里传路径
)

# 5. 调用评估函数
metrics = evaluate_rollout(
    model=model,
    data_loader=test_loader,
    context_length=10,
    rollout_length=20,
    device=device
)

print(metrics)
