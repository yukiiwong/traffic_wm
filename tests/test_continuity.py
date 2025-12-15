"""
快速测试脚本:检查轨迹连续性
"""
import torch
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.world_model import WorldModel

# 创建模型
model = WorldModel(
    input_dim=12,
    latent_dim=256,
    dynamics_type='gru'
)
model.eval()

# 模拟数据
B, T_context, K, F = 2, 65, 10, 12
context_states = torch.randn(B, T_context, K, F)
context_masks = torch.ones(B, T_context, K)

# Rollout
with torch.no_grad():
    output = model.rollout(
        initial_states=context_states,
        initial_masks=context_masks,
        n_steps=15,
        teacher_forcing=False
    )

# 解码context的最后一帧
context_latent = model.encoder(context_states, context_masks)
last_context_decoded, _ = model.decoder(context_latent[:, -1:])

# 预测的第一帧
first_prediction = output['predicted_states'][:, 0:1]

# 计算距离(位置的差异)
distance = torch.norm(
    last_context_decoded[:, :, :, :2] - first_prediction[:, :, :, :2], 
    dim=-1
).mean()

print(f"✅ Context最后一帧 vs Prediction第一帧的平均距离: {distance:.4f}")
print(f"   (期望值: 小的合理距离,如 < 5.0 像素)")

if distance < 10.0:
    print("✅ 连续性良好!")
elif distance < 50.0:
    print("⚠️ 有一定跳跃,但可能在合理范围内")
else:
    print("❌ 跳跃过大,可能还有其他问题")
