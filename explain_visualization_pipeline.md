# 可视化结果生成流程说明

## 时间线
- **12:34** - 训练了两个模型:
  - `gru_z512.pt` (GRU dynamics, epoch 9, 51MB)
  - `lstm_z512.pt` (LSTM dynamics, epoch 8, 63MB)

- **13:18** - 生成了可视化结果 (27个jpg文件,覆盖Site A-I)

## 生成命令(推测)

基于文件时间戳和代码,您很可能运行了类似这样的命令:

```bash
python src/evaluation/visualize_predictions.py \
    --checkpoint gru_z512.pt \  # 或 lstm_z512.pt
    --test_data data/processed/test_episodes.npz \
    --metadata data/processed/metadata.json \
    --site_images_dir src/evaluation/sites \
    --context_length 65 \
    --rollout_horizon 15 \
    --num_samples 3 \
    --output_dir results/visualizations
```

## 生成的文件说明

每个站点生成3个样本:
- Site A: site_A_sample_1.jpg, site_A_sample_2.jpg, site_A_sample_3.jpg
- Site B-I: 同样的命名模式

总计: 9个站点 × 3个样本 = 27个可视化图片

## 可视化内容

每张图片包含:
1. **背景**: 对应站点的航拍图 (SiteA.jpg - SiteI.jpg)
2. **蓝色轨迹**: Context (65帧历史轨迹)
3. **绿色轨迹**: Ground Truth (15帧真实未来轨迹)
4. **红色轨迹**: Prediction (15帧模型预测轨迹)
5. **最多10个车辆**: 每个样本最多显示10个agent的轨迹

## 数据流程

1. 从test_episodes.npz加载测试数据(未归一化的像素坐标)
2. 提取context(前65帧)并归一化
3. 用训练好的模型进行rollout预测(15步)
4. 将预测结果反归一化回像素坐标
5. 在站点航拍图上绘制轨迹
6. 保存为jpg文件

## 模型信息

使用的模型:
- Input dim: 12 (根据metadata.json)
- Latent dim: 512 (从文件名z512推断)
- Dynamics: GRU 或 LSTM
- 训练轮数: ~9-10 epochs

归一化统计:
- Mean[0:2] = [1976.5, 936.7] (像素坐标的中心位置)
- 这些统计从训练集计算,并保存在checkpoint中
