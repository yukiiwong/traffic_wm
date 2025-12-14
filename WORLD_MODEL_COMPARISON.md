# World Model 架构对比: Traffic WM vs DreamerV3

**日期**: 2025-12-14
**版本**: 1.0

---

## 📋 目录

1. [总体架构对比](#总体架构对比)
2. [Encoder 设计对比](#encoder-设计对比)
3. [Dynamics 模型对比](#dynamics-模型对比)
4. [Decoder 设计对比](#decoder-设计对比)
5. [训练策略对比](#训练策略对比)
6. [关键差异总结](#关键差异总结)
7. [优缺点分析](#优缺点分析)

---

## 1. 总体架构对比

### 1.1 Traffic WM (当前实现)

```
输入: Multi-agent states [B, T, K, F]
  ↓
Encoder: MultiAgentEncoder
  - Feature Embedding (连续特征)
  - Site Embedding (离散site_id)
  - Lane Embedding (离散lane_id)
  - Transformer Attention (跨agent维度K)
  - Masked Mean Pooling → [B, T, latent_dim]
  ↓
Dynamics: LatentDynamics (GRU/LSTM/Transformer)
  - 1-step transition: z[t] → z[t+1]
  - Teacher forcing during training
  - Open-loop rollout during evaluation
  ↓
Decoder: StateDecoder
  - MLP layers: latent → states + existence logits
  - Outputs: [B, T, K, F] states, [B, T, K] masks
  ↓
Loss:
  - Reconstruction Loss: L2(states_t, reconstructed_t)
  - Prediction Loss: L2(states_{t+1}, predicted_{t+1})
  - Existence Loss: BCE(masks, predicted_masks)
```

**关键特点**:
- **确定性模型**: 完全确定性的latent表示
- **多智能体特化**: 专门设计用于多agent轨迹预测
- **简单架构**: 直接的encoder-dynamics-decoder结构
- **PyTorch实现**: 使用PyTorch框架

### 1.2 DreamerV3

```
输入: Observations (images + vectors)
  ↓
Encoder: MultiEncoder
  - CNN Encoder (ResNet) for images
  - MLP Encoder for vectors
  - Concatenate features → embed
  ↓
RSSM (Recurrent State-Space Model):
  - Deterministic: GRU(stoch_{t-1}, action_{t-1}) → deter_t
  - Stochastic (Prior): MLP(deter_t) → prior_logits_t
  - Stochastic (Posterior): MLP(deter_t, embed_t) → post_logits_t
  - Sample: stoch_t ~ Categorical(post_logits_t)
  - Latent = [deter_t, stoch_t]
  ↓
Decoder: MultiDecoder
  - CNN Decoder (ResNet) for images
  - MLP Decoder for vectors
  - Reward Head
  - Continuation Head
  ↓
Loss:
  - Dynamics Loss: KL(posterior || prior)
  - Representation Loss: KL(posterior || sg(prior))
  - Reconstruction Loss: -log p(obs | latent)
  - Reward Loss: -log p(reward | latent)
  - Continuation Loss: -log p(cont | latent)
```

**关键特点**:
- **随机模型**: RSSM结合确定性(deter)和随机性(stoch)
- **强化学习特化**: 设计用于model-based RL
- **复杂架构**: 分离的prior和posterior,多个head
- **JAX实现**: 使用JAX框架,支持JIT编译

---

## 2. Encoder 设计对比

### 2.1 Traffic WM: MultiAgentEncoder

**文件**: `src/models/encoder.py`

```python
class MultiAgentEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, n_layers, dropout, use_site_id):
        # 1. Feature Embedding
        self.feature_embedding = nn.Linear(input_dim - num_discrete, hidden_dim)

        # 2. Discrete Embeddings
        self.site_embedding = nn.Embedding(num_sites, site_embed_dim)
        self.lane_embedding = nn.Embedding(num_lanes, lane_embed_dim)

        # 3. Transformer Layers (跨agent注意力)
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(n_layers)
        ])

        # 4. Output Projection
        self.to_latent = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim)
        )

    def forward(self, states, masks):
        B, T, K, F = states.shape

        # 提取离散特征
        site_id = states[..., site_idx].long()
        lane_id = states[..., lane_idx].long()

        # Embedding
        cont_feats = self.feature_embedding(states[..., continuous_indices])
        site_embed = self.site_embedding(site_id)
        lane_embed = self.lane_embedding(lane_id)

        # 拼接
        x = cont_feats + site_embed + lane_embed  # [B, T, K, hidden_dim]

        # Transformer (跨K维度)
        for layer in self.transformer_layers:
            x = layer(x, src_key_padding_mask=~masks.bool())

        # Pooling (跨K维度)
        x = masked_mean_pooling(x, masks)  # [B, T, hidden_dim]

        # Project to latent
        latent = self.to_latent(x)  # [B, T, latent_dim]

        return latent
```

**特点**:
- ✅ **多智能体聚合**: Transformer attention跨agent维度
- ✅ **离散特征处理**: 专门的embedding层用于site_id和lane_id
- ✅ **Masked pooling**: 正确处理padding的agent
- ✅ **确定性输出**: 直接输出latent向量,无随机性
- ⚠️ **无CNN**: 不处理图像输入,仅处理向量特征

### 2.2 DreamerV3: MultiEncoder

**文件**: `dreamerv3/nets.py:211-260`

```python
class MultiEncoder(nj.Module):
    def __init__(self, shapes, cnn_keys, mlp_keys, mlp_layers, mlp_units,
                 cnn='resnet', cnn_depth=48, cnn_blocks=2, ...):
        # 分离CNN和MLP输入
        self.cnn_shapes = {k: v for k, v in shapes.items() if len(v) == 3}  # Images
        self.mlp_shapes = {k: v for k, v in shapes.items() if len(v) in (1, 2)}  # Vectors

        # CNN: ResNet编码器
        if cnn == 'resnet':
            self._cnn = ImageEncoderResnet(cnn_depth, cnn_blocks, ...)

        # MLP: 多层感知机
        if self.mlp_shapes:
            self._mlp = MLP(None, mlp_layers, mlp_units, dist='none')

    def __call__(self, data):
        outputs = []

        # CNN编码图像
        if self.cnn_shapes:
            inputs = jnp.concatenate([data[k] for k in self.cnn_shapes], -1)
            output = self._cnn(inputs)
            outputs.append(output.reshape((output.shape[0], -1)))

        # MLP编码向量
        if self.mlp_shapes:
            inputs = jnp.concatenate([data[k] for k in self.mlp_shapes], -1)
            outputs.append(self._mlp(inputs))

        # 拼接所有编码
        return jnp.concatenate(outputs, -1)
```

**特点**:
- ✅ **多模态输入**: 同时处理图像(CNN)和向量(MLP)
- ✅ **ResNet架构**: 使用深度残差网络编码图像
- ✅ **灵活设计**: 通过regex匹配动态选择CNN/MLP输入
- ⚠️ **无多智能体聚合**: 假设单agent或已聚合的观测
- ⚠️ **无离散特征处理**: 需要one-hot或预处理

---

## 3. Dynamics 模型对比

### 3.1 Traffic WM: LatentDynamics

**文件**: `src/models/dynamics.py`

```python
class LatentDynamics(nn.Module):
    def __init__(self, latent_dim, hidden_dim, n_layers, dropout, model_type):
        if model_type == 'gru':
            self.rnn = nn.GRU(
                input_size=latent_dim,
                hidden_size=hidden_dim,
                num_layers=n_layers,
                dropout=dropout if n_layers > 1 else 0,
                batch_first=True
            )
        elif model_type == 'lstm':
            self.rnn = nn.LSTM(...)
        elif model_type == 'transformer':
            self.transformer = nn.TransformerEncoder(...)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, latent_dim)

    def forward(self, latent, hidden=None):
        # latent: [B, T, latent_dim]

        if self.model_type == 'transformer':
            output = self.transformer(latent)
            return output, None
        else:  # GRU or LSTM
            rnn_out, hidden = self.rnn(latent, hidden)
            output = self.output_proj(rnn_out)  # [B, T, latent_dim]
            return output, hidden
```

**特点**:
- ✅ **简单直接**: 单一的RNN或Transformer模型
- ✅ **确定性转换**: latent[t] → latent[t+1]
- ✅ **灵活选择**: 支持GRU/LSTM/Transformer三种dynamics
- ✅ **无action输入**: 不需要action,适用于纯预测任务
- ⚠️ **无随机性**: 完全确定性,可能欠拟合复杂动态

### 3.2 DreamerV3: RSSM (Recurrent State-Space Model)

**文件**: `dreamerv3/nets.py:22-209`

```python
class RSSM(nj.Module):
    def __init__(self, deter=1024, stoch=32, classes=32, ...):
        self._deter = deter  # 确定性状态维度
        self._stoch = stoch  # 随机状态维度
        self._classes = classes  # 离散类别数

    def obs_step(self, prev_state, prev_action, embed, is_first):
        """观测步骤: 给定观测embed,更新状态"""

        # 1. Prior: img_step预测下一状态
        prior = self.img_step(prev_state, prev_action)

        # 2. Posterior: 结合观测修正
        x = jnp.concatenate([prior['deter'], embed], -1)
        x = self.get('obs_out', Linear)(x)
        stats = self._stats('obs_stats', x)  # → logits or (mean, std)

        # 3. Sample stochastic state
        dist = self.get_dist(stats)
        stoch = dist.sample(seed=nj.rng())

        post = {'stoch': stoch, 'deter': prior['deter'], **stats}
        return post, prior

    def img_step(self, prev_state, prev_action):
        """想象步骤: 仅基于action预测下一状态"""

        prev_stoch = prev_state['stoch']

        # 1. GRU更新确定性状态
        x = jnp.concatenate([prev_stoch, prev_action], -1)
        x = self.get('img_in', Linear)(x)
        x, deter = self._gru(x, prev_state['deter'])

        # 2. 预测随机状态的prior
        x = self.get('img_out', Linear)(x)
        stats = self._stats('img_stats', x)
        dist = self.get_dist(stats)
        stoch = dist.sample(seed=nj.rng())

        prior = {'stoch': stoch, 'deter': deter, **stats}
        return prior

    def _gru(self, x, deter):
        """自定义GRU实现"""
        x = jnp.concatenate([deter, x], -1)
        x = self.get('gru', Linear, units=3 * self._deter)(x)
        reset, cand, update = jnp.split(x, 3, -1)

        reset = jax.nn.sigmoid(reset)
        cand = jnp.tanh(reset * cand)
        update = jax.nn.sigmoid(update - 1)

        deter = update * cand + (1 - update) * deter
        return deter, deter
```

**RSSM核心思想**:

```
Latent State = [Deterministic, Stochastic]
             = [deter_t,      stoch_t     ]

时间演化:
  deter_t = GRU(deter_{t-1}, [stoch_{t-1}, action_{t-1}])

  prior_t ~ p(stoch_t | deter_t)           # 仅基于历史
  post_t  ~ p(stoch_t | deter_t, embed_t)  # 结合当前观测
```

**特点**:
- ✅ **随机+确定**: 分离确定性记忆和随机性变化
- ✅ **Prior-Posterior**: 分别建模预测和修正
- ✅ **离散随机**: 使用Categorical分布(更稳定)或Gaussian
- ✅ **Action-conditioned**: 显式建模action的影响
- ⚠️ **复杂**: 需要维护两个分布,训练更复杂
- ⚠️ **需要action**: 不适用于纯观测预测任务

**关键差异**:

| 方面 | Traffic WM | DreamerV3 RSSM |
|------|-----------|---------------|
| **状态表示** | 纯确定性 latent | 确定性deter + 随机stoch |
| **动态模型** | RNN直接预测latent | GRU更新deter,然后采样stoch |
| **随机性** | 无 | 有(Categorical或Gaussian) |
| **Action** | 不需要 | 必须(action-conditioned) |
| **Prior/Posterior** | 无 | 有(分别建模) |
| **KL Loss** | 无 | 有(约束prior和posterior) |

---

## 4. Decoder 设计对比

### 4.1 Traffic WM: StateDecoder

**文件**: `src/models/decoder.py`

```python
class StateDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, max_agents, n_layers, dropout):
        # MLP decoder
        layers = []
        current_dim = latent_dim

        for _ in range(n_layers):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            current_dim = hidden_dim

        self.decoder = nn.Sequential(*layers)

        # Output heads
        self.state_head = nn.Linear(hidden_dim, max_agents * output_dim)
        self.existence_head = nn.Linear(hidden_dim, max_agents)

    def forward(self, latent):
        # latent: [B, T, latent_dim]

        # Decode
        x = self.decoder(latent)  # [B, T, hidden_dim]

        # State prediction
        states = self.state_head(x)  # [B, T, max_agents * output_dim]
        states = states.reshape(B, T, max_agents, output_dim)

        # Existence prediction
        existence_logits = self.existence_head(x)  # [B, T, max_agents]

        return states, existence_logits
```

**特点**:
- ✅ **简单MLP**: 多层全连接网络
- ✅ **多智能体输出**: 直接输出[K, F]形状的states
- ✅ **Existence prediction**: 预测每个agent slot是否存在
- ✅ **确定性输出**: 直接输出mean,无分布
- ⚠️ **无图像重建**: 仅输出向量状态

### 4.2 DreamerV3: MultiDecoder

**文件**: `dreamerv3/nets.py:263-329`

```python
class MultiDecoder(nj.Module):
    def __init__(self, shapes, cnn='resnet', cnn_depth=48, mlp_layers=4,
                 image_dist='mse', vector_dist='mse', ...):
        # 分离CNN和MLP输出
        self.cnn_shapes = {k: v for k, v in shapes.items() if len(v) == 3}
        self.mlp_shapes = {k: v for k, v in shapes.items() if len(v) == 1}

        # CNN Decoder
        if self.cnn_shapes:
            if cnn == 'resnet':
                self._cnn = ImageDecoderResnet(shape, cnn_depth, ...)

        # MLP Decoder
        if self.mlp_shapes:
            self._mlp = MLP(self.mlp_shapes, mlp_layers, mlp_units, ...)

        self._image_dist = image_dist

    def __call__(self, inputs):
        features = self._inputs(inputs)  # 从latent提取特征
        dists = {}

        # CNN解码图像
        if self.cnn_shapes:
            output = self._cnn(features)
            means = jnp.split(output, split_indices, -1)
            dists.update({
                key: self._make_image_dist(key, mean)
                for (key, shape), mean in zip(self.cnn_shapes.items(), means)
            })

        # MLP解码向量
        if self.mlp_shapes:
            dists.update(self._mlp(features))

        return dists  # 返回分布字典

    def _make_image_dist(self, name, mean):
        if self._image_dist == 'normal':
            return tfd.Independent(tfd.Normal(mean, 1), 3)
        if self._image_dist == 'mse':
            return jaxutils.MSEDist(mean, 3, 'sum')
```

**特点**:
- ✅ **多模态输出**: CNN重建图像,MLP重建向量
- ✅ **ResNet Decoder**: 使用深度残差网络解码图像
- ✅ **概率输出**: 输出分布(Normal或MSE),不是单点估计
- ✅ **灵活**: 支持多种输出类型和分布
- ⚠️ **无多智能体**: 假设单agent或已聚合输出

---

## 5. 训练策略对比

### 5.1 Traffic WM: 监督学习

**文件**: `src/training/train_world_model.py`, `src/training/losses.py`

```python
class WorldModelLoss(nn.Module):
    def forward(self, predictions, targets):
        states = targets['states']
        masks = targets['masks']

        # 1. Reconstruction Loss
        recon_loss = masked_huber_loss(
            predictions['reconstructed_states'],
            states,
            masks
        )

        # 2. One-step Prediction Loss (时间偏移)
        pred_states = predictions['predicted_states'][:, :-1]
        target_states = states[:, 1:]
        target_masks = masks[:, 1:]

        pred_loss = masked_huber_loss(
            pred_states, target_states, target_masks
        )

        # 3. Existence Loss
        existence_loss = BCE(
            predictions['existence_logits'],
            masks
        )

        # Total Loss
        total_loss = (
            self.recon_weight * recon_loss +
            self.pred_weight * pred_loss +
            self.existence_weight * existence_loss
        )

        return {'total': total_loss, ...}
```

**训练流程**:
```python
# Forward pass
states, masks = batch['states'], batch['masks']
output = model(states, masks)

# Compute loss
loss = loss_fn(output, {'states': states, 'masks': masks})

# Backprop
loss['total'].backward()
optimizer.step()
```

**特点**:
- ✅ **简单直接**: 标准监督学习
- ✅ **Teacher forcing**: 训练时使用真实latent序列
- ✅ **明确目标**: 重建当前帧 + 预测下一帧
- ⚠️ **Exposure bias**: 测试时看不到真实latent
- ⚠️ **无正则化**: 缺少latent空间的约束

### 5.2 DreamerV3: 变分推断 + RL

**文件**: `dreamerv3/agent.py:154-187`

```python
class WorldModel:
    def loss(self, data, state):
        # 1. Encode observations
        embed = self.encoder(data)

        # 2. RSSM observe (得到posterior和prior)
        post, prior = self.rssm.observe(
            embed, prev_actions, data['is_first'], prev_latent
        )

        # 3. Decode from posterior
        dists = {}
        feats = {**post, 'embed': embed}
        for name, head in self.heads.items():
            out = head(feats)
            dists.update(out)

        # 4. Compute losses
        losses = {}

        # Dynamics Loss: KL(posterior || prior)
        losses['dyn'] = self.rssm.dyn_loss(post, prior, impl='kl', free=1.0)

        # Representation Loss: KL(posterior || sg(prior))
        losses['rep'] = self.rssm.rep_loss(post, prior, impl='kl', free=1.0)

        # Reconstruction Losses
        for key, dist in dists.items():
            loss = -dist.log_prob(data[key])
            losses[key] = loss

        # 5. Weighted sum
        scaled = {k: v * self.scales[k] for k, v in losses.items()}
        model_loss = sum(scaled.values())

        return model_loss.mean()
```

**关键Loss组件**:

1. **Dynamics Loss (KL Divergence)**:
   ```
   L_dyn = KL(posterior_t || prior_t)
         = KL(q(stoch_t | deter_t, obs_t) || p(stoch_t | deter_t))
   ```
   - 约束posterior不要偏离prior太远
   - 防止posterior过度依赖观测

2. **Representation Loss**:
   ```
   L_rep = KL(posterior_t || sg(prior_t))
   ```
   - sg() = stop gradient
   - 训练posterior拟合数据,不影响prior

3. **Reconstruction Loss**:
   ```
   L_recon = -log p(obs_t | latent_t)
           = -log p(obs_t | deter_t, stoch_t)
   ```

**特点**:
- ✅ **变分推断**: 学习latent的概率分布
- ✅ **KL正则化**: 约束latent空间结构
- ✅ **Posterior-Prior**: 分离观测编码和动态预测
- ✅ **Free bits**: KL loss下界,防止posterior collapse
- ⚠️ **复杂**: 需要平衡多个loss权重
- ⚠️ **RL导向**: 还有actor-critic训练(未在此展示)

---

## 6. 关键差异总结

### 6.1 设计哲学

| 方面 | Traffic WM | DreamerV3 |
|------|-----------|-----------|
| **目标任务** | 多智能体轨迹预测 | 强化学习(model-based RL) |
| **数据类型** | 向量特征(坐标、速度等) | 图像 + 向量 |
| **随机性** | 确定性模型 | 随机模型(RSSM) |
| **Action** | 不需要 | 必须(action-conditioned) |
| **学习范式** | 监督学习 | 变分推断 + RL |

### 6.2 架构细节

#### Encoder

| 特性 | Traffic WM | DreamerV3 |
|------|-----------|-----------|
| **输入处理** | 向量特征 | 图像(CNN) + 向量(MLP) |
| **多智能体** | ✅ Transformer attention | ❌ 单agent或已聚合 |
| **离散特征** | ✅ Embedding层(site, lane) | ⚠️ 需预处理或one-hot |
| **输出** | 确定性latent | 确定性embed |

#### Dynamics

| 特性 | Traffic WM | DreamerV3 RSSM |
|------|-----------|----------------|
| **状态表示** | latent_dim维向量 | deter(确定) + stoch(随机) |
| **时间演化** | RNN(GRU/LSTM/Transformer) | GRU + 随机采样 |
| **随机性** | ❌ 无 | ✅ Categorical/Gaussian |
| **Action依赖** | ❌ 无 | ✅ 有 |
| **训练方式** | Teacher forcing | Posterior-Prior KL |

#### Decoder

| 特性 | Traffic WM | DreamerV3 |
|------|-----------|-----------|
| **输出类型** | 向量states + existence | 图像 + 向量 + reward + cont |
| **重建质量** | MLP | ResNet (for images) |
| **输出形式** | 确定性(mean) | 概率分布 |
| **多智能体** | ✅ [K, F]输出 | ❌ 单agent |

### 6.3 Loss函数

| Loss组件 | Traffic WM | DreamerV3 |
|----------|-----------|-----------|
| **重建loss** | Huber Loss | -log p(obs\|latent) |
| **预测loss** | Huber Loss (1-step) | ❌ (通过KL隐式) |
| **KL loss** | ❌ 无 | ✅ Dyn + Rep |
| **Existence** | ✅ BCE | ❌ (用cont head) |
| **Reward** | ❌ 无 | ✅ (RL需要) |

### 6.4 实现框架

| 方面 | Traffic WM | DreamerV3 |
|------|-----------|-----------|
| **框架** | PyTorch | JAX |
| **自动微分** | torch.autograd | jax.grad |
| **编译** | TorchScript (可选) | JIT (默认) |
| **分布式** | DDP | pmap |
| **随机数** | torch.random | nj.rng() |

---

## 7. 优缺点分析

### 7.1 Traffic WM

#### 优点 ✅

1. **多智能体特化**:
   - Transformer attention跨agent聚合
   - Masked pooling处理可变数量agent
   - Existence prediction显式建模agent出现/消失

2. **简单高效**:
   - 纯确定性模型,训练快速
   - 直接的encoder-dynamics-decoder结构
   - 无需复杂的KL balancing

3. **离散特征处理**:
   - Embedding层处理site_id和lane_id
   - 避免高维one-hot编码

4. **灵活的dynamics**:
   - 支持GRU/LSTM/Transformer选择
   - 适应不同时序复杂度

5. **轨迹预测专用**:
   - 设计契合车辆轨迹预测任务
   - 输出格式符合评估需求(ADE/FDE)

#### 缺点 ⚠️

1. **无随机性**:
   - 完全确定性,无法建模多模态未来
   - 可能欠拟合复杂交互场景
   - 难以捕捉驾驶行为的不确定性

2. **Exposure Bias**:
   - 训练时用真实latent序列
   - 测试时用预测latent(累积误差)
   - 可能导致长期预测漂移

3. **缺少正则化**:
   - 无latent空间结构约束
   - 可能学到不规则的表示
   - 泛化能力可能受限

4. **仅向量输入**:
   - 无法处理图像输入
   - 不能利用视觉信息(如道路图)

5. **无action建模**:
   - 不适用于强化学习
   - 无法做what-if分析

### 7.2 DreamerV3

#### 优点 ✅

1. **强大的表示学习**:
   - RSSM结合确定性和随机性
   - Posterior-Prior分离建模观测和动态
   - KL正则化学习结构化latent空间

2. **多模态支持**:
   - CNN处理图像,MLP处理向量
   - 统一框架处理不同模态

3. **概率建模**:
   - 输出分布而非单点估计
   - 可以采样多个未来轨迹
   - 建模不确定性

4. **RL ready**:
   - Action-conditioned dynamics
   - Reward和continuation预测
   - 支持model-based planning

5. **理论基础**:
   - 基于变分推断
   - 有理论保证的学习目标

#### 缺点 ⚠️

1. **复杂性高**:
   - Prior和Posterior双分支
   - 多个loss需要平衡权重
   - 训练不稳定风险

2. **计算开销**:
   - ResNet编解码器重
   - 采样操作增加计算
   - 需要更多内存

3. **超参数敏感**:
   - KL loss权重
   - Free bits阈值
   - 多个loss scale需要调优

4. **无多智能体**:
   - 假设单agent观测
   - 需要额外设计处理多agent
   - 难以直接应用于交通场景

5. **Action依赖**:
   - 必须有action输入
   - 不适用于纯观测预测
   - 增加数据需求

---

## 8. 改进建议

### 8.1 为Traffic WM添加随机性

**方案1: 简化版RSSM**

```python
class StochasticLatentDynamics(nn.Module):
    def __init__(self, latent_dim, stoch_dim, hidden_dim):
        # 确定性部分
        self.gru = nn.GRU(latent_dim, hidden_dim)

        # 随机部分 (prior)
        self.prior_net = nn.Linear(hidden_dim, stoch_dim * 2)  # mean, logvar

        # 随机部分 (posterior) - 用于训练
        self.posterior_net = nn.Linear(hidden_dim + latent_dim, stoch_dim * 2)

    def forward(self, latent, hidden=None, use_posterior=False, next_latent=None):
        # 1. 更新确定性状态
        deter, hidden = self.gru(latent, hidden)

        # 2. 预测随机状态
        if use_posterior and next_latent is not None:
            # Posterior: q(z_t | h_t, x_t)
            cat = torch.cat([deter, next_latent], -1)
            post_params = self.posterior_net(cat)
            mean, logvar = post_params.chunk(2, -1)
            stoch = self.reparameterize(mean, logvar)

            # Prior: p(z_t | h_t)
            prior_params = self.prior_net(deter)
            prior_mean, prior_logvar = prior_params.chunk(2, -1)

            kl_loss = self.kl_divergence(
                (mean, logvar), (prior_mean, prior_logvar)
            )

            return torch.cat([deter, stoch], -1), hidden, kl_loss
        else:
            # Prior only (inference)
            prior_params = self.prior_net(deter)
            mean, logvar = prior_params.chunk(2, -1)
            stoch = self.reparameterize(mean, logvar)

            return torch.cat([deter, stoch], -1), hidden, None
```

**优点**:
- 保持当前架构基本不变
- 添加多模态预测能力
- 理论上更robust

**缺点**:
- 增加训练复杂度
- 需要调整KL loss权重

### 8.2 为DreamerV3添加多智能体支持

**方案: Multi-Agent RSSM**

```python
class MultiAgentRSSM(nj.Module):
    def __init__(self, deter, stoch, classes, max_agents):
        self.max_agents = max_agents
        self.rssm = RSSM(deter, stoch, classes)

        # Agent aggregation
        self.agent_attention = TransformerEncoder(...)

    def obs_step(self, prev_state, prev_action, embed, masks):
        """
        embed: [B, K, embed_dim] - per-agent embeddings
        masks: [B, K] - agent存在mask
        """

        # 1. Per-agent RSSM
        agent_posts = []
        agent_priors = []

        for k in range(self.max_agents):
            if masks[:, k].any():
                post, prior = self.rssm.obs_step(
                    prev_state[k], prev_action, embed[:, k], ...
                )
                agent_posts.append(post)
                agent_priors.append(prior)

        # 2. Aggregate via attention
        aggregated = self.agent_attention(agent_posts, masks)

        return aggregated, agent_posts, agent_priors
```

---

## 9. 总结

### 核心差异矩阵

| 维度 | Traffic WM | DreamerV3 | 最佳应用 |
|------|-----------|-----------|---------|
| **任务类型** | 轨迹预测 | Model-based RL | Traffic: 预测<br>Dreamer: 决策 |
| **不确定性** | 确定性 | 随机性(RSSM) | Dreamer更robust |
| **多智能体** | ✅ 原生支持 | ❌ 需扩展 | Traffic胜出 |
| **计算效率** | 高(简单MLP) | 中(ResNet+采样) | Traffic更快 |
| **理论基础** | 监督学习 | 变分推断 | Dreamer更严谨 |
| **可解释性** | 高(确定映射) | 中(概率模型) | Traffic更直观 |

### 最终建议

**继续使用Traffic WM的场景**:
- ✅ 纯轨迹预测任务
- ✅ 需要快速训练和推理
- ✅ 多智能体交互建模
- ✅ 计算资源有限

**考虑借鉴DreamerV3的场景**:
- 🔄 需要建模多模态未来
- 🔄 需要不确定性量化
- 🔄 有图像输入需求
- 🔄 计划做强化学习扩展

**混合方案**:
- 保持Traffic WM的多智能体encoder
- 添加简化版RSSM dynamics (仅posterior-prior,无discrete)
- 保持简单的MLP decoder
- 添加KL regularization

---

**文档版本**: 1.0
**生成日期**: 2025-12-14
**作者**: Claude Code Analysis
**项目**: Traffic World Model vs DreamerV3
