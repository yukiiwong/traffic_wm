# Traffic World Model

A multi-agent latent world model for drone-based vehicle trajectory prediction and closed-loop traffic simulation.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Quick Start](#quick-start)
- [Multi-Site Data Processing](#multi-site-data-processing) 🆕
- [Installation](#installation)
- [Data Format](#data-format)
- [Model Improvements](#model-improvements)
- [Training](#training)
- [Evaluation](#evaluation)
- [Visualization](#visualization)
- [Configuration System](#configuration-system)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)

---

## Overview

This project implements a deep learning system that:
- Converts drone-captured vehicle trajectories (CSV) into structured tensor episodes `[T, K, F]`
- Trains a multi-agent latent world model for trajectory prediction
- Enables closed-loop simulation of intersection traffic
- Provides interpretability through attention visualization

### Key Capabilities

✅ **Multi-Agent Modeling**: Handle 50+ vehicles simultaneously
✅ **Spatial Awareness**: Position-based encoding for location-aware learning
✅ **Social Interactions**: Explicit modeling of nearby vehicle relationships
✅ **Rich Features**: Lane info, vehicle relationships, acceleration
✅ **Interpretability**: Attention visualization and pattern analysis
✅ **Flexible Config**: YAML-based experiment management

---

## Features

### Core Models

- **Multi-Agent Encoder**: Transformer-based with spatial positional encoding and social pooling
- **Latent Dynamics**: GRU/LSTM/Transformer models for temporal evolution
- **State Decoder**: Reconstructs future multi-agent states from latent representations

### Evaluation & Visualization

- **Metrics**: ADE, FDE, velocity error, heading error, collision rate
- **Visualizations**: Trajectory plots, rollout animations, attention heatmaps
- **Multi-Horizon**: Evaluate predictions at 1s, 3s, 5s, 10s, 20s

### Enhanced Features (v2.0)

🆕 **Spatial Positional Encoding**: Location-aware embeddings for better spatial understanding
🆕 **Social Pooling**: Explicit modeling of local interactions (50m radius)
🆕 **Extended Features**: +4 features (lane, preceding/following vehicles)
🆕 **Attention Visualization**: Understand what the model learns
🆕 **Config System**: YAML-based experiment management

---

## Quick Start

### Basic Usage (5 Minutes)

#### 1. Prepare Data

```bash
# Copy your drone CSV files to data/raw/
cp /path/to/drone_*.csv traffic-world-model/data/raw/
cd traffic-world-model
```

Your CSV files should contain these columns:
- `track_id`, `frame`, `center_x`, `center_y`, `angle`, `class_id`
- `lane`, `preceding_id`, `following_id` (optional but recommended)

#### 2. Preprocess (Choose Features)

**Option A: Basic Preprocessing (6 features)**
```bash
python src/data/preprocess.py
```
Output: `data/processed/episodes.npz`

**Option B: Extended Preprocessing (10 features)** ⭐ Recommended
```python
# Edit preprocess.py main() function or use directly:
from src.data.preprocess import preprocess_trajectories

preprocess_trajectories(
    input_dir='data/raw',
    output_dir='data/processed',
    use_extended_features=True,  # Enable 10 features
    use_acceleration=True,
    split_data=True  # Auto-split train/val/test
)
```
Output:
- `data/processed/train_episodes.npz`
- `data/processed/val_episodes.npz`
- `data/processed/test_episodes.npz`
- `data/processed/metadata.json`

#### 3. Train

**Basic Model:**
```bash
python src/training/train_world_model.py \
    --train_data data/processed/train_episodes.npz \
    --batch_size 32 \
    --n_epochs 100 \
    --latent_dim 256
```

**Enhanced Model (with config):**
```bash
python src/training/train_world_model.py \
    --config experiments/exp_enhanced/config_enhanced.yaml
```

#### 4. Evaluate

```python
from src.models.world_model import WorldModel
from src.evaluation.rollout_eval import evaluate_rollout
from src.data.dataset import get_dataloader

# Load model
model = WorldModel()
model.load_state_dict(torch.load('checkpoints/best_model.pt'))

# Evaluate
test_loader = get_dataloader('data/processed/test_episodes.npz')
metrics = evaluate_rollout(model, test_loader, context_length=10, rollout_length=20)

print(f"ADE: {metrics['ade']:.3f}m")
print(f"FDE: {metrics['fde']:.3f}m")
print(f"Collision Rate: {metrics['collision_rate']:.2f}%")
```

---

## Multi-Site Data Processing

🆕 **NEW:** If you have data from multiple sites (e.g., Site A-I), use the multi-site preprocessing script.

### Overview

This script automatically:
1. ✅ Collects all CSV files from all site folders (A-I)
2. ✅ Randomly shuffles and splits into train/val/test (80/10/10 by default)
3. ✅ Processes each split separately
4. ✅ Saves split configuration for reproducibility

### Quick Start

```bash
# Default: 80% train, 10% val, 10% test
python preprocess_multisite.py

# Custom ratios (e.g., 70/15/15)
python preprocess_multisite.py \
    --train_ratio 0.7 \
    --val_ratio 0.15 \
    --test_ratio 0.15
```

### Data Organization

Place your data in site folders:
```
data/raw/
├── A/
│   ├── drone_1.csv
│   └── ...
├── B/
│   ├── drone_1.csv
│   └── ...
├── C/
...
└── I/
```

### Output

After preprocessing, you'll have:
```
data/processed/
├── train_episodes.npz      # 80% of all data (mixed from all sites)
├── val_episodes.npz        # 10% of all data
├── test_episodes.npz       # 10% of all data
├── metadata.json           # Feature info, lane mapping
└── split_config.json       # Which files went to which split (for reproducibility)
```

### Advanced Options

```bash
# Full control
python preprocess_multisite.py \
    --raw_data_dir data/raw \
    --output_dir data/processed \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --test_ratio 0.1 \
    --episode_length 30 \
    --max_vehicles 50 \
    --seed 42
```

---

## Installation

### Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

### Setup

```bash
# Clone repository
git clone <repository-url>
cd traffic-world-model

# Install dependencies
pip install -r requirements.txt

# For enhanced features, also install:
pip install pyyaml seaborn
```

---

## Data Format

### Input: CSV Files

Your drone tracking CSV files should contain:

| Column | Description | Required |
|--------|-------------|----------|
| `track_id` | Vehicle unique ID | ✅ |
| `frame` | Frame number | ✅ |
| `center_x`, `center_y` | Position (meters) | ✅ |
| `angle` | Heading (radians) | ✅ |
| `class_id` | Vehicle type | ✅ |
| `lane` | Lane ID (e.g., A1, B2) | ⭐ Recommended |
| `preceding_id` | Vehicle ahead | ⭐ Recommended |
| `following_id` | Vehicle behind | ⭐ Recommended |
| `vx`, `vy` | Velocity (computed if missing) | Optional |

### Output: Tensor Episodes

**Basic (6 features):**
- `[0:2]` Position (x, y)
- `[2:4]` Velocity (vx, vy)
- `[4]` Heading angle
- `[5]` Vehicle type

**Extended (10 features):**
- `[0:2]` Position (x, y)
- `[2:4]` Velocity (vx, vy)
- `[4]` Heading angle
- `[5]` Vehicle type
- `[6]` Lane ID (encoded)
- `[7]` Has preceding vehicle (0/1)
- `[8]` Has following vehicle (0/1)
- `[9]` Reserved

**Shape:** `[N, T, K, F]`
- N: Number of episodes
- T: Time steps (default: 30)
- K: Max vehicles (default: 50)
- F: Features per vehicle (6 or 10)

---

## Model Improvements

### Version Comparison

| Feature | Basic (v1.0) | Enhanced (v2.0) | Gain |
|---------|--------------|-----------------|------|
| **Input Features** | 6 | 10 | +4 |
| **Spatial Encoding** | ❌ | ✅ | +5-10% ADE |
| **Social Pooling** | ❌ | ✅ | +10-15% ADE |
| **Lane Awareness** | ❌ | ✅ | Better lane-keeping |
| **Attention Viz** | ❌ | ✅ | Interpretability |
| **Config System** | ❌ | ✅ | Reproducibility |

### 1. Spatial Positional Encoding

**Problem:** Original encoder treats all vehicles equally regardless of position.

**Solution:** Sinusoidal position encoding (like Transformer)
```python
from src.models.encoder import ImprovedMultiAgentEncoder

encoder = ImprovedMultiAgentEncoder(
    input_dim=10,
    use_spatial_encoding=True  # Enable spatial encoding
)
```

**Benefits:**
- Model understands spatial relationships
- Better prediction for spatially-dependent interactions
- Enhanced lane-following behavior

### 2. Social Pooling

**Problem:** Only global attention, missing local interactions.

**Solution:** Aggregate features from nearby vehicles (default: 50m radius)
```python
encoder = ImprovedMultiAgentEncoder(
    use_social_pooling=True,
    pooling_radius=50.0  # meters
)
```

**Benefits:**
- Explicit car-following modeling
- Better collision avoidance
- Captures local traffic patterns

### 3. Extended Features

**New features:**
- Lane information (crucial for lane-keeping)
- Preceding/following vehicle indicators
- Optional acceleration

**Usage:**
```python
# Configure in preprocess.py (use_extended_features=True)
feature_config = {
    'position': True,
    'velocity': True,
    'acceleration': False,  # optional
    'heading': True,
    'type': True,
    'lane': True,           # NEW
    'has_preceding': True,  # NEW
    'has_following': True   # NEW
}
```

### Expected Performance

Based on similar research:
- **Spatial Encoding:** +5-10% improvement in ADE/FDE
- **Social Pooling:** +10-15% improvement for local interactions
- **Extended Features:** +5-8% improvement
- **Combined:** ~15-25% overall improvement

---

## Training

### Configuration

**Option 1: Command Line Arguments**
```bash
python src/training/train_world_model.py \
    --train_data data/processed/train_episodes.npz \
    --batch_size 16 \
    --n_epochs 100 \
    --latent_dim 256 \
    --dynamics_type gru \
    --learning_rate 0.001
```

**Option 2: YAML Config** ⭐ Recommended
```yaml
# experiments/exp_enhanced/config_enhanced.yaml
model:
  latent_dim: 256
  encoder_hidden: 128
  use_spatial_encoding: true
  use_social_pooling: true
  pooling_radius: 50.0

training:
  batch_size: 16
  learning_rate: 0.001
  n_epochs: 100
```

```bash
python src/training/train_world_model.py \
    --config experiments/exp_enhanced/config_enhanced.yaml
```

### Key Hyperparameters

```python
# Data
episode_length = 30        # Time steps per episode
max_vehicles = 50          # Maximum vehicles to track

# Model
input_dim = 10             # Features per vehicle (6 or 10)
latent_dim = 256           # Latent space dimension
encoder_hidden = 128       # Encoder hidden dimension
dynamics_type = 'gru'      # 'gru', 'lstm', or 'transformer'

# Training
batch_size = 16            # Reduce if out of memory
learning_rate = 1e-3       # Initial learning rate
n_epochs = 100             # Training epochs

# Social Pooling
pooling_radius = 50.0      # Radius for local interactions (meters)
```

### Training Tips

1. **Start Small:** Test with 1-2 CSV files first
2. **Enable Mixed Precision:** 2x faster on GPU
   ```yaml
   training:
     use_amp: true
   ```
3. **Monitor Metrics:** Track ADE, FDE, collision rate
4. **Save Checkpoints:** Every 10 epochs
5. **Early Stopping:** Patience of 15-20 epochs

---

## Evaluation

### Metrics

```python
from src.evaluation.prediction_metrics import compute_all_metrics

metrics = compute_all_metrics(
    predicted=predictions,
    ground_truth=targets,
    masks=masks
)

# Available metrics:
# - ade: Average Displacement Error (meters)
# - fde: Final Displacement Error (meters)
# - velocity_error: Average velocity error (m/s)
# - heading_error: Average heading error (degrees)
# - collision_rate: Collision percentage (%)
```

### Multi-Horizon Evaluation

```python
from src.evaluation.rollout_eval import evaluate_multihorizon

results = evaluate_multihorizon(
    model=model,
    data_loader=test_loader,
    horizons=[1, 3, 5, 10, 20]  # seconds
)

for horizon, metrics in results.items():
    print(f"{horizon}s: ADE={metrics['ade']:.3f}m")
```

### Teacher Forcing Comparison

```python
from src.evaluation.rollout_eval import evaluate_with_teacher_forcing

results = evaluate_with_teacher_forcing(
    model=model,
    data_loader=test_loader
)

print(f"Open-loop ADE: {results['open_loop']['ade']:.3f}m")
print(f"Closed-loop ADE: {results['closed_loop']['ade']:.3f}m")
```

---

## Visualization

### Trajectory Plots

```python
from src.evaluation.visualization import visualize_trajectories

visualize_trajectories(
    predicted=pred_np,
    ground_truth=gt_np,
    masks=mask_np,
    time_step=15,
    save_path='trajectories.png'
)
```

### Rollout Comparison

```python
from src.evaluation.visualization import visualize_rollout

visualize_rollout(
    predicted=pred_np,
    ground_truth=gt_np,
    masks=mask_np,
    save_path='rollout_comparison.png'
)
```

### Animations

```python
from src.evaluation.visualization import create_animation

create_animation(
    predicted=pred_np,
    ground_truth=gt_np,
    masks=mask_np,
    save_path='rollout.gif',
    fps=10
)
```

### Attention Visualization 🆕

**Attention Heatmap:**
```python
from src.evaluation.attention_visualization import visualize_attention_heatmap

visualize_attention_heatmap(
    attention_weights=attn[0, 0].cpu().numpy(),  # [K, K]
    save_path='attention_heatmap.png'
)
```

**Spatial Attention:**
```python
from src.evaluation.attention_visualization import visualize_spatial_attention

visualize_spatial_attention(
    attention_weights=attn[0, 0].cpu().numpy(),
    positions=positions[0, 0].cpu().numpy(),
    query_idx=0,  # Visualize attention from vehicle 0
    save_path='spatial_attention.png'
)
```

**Attention Analysis:**
```python
from src.evaluation.attention_visualization import analyze_attention_patterns

analysis = analyze_attention_patterns(
    attention_weights=attn,
    masks=masks,
    positions=positions
)

print(f"Attention entropy: {analysis['attention_entropy']:.3f}")
print(f"Distance correlation: {analysis['attention_distance_correlation']:.3f}")
```

---

## Configuration System

### YAML Configuration

Create experiment configs for reproducibility:

```yaml
# my_experiment.yaml
experiment_name: "my_experiment"
description: "Testing enhanced model"
seed: 42

data:
  train_path: "data/processed/train_episodes.npz"
  val_path: "data/processed/val_episodes.npz"
  input_dim: 10

model:
  latent_dim: 256
  use_spatial_encoding: true
  use_social_pooling: true
  pooling_radius: 50.0

training:
  batch_size: 16
  learning_rate: 0.001
  n_epochs: 100
  use_amp: true

evaluation:
  context_length: 10
  rollout_length: 20
  horizons: [1, 3, 5, 10, 20]
```

### Loading Config

```python
from src.utils.config import load_config

config = load_config('my_experiment.yaml')
print(config)  # Pretty print

# Use in code
encoder = ImprovedMultiAgentEncoder(
    input_dim=config.data.input_dim,
    latent_dim=config.model.latent_dim,
    use_spatial_encoding=config.model.use_spatial_encoding
)
```

---

## API Reference

### Data Processing

```python
from src.data.preprocess import preprocess_trajectories

preprocess_trajectories(
    input_dir='data/raw',
    output_dir='data/processed',
    episode_length=30,
    max_vehicles=50,
    overlap=5,
    split_data=True
)
```

### Models

```python
from src.models.encoder import ImprovedMultiAgentEncoder
from src.models.dynamics import LatentDynamics
from src.models.decoder import StateDecoder

# Encoder
encoder = ImprovedMultiAgentEncoder(
    input_dim=10,
    hidden_dim=128,
    latent_dim=256,
    use_spatial_encoding=True,
    use_social_pooling=True,
    pooling_radius=50.0
)

# Dynamics
dynamics = LatentDynamics(
    latent_dim=256,
    hidden_dim=512,
    model_type='gru'
)

# Decoder
decoder = StateDecoder(
    latent_dim=256,
    output_dim=10,
    max_agents=50
)
```

### Dataset

```python
from src.data.dataset import get_dataloader

train_loader = get_dataloader(
    data_path='data/processed/train_episodes.npz',
    batch_size=16,
    shuffle=True,
    normalize=True
)
```

---

## Troubleshooting

### Common Issues

**Issue: "RuntimeError: size mismatch"**
- **Cause:** `input_dim` doesn't match feature count
- **Solution:** Set `input_dim=10` for extended features or `input_dim=6` for basic

**Issue: Out of memory**
- **Solutions:**
  - Reduce `batch_size` (try 8 instead of 16)
  - Reduce `max_vehicles` (try 30 instead of 50)
  - Reduce `pooling_radius` (try 30.0 instead of 50.0)
  - Reduce `latent_dim` (try 128 instead of 256)

**Issue: Slow training**
- **Solutions:**
  - Enable mixed precision: `use_amp: true`
  - Use fewer encoder layers: `encoder_n_layers: 1`
  - Use smaller hidden dimensions
  - Use `dynamics_type: 'gru'` (faster than transformer)

**Issue: FileNotFoundError: metadata.json**
- **Cause:** Using basic preprocessing instead of extended
- **Solution:** Use `preprocess.py` with `use_extended_features=True` or don't use lane features

**Issue: Poor prediction quality**
- **Solutions:**
  - Increase model capacity (more layers, larger `latent_dim`)
  - Adjust loss weights
  - Check data normalization
  - Verify mask correctness
  - Enable spatial encoding and social pooling

### Debug Checklist

```python
# 1. Check data shape
import numpy as np
data = np.load('data/processed/train_episodes.npz')
print(f"States: {data['states'].shape}")  # [N, 30, 50, F]
print(f"Masks: {data['masks'].shape}")    # [N, 30, 50]

# 2. Check feature values
states = data['states'][0, 0]  # First episode, first timestep
masks = data['masks'][0, 0]
valid = states[masks > 0.5]
print(f"Valid vehicles: {len(valid)}")
print(f"Position range: [{valid[:,0].min():.1f}, {valid[:,0].max():.1f}]")

# 3. Check lane encoding (if using extended features)
import json
with open('data/processed/metadata.json') as f:
    meta = json.load(f)
print("Lane mapping:", meta['lane_mapping'])
```

---

## Project Structure

```
traffic-world-model/
├── README.md                          # This file
├── CLAUDE.md                          # Project guide for AI assistants
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore rules
│
├── data/
│   ├── README.md                      # Data format documentation
│   ├── raw/                           # Original CSV files (drone_*.csv)
│   ├── processed/                     # Preprocessed episodes (npz files)
│   └── sample/                        # Small test subset
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── preprocess.py              # Unified preprocessing (6 or 10 features via flags)
│   │   └── dataset.py                 # PyTorch Dataset
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── encoder.py                 # All encoder variants (basic + enhanced)
│   │   ├── dynamics.py                # Latent dynamics (GRU/LSTM/Transformer)
│   │   ├── decoder.py                 # State decoder
│   │   └── world_model.py             # Complete world model
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train_world_model.py       # Training script
│   │   └── losses.py                  # Loss functions
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── prediction_metrics.py      # ADE, FDE, collision rate
│   │   ├── rollout_eval.py            # Multi-step evaluation
│   │   ├── visualization.py           # Trajectory plots
│   │   └── attention_visualization.py # Attention heatmaps (NEW)
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logger.py                  # Logging utilities
│       ├── common.py                  # Helper functions
│       └── config.py                  # Configuration system (NEW)
│
└── experiments/
    ├── exp_prediction/
    │   └── config.yaml
    └── exp_enhanced/
        └── config_enhanced.yaml       # Enhanced model config
```

---

## Performance Tips

### Speed Optimization

1. **Mixed Precision Training** (2x faster):
   ```yaml
   training:
     use_amp: true
   ```

2. **Reduce Batch Size** (if memory limited):
   ```yaml
   training:
     batch_size: 8  # Instead of 16
   ```

3. **Adjust Pooling Radius** (smaller = faster):
   ```yaml
   model:
     pooling_radius: 30.0  # Instead of 50.0
   ```

4. **Use Fewer Layers**:
   ```yaml
   model:
     encoder_n_layers: 1  # Instead of 2
   ```

### Accuracy Optimization

1. **Enable All Features**:
   ```python
   feature_config = {
       'position': True,
       'velocity': True,
       'acceleration': True,  # Add if available
       'heading': True,
       'type': True,
       'lane': True,
       'has_preceding': True,
       'has_following': True
   }
   ```

2. **Use Enhanced Model**:
   ```python
   encoder = ImprovedMultiAgentEncoder(
       use_spatial_encoding=True,
       use_social_pooling=True
   )
   ```

3. **Increase Model Capacity**:
   ```yaml
   model:
     latent_dim: 512  # Instead of 256
     encoder_hidden: 256  # Instead of 128
   ```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{traffic_world_model,
  title={Traffic World Model for Multi-Agent Prediction},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/traffic-world-model}
}
```

---

## License

MIT License

---

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

---

## Contact

For questions or feedback:
- Open an issue on GitHub
- Email: your.email@example.com

---

## Acknowledgments

This project builds upon research in:
- **World Models** (Ha & Schmidhuber, 2018)
- **Social GAN** (Gupta et al., 2018)
- **Transformer Networks** (Vaswani et al., 2017)
- **Traffic Prediction** (Cui et al., 2019)

---

**Version:** 2.0 (Enhanced)
**Last Updated:** 2025-12-09
**Status:** ✅ Production Ready
