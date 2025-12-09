# Claude Code Project Guide

## 1. Project Overview

This repository implements a **drone-based traffic world model**.

The goal is to:
- Read **drone-captured vehicle trajectories from CSV files**,
- Convert them into fixed-shape tensors of shape **[T, K, F]**:
  - T: time steps (1 Hz, e.g. 20–60 seconds per episode),
  - K: max number of vehicles per frame (with padding + masks),
  - F: per-vehicle features (e.g. x, y, vx, vy, heading, type),
- Train a **multi-agent latent world model** that can:
  - Predict / roll out future multi-agent states,
  - Perform closed-loop simulation of intersection traffic,
  - Optionally support a diffusion-based head for multi-modal generation.

The code should be clean, modular, and research-friendly (easy to run experiments).

---

## 2. Tech Stack and Environment

- Language: **Python 3.10+**
- Main libraries:
  - `numpy`
  - `pandas`
  - `torch` (PyTorch) for models and training
  - `matplotlib` for simple visualization (optional)
- Dependencies are listed in `requirements.txt`.

Please:
- Prefer **PyTorch** for deep learning code.
- Use **type hints** where reasonable.
- Avoid introducing extra heavy dependencies unless necessary.

---

## 3. Data Format

Raw trajectory data is stored as CSV files under `data/raw/`.

Each row roughly corresponds to:
- `scene_id`   : integer, which site / intersection this frame belongs to
- `frame`      : integer time index (1 Hz)
- `track_id`   : integer vehicle identifier
- `x`, `y`     : vehicle position in meters (global or local coordinates)
- `vx`, `vy`   : velocity components (optional, can be computed from positions)
- `heading`    : vehicle heading (radians or degrees)
- `type`       : vehicle type (car / truck etc., can be encoded as integers)

We want to preprocess data into episodes:
- Each episode is a sequence of length **T** (e.g. T=20–60 seconds).
- For each time step, we keep at most **K** vehicles.
  - Vehicles are sorted by a chosen rule (e.g. by `track_id` or distance to intersection center).
  - If fewer than K vehicles exist, we pad with zeros.
  - We create a `mask` tensor indicating which vehicles are real vs padding.

Target processed format (per episode):
- `states`: shape `[T, K, F]` as `float32`
- `masks` : shape `[T, K]` as `float32` (1.0 = valid vehicle, 0.0 = padding)
- `scene_id`: integer ID

These episodes will be fed into a `torch.utils.data.Dataset` that returns batches of:
- `states`: `[B, T, K, F]`
- `masks`: `[B, T, K]`
- `scene_id`: `[B]`

When writing code that consumes data, assume:
- `states[..., 0:2]` is `(x, y)` in meters,
- other feature indices should be documented in comments.

---

## 4. Implementation Status

### Completed Modules

✅ **Data Processing** (`src/data/`)
- `preprocess.py`: CSV to episode conversion
- `dataset.py`: PyTorch Dataset and DataLoader

✅ **Models** (`src/models/`)
- `encoder.py`: Multi-agent encoder (Transformer + MLP variants)
- `dynamics.py`: Latent dynamics (GRU/LSTM/Transformer/RSSM)
- `decoder.py`: State decoder (standard + autoregressive)
- `world_model.py`: Complete world model

✅ **Training** (`src/training/`)
- `train_world_model.py`: Training script with full loop
- `losses.py`: Loss functions (reconstruction, prediction, rollout, contrastive)

✅ **Evaluation** (`src/evaluation/`)
- `prediction_metrics.py`: ADE, FDE, velocity, heading, collision metrics
- `rollout_eval.py`: Multi-step evaluation utilities
- `visualization.py`: Plotting and animation tools

✅ **Utils** (`src/utils/`)
- `logger.py`: Logging setup
- `common.py`: Helper functions (seed, parameter counting, early stopping)

### Next Steps

When working on this project, consider:

1. **Data Integration**: Connect the preprocessing to actual drone CSV files from `D:\DRIFT\A\`
2. **Hyperparameter Tuning**: Experiment with different model configurations
3. **Advanced Features**:
   - Multi-modal prediction with diffusion models
   - Attention visualizations
   - Real-time inference optimization
4. **CARLA Integration**: Export predictions to CARLA simulator format

---

## 5. Common Tasks

### Training a Model

```bash
python src/training/train_world_model.py \
    --train_data data/processed/episodes.npz \
    --batch_size 32 \
    --n_epochs 100 \
    --latent_dim 256 \
    --dynamics_type gru
```

### Running Evaluation

```python
from src.models.world_model import WorldModel
from src.evaluation.rollout_eval import evaluate_rollout

model = WorldModel()
model.load_state_dict(torch.load('checkpoints/best_model.pt'))

metrics = evaluate_rollout(
    model=model,
    data_loader=test_loader,
    context_length=10,
    rollout_length=20
)
```

### Preprocessing New Data

```bash
python src/data/preprocess.py
```

---

## 6. Code Style Guidelines

- Use clear, descriptive variable names
- Add docstrings to all functions and classes
- Type hints for function arguments and returns
- Keep functions focused and modular
- Comment complex logic
- Follow PEP 8 style guide

---

## 7. Troubleshooting

### CUDA Out of Memory
- Reduce `batch_size`
- Reduce `max_agents` or `latent_dim`
- Use gradient accumulation

### Slow Training
- Reduce `n_encoder_layers` or `n_dynamics_layers`
- Use smaller `hidden_dim`
- Enable mixed precision training

### Poor Prediction Quality
- Increase model capacity
- Adjust loss weights
- Check data normalization
- Verify mask correctness

---

## 8. References

- World Models: https://arxiv.org/abs/1803.10122
- Dreamer: https://arxiv.org/abs/1912.01603
- Trajectory Prediction: https://arxiv.org/abs/2104.10133
