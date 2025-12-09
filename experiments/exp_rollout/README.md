# Rollout Experiments

This directory contains experiments focused on long-horizon trajectory rollouts.

## Objectives

- Evaluate multi-step prediction capability
- Test different rollout strategies (open-loop vs closed-loop)
- Analyze error accumulation over time
- Compare different dynamics models (GRU vs LSTM vs Transformer)

## Experiments

### 1. Baseline Open-Loop Rollout
- Pure autoregressive prediction without ground truth
- Measure how errors accumulate over time

### 2. Teacher Forcing Comparison
- Compare open-loop (no teacher forcing) vs closed-loop (with teacher forcing)
- Analyze when teacher forcing provides benefits

### 3. Multi-Horizon Analysis
- Evaluate at multiple prediction horizons: 1s, 3s, 5s, 10s, 20s
- Identify at which horizon performance degrades significantly

## Running Experiments

```bash
# Run baseline rollout evaluation
python ../../src/evaluation/rollout_eval.py \
    --model_path ../../checkpoints/best_model.pt \
    --data_path ../../data/processed/test_episodes.npz \
    --context_length 10 \
    --rollout_length 20
```

## Results

Results will be saved in this directory with timestamps.
