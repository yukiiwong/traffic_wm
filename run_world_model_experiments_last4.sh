#!/usr/bin/env bash
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

TRAIN_SCRIPT="src/training/train_world_model.py"

TRAIN_DATA="data/processed/train_episodes.npz"
VAL_DATA="data/processed/val_episodes.npz"

LOG_DIR="logs/world_model"
CKPT_DIR="checkpoints/world_model"

mkdir -p "$LOG_DIR" "$CKPT_DIR"

# ===== 自动读取特征维度 =====
INPUT_DIM=$(python - << 'PY'
import json
with open("data/processed/metadata.json") as f:
    print(json.load(f)["n_features"])
PY
)

echo "Detected INPUT_DIM: $INPUT_DIM"

# ===== 固定训练超参 =====
BATCH_SIZE=128
N_EPOCHS=200
LR=3e-4

# ===== 你需要补跑的实验组合 =====
declare -a EXPERIMENTS=(
  "transformer 256 3"
  "transformer 512 1"
  "transformer 512 2"
  "transformer 512 3"
)

# GPU 轮询
GPU_IDS=(0 1)
NUM_GPUS=${#GPU_IDS[@]}
JOB_IDX=0

wait_for_slot() {
  while true; do
    running=$(jobs -r | wc -l)
    if (( running < NUM_GPUS )); then
      break
    fi
    sleep 3
  done
}

# ===== 主循环 =====
for exp in "${EXPERIMENTS[@]}"; do
  read dyn latent seed <<< "$exp"

  EXP_NAME="dyn_${dyn}_z${latent}_seed${seed}"
  LOG_FILE="${LOG_DIR}/${EXP_NAME}.log"
  CKPT_SUBDIR="${CKPT_DIR}/${EXP_NAME}"

  mkdir -p "$CKPT_SUBDIR"

  echo "▶ Launching: $EXP_NAME"
  echo "  GPU slot check..."

  wait_for_slot

  GPU_ID=${GPU_IDS[$(( JOB_IDX % NUM_GPUS ))]}
  JOB_IDX=$(( JOB_IDX + 1 ))

  echo "  → Using GPU: $GPU_ID"
  echo "  → Log: $LOG_FILE"

  CUDA_VISIBLE_DEVICES=$GPU_ID \
  python "$TRAIN_SCRIPT" \
    --train_data "$TRAIN_DATA" \
    --val_data "$VAL_DATA" \
    --input_dim "$INPUT_DIM" \
    --latent_dim "$latent" \
    --dynamics_type "$dyn" \
    --batch_size "$BATCH_SIZE" \
    --n_epochs "$N_EPOCHS" \
    --learning_rate "$LR" \
    --seed "$seed" \
    --checkpoint_dir "$CKPT_SUBDIR" \
    --log_dir "$CKPT_SUBDIR" \
    > "$LOG_FILE" 2>&1 &
done

echo "🎉 All remaining experiments submitted!"
echo "Use: tail -f logs/world_model/dyn_transformer_z512_seed1.log"
