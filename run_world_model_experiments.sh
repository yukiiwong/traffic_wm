#!/usr/bin/env bash
set -euo pipefail

# ========= 基本路径设置 =========
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

TRAIN_SCRIPT="src/training/train_world_model.py"

TRAIN_DATA="data/processed/train_episodes.npz"
VAL_DATA="data/processed/val_episodes.npz"

LOG_DIR="logs/world_model"
CKPT_DIR="checkpoints/world_model"

mkdir -p "$LOG_DIR" "$CKPT_DIR"

# ========= 固定参数（可以按需改） =========
INPUT_DIM=11              # 和 metadata.json 里的 n_features 一致
BATCH_SIZE=128
N_EPOCHS=200
LR=3e-4

ENCODER_HIDDEN=128
DYNAMICS_HIDDEN=512
LATENT_DIMS=(256 512)     # 对比两个 latent dim

# 支持的 dynamics_type：gru / lstm / transformer
DYNAMICS_TYPES=("gru" "lstm" "transformer")

# 多随机种子（可选）
SEEDS=(1 2 3)

# 控制并行数（比如同时跑 3 个实验，如果你想全部串行，就设成 1）
MAX_JOBS=3

# ========= 一个小函数：限制并行任务数 =========
wait_for_free_slot() {
  while true; do
    # 统计当前后台运行的任务数
    local njobs
    njobs=$(jobs -r | wc -l)
    if (( njobs < MAX_JOBS )); then
      break
    fi
    sleep 5
  done
}

# ========= 主循环：扫所有组合 =========
for dyn in "${DYNAMICS_TYPES[@]}"; do
  for latent in "${LATENT_DIMS[@]}"; do
    for seed in "${SEEDS[@]}"; do

      EXP_NAME="dyn_${dyn}_z${latent}_seed${seed}"
      LOG_FILE="${LOG_DIR}/${EXP_NAME}.log"
      CKPT_SUBDIR="${CKPT_DIR}/${EXP_NAME}"
      mkdir -p "$CKPT_SUBDIR"

      echo "▶ Starting experiment: ${EXP_NAME}"
      echo "  dynamics_type=${dyn}, latent_dim=${latent}, seed=${seed}"
      echo "  logs:   ${LOG_FILE}"
      echo "  ckpt:   ${CKPT_SUBDIR}"

      wait_for_free_slot

      CUDA_VISIBLE_DEVICES=0 \
      python "$TRAIN_SCRIPT" \
        --train_data "$TRAIN_DATA" \
        --val_data   "$VAL_DATA" \
        --input_dim  "$INPUT_DIM" \
        --latent_dim "$latent" \
        --encoder_hidden  "$ENCODER_HIDDEN" \
        --dynamics_hidden "$DYNAMICS_HIDDEN" \
        --dynamics_type   "$dyn" \
        --batch_size  "$BATCH_SIZE" \
        --n_epochs    "$N_EPOCHS" \
        --learning_rate "$LR" \
        --seed        "$seed" \
        --output_dir  "$CKPT_SUBDIR" \
        >"$LOG_FILE" 2>&1 &

      # 去掉上面的 & 则改为串行运行

    done
  done
done

echo "🎉 All experiments submitted. Use 'jobs -l' to check running status."
