#!/usr/bin/env bash
set -euo pipefail

# 避免显存碎片导致的 OOM
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ====== 路径设置 ======
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

TRAIN_SCRIPT="src/training/train_world_model.py"

TRAIN_DATA="data/processed/train_episodes.npz"
VAL_DATA="data/processed/val_episodes.npz"

LOG_DIR="logs/world_model"
CKPT_DIR="checkpoints/world_model"

mkdir -p "$LOG_DIR" "$CKPT_DIR"

# ====== GPU 设置（这里用 0 和 1 两块卡）======
GPU_IDS=(0 1)                    # 可用 GPU 列表
NUM_GPUS=${#GPU_IDS[@]}          # GPU 数量
MAX_JOBS=$NUM_GPUS               # 同时最多跑几个实验（这里 = GPU 数）

JOB_IDX=0                        # 实验计数器，用于轮流分配 GPU

# ====== 从 metadata.json 自动读取 input_dim ======
INPUT_DIM=$(python - << 'PY'
import json
with open("data/processed/metadata.json") as f:
    meta = json.load(f)
print(meta["n_features"])
PY
)

echo "Detected INPUT_DIM from metadata.json: ${INPUT_DIM}"

# ====== 固定训练超参（按需修改） ======
BATCH_SIZE=128          # 稍微保守，防止 OOM；你可以试着改回 128
N_EPOCHS=200
LR=3e-4

LATENT_DIMS=(256 512)                 # 对比两种 latent_dim
DYNAMICS_TYPES=("gru" "lstm" "transformer")  # 三种 dynamics
SEEDS=(1 2 3)                         # 多随机种子

# ====== 并行控制：限制后台任务数 ======
wait_for_free_slot() {
  while true; do
    local njobs
    njobs=$(jobs -r | wc -l)   # 当前正在运行的后台任务数
    if (( njobs < MAX_JOBS )); then
      break
    fi
    sleep 5
  done
}

# ====== 主循环：扫所有组合，并轮流分配到不同 GPU ======
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

      # 等待有空闲“槽位”（不超过 MAX_JOBS 个并行任务）
      wait_for_free_slot

      # 轮流选择 GPU：0,1,0,1,...
      GPU_ID=${GPU_IDS[$(( JOB_IDX % NUM_GPUS ))]}
      JOB_IDX=$(( JOB_IDX + 1 ))
      echo "  using GPU: ${GPU_ID}"

      # 启动训练（放到后台跑，输出写入 log）
      CUDA_VISIBLE_DEVICES=${GPU_ID} \
      python "$TRAIN_SCRIPT" \
        --train_data     "$TRAIN_DATA" \
        --val_data       "$VAL_DATA" \
        --input_dim      "$INPUT_DIM" \
        --latent_dim     "$latent" \
        --dynamics_type  "$dyn" \
        --batch_size     "$BATCH_SIZE" \
        --n_epochs       "$N_EPOCHS" \
        --learning_rate  "$LR" \
        --seed           "$seed" \
        --checkpoint_dir "$CKPT_SUBDIR" \
        --log_dir        "$CKPT_SUBDIR" \
        >"$LOG_FILE" 2>&1 &

    done
  done
done

echo "🎉 All experiments submitted. Use 'nvidia-smi' to monitor GPUs, and 'tail -f logs/world_model/xxx.log' to watch training."
