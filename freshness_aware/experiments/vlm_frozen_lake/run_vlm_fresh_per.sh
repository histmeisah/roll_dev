#!/bin/bash
# ==============================================================================
# VLM + FreshPER: GroupReplayBuffer on FrozenLake RGB
# Run inside tmux: tmux new -s vlm_fresh && bash run_vlm_fresh_per.sh
# ==============================================================================
# Purpose: Validate VLM + replay buffer end-to-end on Qwen2.5-VL-3B.
#          Expected risk: multimodal tensors (pixel_values / image_grid_thw)
#          are not preserved by GroupReplayBuffer. Use this run to identify
#          and fix the issue.
# Server:  aicoder (main training)
# ==============================================================================
set -e

CONFIG_NAME="vlm_fresh_per"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -f "${CONFIG_NAME}.yaml" ]; then
    echo "Error: Config file '${CONFIG_NAME}.yaml' not found!"
    ls -1 *.yaml 2>/dev/null | sed 's/.yaml$//'
    exit 1
fi

export ROLL_PATH="/mnt/project_modelware_roce/zhaojian/weiyu/freshness_replaybuffer/ROLL"
export PYTHONPATH="$ROLL_PATH:$PYTHONPATH"

unset MASTER_ADDR
unset MASTER_PORT
unset RAY_ADDRESS
unset RAY_NODE_IP_ADDRESS

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export DS_SKIP_CUDA_CHECK=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_V1=0

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
export TRAINING_TIMESTAMP="$TIMESTAMP"
OUTPUT_DIR="$SCRIPT_DIR/output/$TIMESTAMP"

export WANDB_MODE=offline
export WANDB_API_KEY=local
export WANDB_DIR="$OUTPUT_DIR/wandb"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

mkdir -p "$OUTPUT_DIR/logs" "$OUTPUT_DIR/models" "$OUTPUT_DIR/render" "$OUTPUT_DIR/wandb"

LOG_FILE="$OUTPUT_DIR/logs/${CONFIG_NAME}_${TIMESTAMP}.log"
exec > >(tee -a "$LOG_FILE")
exec 2>&1

echo "========================================"
echo "VLM + FreshPER experiment"
echo "========================================"
echo "Config:    ${CONFIG_NAME}"
echo "Timestamp: $TIMESTAMP"
echo "Log:       $LOG_FILE"
echo "Output:    $OUTPUT_DIR"
echo "ROLL path: $ROLL_PATH"
echo "Model:     Qwen2.5-VL-3B-Instruct (VLTrajEnvManager)"
echo "Env:       FrozenLake (render_mode=rgb_array)"
echo "Steps:     100"
echo "Replay:    group-level, reward_fresh priority, age_decay=1000"
echo "========================================"

if [ -n "$TMUX" ]; then
    echo "tmux session: $(tmux display-message -p '#S'), window: $(tmux display-message -p '#W')"
fi

echo ""
echo "--- GPU availability ---"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
fi
echo ""

echo "--- cleaning up previous Ray clusters ---"
ray stop --force 2>/dev/null || true
pkill -9 -u $USER ray 2>/dev/null || true
rm -rf /tmp/ray/* 2>/dev/null || true
sleep 2
echo ""

cd "$ROLL_PATH"
echo "Training [${CONFIG_NAME}] started at $(date)"
echo "CWD: $ROLL_PATH"
echo ""

python examples/start_agentic_pipeline.py \
    --config_path ../../experiments/vlm_frozen_lake \
    --config_name "$CONFIG_NAME"

echo ""
echo "========================================"
echo "Training [${CONFIG_NAME}] completed at $(date)"
echo "========================================"
echo "Output: $OUTPUT_DIR"
echo "Log:    $LOG_FILE"
echo "========================================"
