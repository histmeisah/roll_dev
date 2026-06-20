#!/bin/bash
# ==============================================================================
# VLM MathVista Experiments Runner - 4x H100
# ==============================================================================
set -e

CONFIG_NAME="traj_baseline"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -f "${CONFIG_NAME}.yaml" ]; then
    echo "Error: Config file '${CONFIG_NAME}.yaml' not found!"
    ls -1 *.yaml 2>/dev/null | sed 's/.yaml$//' || echo "No config files found"
    exit 1
fi

export ROLL_PATH="/mnt/project_modelware_roce/zhaojian/liangsirui/weiyu/projects/local_roll_dev/roll_dev/ROLL"
export PYTHONPATH="$ROLL_PATH:$PYTHONPATH"

unset MASTER_ADDR MASTER_PORT RAY_ADDRESS RAY_NODE_IP_ADDRESS

export CUDA_VISIBLE_DEVICES=0,1,2,3

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
export TRAINING_TIMESTAMP="$TIMESTAMP"
OUTPUT_DIR="$SCRIPT_DIR/output/$TIMESTAMP"

export WANDB_MODE=offline
export WANDB_API_KEY=local
export WANDB_DIR="$OUTPUT_DIR/wandb"

# HuggingFace offline mode (use cached datasets, skip Hub connection)
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

mkdir -p "$OUTPUT_DIR/logs" "$OUTPUT_DIR/models" "$OUTPUT_DIR/render" "$OUTPUT_DIR/wandb"

LOG_FILE="$OUTPUT_DIR/logs/training_${TIMESTAMP}.log"
exec > >(tee -a "$LOG_FILE")
exec 2>&1

echo "========================================"
echo "VLM MathVista Experiment: ${CONFIG_NAME}"
echo "========================================"
echo "Timestamp: $TIMESTAMP"
echo "Config: ${CONFIG_NAME}.yaml"
echo "Output directory: $OUTPUT_DIR"
echo "Hardware: 4x H100"
echo "========================================"

if [ -n "$TMUX" ]; then
    echo "Running in tmux: $(tmux display-message -p '#S:#W')"
fi

if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
fi

ray stop --force 2>/dev/null || true
pkill -9 -u $USER ray 2>/dev/null || true
rm -rf /tmp/ray/* 2>/dev/null || true
sleep 2

cd "$ROLL_PATH"
echo "Training started at $(date)"

python examples/start_agentic_pipeline.py \
    --config_path ../../experiments/vlm_math_vista \
    --config_name "$CONFIG_NAME"

echo "Training completed at $(date)"
echo "Output directory: $OUTPUT_DIR"
