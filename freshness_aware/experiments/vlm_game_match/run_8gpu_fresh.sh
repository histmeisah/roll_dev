#!/bin/bash
# ==============================================================================
# VLM GameMatch Experiments Runner - 8x H100
# ==============================================================================
# Available configs:
#   Baseline (no replay):
#     - traj_baseline_8gpu      : VLM Trajectory baseline
#
#   Off-Policy with Standard PER:
#     - traj_per_8gpu           : VLM + Trajectory level + reward priority + IS
#
#   Off-Policy with Reward-Fresh (our extension):
#     - traj_reward_fresh_8gpu  : VLM + Trajectory level + reward_fresh + IS
# ==============================================================================
set -e

# =============================================================================
# CONFIG NAME - Modify this to run different experiments
# =============================================================================
CONFIG_NAME="traj_reward_fresh_8gpu"
# =============================================================================

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Verify config file exists
if [ ! -f "${CONFIG_NAME}.yaml" ]; then
    echo "Error: Config file '${CONFIG_NAME}.yaml' not found!"
    echo "Available configs:"
    ls -1 *.yaml 2>/dev/null | sed 's/.yaml$//' || echo "No config files found"
    exit 1
fi

# Set ROLL path
export ROLL_PATH="/mnt/project_modelware_roce/zhaojian/liangsirui/weiyu/projects/local_roll_dev/roll_dev/ROLL"
export PYTHONPATH="$ROLL_PATH:$PYTHONPATH"

# Clear Ray environment variables (let ROLL manage Ray)
unset MASTER_ADDR
unset MASTER_PORT
unset RAY_ADDRESS
unset RAY_NODE_IP_ADDRESS

# Set visible GPUs (8x H100)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Generate unified timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
export TRAINING_TIMESTAMP="$TIMESTAMP"
OUTPUT_DIR="$SCRIPT_DIR/output/$TIMESTAMP"

# Set wandb to offline mode (no login required)
export WANDB_MODE=offline
export WANDB_API_KEY=local
export WANDB_DIR="$OUTPUT_DIR/wandb"

# HuggingFace offline mode (use cached datasets, skip Hub connection)
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

# Create output directories
mkdir -p "$OUTPUT_DIR/logs"
mkdir -p "$OUTPUT_DIR/models"
mkdir -p "$OUTPUT_DIR/render"
mkdir -p "$OUTPUT_DIR/wandb"

# Set log file
LOG_FILE="$OUTPUT_DIR/logs/training_${TIMESTAMP}.log"

# Redirect stdout and stderr to both screen and log file
exec > >(tee -a "$LOG_FILE")
exec 2>&1

echo "========================================"
echo "VLM GameMatch Experiment: ${CONFIG_NAME}"
echo "========================================"
echo "Timestamp: $TIMESTAMP"
echo "Config: ${CONFIG_NAME}.yaml"
echo "Log file: $LOG_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "Hardware: 8x H100"
echo "  GPU 0-3: actor_train + reference"
echo "  GPU 4-7: actor_infer (vLLM)"
echo "========================================"

# Check tmux session info
if [ -n "$TMUX" ]; then
    TMUX_SESSION=$(tmux display-message -p '#S')
    TMUX_WINDOW=$(tmux display-message -p '#W')
    echo "Running in tmux session: $TMUX_SESSION, window: $TMUX_WINDOW"
fi

# Check GPU availability
echo ""
echo "========================================"
echo "Checking GPU availability..."
echo "========================================"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
else
    echo "nvidia-smi not available, skipping GPU check"
fi
echo ""

# Clean up existing Ray clusters
echo "========================================"
echo "Cleaning up existing Ray clusters..."
echo "========================================"
ray stop --force 2>/dev/null || true
pkill -9 -u $USER ray 2>/dev/null || true
rm -rf /tmp/ray/* 2>/dev/null || true
sleep 2
echo "Cleanup completed. ROLL will manage Ray initialization."
echo "========================================"
echo ""

# Start training
cd "$ROLL_PATH"
echo "Training started at $(date)"
echo "Working directory: $ROLL_PATH"
echo ""

python examples/start_agentic_pipeline.py \
    --config_path ../../experiments/vlm_game_match \
    --config_name "$CONFIG_NAME"

echo ""
echo "========================================"
echo "Training completed at $(date)"
echo "========================================"
echo "Config: ${CONFIG_NAME}"
echo "Output directory: $OUTPUT_DIR"
echo "Training log: $LOG_FILE"
echo "========================================"
