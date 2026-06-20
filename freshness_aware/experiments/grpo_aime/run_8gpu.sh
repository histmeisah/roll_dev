#!/bin/bash
# ==============================================================================
# GRPO + GroupReplayBuffer - 8x H100/H200 GPU Runner (shared by all exp configs)
# Usage: bash run_8gpu.sh <config_name_without_yaml>
# Example: bash run_8gpu.sh exp1_grpo_baseline
# ==============================================================================
set -e
set -o pipefail

# =============================================================================
# CONFIG NAME (from first argument, default: exp1_grpo_baseline)
# =============================================================================
CONFIG_NAME="${1:-exp1_grpo_baseline}"

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

# Set ROLL path (freshness_replaybuffer repo with GroupReplayBuffer)
export ROLL_PATH="/mnt/data/u/maw0a/python_project/freshness_aware/ROLL"
export PYTHONPATH="$ROLL_PATH:$PYTHONPATH"

# Clear stale Ray environment variables. The runner assigns a per-job Ray port
# below to avoid colliding with stale sessions on shared nodes.
unset MASTER_ADDR
unset RAY_ADDRESS
unset RAY_NODE_IP_ADDRESS

# Set visible GPUs for direct shell runs. Under SLURM, keep the allocation that
# SLURM provides via CUDA_VISIBLE_DEVICES.
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
fi

# NCCL settings for H100/H200
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export DS_SKIP_CUDA_CHECK=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_V1=0

# Generate unified timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
export TRAINING_TIMESTAMP="$TIMESTAMP"
OUTPUT_DIR="$SCRIPT_DIR/output/$TIMESTAMP"

# Wandb offline mode
export WANDB_MODE=offline
export WANDB_API_KEY=local
export WANDB_DIR="$OUTPUT_DIR/wandb"

# HuggingFace offline mode
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

# Create output directories
mkdir -p "$OUTPUT_DIR/logs"
mkdir -p "$OUTPUT_DIR/models"
mkdir -p "$OUTPUT_DIR/render"
mkdir -p "$OUTPUT_DIR/wandb"

# Keep Ray/TMP state scoped to this job-specific short path. This avoids touching
# /tmp/ray or other jobs from the same user on shared public nodes. Ray socket
# paths must stay short because AF_UNIX has a 107-byte path limit.
JOB_TMP_ID="${SLURM_JOB_ID:-$$}"
export RAY_TMPDIR="/tmp/ray_${USER}_${JOB_TMP_ID}"
export TMPDIR="/tmp/tmp_${USER}_${JOB_TMP_ID}"
export MASTER_PORT="$((15000 + JOB_TMP_ID % 20000))"
export DASHBOARD_PORT="$((35000 + JOB_TMP_ID % 20000))"
mkdir -p "$RAY_TMPDIR"
mkdir -p "$TMPDIR"

# Set log file (include config name for easy identification)
LOG_FILE="$OUTPUT_DIR/logs/${CONFIG_NAME}_${TIMESTAMP}.log"

# Redirect stdout and stderr to both screen and log file
exec > >(tee -a "$LOG_FILE")
exec 2>&1

echo "========================================"
echo "GRPO + ReplayBuffer Experiment"
echo "========================================"
echo "Config: ${CONFIG_NAME}"
echo "Timestamp: $TIMESTAMP"
echo "Log file: $LOG_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "ROLL path: $ROLL_PATH"
echo "Hardware: 8x H100/H200"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "RAY_TMPDIR: $RAY_TMPDIR"
echo "MASTER_PORT: $MASTER_PORT"
echo "DASHBOARD_PORT: $DASHBOARD_PORT"
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
echo "GPU availability..."
echo "========================================"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
else
    echo "nvidia-smi not available"
fi
echo ""

# Clean up Ray state owned by this process context only. Do not pkill all user
# Ray processes or clear /tmp/ray on public shared nodes.
echo "========================================"
echo "Cleaning up local Ray state..."
echo "========================================"
ray stop --force 2>/dev/null || true
sleep 2
echo "Cleanup completed."
echo "========================================"
echo ""

# Start training
cd "$ROLL_PATH"
echo "Training [${CONFIG_NAME}] started at $(date)"
echo "Working directory: $ROLL_PATH"
echo ""

python examples/start_agentic_pipeline.py \
    --config_path ../../experiments/grpo_aime \
    --config_name "$CONFIG_NAME"

echo ""
echo "========================================"
echo "Training [${CONFIG_NAME}] completed at $(date)"
echo "========================================"
echo "Output directory: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo "========================================"
