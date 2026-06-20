#!/bin/bash
# ==============================================================================
# Sokoban Hard + KL-FreshPER - 8 GPU runner for Wuwen platform shell windows.
#
# Usage:
#   bash run_8gpu.sh
#   bash run_8gpu.sh sokoban_hard_reinforce_kl_fresh_qwen3_8b_8gpu_smoke
# ==============================================================================
set -e
set -o pipefail

CONFIG_NAME="${1:-sokoban_hard_reinforce_kl_fresh_qwen3_8b_8gpu_smoke}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -f "${CONFIG_NAME}.yaml" ]; then
    echo "Error: config '${CONFIG_NAME}.yaml' not found in ${SCRIPT_DIR}"
    echo "Available configs:"
    ls -1 *.yaml 2>/dev/null | sed 's/.yaml$//' || true
    exit 1
fi

export REPO_ROOT="${REPO_ROOT:-/mnt/project_modelware_roce/zhaojian/weiyu/freshness_aware}"
export ROLL_PATH="${ROLL_PATH:-${REPO_ROOT}/ROLL}"
export MODEL_ROOT="${MODEL_ROOT:-/mnt/project_modelware_roce/zhaojian/liangsirui/Model}"
export MODEL_PATH="${MODEL_PATH:-${MODEL_ROOT}/Qwen3-8B}"
export PYTHONPATH="$ROLL_PATH:$PYTHONPATH"

CONDA_SH="${CONDA_SH:-/mnt/project_modelware_roce/zhaojian/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-/mnt/project_modelware_roce/zhaojian/envs/roll}"
if [ -f "$CONDA_SH" ]; then
    source "$CONDA_SH"
    conda activate "$CONDA_ENV"
else
    echo "Error: conda.sh not found at ${CONDA_SH}"
    exit 1
fi

export PATH="$CONDA_PREFIX/bin:$PATH"

if [ ! -d "$ROLL_PATH" ]; then
    echo "Error: ROLL_PATH does not exist: ${ROLL_PATH}"
    exit 1
fi

if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: MODEL_PATH does not exist: ${MODEL_PATH}"
    exit 1
fi

unset MASTER_ADDR
unset RAY_ADDRESS
unset RAY_NODE_IP_ADDRESS

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
fi

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export DS_SKIP_CUDA_CHECK=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_V1=0

export WANDB_MODE=offline
export WANDB_API_KEY=local
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
export TRAINING_TIMESTAMP="$TIMESTAMP"
OUTPUT_DIR="$SCRIPT_DIR/output/$TIMESTAMP"
mkdir -p "$OUTPUT_DIR/logs" "$OUTPUT_DIR/models" "$OUTPUT_DIR/render" "$OUTPUT_DIR/wandb"
export WANDB_DIR="$OUTPUT_DIR/wandb"

JOB_TMP_ID="${SLURM_JOB_ID:-$$}"
export RAY_TMPDIR="/tmp/ray_${USER}_${JOB_TMP_ID}"
export TMPDIR="/tmp/tmp_${USER}_${JOB_TMP_ID}"
# Ray's default worker port range is 10002-19999. Keep control ports outside
# that range; otherwise ray start may fail when MASTER_PORT lands in it.
export MASTER_PORT="$((24000 + JOB_TMP_ID % 8000))"
export DASHBOARD_PORT="$((34000 + JOB_TMP_ID % 8000))"
mkdir -p "$RAY_TMPDIR" "$TMPDIR"

LOG_FILE="$OUTPUT_DIR/logs/${CONFIG_NAME}_${TIMESTAMP}.log"
exec > >(tee -a "$LOG_FILE")
exec 2>&1

echo "========================================"
echo "Sokoban Hard KL-FreshPER"
echo "========================================"
echo "Config: ${CONFIG_NAME}"
echo "Timestamp: ${TIMESTAMP}"
echo "Repo: ${REPO_ROOT}"
echo "ROLL path: ${ROLL_PATH}"
echo "Model path: ${MODEL_PATH}"
echo "Conda env: ${CONDA_PREFIX}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Log file: ${LOG_FILE}"
echo "RAY_TMPDIR: ${RAY_TMPDIR}"
echo "MASTER_PORT: ${MASTER_PORT}"
echo "========================================"

if [ -n "${TMUX:-}" ]; then
    TMUX_SESSION=$(tmux display-message -p '#S' 2>/dev/null || true)
    TMUX_WINDOW=$(tmux display-message -p '#W' 2>/dev/null || true)
    echo "tmux: session=${TMUX_SESSION:-unknown}, window=${TMUX_WINDOW:-unknown}"
fi

echo ""
echo "GPU availability:"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
else
    echo "nvidia-smi not found"
fi

echo ""
echo "Cleaning local Ray state..."
ray stop --force 2>/dev/null || true
sleep 2

cd "$ROLL_PATH"
echo ""
echo "Training started at $(date)"
echo "Working directory: $ROLL_PATH"
echo ""

python examples/start_agentic_pipeline.py \
    --config_path ../../experiments/sokoban_hard_8gpu_kl_fresh \
    --config_name "$CONFIG_NAME"

echo ""
echo "Training completed at $(date)"
echo "Output directory: ${OUTPUT_DIR}"
echo "Log file: ${LOG_FILE}"
