#!/bin/bash
set -e
set -o pipefail

CONFIG_NAME="${1:-sokoban_hard_reinforce_baseline_qwen3_8b_8gpu}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -f "${CONFIG_NAME}.yaml" ]; then
    echo "Error: config '${CONFIG_NAME}.yaml' not found in ${SCRIPT_DIR}"
    echo "Available configs:"
    ls -1 *.yaml 2>/dev/null | sed 's/.yaml$//' || true
    exit 1
fi

export REPO_ROOT="${REPO_ROOT:-/mnt/data/u/maw0a/python_project/freshness_aware}"
export ROLL_PATH="${ROLL_PATH:-${REPO_ROOT}/ROLL}"
export MODEL_ROOT="${MODEL_ROOT:-/mnt/data/u/maw0a/models}"
export MODEL_PATH="${MODEL_PATH:-${MODEL_ROOT}/Qwen3-8B}"
export PYTHONPATH="$ROLL_PATH:$PYTHONPATH"

CONDA_SH="${CONDA_SH:-/mnt/data/u/maw0a/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-/mnt/data/u/maw0a/miniconda3/envs/roll}"
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
    echo "Set MODEL_PATH to an existing Qwen3-8B checkout, or download it before submitting:"
    echo "  huggingface-cli download Qwen/Qwen3-8B --local-dir ${MODEL_PATH}"
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
export WANDB_API_KEY="${WANDB_API_KEY:-local}"
export HF_HOME="${HF_HOME:-/mnt/data/u/maw0a/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
export TRAINING_TIMESTAMP="$TIMESTAMP"
export OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-$SCRIPT_DIR/output}"
OUTPUT_DIR="$OUTPUT_BASE_DIR/$TIMESTAMP"
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
echo "Sokoban Hard 8GPU Ablation"
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

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
    GPU_COUNT="$(nvidia-smi -L | wc -l | tr -d ' ')"
    if [ "${GPU_COUNT:-0}" -lt 8 ] && [ "${ALLOW_FEWER_GPUS:-0}" != "1" ]; then
        echo "Error: expected 8 visible GPUs, got ${GPU_COUNT}."
        echo "Submit with sbatch_8gpu.sh or set ALLOW_FEWER_GPUS=1 only for debugging."
        exit 1
    fi
else
    echo "nvidia-smi not found"
    exit 1
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
    --config_path ../../experiments/sokoban_hard_8gpu_ablation \
    --config_name "$CONFIG_NAME"

echo ""
echo "Training completed at $(date)"
echo "Output directory: ${OUTPUT_DIR}"
echo "Log file: ${LOG_FILE}"
