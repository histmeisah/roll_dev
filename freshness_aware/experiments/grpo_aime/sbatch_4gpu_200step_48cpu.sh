#!/bin/bash
# ==============================================================================
# 4-GPU AIME GRPO 200-step job with reduced CPU request.
# Usage: sbatch -J aime-grpo4-freshper-48cpu experiments/grpo_aime/sbatch_4gpu_200step_48cpu.sh aime_grpo_4h200_reward_fresh_per_200step
# ==============================================================================
#SBATCH --partition=freecycle-h200
#SBATCH --qos=freecycle
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=768G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/data/u/maw0a/python_project/freshness_aware/experiments/grpo_aime/output/slurm_%x_%j.out
#SBATCH --error=/mnt/data/u/maw0a/python_project/freshness_aware/experiments/grpo_aime/output/slurm_%x_%j.err

set -e
set -o pipefail

CONFIG_NAME="${1:-${CONFIG_NAME:-aime_grpo_4h200_reward_fresh_per_200step}}"
SCRIPT_DIR="/mnt/data/u/maw0a/python_project/freshness_aware/experiments/grpo_aime"
CONFIG_FILE="$SCRIPT_DIR/${CONFIG_NAME}.yaml"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: config file not found: $CONFIG_FILE"
    exit 1
fi

UTIL_LOG="$SCRIPT_DIR/output/gpu_util_${CONFIG_NAME}_${SLURM_JOB_ID:-local}.csv"
UTIL_PID=""

cleanup() {
    if [ -n "$UTIL_PID" ]; then
        kill "$UTIL_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "================================================================"
echo "JOB:      ${SLURM_JOB_NAME:-local} (${SLURM_JOB_ID:-no-slurm-id})"
echo "NODE:     ${SLURMD_NODENAME:-$(hostname)}"
echo "GPUs:     CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} (gres=${SLURM_JOB_GRES:-unset})"
echo "CONFIG:   $CONFIG_NAME"
echo "QUEUE:    partition=${SLURM_JOB_PARTITION:-freecycle-h200} qos=${SLURM_JOB_QOS:-freecycle}"
echo "UTIL_LOG: $UTIL_LOG"
echo "================================================================"

nvidia-smi --query-gpu=index,name,memory.total --format=csv || true

if command -v nvidia-smi >/dev/null 2>&1; then
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        nvidia-smi --id="${CUDA_VISIBLE_DEVICES}" \
            --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
            --format=csv -l 5 > "$UTIL_LOG" &
    else
        nvidia-smi \
            --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
            --format=csv -l 5 > "$UTIL_LOG" &
    fi
    UTIL_PID="$!"
fi

source /mnt/data/u/maw0a/miniconda3/bin/activate roll
export PATH="$CONDA_PREFIX/bin:$PATH"

cd "$SCRIPT_DIR"
bash run_8gpu.sh "$CONFIG_NAME"
