#!/bin/bash
# ==============================================================================
# AIME GRPO training as a SLURM job on the public batch H200 queue.
# Usage: sbatch -J aime-exp1 sbatch_8gpu.sh exp1_grpo_baseline
# ==============================================================================
# Public batch H200 queue; do not use the PI-reserved pi-elhosemh partition/qos.
# ORIX still shows Account=pi-elhosemh because that is the user's accounting
# association; the actual resource pool is selected by Partition=batch-h200 and
# QOS=batch below.
#SBATCH --partition=batch-h200
#SBATCH --qos=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=1400G
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/data/u/maw0a/python_project/freshness_aware/experiments/grpo_aime/output/slurm_%x_%j.out
#SBATCH --error=/mnt/data/u/maw0a/python_project/freshness_aware/experiments/grpo_aime/output/slurm_%x_%j.err

set -e
set -o pipefail

CONFIG_NAME="${1:-${CONFIG_NAME:-exp1_grpo_baseline}}"
SCRIPT_DIR="/mnt/data/u/maw0a/python_project/freshness_aware/experiments/grpo_aime"
CONFIG_FILE="$SCRIPT_DIR/${CONFIG_NAME}.yaml"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: config file not found: $CONFIG_FILE"
    exit 1
fi

echo "================================================================"
echo "JOB:     ${SLURM_JOB_NAME:-local} (${SLURM_JOB_ID:-no-slurm-id})"
echo "NODE:    ${SLURMD_NODENAME:-$(hostname)}"
echo "GPUs:    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} (gres=${SLURM_JOB_GRES:-unset})"
echo "CONFIG:  $CONFIG_NAME"
echo "QUEUE:   partition=${SLURM_JOB_PARTITION:-batch-h200} qos=${SLURM_JOB_QOS:-batch}"
echo "================================================================"

nvidia-smi --query-gpu=index,name,memory.total --format=csv || true

source /mnt/data/u/maw0a/miniconda3/bin/activate roll
export PATH="$CONDA_PREFIX/bin:$PATH"

cd "$SCRIPT_DIR"
bash run_8gpu.sh "$CONFIG_NAME"
