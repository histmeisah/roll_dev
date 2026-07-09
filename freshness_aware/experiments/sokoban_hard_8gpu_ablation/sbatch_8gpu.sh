#!/bin/bash
#SBATCH --job-name=roll-soko-hard
#SBATCH --partition=pi-elhosemh
#SBATCH --qos=pi-elhosemh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=1400G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:8
#SBATCH --time=3-00:00:00
#SBATCH --chdir=/mnt/data/u/maw0a/python_project/freshness_aware/experiments/sokoban_hard_8gpu_ablation
#SBATCH --output=/mnt/data/u/maw0a/python_project/freshness_aware/experiments/sokoban_hard_8gpu_ablation/output/slurm/%x_%j.out
#SBATCH --error=/mnt/data/u/maw0a/python_project/freshness_aware/experiments/sokoban_hard_8gpu_ablation/output/slurm/%x_%j.err

set -euo pipefail

CONFIG_NAME="${1:-sokoban_hard_reinforce_ppo2_reward_fresh_filter_qwen3_8b_8gpu}"
SCRIPT_DIR="/mnt/data/u/maw0a/python_project/freshness_aware/experiments/sokoban_hard_8gpu_ablation"
mkdir -p "$SCRIPT_DIR/output/slurm"

export REPO_ROOT="${REPO_ROOT:-/mnt/data/u/maw0a/python_project/freshness_aware}"
export ROLL_PATH="${ROLL_PATH:-${REPO_ROOT}/ROLL}"
export CONDA_SH="${CONDA_SH:-/mnt/data/u/maw0a/miniconda3/etc/profile.d/conda.sh}"
export CONDA_ENV="${CONDA_ENV:-/mnt/data/u/maw0a/miniconda3/envs/roll}"
export MODEL_ROOT="${MODEL_ROOT:-/mnt/data/u/maw0a/models}"
export MODEL_PATH="${MODEL_PATH:-${MODEL_ROOT}/Qwen3-8B}"
export OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-$SCRIPT_DIR/output}"
export HF_HOME="${HF_HOME:-/mnt/data/u/maw0a/hf_cache}"

echo "Submitting config: $CONFIG_NAME"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-unset}"
echo "Host: $(hostname)"
echo "Partition: ${SLURM_JOB_PARTITION:-unset}"
echo "GPUs: ${SLURM_JOB_GPUS:-unset}"

srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-128}" \
    bash "$SCRIPT_DIR/run_8gpu.sh" "$CONFIG_NAME"
