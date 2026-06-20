#!/bin/bash
# ==============================================================================
# Single-seed Sokoban training as a SLURM job on the public batch H200 queue.
# Usage:  sbatch -J <job_name> sbatch_one_seed.sh <CONFIG_NAME>
# ==============================================================================
# Public batch H200 queue; do not use the PI-reserved pi-elhosemh partition/qos.
# ORIX still shows Account=pi-elhosemh because that is the user's accounting
# association; the actual resource pool is selected by Partition=batch-h200 and
# QOS=batch below.
#SBATCH --partition=batch-h200
#SBATCH --qos=batch
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/data/u/maw0a/python_project/freshness_aware/experiments/sokoban_hard_2a100_replaybuffer/output/slurm_%x_%j.out
#SBATCH --error=/mnt/data/u/maw0a/python_project/freshness_aware/experiments/sokoban_hard_2a100_replaybuffer/output/slurm_%x_%j.err

set -e
set -o pipefail   # propagate non-zero exit through `tee` so a training crash
                  # surfaces as a SLURM-FAILED job instead of silent COMPLETED.

CONFIG_NAME="${1:-${CONFIG_NAME}}"
if [ -z "$CONFIG_NAME" ]; then
    echo "ERROR: pass CONFIG_NAME as positional arg, e.g. sbatch sbatch_one_seed.sh sokoban_traj_advantage_per_configA_seed43"
    exit 1
fi

SCRIPT_DIR="/mnt/data/u/maw0a/python_project/freshness_aware/experiments/sokoban_hard_2a100_replaybuffer"
CONFIG_FILE="$SCRIPT_DIR/${CONFIG_NAME}.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: config file not found: $CONFIG_FILE"
    exit 1
fi

echo "================================================================"
echo "JOB:     $SLURM_JOB_NAME ($SLURM_JOB_ID)"
echo "NODE:    $SLURMD_NODENAME"
echo "GPUs:    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (gres=$SLURM_JOB_GRES)"
echo "CONFIG:  $CONFIG_NAME"
echo "================================================================"

nvidia-smi --query-gpu=index,name,memory.total --format=csv || true

export ROLL_PATH="/mnt/data/u/maw0a/python_project/freshness_aware/ROLL"
export PYTHONPATH="$ROLL_PATH:$PYTHONPATH"

# NOTE: pass env name to activate explicitly — without it, the activate script
# would interpret $1 (our CONFIG_NAME) as the env name and fail under `set -e`.
source /mnt/data/u/maw0a/miniconda3/bin/activate roll
export PATH="$CONDA_PREFIX/bin:$PATH"

# Let SLURM-provided CUDA_VISIBLE_DEVICES stand; clear any inherited Ray/torch
# distributed env from the launching shell.
unset MASTER_ADDR MASTER_PORT RAY_ADDRESS RAY_NODE_IP_ADDRESS

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
export TRAINING_TIMESTAMP="$TIMESTAMP"
OUTPUT_DIR="$SCRIPT_DIR/output/$TIMESTAMP"
export WANDB_MODE=offline
export WANDB_DIR="$OUTPUT_DIR/wandb"

mkdir -p "$OUTPUT_DIR/logs" "$OUTPUT_DIR/models" "$OUTPUT_DIR/tensorboard" \
         "$OUTPUT_DIR/render" "$OUTPUT_DIR/wandb"

LOG_FILE="$OUTPUT_DIR/logs/training_${TIMESTAMP}.log"
echo "Output dir: $OUTPUT_DIR"
echo "Log file:   $LOG_FILE"

cd "$ROLL_PATH"
# NOTE: hydra.initialize() requires --config_path to be RELATIVE (to the
# location of start_agentic_pipeline.py, i.e. ROLL/examples/).
python examples/start_agentic_pipeline.py \
    --config_path "../../experiments/sokoban_hard_2a100_replaybuffer" \
    --config_name "$CONFIG_NAME" 2>&1 | tee "$LOG_FILE"

echo "================================================================"
echo "[$(date)] DONE: $CONFIG_NAME"
echo "================================================================"
