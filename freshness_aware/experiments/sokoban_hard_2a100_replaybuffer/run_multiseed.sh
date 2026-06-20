#!/bin/bash
# ==============================================================================
# Sokoban Multi-Seed Runner — runs 4 configs sequentially.
# Usage:  bash run_multiseed.sh
# ==============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIGS=(
    "sokoban_traj_reward_fresh_configA_age1000_seed43"
    "sokoban_traj_reward_fresh_configA_age1000_seed44"
    "sokoban_traj_advantage_per_configA_seed43"
    "sokoban_traj_advantage_per_configA_seed44"
)

# ===== Environment (one-time) =====
export ROLL_PATH="/mnt/data/u/maw0a/python_project/freshness_aware/ROLL"
export PYTHONPATH="$ROLL_PATH:$PYTHONPATH"

source /mnt/data/u/maw0a/miniconda3/bin/activate
conda activate roll
export PATH="$CONDA_PREFIX/bin:$PATH"

unset MASTER_ADDR MASTER_PORT RAY_ADDRESS RAY_NODE_IP_ADDRESS
export CUDA_VISIBLE_DEVICES=0,1
export WANDB_MODE=offline

# ===== Sequential loop over 4 configs =====
for CONFIG_NAME in "${CONFIGS[@]}"; do
    echo ""
    echo "========================================"
    echo "[$(date +%Y-%m-%d\ %H:%M:%S)] START: $CONFIG_NAME"
    echo "========================================"

    if [ ! -f "${CONFIG_NAME}.yaml" ]; then
        echo "ERROR: ${CONFIG_NAME}.yaml not found, skipping."
        continue
    fi

    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    export TRAINING_TIMESTAMP="$TIMESTAMP"
    OUTPUT_DIR="$SCRIPT_DIR/output/$TIMESTAMP"
    export WANDB_DIR="$OUTPUT_DIR/wandb"

    mkdir -p "$OUTPUT_DIR/logs" "$OUTPUT_DIR/models" "$OUTPUT_DIR/tensorboard" \
             "$OUTPUT_DIR/render" "$OUTPUT_DIR/wandb"

    LOG_FILE="$OUTPUT_DIR/logs/training_${TIMESTAMP}.log"
    echo "Output dir: $OUTPUT_DIR"
    echo "Log file:   $LOG_FILE"

    # Clean up Ray between runs
    ray stop --force 2>/dev/null || true
    pkill -9 -u "$USER" ray 2>/dev/null || true
    rm -rf /tmp/ray/* 2>/dev/null || true
    sleep 2

    # Run training (output to both screen and per-run log file)
    cd "$ROLL_PATH"
    python examples/start_agentic_pipeline.py \
        --config_path ../../experiments/sokoban_hard_2a100_replaybuffer \
        --config_name "$CONFIG_NAME" 2>&1 | tee "$LOG_FILE"

    cd "$SCRIPT_DIR"
    echo "[$(date +%Y-%m-%d\ %H:%M:%S)] DONE:  $CONFIG_NAME"
done

echo ""
echo "========================================"
echo "[$(date +%Y-%m-%d\ %H:%M:%S)] ALL 4 SEEDS COMPLETED"
echo "========================================"
