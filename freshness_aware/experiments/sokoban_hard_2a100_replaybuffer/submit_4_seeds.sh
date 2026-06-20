#!/bin/bash
# ==============================================================================
# Submit all 4 sokoban multi-seed runs as independent SLURM jobs.
# Usage:  bash submit_4_seeds.sh
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIGS=(
    "sokoban_traj_reward_fresh_configA_age1000_seed43"
    "sokoban_traj_reward_fresh_configA_age1000_seed44"
    "sokoban_traj_advantage_per_configA_seed43"
    "sokoban_traj_advantage_per_configA_seed44"
)

mkdir -p "$SCRIPT_DIR/output"

echo "Submitting ${#CONFIGS[@]} jobs to SLURM (partition/qos read from sbatch_one_seed.sh, --gres=gpu:2)..."
echo "Target resource pool: public batch-h200 with qos=batch, not pi-elhosemh/qos=pi-elhosemh."
echo "Jobs are CHAINED via --dependency=afterany so they run sequentially."
echo "Reason: ROLL/Ray uses fixed local ports and /tmp/ray sessions; sequential"
echo "        jobs avoid port/session conflicts on shared public GPU nodes."
PREV_ID=""
for CFG in "${CONFIGS[@]}"; do
    JOB_NAME="sokoban-${CFG#sokoban_traj_}"  # trim shared prefix to keep job name short
    if [ -z "$PREV_ID" ]; then
        JOB_ID=$(sbatch --parsable -J "$JOB_NAME" sbatch_one_seed.sh "$CFG")
    else
        JOB_ID=$(sbatch --parsable --dependency=afterany:"$PREV_ID" -J "$JOB_NAME" sbatch_one_seed.sh "$CFG")
    fi
    echo "  submitted  job_id=$JOB_ID  name=$JOB_NAME  depends=${PREV_ID:-none}  config=$CFG"
    PREV_ID="$JOB_ID"
done

echo ""
echo "Submitted. Track with:"
echo "  squeue -u \$USER"
echo "  tail -f $SCRIPT_DIR/output/slurm_<jobname>_<jobid>.out"
