#!/bin/bash
# ============================================================================
# Run EXP 3: GRPO + GroupReplayBuffer uniform sampling
# Run inside tmux: tmux new -s exp3 && bash run_exp3.sh
# ============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================================"
echo " EXP 3: GRPO + GroupReplayBuffer (uniform sampling)"
echo "   Algorithm: GRPO (group_size=4), ppo_epochs=1"
echo "   Replay: group-level, uniform, train_steps=1"
echo "   Purpose: Test standard replay effect vs exp2 ppo_epochs=2"
echo " Started at $(date)"
echo "========================================================"

bash run_8gpu.sh exp3_grpo_replay_uniform

echo ""
echo "========================================================"
echo " EXP 3 COMPLETED at $(date)"
echo "========================================================"
