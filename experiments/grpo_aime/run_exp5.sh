#!/bin/bash
# ============================================================================
# Run EXP 5: GRPO + GroupReplayBuffer FreshPER (reward × age_decay)
# Run inside tmux: tmux new -s exp5 && bash run_exp5.sh
# ============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================================"
echo " EXP 5: GRPO + GroupReplayBuffer FreshPER"
echo "   priority = (|reward| + eps) * exp(-age / age_decay=1000)"
echo "   Algorithm: GRPO (group_size=4), ppo_epochs=1"
echo "   Replay: group-level, reward_fresh priority + IS correction, train_steps=1"
echo "   Purpose: Test if age decay can fix exp4's stale-reward exploitation"
echo " Started at $(date)"
echo "========================================================"

bash run_8gpu.sh exp5_grpo_replay_fresh_per

echo ""
echo "========================================================"
echo " EXP 5 COMPLETED at $(date)"
echo "========================================================"
