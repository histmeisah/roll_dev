#!/bin/bash
# ============================================================================
# Run EXP 4: GRPO + GroupReplayBuffer reward-based PER
# Run inside tmux: tmux new -s exp4 && bash run_exp4.sh
# ============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================================"
echo " EXP 4: GRPO + GroupReplayBuffer (reward PER, alpha=0.6, beta=0.4)"
echo "   Algorithm: GRPO (group_size=4), ppo_epochs=1"
echo "   Replay: group-level, reward priority + IS correction, train_steps=1"
echo "   Purpose: Test PER incremental value vs exp3 uniform replay"
echo " Started at $(date)"
echo "========================================================"

bash run_8gpu.sh exp4_grpo_replay_per_reward

echo ""
echo "========================================================"
echo " EXP 4 COMPLETED at $(date)"
echo "========================================================"
