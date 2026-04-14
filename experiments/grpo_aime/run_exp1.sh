#!/bin/bash
# ============================================================================
# Run EXP 1: GRPO Pure On-Policy Baseline (ppo_epochs=1, no replay)
# Run inside tmux: tmux new -s exp1 && bash run_exp1.sh
# ============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================================"
echo " EXP 1: GRPO Pure On-Policy Baseline"
echo "   Algorithm: GRPO (group_size=4), ppo_epochs=1"
echo "   Replay: DISABLED"
echo " Started at $(date)"
echo "========================================================"

bash run_8gpu.sh exp1_grpo_baseline

echo ""
echo "========================================================"
echo " EXP 1 COMPLETED at $(date)"
echo "========================================================"
