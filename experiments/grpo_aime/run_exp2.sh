#!/bin/bash
# ============================================================================
# Run EXP 2: GRPO ppo_epochs=2 Baseline (gradient-aligned vs exp3/4)
# Run inside tmux: tmux new -s exp2 && bash run_exp2.sh
# ============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================================"
echo " EXP 2: GRPO ppo_epochs=2 Baseline (gradient-matched control)"
echo "   Algorithm: GRPO (group_size=4), ppo_epochs=2"
echo "   Replay: DISABLED"
echo "   Purpose: Match total gradient updates of exp3/exp4"
echo " Started at $(date)"
echo "========================================================"

bash run_8gpu.sh exp2_grpo_ppo2_baseline

echo ""
echo "========================================================"
echo " EXP 2 COMPLETED at $(date)"
echo "========================================================"
