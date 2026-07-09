#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

sbatch "$SCRIPT_DIR/sbatch_8gpu.sh" sokoban_hard_reinforce_ppo2_reward_fresh_filter_qwen3_8b_8gpu
sbatch "$SCRIPT_DIR/sbatch_8gpu.sh" sokoban_hard_reinforce_ppo2_kl_fresh_filter_qwen3_8b_8gpu
