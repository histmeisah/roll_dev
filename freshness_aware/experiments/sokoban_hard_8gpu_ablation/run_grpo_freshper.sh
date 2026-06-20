#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$SCRIPT_DIR/run_8gpu.sh" sokoban_hard_grpo_freshper_qwen3_8b_8gpu
