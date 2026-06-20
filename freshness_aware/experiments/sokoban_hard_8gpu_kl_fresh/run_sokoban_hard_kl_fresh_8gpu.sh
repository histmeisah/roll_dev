#!/bin/bash
set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

bash run_8gpu.sh sokoban_hard_reinforce_kl_fresh_qwen3_8b_8gpu_smoke
