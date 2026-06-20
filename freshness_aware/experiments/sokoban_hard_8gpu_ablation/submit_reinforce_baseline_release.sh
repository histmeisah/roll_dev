#!/bin/bash
set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p output/release_entry_logs
ENTRY_LOG="output/release_entry_logs/reinforce_baseline_$(date +%Y%m%d_%H%M%S).log"

{
    echo "Release entry started at $(date)"
    echo "Host: $(hostname)"
    echo "PWD: $(pwd)"
    echo "User: $(whoami)"
    echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
    echo "Launching run_reinforce_baseline.sh"
} | tee -a "$ENTRY_LOG"

bash "$SCRIPT_DIR/run_reinforce_baseline.sh" 2>&1 | tee -a "$ENTRY_LOG"
