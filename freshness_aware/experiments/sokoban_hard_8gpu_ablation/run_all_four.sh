#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "$SCRIPT_DIR/run_reinforce_baseline.sh"
bash "$SCRIPT_DIR/run_grpo_baseline.sh"
bash "$SCRIPT_DIR/run_reinforce_freshper.sh"
bash "$SCRIPT_DIR/run_grpo_freshper.sh"
