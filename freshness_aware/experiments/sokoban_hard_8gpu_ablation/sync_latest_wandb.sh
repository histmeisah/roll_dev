#!/bin/bash
set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="${WANDB_PROJECT:-roll-sokoban-hard-8gpu-ablation}"
RUN_DIR="${1:-}"

if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "Error: WANDB_API_KEY is not set."
    exit 1
fi

if [ -z "$RUN_DIR" ]; then
    RUN_DIR="$(find "$SCRIPT_DIR/output" -type d -name 'offline-run-*' -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2-)"
fi

if [ -z "$RUN_DIR" ] || [ ! -d "$RUN_DIR" ]; then
    echo "Error: offline W&B run directory not found: ${RUN_DIR:-<empty>}"
    exit 1
fi

CONDA_SH="${CONDA_SH:-/mnt/project_modelware_roce/zhaojian/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-/mnt/project_modelware_roce/zhaojian/envs/roll}"
source "$CONDA_SH"
conda activate "$CONDA_ENV"

export WANDB_MODE=online
python -m wandb sync --project "$PROJECT" "$RUN_DIR"
