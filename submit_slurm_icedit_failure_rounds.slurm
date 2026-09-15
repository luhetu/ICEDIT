#!/bin/bash
#SBATCH --gres=gpu:AA_A6000_0:1
#SBATCH -p gpu-bigmem
#SBATCH --qos=long-high-prio
#SBATCH --time=02:00:00
#SBATCH -J icedit_rounds
#SBATCH -o logs/%x%j.out
#SBATCH --error=logs/%x%j.err

set -euo pipefail

if command -v module >/dev/null 2>&1; then
    module load cuda/11.1
fi

cd /home3/dnrx52/ICEdit
mkdir -p logs

PYTHON=/home3/dnrx52/.venvs/icedit/bin/python
ROUNDS=${1:-3}
OUTPUT_DIR=${OUTPUT_DIR:-failure_rounds}
shift || true

${PYTHON} scripts/inference_failure_rounds.py \
    --cases edit_topology/examples/failure_cases.json \
    --output-dir "${OUTPUT_DIR}" \
    --rounds "${ROUNDS}" \
    --flux-path models/flux.1-fill-dev \
    --lora-path models/ICEdit-normal-LoRA \
    "$@"
