#!/bin/bash
#SBATCH --gres=gpu:AA_A6000_0:1
#SBATCH -p gpu-bigmem
#SBATCH --qos=long-high-prio
#SBATCH --time=01:00:00
#SBATCH -J icedit_contract_probe
#SBATCH -o logs/%x%j.out
#SBATCH --error=logs/%x%j.err

set -euo pipefail

if command -v module >/dev/null 2>&1; then
    module load cuda/11.1
fi

cd /home3/dnrx52/ICEdit
mkdir -p logs

PYTHON=/home3/dnrx52/.venvs/icedit/bin/python
SEED=${1:-731301}
OUTPUT_DIR=${2:-research_outputs/contract_trajectory_coca_cola_seed_${SEED}}

${PYTHON} scripts/research_contract_trajectory.py \
    --case-id remove_coca_cola_signs \
    --output-dir "${OUTPUT_DIR}" \
    --flux-path models/flux.1-fill-dev \
    --lora-path models/ICEdit-normal-LoRA \
    --seed "${SEED}" \
    --num-inference-steps 28 \
    --checkpoint-count 6 \
    --guidance-scale 50
