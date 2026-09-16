#!/bin/bash
#SBATCH -p res-gpu-small
#SBATCH --qos=short
#SBATCH --gres=gpu:ampere:1
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --array=0-2%1
#SBATCH -J objectclear_removal
#SBATCH -o logs/%x%A_%a.out
#SBATCH --error=logs/%x%A_%a.err
set -euo pipefail
cd /home3/dnrx52/ICEdit
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
SEED=$((731001 + SLURM_ARRAY_TASK_ID))
for AGF_MODE in on off; do
    /home3/dnrx52/.venvs/objectclear/bin/python scripts/objectclear_smoke.py \
        --source research_outputs/removal_pilot_1042473/source.png \
        --mask research_outputs/removal_pilot_1042473/mask.png \
        --seed "$SEED" --agf "$AGF_MODE" \
        --out "research_outputs/objectclear_removal_${SLURM_ARRAY_JOB_ID}_s${SEED}_${AGF_MODE}"
done
