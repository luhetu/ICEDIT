#!/bin/bash
#SBATCH -p res-gpu-small
#SBATCH --qos=short
#SBATCH --gres=gpu:ampere:1
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=2
#SBATCH -J objectclear_smoke
#SBATCH -o logs/%x%j.out
#SBATCH --error=logs/%x%j.err
set -euo pipefail
cd /home3/dnrx52/ICEdit
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
/home3/dnrx52/.venvs/objectclear/bin/python scripts/objectclear_smoke.py \
    --out "research_outputs/objectclear_${SLURM_JOB_ID}_agf_on" --agf on
/home3/dnrx52/.venvs/objectclear/bin/python scripts/objectclear_smoke.py \
    --out "research_outputs/objectclear_${SLURM_JOB_ID}_agf_off" --agf off
