#!/bin/bash
#SBATCH --gres=gpu:AA_A6000_0:1
#SBATCH -p gpu-bigmem
#SBATCH --qos=long-high-prio
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH -J icedit_removal_pilot
#SBATCH -o logs/%x%j.out
#SBATCH --error=logs/%x%j.err

set -euo pipefail
cd /home3/dnrx52/ICEdit
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
OUT="research_outputs/removal_pilot_${SLURM_JOB_ID}"
/home3/dnrx52/.venvs/icedit/bin/python scripts/research_removal_pilot.py \
    --out "$OUT" --steps 28 --seeds 731001 731002 731003
