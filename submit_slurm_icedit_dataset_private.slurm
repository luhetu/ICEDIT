#!/bin/bash
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --job-name=icedit_removal
#SBATCH --output=logs/icedit_removal_%j.out
#SBATCH --error=logs/icedit_removal_%j.err
#SBATCH --time=144:00:00
#SBATCH --gres=gpu:ampere:1
#SBATCH --qos=long-high-prio
#SBATCH --partition=gpu-private
#SBATCH --mem=56G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dnrx52@durham.ac.uk

if command -v module >/dev/null 2>&1; then
    module load cuda/11.1
fi

set -euo pipefail
mkdir -p /home3/dnrx52/ICEdit/logs
cd /home3/dnrx52/ICEdit

PYTHON=/home3/dnrx52/.venvs/icedit/bin/python
PARQUET=${1:-train/parquet/train-00115-of-00571.parquet}
OUTPUT_DIR=${2:-dataset_outputs/omni_removal_shard_00115}
MAX_SAMPLES=${3:-0}
START_INDEX=${4:-0}
BASE_SEED=${5:-304897401}
PROMPT_INDEX=${6:-1}

FLUX_PATH=/home3/dnrx52/ICEdit/models/flux.1-fill-dev
LORA_PATH=/home3/dnrx52/ICEdit/models/ICEdit-normal-LoRA

echo "Python: ${PYTHON}"
echo "Dataset: ${PARQUET}"
echo "Output: ${OUTPUT_DIR}"
echo "Rows: start=${START_INDEX}, max=${MAX_SAMPLES}"
echo "Prompt index: ${PROMPT_INDEX}"
${PYTHON} -c "import torch, pyarrow; print('cuda:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0)); print('pyarrow:', pyarrow.__version__)"

${PYTHON} -u scripts/inference_dataset.py \
    --parquet "${PARQUET}" \
    --output-dir "${OUTPUT_DIR}" \
    --max-samples "${MAX_SAMPLES}" \
    --start-index "${START_INDEX}" \
    --seed "${BASE_SEED}" \
    --prompt-index "${PROMPT_INDEX}" \
    --flux-path "${FLUX_PATH}" \
    --lora-path "${LORA_PATH}"
