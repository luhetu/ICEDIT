#!/bin/bash
#SBATCH --gres=gpu:AA_A6000_0:1
#SBATCH -p gpu-bigmem
#SBATCH --qos=long-high-prio
#SBATCH --time=1-00:00:00
#SBATCH -c 4
#SBATCH --mem=32g
#SBATCH -J icedit_dataset
#SBATCH -o logs/%x%j.out
#SBATCH --error=logs/%x%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dnrx52@durham.ac.uk

if command -v module >/dev/null 2>&1; then
    module load cuda/11.1
fi

set -euo pipefail
mkdir -p /home3/dnrx52/ICEdit/logs
cd /home3/dnrx52/ICEdit
if [[ -f .env ]]; then
    set -a
    source .env
    set +a
fi

PYTHON=/home3/dnrx52/.venvs/icedit/bin/python
PARQUET=${1:-train/parquet/train-00000-of-00105.parquet}
OUTPUT_DIR=${2:-dataset_outputs/omni_mini_shard_00000}
MAX_SAMPLES=${3:-10}
START_INDEX=${4:-0}
BASE_SEED=${5:-304897401}
PROMPT_INDEX=${6:-1}

PIPELINE_MODE=${PIPELINE_MODE:-raw}
PLANNER=${PLANNER:-rule}
PLANNER_MODEL=${PLANNER_MODEL:-}
TASK_FILTER=${TASK_FILTER:-}
MASK_DIR=${MASK_DIR:-}

FLUX_PATH=/home3/dnrx52/ICEdit/models/flux.1-fill-dev
LORA_PATH=/home3/dnrx52/ICEdit/models/ICEdit-normal-LoRA

echo "Python: ${PYTHON}"
echo "Dataset: ${PARQUET}"
echo "Output: ${OUTPUT_DIR}"
echo "Rows: start=${START_INDEX}, max=${MAX_SAMPLES}"
echo "Prompt index: ${PROMPT_INDEX}"
echo "Pipeline mode: ${PIPELINE_MODE}"
echo "Planner: ${PLANNER}${PLANNER_MODEL:+ (${PLANNER_MODEL})}"
echo "Task filter: ${TASK_FILTER:-all}"
echo "Mask directory: ${MASK_DIR:-none}"
${PYTHON} -c "import torch, pyarrow; print('cuda:', torch.cuda.is_available()); print('pyarrow:', pyarrow.__version__)"

EXTRA_ARGS=(
    --pipeline-mode "${PIPELINE_MODE}"
    --planner "${PLANNER}"
)
if [[ -n "${PLANNER_MODEL}" ]]; then
    EXTRA_ARGS+=(--planner-model "${PLANNER_MODEL}")
fi
if [[ -n "${TASK_FILTER}" ]]; then
    IFS=',' read -ra TASKS <<< "${TASK_FILTER}"
    for TASK in "${TASKS[@]}"; do
        EXTRA_ARGS+=(--task "${TASK}")
    done
fi
if [[ -n "${MASK_DIR}" ]]; then
    EXTRA_ARGS+=(--mask-dir "${MASK_DIR}")
fi

${PYTHON} scripts/inference_dataset.py \
    --parquet "${PARQUET}" \
    --output-dir "${OUTPUT_DIR}" \
    --max-samples "${MAX_SAMPLES}" \
    --start-index "${START_INDEX}" \
    --seed "${BASE_SEED}" \
    --prompt-index "${PROMPT_INDEX}" \
    --flux-path "${FLUX_PATH}" \
    --lora-path "${LORA_PATH}" \
    "${EXTRA_ARGS[@]}"
