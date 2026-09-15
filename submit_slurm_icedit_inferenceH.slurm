#!/bin/bash
#SBATCH -N 1
#SBATCH -w gpu13
#SBATCH -c 8
#SBATCH --mem=56g
#SBATCH --gres=gpu:hopper:1
#SBATCH -p gpu-bigmem
#SBATCH --qos=long-high-prio
#SBATCH --time=6-23:59:59
#SBATCH -J icedit_infer
#SBATCH -o logs/%x%j.out
#SBATCH --error=logs/%x%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dnrx52@durham.ac.uk

if command -v module >/dev/null 2>&1; then
    module load cuda/11.1
fi

mkdir -p logs
cd /home3/dnrx52/ICEdit

PYTHON=/home3/dnrx52/.venvs/icedit/bin/python
IMAGE=${1:-assets/girl.png}
INSTRUCTION=${2:-"Make her hair dark green and her clothes checked."}
SEED=${3:-304897401}

FLUX_PATH=/home3/dnrx52/ICEdit/models/flux.1-fill-dev
LORA_PATH=/home3/dnrx52/ICEdit/models/ICEdit-normal-LoRA

echo "Python: $PYTHON"
$PYTHON -c "import sys; print('sys.executable:', sys.executable); import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

if [[ ! -d "${FLUX_PATH}" ]]; then
    echo "错误: 未找到 Flux 模型，请先运行: bash scripts/finish_setup.sh"
    exit 1
fi

$PYTHON scripts/inference.py \
    --image "${IMAGE}" \
    --instruction "${INSTRUCTION}" \
    --seed "${SEED}" \
    --flux-path "${FLUX_PATH}" \
    --lora-path "${LORA_PATH}"
