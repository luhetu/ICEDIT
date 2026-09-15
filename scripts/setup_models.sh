#!/bin/bash
# 下载 ICEdit 所需模型权重
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="${PROJECT_DIR}/models"

source "${HOME}/.venvs/icedit/bin/activate"

mkdir -p "${MODELS_DIR}"

echo "==> 下载 ICEdit LoRA 权重..."
hf download RiverZ/normal-lora --local-dir "${MODELS_DIR}/ICEdit-normal-LoRA"

echo "==> 下载 GGUF 量化模型（12GB 显存推荐）..."
hf download YarvixPA/FLUX.1-Fill-dev-gguf flux1-fill-dev-Q4_0.gguf --local-dir "${MODELS_DIR}"
hf download city96/t5-v1_1-xxl-encoder-gguf t5-v1_1-xxl-encoder-Q8_0.gguf --local-dir "${MODELS_DIR}"

echo "==> 下载 Flux.1-fill-dev 基础模型（需在 HuggingFace 接受许可并登录）..."
echo "    1. 访问 https://huggingface.co/black-forest-labs/flux.1-fill-dev 点击同意条款"
echo "    2. 运行: hf auth login"
if hf auth whoami &>/dev/null; then
    hf download black-forest-labs/flux.1-fill-dev --local-dir "${MODELS_DIR}/flux.1-fill-dev"
    echo "==> Flux 模型下载完成"
else
    echo "==> 未登录 HuggingFace，跳过 Flux 下载。请先运行: hf auth login"
    exit 1
fi

echo "==> 全部模型下载完成，保存在 ${MODELS_DIR}"
