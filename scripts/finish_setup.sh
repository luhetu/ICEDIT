#!/bin/bash
# 完成部署：登录 HuggingFace 并下载 Flux 基础模型
set -euo pipefail

source "${HOME}/.venvs/icedit/bin/activate"

echo "=========================================="
echo "ICEdit 部署 - 最后一步：下载 Flux 模型"
echo "=========================================="
echo ""
echo "Flux 模型需要 HuggingFace 授权，请按以下步骤操作："
echo "  1. 打开 https://huggingface.co/black-forest-labs/flux.1-fill-dev"
echo "  2. 登录并点击「Agree and access repository」同意条款"
echo "  3. 在 https://huggingface.co/settings/tokens 创建 Access Token"
echo "  4. 运行: hf auth login"
echo ""

if ! hf auth whoami &>/dev/null; then
    echo "当前未登录 HuggingFace，请先运行: hf auth login"
    exit 1
fi

echo "已登录 HuggingFace: $(hf auth whoami)"
echo "开始下载 Flux.1-fill-dev（约 30GB，需要一些时间）..."

hf download black-forest-labs/flux.1-fill-dev \
    --local-dir "${HOME}/ICEdit/models/flux.1-fill-dev"

echo ""
echo "=========================================="
echo "部署完成！"
echo "=========================================="
echo ""
echo "命令行推理:"
echo "  sbatch ${HOME}/ICEdit/submit_slurm_icedit_inferenceH.slurm   # Hopper (gpu13)"
echo "  sbatch ${HOME}/ICEdit/submit_slurm_icedit_inference.slurm    # A6000"
echo ""
echo "Gradio Web 界面:"
echo "  sbatch ${HOME}/ICEdit/submit_slurm_icedit_gradioH.slurm      # Hopper (gpu13)"
echo "  sbatch ${HOME}/ICEdit/submit_slurm_icedit_gradio.slurm       # A6000"
echo "  然后 SSH 端口转发: ssh -L 7860:<节点名>:7860 ${USER}@$(hostname -f)"
echo ""
echo "手动激活环境:"
echo "  source ${HOME}/ICEdit/activate_icedit.sh"
