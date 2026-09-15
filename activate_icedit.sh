#!/bin/bash
# 激活 ICEdit 环境
source "${HOME}/.venvs/icedit/bin/activate"
export PYTHONPATH="${HOME}/ICEdit:${PYTHONPATH:-}"
cd "${HOME}/ICEdit"

if [[ -f .env ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi
