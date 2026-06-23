#!/usr/bin/env bash
set -euo pipefail

export CONDA_ENVS_PATH="${CONDA_ENVS_PATH:-/root/autodl-tmp/conda-envs}"
LOCAL_NO_PROXY="127.0.0.1,localhost,0.0.0.0"
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}$LOCAL_NO_PROXY"
export no_proxy="${no_proxy:+$no_proxy,}$LOCAL_NO_PROXY"
for var in OMP_NUM_THREADS MKL_NUM_THREADS; do
    if [[ -z "${!var:-}" || "${!var}" == "0" ]]; then
        export "$var=8"
    fi
done
eval "$(conda shell.bash hook)"
conda activate "${QWEN_ENV_NAME:-fdbc-qwen3o-vllm}"

export FDBC_VLLM_URL="${FDBC_VLLM_URL:-http://127.0.0.1:10003/v1/chat/completions}"
export FDBC_QWEN_MODEL="${FDBC_QWEN_MODEL:-Qwen3-Omni-30B-A3B-Instruct}"

exec python src/qwen3_api.py
