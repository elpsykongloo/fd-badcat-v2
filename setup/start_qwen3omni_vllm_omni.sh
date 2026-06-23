#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(dirname "$SCRIPT_DIR")

export CONDA_ENVS_PATH="${CONDA_ENVS_PATH:-/root/autodl-tmp/conda-envs}"
LOCAL_NO_PROXY="127.0.0.1,localhost,0.0.0.0"
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}$LOCAL_NO_PROXY"
export no_proxy="${no_proxy:+$no_proxy,}$LOCAL_NO_PROXY"
for var in OMP_NUM_THREADS MKL_NUM_THREADS; do
    if [[ -z "${!var:-}" || "${!var}" == "0" ]]; then
        export "$var=8"
    fi
done
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
eval "$(conda shell.bash hook)"
conda activate "${QWEN_ENV_NAME:-fdbc-qwen3o-vllm}"
python "$ROOT_DIR/scripts/patch_prometheus_instrumentator.py"

MODEL_DIR="${QWEN_MODEL_DIR:-$ROOT_DIR/model/Qwen3-Omni-30B-A3B-Instruct}"
SERVED_MODEL_NAME="${FDBC_QWEN_MODEL:-Qwen3-Omni-30B-A3B-Instruct}"
DEPLOY_CONFIG="${QWEN_DEPLOY_CONFIG:-$ROOT_DIR/configs/qwen3_omni_text_only.yaml}"
PORT="${QWEN_PORT:-10003}"
HOST="${QWEN_HOST:-0.0.0.0}"
TP_SIZE="${QWEN_TP:-1}"
MAX_MODEL_LEN="${QWEN_MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${QWEN_GPU_MEMORY_UTILIZATION:-0.78}"

exec vllm serve "$MODEL_DIR" \
    --omni \
    --deploy-config "$DEPLOY_CONFIG" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --host "$HOST" \
    --port "$PORT" \
    --dtype bfloat16 \
    --max-model-len "$MAX_MODEL_LEN" \
    --allowed-local-media-path / \
    --tensor-parallel-size "$TP_SIZE" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    ${QWEN_EXTRA_ARGS:-}
