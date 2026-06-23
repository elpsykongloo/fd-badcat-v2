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
eval "$(conda shell.bash hook)"
conda activate "${INDEX_TTS_ENV_NAME:-index-tts-vllm}"
python "$ROOT_DIR/scripts/patch_index_tts_vllm.py" "$ROOT_DIR/model/index-tts-vllm"
export INDEX_TTS_ENFORCE_EAGER="${INDEX_TTS_ENFORCE_EAGER:-1}"

cd "$ROOT_DIR/model/index-tts-vllm"
exec python api_server.py \
    --host "${INDEX_TTS_HOST:-0.0.0.0}" \
    --port "${INDEX_TTS_PORT:-19000}" \
    --model_dir "${INDEX_TTS_MODEL_DIR:-$ROOT_DIR/model/index-tts-vllm/checkpoints/Index-TTS-1.5-vLLM}" \
    --gpu_memory_utilization "${INDEX_TTS_GPU_MEMORY_UTILIZATION:-0.25}"
