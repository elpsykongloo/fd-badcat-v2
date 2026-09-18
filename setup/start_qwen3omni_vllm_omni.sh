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
python "$ROOT_DIR/scripts/patch_omni_verbatim_tts.py"

MODEL_DIR="${QWEN_MODEL_DIR:-$ROOT_DIR/model/Qwen3-Omni-30B-A3B-Instruct}"
SERVED_MODEL_NAME="${FDBC_QWEN_MODEL:-Qwen3-Omni-30B-A3B-Instruct}"
DEPLOY_CONFIG="${QWEN_DEPLOY_CONFIG:-$ROOT_DIR/configs/qwen3_omni_text_only.yaml}"
PORT="${QWEN_PORT:-10003}"
HOST="${QWEN_HOST:-0.0.0.0}"
TP_SIZE="${QWEN_TP:-1}"
MAX_MODEL_LEN="${QWEN_MAX_MODEL_LEN-32768}"
GPU_MEMORY_UTILIZATION="${QWEN_GPU_MEMORY_UTILIZATION-0.78}"
SCHEDULING_POLICY="${QWEN_SCHEDULING_POLICY:-fcfs}"

# Only the demo opts in. Derive the stage env from the caller's existing YAML;
# this vLLM-Omni release does not forward runtime env via --stage-overrides.
VOICE_PATCH_ARGS=()
DEMO_NUMERICS="${FDBC_DEMO_TALKER_NUMERICS:-off}"
if [[ "$DEMO_NUMERICS" == "native" || "$DEMO_NUMERICS" == "invariant" ]]; then
    if [[ "${FDBC_DEMO_VOICE_ADAPTER:-0}" != "1" ]]; then
        echo "FDBC_DEMO_TALKER_NUMERICS requires FDBC_DEMO_VOICE_ADAPTER=1" >&2
        exit 1
    fi
    DEMO_DEPLOY_CONFIG=$(mktemp "${TMPDIR:-/tmp}/fd-demo-voice-XXXXXX.yaml")
    VOICE_PATCH_ARGS+=(--talker-numerics-deploy "$DEMO_NUMERICS" "$DEPLOY_CONFIG" "$DEMO_DEPLOY_CONFIG")
    DEPLOY_CONFIG="$DEMO_DEPLOY_CONFIG"
elif [[ "$DEMO_NUMERICS" != "off" ]]; then
    echo "FDBC_DEMO_TALKER_NUMERICS must be off, native or invariant" >&2
    exit 1
fi
if [[ "${FDBC_DEMO_VOICE_ADAPTER:-0}" == "1" ]]; then
    python "$ROOT_DIR/scripts/patch_omni_demo_voice.py" "${VOICE_PATCH_ARGS[@]}"
fi
DEMO_CHUNKS="${FDBC_DEMO_CODEC_CHUNKS:-off}"
if [[ "$DEMO_CHUNKS" != "off" ]]; then
    if [[ "${FDBC_DEMO_VOICE_ADAPTER:-0}" != "1" ]]; then
        echo "FDBC_DEMO_CODEC_CHUNKS requires FDBC_DEMO_VOICE_ADAPTER=1" >&2
        exit 1
    fi
    DEMO_STREAM_CONFIG=$(mktemp "${TMPDIR:-/tmp}/fd-demo-stream-XXXXXX.yaml")
    python "$ROOT_DIR/scripts/demo_stream_config.py" "$DEPLOY_CONFIG" "$DEMO_STREAM_CONFIG" "$DEMO_CHUNKS"
    DEPLOY_CONFIG="$DEMO_STREAM_CONFIG"
fi

ARGS=(
    vllm serve "$MODEL_DIR"
    --omni
    --deploy-config "$DEPLOY_CONFIG"
    --served-model-name "$SERVED_MODEL_NAME"
    --host "$HOST"
    --port "$PORT"
    --dtype bfloat16
    --allowed-local-media-path /
    --tensor-parallel-size "$TP_SIZE"
    --scheduling-policy "$SCHEDULING_POLICY"
)

if [[ -n "$MAX_MODEL_LEN" ]]; then
    ARGS+=(--max-model-len "$MAX_MODEL_LEN")
fi
if [[ -n "$GPU_MEMORY_UTILIZATION" ]]; then
    ARGS+=(--gpu-memory-utilization "$GPU_MEMORY_UTILIZATION")
fi

exec "${ARGS[@]}" \
    ${QWEN_EXTRA_ARGS:-}
