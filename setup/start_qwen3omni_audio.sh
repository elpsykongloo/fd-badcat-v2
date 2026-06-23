#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(dirname "$SCRIPT_DIR")

export QWEN_DEPLOY_CONFIG="${QWEN_DEPLOY_CONFIG:-$ROOT_DIR/configs/qwen3_omni_audio_single_gpu.yaml}"
# Leave these empty by default so the per-stage values in the deploy YAML are
# not overwritten by global CLI options.
export QWEN_GPU_MEMORY_UTILIZATION="${QWEN_GPU_MEMORY_UTILIZATION:-}"
export QWEN_MAX_MODEL_LEN="${QWEN_MAX_MODEL_LEN:-}"
export FDBC_QWEN_MODEL="${FDBC_QWEN_MODEL:-Qwen3-Omni-30B-A3B-Instruct}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

exec "$SCRIPT_DIR/start_qwen3omni_vllm_omni.sh"
