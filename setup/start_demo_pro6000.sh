#!/usr/bin/env bash
# Scoped PRO 6000 profile. Run in tmux; Ctrl+C owns and stops its children.
set -euo pipefail
DEMO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="/root/miniconda3/bin:$PATH"
export QWEN_DEPLOY_CONFIG="${QWEN_DEPLOY_CONFIG:-$DEMO_ROOT/configs/qwen3_omni_audio_pro6000.yaml}"
if [[ "${FDBC_DEMO_MOE_TUNING:-1}" == "1" ]]; then
    export VLLM_TUNED_CONFIG_FOLDER="${VLLM_TUNED_CONFIG_FOLDER:-$DEMO_ROOT/configs/moe_pro6000_bf16}"
    for width in 768 384; do
        file="$VLLM_TUNED_CONFIG_FOLDER/E=128,N=$width,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition.json"
        [[ -s "$file" ]] || { echo "Missing validated MoE configuration: $file" >&2; exit 1; }
    done
else
    unset VLLM_TUNED_CONFIG_FOLDER
fi
exec bash "$DEMO_ROOT/setup/start_demo.sh" "$@"
