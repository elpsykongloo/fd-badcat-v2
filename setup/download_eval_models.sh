#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${FDBC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_NAME="${ENV_NAME:-fd-sds}"

if [[ -z "${OMP_NUM_THREADS:-}" || "${OMP_NUM_THREADS:-}" == "0" ]]; then
  export OMP_NUM_THREADS=8
fi
if [[ -z "${MKL_NUM_THREADS:-}" || "${MKL_NUM_THREADS:-}" == "0" ]]; then
  export MKL_NUM_THREADS=8
fi
export MODELSCOPE_DOWNLOAD_PARALLELS="${MODELSCOPE_DOWNLOAD_PARALLELS:-16}"
export HF_HOME="${HF_HOME:-$ROOT_DIR/.cache/huggingface}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$ROOT_DIR/.cache/modelscope}"
export FDBC_FUNASR_MODEL="${FDBC_FUNASR_MODEL:-paraformer-zh}"
export FDBC_FUNASR_MODEL_REVISION="${FDBC_FUNASR_MODEL_REVISION:-v2.0.4}"
export FDBC_FUNASR_VAD_MODEL="${FDBC_FUNASR_VAD_MODEL:-fsmn-vad}"
export FDBC_FUNASR_VAD_MODEL_REVISION="${FDBC_FUNASR_VAD_MODEL_REVISION:-v2.0.4}"
export FDBC_PARAKEET_MODEL="${FDBC_PARAKEET_MODEL:-nvidia/parakeet-tdt-0.6b-v2}"
export FDBC_PARAKEET_DIR="${FDBC_PARAKEET_DIR:-$ROOT_DIR/model/parakeet-tdt-0.6b-v2}"
export FDBC_PARAKEET_NEMO="${FDBC_PARAKEET_NEMO:-$FDBC_PARAKEET_DIR/parakeet-tdt-0.6b-v2.nemo}"

if [[ "${USE_NETWORK_TURBO:-1}" == "1" && -f /etc/network_turbo ]]; then
  source /etc/network_turbo
fi

source /root/miniconda3/etc/profile.d/conda.sh

conda activate "$ENV_NAME"
mkdir -p "$FDBC_PARAKEET_DIR" "$HF_HOME" "$MODELSCOPE_CACHE"

echo "Downloading Parakeet NeMo model to: $FDBC_PARAKEET_NEMO"
if [[ -s "$FDBC_PARAKEET_NEMO" ]]; then
  echo "Parakeet model already exists: $FDBC_PARAKEET_NEMO"
elif command -v aria2c >/dev/null 2>&1; then
  unset all_proxy ALL_PROXY
  aria2c -c \
    -x "${PARAKEET_ARIA2_X:-16}" \
    -s "${PARAKEET_ARIA2_S:-16}" \
    -j 1 \
    --file-allocation=none \
    --summary-interval=30 \
    --max-tries=20 \
    --retry-wait=5 \
    -d "$FDBC_PARAKEET_DIR" \
    -o "parakeet-tdt-0.6b-v2.nemo" \
    "https://huggingface.co/$FDBC_PARAKEET_MODEL/resolve/main/parakeet-tdt-0.6b-v2.nemo"
else
  hf download "$FDBC_PARAKEET_MODEL" \
    parakeet-tdt-0.6b-v2.nemo \
    --local-dir "$FDBC_PARAKEET_DIR" \
    --max-workers "${HF_MAX_WORKERS:-8}"
fi

echo "Preloading FunASR models into MODELSCOPE_CACHE=$MODELSCOPE_CACHE"
python - <<'PY'
import os
from funasr import AutoModel

model = AutoModel(
    model=os.getenv("FDBC_FUNASR_MODEL", "paraformer-zh"),
    model_revision=os.getenv("FDBC_FUNASR_MODEL_REVISION", "v2.0.4"),
    vad_model=os.getenv("FDBC_FUNASR_VAD_MODEL", "fsmn-vad"),
    vad_model_revision=os.getenv("FDBC_FUNASR_VAD_MODEL_REVISION", "v2.0.4"),
    disable_update=True,
)
print("FunASR model loaded:", type(model).__name__)
PY

echo "Evaluation models are ready."
