#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-fd-sds}"
PYPI_INDEX_URL="${PYPI_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple}"
PYPI_TRUSTED_HOST="${PYPI_TRUSTED_HOST:-mirrors.aliyun.com}"

if [[ -z "${OMP_NUM_THREADS:-}" || "${OMP_NUM_THREADS:-}" == "0" ]]; then
  export OMP_NUM_THREADS=8
fi
if [[ -z "${MKL_NUM_THREADS:-}" || "${MKL_NUM_THREADS:-}" == "0" ]]; then
  export MKL_NUM_THREADS=8
fi
export PIP_NO_CACHE_DIR="${PIP_NO_CACHE_DIR:-1}"

source /root/miniconda3/etc/profile.d/conda.sh

conda activate "$ENV_NAME"

python -m pip install --no-cache-dir \
  openai \
  huggingface_hub \
  funasr==1.3.11 \
  "nemo_toolkit[asr]==2.7.3" \
  -i "$PYPI_INDEX_URL" \
  --trusted-host "$PYPI_TRUSTED_HOST"

python - <<'PY'
import torch
import torchaudio
import funasr
import nemo.collections.asr as nemo_asr
import openai

print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("torchaudio", torchaudio.__version__)
print("funasr", getattr(funasr, "__version__", "unknown"))
print("nemo_asr", nemo_asr.__name__)
print("openai", openai.__version__)
PY
